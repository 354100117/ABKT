"""Adaptive quantization/dequantization for mixed-precision KV Cache.

Supports FP16 (passthrough), FP8 (symmetric int8), INT4 (asymmetric uint4),
and INT2 (asymmetric uint2) with per-group precision within each layer.

The precision_map from PrecisionAllocator determines which precision each
group uses. Each layer is split into NUM_GROUPS along the seq_len dimension;
each group is quantized independently with its own precision and scales.

KIVI-style quantization:
  - Keys: per-channel quantization (dim=3: head_dim) within each group
  - Values: per-token quantization (dim=2: seq_len) within each group
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from backend.precision_allocator import NUM_GROUPS, Precision


class AdaptiveQuantizer:
    """Quantize and dequantize KV cache layers per precision map."""

    def quantize(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        precision_map: Dict[int, Dict[int, torch.Tensor]],
    ) -> Tuple[Dict, Dict]:
        """Quantize KV cache layers according to precision_map.

        precision_map values are tensors of shape [NUM_GROUPS] (per-group
        precision) or [seq_len] (per-token, legacy). When NUM_GROUPS values
        are detected, each group is quantized independently.

        Returns:
            (quantized_kv, metadata) ready for transport.
        """
        quantized = {}
        metadata = {}

        for dnode, layer_cache in kv_cache.items():
            quantized[dnode] = {}
            metadata[dnode] = {}

            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                prec_t = precision_map.get(dnode, {}).get(lidx)

                if prec_t is None:
                    quantized[dnode][lidx] = (k, v)
                    metadata[dnode][lidx] = {"precision": 16}
                    continue

                # Detect per-group vs per-token/legacy
                if len(prec_t) == NUM_GROUPS:
                    # Per-group allocation: use the most common precision
                    # across groups as the layer's uniform precision.
                    # This avoids mixed-dtype concatenation issues while
                    # still benefiting from per-group importance scoring
                    # in the allocator (S3 quality-weighted upgrade).
                    vals = [int(p.item()) for p in prec_t]
                    from collections import Counter
                    most_common_val = Counter(vals).most_common(1)[0][0]
                    prec = Precision(most_common_val) if most_common_val in {16, 8, 4, 2} else Precision.FP16
                    qk, qv, meta = self._quantize_single(k, v, prec)
                    quantized[dnode][lidx] = (qk, qv)
                    metadata[dnode][lidx] = meta
                else:
                    # Legacy per-token tensor — collapse to single precision
                    avg_prec = int(round(float(prec_t.float().mean())))
                    prec = Precision(avg_prec) if avg_prec in {16, 8, 4, 2} else Precision.FP16
                    qk, qv, meta = self._quantize_single(k, v, prec)
                    quantized[dnode][lidx] = (qk, qv)
                    metadata[dnode][lidx] = meta

        return quantized, metadata

    def dequantize(
        self,
        quantized_kv: Dict,
        metadata: Dict,
    ) -> Dict:
        """Dequantize all layers back to FP16 for decode consumption."""
        result = {}
        for dnode, layer_cache in quantized_kv.items():
            result[dnode] = {}
            meta = metadata.get(dnode, {})

            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                m = meta.get(lidx, {})
                prec = m.get("precision", 16)

                if prec == 16:
                    result[dnode][lidx] = kv
                elif "groups" in m:
                    # Per-group dequantization
                    result[dnode][lidx] = self._dequantize_per_group(kv, m)
                else:
                    k, v = kv
                    p = Precision(prec)
                    dk = self._dequantize(k, m.get("scale_k"), m.get("zero_k"), p)
                    dv = self._dequantize(v, m.get("scale_v"), m.get("zero_v"), p)
                    result[dnode][lidx] = (dk, dv)

        return result

    # ── Per-group quantization ──

    def _quantize_per_group(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        prec_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Quantize a layer split into NUM_GROUPS with independent precisions.

        All groups are stored as uint8 to enable concatenation across groups
        with different precisions. FP16 groups reinterpret float16 bytes as
        uint8; FP8 groups cast int8 → uint8; INT4/INT2 are already uint8.
        """
        seq_len = k.shape[2]
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []
        groups_meta: List[dict] = []

        for gi in range(NUM_GROUPS):
            g_start = gi * seq_len // NUM_GROUPS
            g_end = (gi + 1) * seq_len // NUM_GROUPS
            prec_val = int(prec_tensor[gi].item())
            prec = Precision(prec_val) if prec_val in {16, 8, 4, 2} else Precision.FP16

            k_slice = k[:, :, g_start:g_end, :].contiguous()
            v_slice = v[:, :, g_start:g_end, :].contiguous()

            if prec == Precision.FP16:
                # Reinterpret float16 raw bytes as uint8 for uniform dtype
                k_parts.append(k_slice.view(torch.uint8))
                v_parts.append(v_slice.view(torch.uint8))
                groups_meta.append({"precision": 16})
            elif prec == Precision.FP8:
                qk, mk = self._quantize(k_slice, prec)
                qv, mv = self._quantize(v_slice, prec)
                # Cast int8 → uint8 (bit pattern preserved, dequantize handles it)
                k_parts.append(qk.view(torch.uint8))
                v_parts.append(qv.view(torch.uint8))
                groups_meta.append({
                    "precision": 8,
                    "scale_k": mk["scale"], "zero_k": mk["zero_point"],
                    "scale_v": mv["scale"], "zero_v": mv["zero_point"],
                })
            else:
                # INT4/INT2: KIVI-style per-channel K, per-token V (already uint8)
                qk, mk = self._quantize_per_channel(k_slice, prec)
                qv, mv = self._quantize_per_token(v_slice, prec)
                k_parts.append(qk)
                v_parts.append(qv)
                groups_meta.append({
                    "precision": prec_val,
                    "scale_k": mk["scale"], "zero_k": mk["zero_point"],
                    "scale_v": mv["scale"], "zero_v": mv["zero_point"],
                })

        k_cat = torch.cat(k_parts, dim=2)
        v_cat = torch.cat(v_parts, dim=2)
        meta = {
            "precision": max(g["precision"] for g in groups_meta),
            "groups": groups_meta,
        }
        return k_cat, v_cat, meta

    @staticmethod
    def _dequantize_per_group(
        kv: Tuple[torch.Tensor, torch.Tensor],
        meta: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize a per-group quantized layer.

        Data is stored as uint8. FP16 groups need view(float16) to recover
        the original float16 bytes. FP8 groups need view(int8) before
        dequantization. INT4/INT2 groups are already uint8.
        """
        k, v = kv
        groups_meta: List[dict] = meta["groups"]
        seq_len = k.shape[2]
        k_parts: List[torch.Tensor] = []
        v_parts: List[torch.Tensor] = []

        for gi, gm in enumerate(groups_meta):
            g_start = gi * seq_len // NUM_GROUPS
            g_end = (gi + 1) * seq_len // NUM_GROUPS
            prec_val = gm.get("precision", 16)

            k_slice = k[:, :, g_start:g_end, :]
            v_slice = v[:, :, g_start:g_end, :]

            if prec_val == 16:
                # Reinterpret uint8 bytes back to float16
                k_parts.append(k_slice.view(torch.float16))
                v_parts.append(v_slice.view(torch.float16))
            elif prec_val == 8:
                # Reinterpret uint8 back to int8, then dequantize
                p = Precision.FP8
                dk = AdaptiveQuantizer._dequantize(
                    k_slice.view(torch.int8),
                    gm.get("scale_k"), gm.get("zero_k"), p)
                dv = AdaptiveQuantizer._dequantize(
                    v_slice.view(torch.int8),
                    gm.get("scale_v"), gm.get("zero_v"), p)
                k_parts.append(dk)
                v_parts.append(dv)
            else:
                p = Precision(prec_val)
                dk = AdaptiveQuantizer._dequantize(
                    k_slice, gm.get("scale_k"), gm.get("zero_k"), p)
                dv = AdaptiveQuantizer._dequantize(
                    v_slice, gm.get("scale_v"), gm.get("zero_v"), p)
                k_parts.append(dk)
                v_parts.append(dv)

        return torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)

    # ── Single-precision quantization (legacy / FP8) ──

    @staticmethod
    def _quantize_single(
        k: torch.Tensor, v: torch.Tensor, prec: Precision,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Quantize entire layer with a single precision."""
        if prec == Precision.FP16:
            return k, v, {"precision": 16}
        elif prec == Precision.FP8:
            qk, mk = AdaptiveQuantizer._quantize(k, prec)
            qv, mv = AdaptiveQuantizer._quantize(v, prec)
            return qk, qv, {
                "precision": 8,
                "scale_k": mk["scale"], "zero_k": mk["zero_point"],
                "scale_v": mv["scale"], "zero_v": mv["zero_point"],
            }
        else:
            qk, mk = AdaptiveQuantizer._quantize_per_channel(k, prec)
            qv, mv = AdaptiveQuantizer._quantize_per_token(v, prec)
            return qk, qv, {
                "precision": prec.value,
                "scale_k": mk["scale"], "zero_k": mk["zero_point"],
                "scale_v": mv["scale"], "zero_v": mv["zero_point"],
            }

    # ── Internal quantization routines ──

    @staticmethod
    def _quantize(tensor: torch.Tensor, precision: Precision) -> Tuple[torch.Tensor, dict]:
        """Per-layer quantization (one scale/zero per tensor). Used for FP8."""
        f = tensor.float()
        if precision == Precision.FP8:
            scale = f.abs().max() / 127.0
            if scale < 1e-10:
                scale = 1.0
            q = torch.clamp(torch.round(f / scale), -128, 127).to(torch.int8)
            return q, {"scale": scale, "zero_point": 0}

        elif precision == Precision.INT4:
            mn, mx = f.min(), f.max()
            scale = (mx - mn) / 15.0
            if scale < 1e-10:
                scale = 1.0
            zp = torch.round(-mn / scale).clamp(0, 15)
            q = torch.clamp(torch.round(f / scale + zp), 0, 15).to(torch.uint8)
            return q, {"scale": scale, "zero_point": zp}

        elif precision == Precision.INT2:
            mn, mx = f.min(), f.max()
            scale = (mx - mn) / 3.0
            if scale < 1e-10:
                scale = 1.0
            zp = torch.round(-mn / scale).clamp(0, 3)
            q = torch.clamp(torch.round(f / scale + zp), 0, 3).to(torch.uint8)
            return q, {"scale": scale, "zero_point": zp}

        return tensor, {"scale": None, "zero_point": None}

    @staticmethod
    def _quantize_per_channel(tensor: torch.Tensor, precision: Precision) -> Tuple[torch.Tensor, dict]:
        """Per-channel quantization for Keys (dim=3: head_dim).

        K tensors have per-channel outliers — each channel gets its own
        scale/zero_point. This is the KIVI approach for Key quantization.
        """
        f = tensor.float()
        n_levels = {Precision.INT4: 15, Precision.INT2: 3}[precision]
        max_val = {Precision.INT4: 15, Precision.INT2: 3}[precision]

        # min/max per channel: reduce over dims 0,1,2 → shape [1,1,1,head_dim]
        mn = f.amin(dim=(0, 1, 2), keepdim=True)
        mx = f.amax(dim=(0, 1, 2), keepdim=True)
        scale = (mx - mn) / n_levels
        scale = scale.clamp(min=1e-10)
        zp = torch.round(-mn / scale).clamp(0, max_val)
        q = torch.clamp(torch.round(f / scale + zp), 0, max_val).to(torch.uint8)
        return q, {"scale": scale, "zero_point": zp}

    @staticmethod
    def _quantize_per_token(tensor: torch.Tensor, precision: Precision) -> Tuple[torch.Tensor, dict]:
        """Per-token quantization for Values (dim=2: seq_len).

        V tensors have per-token outliers — each token position gets its own
        scale/zero_point. This is the KIVI approach for Value quantization.
        """
        f = tensor.float()
        n_levels = {Precision.INT4: 15, Precision.INT2: 3}[precision]
        max_val = {Precision.INT4: 15, Precision.INT2: 3}[precision]

        # min/max per token: reduce over dims 0,1,3 → shape [1,1,seq_len,1]
        mn = f.amin(dim=(0, 1, 3), keepdim=True)
        mx = f.amax(dim=(0, 1, 3), keepdim=True)
        scale = (mx - mn) / n_levels
        scale = scale.clamp(min=1e-10)
        zp = torch.round(-mn / scale).clamp(0, max_val)
        q = torch.clamp(torch.round(f / scale + zp), 0, max_val).to(torch.uint8)
        return q, {"scale": scale, "zero_point": zp}

    @staticmethod
    def _dequantize(qt: torch.Tensor, scale, zero_point, precision: Precision) -> torch.Tensor:
        if scale is None:
            return qt
        if precision in (Precision.INT4, Precision.INT2):
            return ((qt.float() - zero_point) * scale).half()
        return (qt.float() * scale).half()
