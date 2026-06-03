"""Adaptive quantization/dequantization for mixed-precision KV Cache.

Supports FP16 (passthrough), FP8 (symmetric int8), INT4 (asymmetric uint4),
and INT2 (asymmetric uint2) at the per-layer level.

The precision_map from PrecisionAllocator determines which precision
each (layer, token) entry uses. Currently, quantization is applied
uniformly per-layer (using the layer's average precision) to keep
serialization practical.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from backend.precision_allocator import Precision


class AdaptiveQuantizer:
    """Quantize and dequantize KV cache layers per precision map."""

    def quantize(
        self,
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        precision_map: Dict[int, Dict[int, torch.Tensor]],
    ) -> Tuple[Dict, Dict]:
        """Quantize KV cache layers according to precision_map.

        Returns:
            (quantized_kv, metadata) ready for transport.
            quantized_kv: same structure as kv_cache but with quantized tensors.
            metadata: {decode_node: {layer: {precision, scale_k, ...}}}.
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

                avg_prec = int(round(float(prec_t.float().mean())))
                prec = Precision(avg_prec) if avg_prec in {16, 8, 4, 2} else Precision.FP16

                if prec == Precision.FP16:
                    quantized[dnode][lidx] = (k, v)
                    metadata[dnode][lidx] = {"precision": 16}
                else:
                    qk, mk = self._quantize(k, prec)
                    qv, mv = self._quantize(v, prec)
                    quantized[dnode][lidx] = (qk, qv)
                    metadata[dnode][lidx] = {
                        "precision": prec.value,
                        "scale_k": mk["scale"],
                        "zero_k": mk["zero_point"],
                        "scale_v": mv["scale"],
                        "zero_v": mv["zero_point"],
                    }

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
                else:
                    k, v = kv
                    p = Precision(prec)
                    dk = self._dequantize(k, m.get("scale_k"), m.get("zero_k"), p)
                    dv = self._dequantize(v, m.get("scale_v"), m.get("zero_v"), p)
                    result[dnode][lidx] = (dk, dv)

        return result

    # ── Internal quantization routines ──

    @staticmethod
    def _quantize(tensor: torch.Tensor, precision: Precision) -> Tuple[torch.Tensor, dict]:
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
    def _dequantize(qt: torch.Tensor, scale, zero_point, precision: Precision) -> torch.Tensor:
        if scale is None:
            return qt
        if precision in (Precision.INT4, Precision.INT2):
            return ((qt.float() - zero_point) * scale).half()
        return (qt.float() * scale).half()
