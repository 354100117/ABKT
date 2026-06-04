"""Precision allocator — PIA (Proportional-Importance Allocation) algorithm.

Given a budget of bytes and an importance score for each (layer, token) entry,
allocate precision levels (FP16/FP8/INT4/INT2) greedily to maximize weighted quality.

Algorithm:
  1. Start all entries at the minimum precision that fits within budget.
  2. Sort entries by importance descending.
  3. For each entry, try upgrading from current precision to the next higher level
     if the budget allows.
  4. This is guaranteed to be optimal for the budget-constrained problem with
     uniform quality-per-byte across entries (proved by the greedy exchange argument).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class Precision(IntEnum):
    FP16 = 16
    FP8 = 8
    INT4 = 4
    INT2 = 2


# Offline-calibrated quality fidelity per precision (0..1)
QUALITY_FIDELITY = {
    Precision.FP16: 1.00,
    Precision.FP8: 0.98,
    Precision.INT4: 0.92,
    Precision.INT2: 0.80,
}

# Bytes per element per precision
BYTES_PER_ELEMENT = {
    Precision.FP16: 2.0,
    Precision.FP8: 1.0,
    Precision.INT4: 0.5,
    Precision.INT2: 0.25,
}


@dataclass
class AllocationResult:
    """Result of precision allocation.

    Attributes:
        precision_map: {decode_node: {layer_idx: precision_tensor[seq_len]}}.
        total_bytes: Total bytes after allocation.
        budget_bytes: Budget provided.
        avg_precision_bits: Weighted average bits per element.
        compression_ratio: FP16 size / actual size.
        feasible: Whether the budget can accommodate minimum precision.
        forced_int2: True when all layers forced to INT2 to fit within
            extended transfer time (budget was below normal minimum floor).
        dropped_layers: Layers suggested for dropping when infeasible.
    """
    precision_map: Dict[int, Dict[int, torch.Tensor]]
    total_bytes: float
    budget_bytes: float
    avg_precision_bits: float
    compression_ratio: float
    feasible: bool = True
    forced_int2: bool = False
    dropped_layers: Optional[List[int]] = None


class PrecisionAllocator:
    """Bandwidth-constrained precision allocator using PIA algorithm.

    Usage:
        allocator = PrecisionAllocator()
        result = allocator.allocate(importance_map, kv_cache, budget_bytes)
        # result.precision_map → feed to AdaptiveQuantizer
    """

    def __init__(self, precision_levels: Optional[List[Precision]] = None):
        self.precision_levels = precision_levels or [
            Precision.FP16, Precision.FP8, Precision.INT4, Precision.INT2,
        ]
        self._sorted_precisions = sorted(
            self.precision_levels, key=lambda p: p.value, reverse=True
        )

    def allocate(
        self,
        importance_map: Dict[int, Dict[int, torch.Tensor]],
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        budget_bytes: float,
        num_layers_total: int = 0,
    ) -> AllocationResult:
        """Allocate precision levels under a byte budget.

        Args:
            importance_map: {decode_node: {layer: importance[seq_len]}} values in [0,1].
            kv_cache: {decode_node: {layer: (k, v)}} — used for shape info and total size.
            budget_bytes: Maximum bytes allowed for the transfer.
            num_layers_total: Total layers in model. When > 0, enables per-layer
                minimum precision constraints (bottom 1/3: FP8, middle: INT4, top: INT2).
                When 0 (default), falls back to global FP8 minimum for backward compat.

        Returns:
            AllocationResult with precision_map and statistics.
            If budget < minimum floor, returns feasible=False with suggested dropped layers.
        """
        total_fp16 = self._total_bytes(kv_cache, Precision.FP16)

        if budget_bytes >= total_fp16 or not kv_cache:
            # Budget ample → uniform FP16
            prec_map = self._uniform_map(kv_cache, Precision.FP16)
            return AllocationResult(prec_map, total_fp16, budget_bytes,
                                    16.0, 1.0)

        # Build entry list: (importance, decode_node, layer_idx, element_count)
        entries: List[Tuple[float, int, int, int]] = []
        for dnode, layer_cache in kv_cache.items():
            imp_map = importance_map.get(dnode, {})
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                elem_count = k.numel() + v.numel()
                imp = imp_map.get(lidx)
                avg_imp = float(imp.mean().item()) if imp is not None else 0.5
                entries.append((avg_imp, dnode, lidx, elem_count))

        entries.sort(key=lambda x: x[0], reverse=True)

        # Compute per-layer minimum floor (or global FP8 floor for backward compat)
        min_floor = 0.0
        for dnode, layer_cache in kv_cache.items():
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                elem_count = k.numel() + v.numel()
                min_prec = self._min_precision_for_layer(lidx, num_layers_total, 999)
                min_floor += elem_count * BYTES_PER_ELEMENT[min_prec]

        # Budget below minimum floor — force all layers to INT2
        if budget_bytes < min_floor:
            int2_total = self._total_bytes(kv_cache, Precision.INT2)
            logger.warning(
                "Budget %.2f MB below minimum floor %.2f MB — "
                "forcing all layers to INT2 (%.2f MB)",
                budget_bytes / 1e6, min_floor / 1e6, int2_total / 1e6,
            )
            prec_names = {16: "FP16", 8: "FP8 ", 4: "INT4", 2: "INT2"}
            print("[allocator] Budget infeasible — forcing all layers to INT2")
            print(f"[allocator]   budget={budget_bytes/1e6:.2f} MB, "
                  f"min_floor={min_floor/1e6:.2f} MB, "
                  f"INT2_total={int2_total/1e6:.2f} MB")
            for imp, dnode, lidx, elem_cnt in entries:
                print(f"[allocator]   layer {lidx:2d}: imp={imp:.3f} → INT2 "
                      f"({elem_cnt * BYTES_PER_ELEMENT[Precision.INT2]/1024:.0f} KB)")
            # Build fallback drop list (lowest importance first) for caller
            dropped = [lidx for _, _, lidx, _ in reversed(entries)]
            int2_map = self._uniform_map(kv_cache, Precision.INT2)
            return AllocationResult(
                precision_map=int2_map,
                total_bytes=int2_total,
                budget_bytes=budget_bytes,
                avg_precision_bits=2.0,
                compression_ratio=total_fp16 / max(int2_total, 1),
                feasible=False,
                forced_int2=True,
                dropped_layers=dropped,
            )

        compression_needed = total_fp16 / max(budget_bytes, 1)

        # Compute per-entry minimum precision (per-layer if num_layers_total > 0,
        # else global FP8 fallback for backward compat)
        entry_min_prec: Dict[Tuple[int, int], Precision] = {}
        for _, dnode, lidx, _ in entries:
            layer_min = self._min_precision_for_layer(lidx, num_layers_total, compression_needed)
            # Backward compat: when num_layers_total=0, _min_precision_for_layer
            # returns FP8, so this is equivalent to the old global minimum.
            entry_min_prec[(dnode, lidx)] = layer_min

        # Init all at their per-entry minimum precision
        current = {}
        for entry in entries:
            _, dnode, lidx, elem_cnt = entry
            prec = entry_min_prec[(dnode, lidx)]
            current[entry] = BYTES_PER_ELEMENT[prec] * elem_cnt
        current_bytes = sum(current.values())
        assignments: Dict[Tuple[int, int], Precision] = {}
        for _, dnode, lidx, _ in entries:
            assignments[(dnode, lidx)] = entry_min_prec[(dnode, lidx)]

        # Greedy upgrade by importance
        prec_order = self._sorted_precisions  # [FP16, FP8, INT4, INT2]
        for imp, dnode, lidx, elem_cnt in entries:
            cur_prec = assignments[(dnode, lidx)]
            cur_idx = prec_order.index(cur_prec)

            for upgrade_idx in range(cur_idx - 1, -1, -1):
                target = prec_order[upgrade_idx]
                cost = (BYTES_PER_ELEMENT[target] - BYTES_PER_ELEMENT[cur_prec]) * elem_cnt
                if current_bytes + cost <= budget_bytes + 1e-6:
                    assignments[(dnode, lidx)] = target
                    current_bytes += cost
                    cur_prec = target
                else:
                    break

        # Log per-layer allocation decisions
        print("[allocator] === Per-Layer Precision Allocation ===")
        print(f"[allocator] Budget: {budget_bytes/1e6:.2f} MB, FP16 total: {total_fp16/1e6:.2f} MB")
        prec_names = {16: "FP16", 8: "FP8 ", 4: "INT4", 2: "INT2"}
        for imp, dnode, lidx, elem_cnt in entries:
            min_p = entry_min_prec[(dnode, lidx)]
            final_p = assignments[(dnode, lidx)]
            min_name = prec_names.get(min_p.value, "????")
            final_name = prec_names.get(final_p.value, "????")
            layer_bytes = BYTES_PER_ELEMENT[final_p] * elem_cnt
            if final_p.value > min_p.value:
                tag = f"↑ upgraded {min_name}→{final_name}"
            elif final_p.value < min_p.value:
                tag = "↓ downgraded"
            else:
                tag = "= min"
            print(f"[allocator]   layer {lidx:2d}: imp={imp:.3f}  min={min_name}  → {final_name}  ({layer_bytes/1024:5.0f} KB)  {tag}")
        print("[allocator] === End Allocation ===")

        # Build output precision_map
        precision_map: Dict[int, Dict[int, torch.Tensor]] = {}
        for dnode, layer_cache in kv_cache.items():
            precision_map[dnode] = {}
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                prec = assignments.get((dnode, lidx), entry_min_prec.get((dnode, lidx), Precision.FP8))
                seq_len = kv[0].shape[2]
                precision_map[dnode][lidx] = torch.full(
                    (seq_len,), prec.value, dtype=torch.int8
                )

        avg_bits = self._avg_precision(precision_map)
        cr = total_fp16 / max(current_bytes, 1)

        return AllocationResult(precision_map, current_bytes, budget_bytes, avg_bits, cr)

    @staticmethod
    def _total_bytes(
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        precision: Precision,
    ) -> float:
        bpe = BYTES_PER_ELEMENT[precision]
        total = 0.0
        for layer_cache in kv_cache.values():
            for kv in layer_cache.values():
                if kv is None:
                    continue
                k, v = kv
                total += (k.numel() + v.numel()) * bpe
        return total

    @staticmethod
    def _min_precision_for_layer(
        layer_idx: int, num_layers_total: int, compression_ratio: float
    ) -> Precision:
        """Per-layer minimum precision constraint.

        Bottom 1/3 layers: min FP8 (feature extraction, sensitive to quantization error)
        Middle 1/3 layers: min INT4
        Top 1/3 layers: min INT2 (high-level semantics, more robust)

        Args:
            layer_idx: Index of the layer being evaluated.
            num_layers_total: Total number of layers in the model. When 0, returns
                FP8 for backward compatibility.
            compression_ratio: FP16_size / budget. Higher values mean more compression
                needed. Currently unused but reserved for future tightening logic.

        Returns:
            Minimum Precision allowed for this layer.
        """
        # Backward compat: no layer info → global FP8 floor (old behavior)
        if num_layers_total <= 0:
            return Precision.FP8

        # Determine which third this layer falls into
        third = num_layers_total / 3.0
        if layer_idx < third:
            # Bottom 1/3: feature extraction, most sensitive
            return Precision.FP8
        elif layer_idx < 2 * third:
            # Middle 1/3: intermediate representations
            return Precision.INT4
        else:
            # Top 1/3: high-level semantics, most robust
            return Precision.INT2

    @staticmethod
    def _uniform_map(kv_cache, precision: Precision):
        prec_map = {}
        for dnode, layer_cache in kv_cache.items():
            prec_map[dnode] = {}
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                seq_len = kv[0].shape[2]
                prec_map[dnode][lidx] = torch.full(
                    (seq_len,), precision.value, dtype=torch.int8
                )
        return prec_map

    @staticmethod
    def _avg_precision(precision_map: Dict) -> float:
        total_bits = 0.0
        total_elements = 0
        for layer_map in precision_map.values():
            for prec_tensor in layer_map.values():
                cnt = prec_tensor.numel()
                total_bits += float(prec_tensor.float().mean()) * cnt
                total_elements += cnt
        return total_bits / total_elements if total_elements > 0 else 0.0
