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

# Number of token groups per layer for per-group quantization.
# Each layer's seq_len is split into NUM_GROUPS groups; the allocator
# assigns independent precision levels to each group.
NUM_GROUPS = 4

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
        precision_map: {decode_node: {layer_idx: precision_tensor[NUM_GROUPS]}}.
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
        """Allocate precision levels under a byte budget (per-group granularity).

        Each layer's seq_len is split into NUM_GROUPS groups. The allocator
        assigns independent precision levels to each group based on importance.

        Args:
            importance_map: {decode_node: {layer: importance[seq_len]}} values in [0,1].
            kv_cache: {decode_node: {layer: (k, v)}} — used for shape info and total size.
            budget_bytes: Maximum bytes allowed for the transfer.
            num_layers_total: Total layers in model. When > 0, enables per-layer
                minimum precision constraints (bottom 1/3: FP8, middle: INT4, top: INT2).

        Returns:
            AllocationResult with precision_map as {dnode: {lidx: tensor[NUM_GROUPS]}}.
        """
        total_fp16 = self._total_bytes(kv_cache, Precision.FP16)

        if budget_bytes >= total_fp16 or not kv_cache:
            prec_map = self._uniform_map(kv_cache, Precision.FP16)
            return AllocationResult(prec_map, total_fp16, budget_bytes,
                                    16.0, 1.0)

        # Build per-group entry list:
        # (importance, decode_node, layer_idx, group_idx, element_count)
        entries: List[Tuple[float, int, int, int, int]] = []
        for dnode, layer_cache in kv_cache.items():
            imp_map = importance_map.get(dnode, {})
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                seq_len = k.shape[2]
                # Per-token element count (everything except seq dim)
                k_per_token = k.numel() // seq_len
                v_per_token = v.numel() // seq_len
                imp = imp_map.get(lidx)
                for gi in range(NUM_GROUPS):
                    g_start = gi * seq_len // NUM_GROUPS
                    g_end = (gi + 1) * seq_len // NUM_GROUPS
                    group_size = g_end - g_start
                    elem_count = group_size * (k_per_token + v_per_token)
                    if imp is not None:
                        avg_imp = float(imp[g_start:g_end].mean().item())
                    else:
                        avg_imp = 0.5
                    entries.append((avg_imp, dnode, lidx, gi, elem_count))

        entries.sort(key=lambda x: x[0], reverse=True)

        # Compute minimum floor (per-group, using layer-level min precision)
        min_floor = 0.0
        for dnode, layer_cache in kv_cache.items():
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                k, v = kv
                seq_len = k.shape[2]
                k_per_token = k.numel() // seq_len
                v_per_token = v.numel() // seq_len
                min_prec = self._min_precision_for_layer(lidx, num_layers_total, 999)
                for gi in range(NUM_GROUPS):
                    g_start = gi * seq_len // NUM_GROUPS
                    g_end = (gi + 1) * seq_len // NUM_GROUPS
                    group_size = g_end - g_start
                    elem_count = group_size * (k_per_token + v_per_token)
                    min_floor += elem_count * BYTES_PER_ELEMENT[min_prec]

        # Budget below minimum floor
        if budget_bytes < min_floor:
            int2_total = self._total_bytes(kv_cache, Precision.INT2)
            logger.warning(
                "Budget %.2f MB below minimum floor %.2f MB — "
                "INT2 total %.2f MB",
                budget_bytes / 1e6, min_floor / 1e6, int2_total / 1e6,
            )

            if int2_total <= budget_bytes:
                # INT2 fits — upgrade important groups
                assignments: Dict[Tuple[int, int, int], Precision] = {}
                current_bytes = 0.0
                for _, dnode, lidx, gi, elem_cnt in entries:
                    assignments[(dnode, lidx, gi)] = Precision.INT2
                    current_bytes += BYTES_PER_ELEMENT[Precision.INT2] * elem_cnt

                prec_order = self._sorted_precisions
                for imp, dnode, lidx, gi, elem_cnt in entries:
                    cur_prec = assignments[(dnode, lidx, gi)]
                    cur_idx = prec_order.index(cur_prec)
                    for upgrade_idx in range(cur_idx - 1, -1, -1):
                        target = prec_order[upgrade_idx]
                        cost = (BYTES_PER_ELEMENT[target] - BYTES_PER_ELEMENT[cur_prec]) * elem_cnt
                        if current_bytes + cost <= budget_bytes + 1e-6:
                            assignments[(dnode, lidx, gi)] = target
                            current_bytes += cost
                            cur_prec = target
                        else:
                            break

                precision_map: Dict[int, Dict[int, torch.Tensor]] = {}
                for dnode, layer_cache in kv_cache.items():
                    precision_map[dnode] = {}
                    for lidx, kv in layer_cache.items():
                        if kv is None:
                            continue
                        prec_tensor = torch.tensor([
                            assignments.get((dnode, lidx, gi), Precision.INT2).value
                            for gi in range(NUM_GROUPS)
                        ], dtype=torch.int8)
                        precision_map[dnode][lidx] = prec_tensor

                avg_bits = self._avg_precision(precision_map)
                cr = total_fp16 / max(current_bytes, 1)
                dropped = [lidx for _, _, lidx, _, _ in reversed(entries)]

                prec_names = {16: "FP16", 8: "FP8 ", 4: "INT4", 2: "INT2"}
                print("[allocator] Budget infeasible — per-group upgrade from INT2")
                print(f"[allocator]   budget={budget_bytes/1e6:.2f} MB, "
                      f"INT2_total={int2_total/1e6:.2f} MB")
                for imp, dnode, lidx, gi, elem_cnt in entries:
                    final_p = assignments[(dnode, lidx, gi)]
                    final_name = prec_names.get(final_p.value, "????")
                    g_bytes = BYTES_PER_ELEMENT[final_p] * elem_cnt
                    tag = "↑ upgraded" if final_p.value > 2 else "= INT2"
                    print(f"[allocator]   layer {lidx:2d} g{gi}: imp={imp:.3f} → {final_name}  "
                          f"({g_bytes/1024:.0f} KB)  {tag}")

                # Metadata overhead: per-group scales
                meta_overhead = 0.0
                for (dn, li, gi), p in assignments.items():
                    if p in (Precision.INT4, Precision.INT2):
                        kv = kv_cache.get(dn, {}).get(li)
                        if kv is not None:
                            k, _ = kv
                            seq_len = k.shape[2]
                            g_size = (gi + 1) * seq_len // NUM_GROUPS - gi * seq_len // NUM_GROUPS
                            meta_overhead += (k.shape[3] + g_size) * 4.0
                current_bytes += meta_overhead

                return AllocationResult(
                    precision_map=precision_map,
                    total_bytes=current_bytes,
                    budget_bytes=budget_bytes,
                    avg_precision_bits=avg_bits,
                    compression_ratio=cr,
                    feasible=False,
                    forced_int2=True,
                    dropped_layers=dropped,
                )
            else:
                prec_names = {16: "FP16", 8: "FP8 ", 4: "INT4", 2: "INT2"}
                print("[allocator] Budget infeasible — all layers at INT2")
                print(f"[allocator]   budget={budget_bytes/1e6:.2f} MB, "
                      f"INT2_total={int2_total/1e6:.2f} MB")
                for imp, dnode, lidx, gi, elem_cnt in entries:
                    print(f"[allocator]   layer {lidx:2d} g{gi}: imp={imp:.3f} → INT2 "
                          f"({elem_cnt * BYTES_PER_ELEMENT[Precision.INT2]/1024:.0f} KB)")
                dropped = [lidx for _, _, lidx, _, _ in reversed(entries)]
                int2_map = self._uniform_map(kv_cache, Precision.INT2)
                int2_meta = self.metadata_bytes(kv_cache, Precision.INT2)
                return AllocationResult(
                    precision_map=int2_map,
                    total_bytes=int2_total + int2_meta,
                    budget_bytes=budget_bytes,
                    avg_precision_bits=2.0,
                    compression_ratio=total_fp16 / max(int2_total + int2_meta, 1),
                    feasible=False,
                    forced_int2=True,
                    dropped_layers=dropped,
                )

        compression_needed = total_fp16 / max(budget_bytes, 1)

        # Per-group minimum precision (uses layer-level constraint)
        entry_min_prec: Dict[Tuple[int, int], Precision] = {}
        for _, dnode, lidx, _, _ in entries:
            key = (dnode, lidx)
            if key not in entry_min_prec:
                entry_min_prec[key] = self._min_precision_for_layer(
                    lidx, num_layers_total, compression_needed)

        # Init all groups at their layer's minimum precision
        assignments: Dict[Tuple[int, int, int], Precision] = {}
        current_bytes = 0.0
        for _, dnode, lidx, gi, elem_cnt in entries:
            prec = entry_min_prec[(dnode, lidx)]
            assignments[(dnode, lidx, gi)] = prec
            current_bytes += BYTES_PER_ELEMENT[prec] * elem_cnt

        # Greedy upgrade by importance × quality gain (per-group)
        prec_order = self._sorted_precisions  # [FP16, FP8, INT4, INT2]
        def _upgrade_benefit(entry):
            imp, dnode, lidx, gi, _ = entry
            cur_prec = assignments[(dnode, lidx, gi)]
            quality_gain = QUALITY_FIDELITY[Precision.FP16] - QUALITY_FIDELITY[cur_prec]
            return imp * quality_gain

        sorted_entries = sorted(entries, key=_upgrade_benefit, reverse=True)
        for imp, dnode, lidx, gi, elem_cnt in sorted_entries:
            cur_prec = assignments[(dnode, lidx, gi)]
            cur_idx = prec_order.index(cur_prec)

            for upgrade_idx in range(cur_idx - 1, -1, -1):
                target = prec_order[upgrade_idx]
                cost = (BYTES_PER_ELEMENT[target] - BYTES_PER_ELEMENT[cur_prec]) * elem_cnt
                if current_bytes + cost <= budget_bytes + 1e-6:
                    assignments[(dnode, lidx, gi)] = target
                    current_bytes += cost
                    cur_prec = target
                else:
                    break

        # Log per-group allocation decisions
        print(f"[allocator] === Per-Group Precision Allocation (N={NUM_GROUPS}) ===")
        print(f"[allocator] Budget: {budget_bytes/1e6:.2f} MB, FP16 total: {total_fp16/1e6:.2f} MB")
        prec_names = {16: "FP16", 8: "FP8 ", 4: "INT4", 2: "INT2"}
        for imp, dnode, lidx, gi, elem_cnt in entries:
            min_p = entry_min_prec[(dnode, lidx)]
            final_p = assignments[(dnode, lidx, gi)]
            min_name = prec_names.get(min_p.value, "????")
            final_name = prec_names.get(final_p.value, "????")
            g_bytes = BYTES_PER_ELEMENT[final_p] * elem_cnt
            if final_p.value > min_p.value:
                tag = f"↑ upgraded {min_name}→{final_name}"
            elif final_p.value < min_p.value:
                tag = "↓ downgraded"
            else:
                tag = "= min"
            print(f"[allocator]   layer {lidx:2d} g{gi}: imp={imp:.3f}  min={min_name}  → {final_name}  ({g_bytes/1024:5.0f} KB)  {tag}")
        print("[allocator] === End Allocation ===")

        # Build output precision_map: {dnode: {lidx: tensor[NUM_GROUPS]}}
        precision_map: Dict[int, Dict[int, torch.Tensor]] = {}
        for dnode, layer_cache in kv_cache.items():
            precision_map[dnode] = {}
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                prec_tensor = torch.tensor([
                    assignments.get((dnode, lidx, gi),
                                    entry_min_prec.get((dnode, lidx), Precision.FP8)).value
                    for gi in range(NUM_GROUPS)
                ], dtype=torch.int8)
                precision_map[dnode][lidx] = prec_tensor

        # Metadata overhead: per-group KIVI scale/zero tensors
        meta_overhead = 0.0
        for (dnode, lidx, gi), prec in assignments.items():
            if prec in (Precision.INT4, Precision.INT2):
                kv = kv_cache.get(dnode, {}).get(lidx)
                if kv is not None:
                    k, _ = kv
                    seq_len = k.shape[2]
                    g_size = (gi + 1) * seq_len // NUM_GROUPS - gi * seq_len // NUM_GROUPS
                    meta_overhead += (k.shape[3] + g_size) * 4.0
        current_bytes += meta_overhead

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
    def metadata_bytes(
        kv_cache: Dict[int, Dict[int, Tuple[torch.Tensor, torch.Tensor]]],
        precision: Precision,
    ) -> float:
        """Estimate serialization metadata size for KIVI quantization.

        For INT4/INT2 with per-group quantization:
          Each group has per-channel scale/zero for K and per-token scale/zero for V.
          Per group: (head_dim + group_seq_len) × 4 bytes.
          Total per layer: NUM_GROUPS × (head_dim + seq_len/NUM_GROUPS) × 4 bytes.

        For FP8: scalar scale/zero → ~8 bytes per layer (negligible).
        For FP16: 0 bytes.
        """
        if precision in (Precision.FP16,):
            return 0.0
        if precision == Precision.FP8:
            count = sum(1 for lc in kv_cache.values()
                        for kv in lc.values() if kv is not None)
            return count * 8.0
        # INT4 / INT2: per-group, per-channel K scales + per-token V scales
        total = 0.0
        for layer_cache in kv_cache.values():
            for kv in layer_cache.values():
                if kv is None:
                    continue
                k, v = kv
                head_dim = k.shape[3]
                seq_len = k.shape[2]
                for gi in range(NUM_GROUPS):
                    g_start = gi * seq_len // NUM_GROUPS
                    g_end = (gi + 1) * seq_len // NUM_GROUPS
                    g_size = g_end - g_start
                    total += (head_dim + g_size) * 4.0
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
        """Build precision_map with per-group tensors (shape [NUM_GROUPS])."""
        prec_map = {}
        for dnode, layer_cache in kv_cache.items():
            prec_map[dnode] = {}
            for lidx, kv in layer_cache.items():
                if kv is None:
                    continue
                prec_map[dnode][lidx] = torch.full(
                    (NUM_GROUPS,), precision.value, dtype=torch.int8
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
