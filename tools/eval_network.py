#!/usr/bin/env python3
"""End-to-end network fluctuation evaluation: ABKT vs fixed-precision strategies.

Combines PPL measurement (from eval_ppl.py) with simulated network transfer
(from simulate_network.py) to show ABKT's advantage under bandwidth fluctuation.

For each network scenario, measures both quality (PPL) and speed (transfer time)
for ABKT and fixed-precision baselines.

Usage:
    # Quick smoke test
    python tools/eval_network.py --model /ssd/models/qwen2.5-3b --max-windows 5

    # Full evaluation, all scenarios
    python tools/eval_network.py --model /ssd/models/qwen2.5-3b --output results_network.json

    # Specific scenarios
    python tools/eval_network.py --model /ssd/models/qwen2.5-3b --scenario oscillate mid_drop
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import os
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from backend.precision_allocator import PrecisionAllocator, Precision
from backend.adaptive_quant import AdaptiveQuantizer
from backend.token_importance import TokenImportanceEvaluator
from backend.mock_network import (
    MockNetworkProbeClient, NetworkProfile,
    step_scenario, ramp_scenario, oscillate_scenario, random_walk_scenario,
)
from pd_inference.kv_cache import KVCache


# ── Network scenarios ──

SCENARIOS = {
    "stable_high": {
        "desc": "Stable high bandwidth (8 MB/s)",
        "factory": lambda: NetworkProfile([(0, 8e6, 2.0), (600, 8e6, 2.0)]),
    },
    "stable_low": {
        "desc": "Stable low bandwidth (2 MB/s)",
        "factory": lambda: NetworkProfile([(0, 2e6, 10.0), (600, 2e6, 10.0)]),
    },
    "mid_drop": {
        "desc": "Sudden bandwidth drop (8→2 MB/s at t=10s)",
        "factory": lambda: step_scenario(8e6, 2e6, 10.0),
    },
    "gradual": {
        "desc": "Gradual bandwidth degradation (8→2 MB/s over 30s)",
        "factory": lambda: ramp_scenario(8e6, 2e6, 30.0),
    },
    "oscillate": {
        "desc": "Bandwidth oscillation (6↔2 MB/s, period 5s)",
        "factory": lambda: oscillate_scenario(6e6, 2e6, 5.0, 10),
    },
}

ALL_STRATEGIES = ["fp16", "abkt", "uniform_int8", "uniform_int4", "uniform_int2"]


def evaluate_scenario(
    model,
    windows: List[List[int]],
    scenario_name: str,
    strategies: List[str],
    budget_ratio: float = 0.5,
    context_len: int = 2048,
    stride: int = 512,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, dict]:
    """Run all strategies under one network scenario, return per-strategy results."""

    profile = SCENARIOS[scenario_name]["factory"]()
    results = {}

    for strategy in strategies:
        # Each strategy gets its own mock client (same profile, independent EWMA state)
        mock = MockNetworkProbeClient(profile)
        mock.start()

        evaluator = TokenImportanceEvaluator()
        allocator = PrecisionAllocator()
        quantizer = AdaptiveQuantizer()

        total_loss = 0.0
        total_tokens = 0
        total_bytes_sum = 0.0
        total_fp16_sum = 0.0
        total_transfer_time = 0.0
        timeout_count = 0
        state_transitions = 0
        prev_state = None
        timeline = []

        t_start = time.time()

        for win_idx, window_ids in enumerate(windows):
            # 1. Advance virtual time and probe network
            mock.advance_time(2.0)
            mock.probe_now()
            snapshot = mock.get_snapshot()

            # Track state transitions
            if prev_state is not None and snapshot.state != prev_state:
                state_transitions += 1
            prev_state = snapshot.state

            # 2. Extract KV cache
            context_input = torch.tensor([window_ids[:stride]], device=device)
            with torch.no_grad():
                context_out = model(context_input, use_cache=True)
            orig_kv = context_out.past_key_values
            kv_cache_obj = KVCache.from_dynamic_cache(orig_kv)
            abkt_kv = kv_cache_obj.to_abkt_dict()
            num_layers = kv_cache_obj.num_layers
            seq_len = kv_cache_obj.seq_len
            del orig_kv, context_out

            # 3. Apply strategy
            # Use raw probe bandwidth for budget (not EWMA-smoothed)
            raw_bw, _ = profile.interpolate(mock.virtual_time)
            raw_budget = raw_bw * 2.0  # TARGET_TRANSFER_TIME = 2s

            if strategy == "fp16":
                result = allocator.allocate_uniform(abkt_kv, Precision.FP16)
            elif strategy == "uniform_int8":
                result = allocator.allocate_uniform(abkt_kv, Precision.INT8)
            elif strategy == "uniform_int4":
                result = allocator.allocate_uniform(abkt_kv, Precision.INT4)
            elif strategy == "uniform_int2":
                result = allocator.allocate_uniform(abkt_kv, Precision.INT2)
            elif strategy == "abkt":
                importance = evaluator.compute(abkt_kv, num_layers, seq_len)
                total_fp16 = PrecisionAllocator._total_bytes(abkt_kv, Precision.FP16)
                budget = min(raw_budget, total_fp16 * budget_ratio)
                result = allocator.allocate(
                    importance, abkt_kv, budget,
                    num_layers_total=num_layers,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # 4. Quantize → dequantize
            if strategy == "fp16":
                reconstructed_kv = abkt_kv
            else:
                quantized_kv, metadata = quantizer.quantize(abkt_kv, result.precision_map)
                reconstructed_kv = quantizer.dequantize(quantized_kv, metadata)

            # 5. Compute PPL on continuation tokens
            recon_obj = KVCache.from_abkt_dict(reconstructed_kv)
            dyn_cache = recon_obj.to_dynamic_cache(device)

            cont_ids = window_ids[stride:context_len]
            cont_input = torch.tensor([cont_ids], device=device)
            position_ids = torch.arange(
                stride, stride + len(cont_ids), device=device
            ).unsqueeze(0)

            with torch.no_grad():
                cont_out = model(
                    cont_input,
                    past_key_values=dyn_cache,
                    position_ids=position_ids,
                )

            logits = cont_out.logits[:, :-1, :].contiguous().float()
            targets = cont_input[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
            total_bytes_sum += result.total_bytes
            total_fp16_sum += PrecisionAllocator._total_bytes(abkt_kv, Precision.FP16)

            # 6. Simulate transfer time (use raw bandwidth)
            transfer_time = result.total_bytes / max(raw_bw, 1.0)
            total_transfer_time += transfer_time
            if transfer_time > 60.0:
                timeout_count += 1

            # 7. Record transfer for EWMA feedback
            mock.record_transfer(
                int(result.total_bytes), transfer_time, result.compression_ratio
            )

            timeline.append({
                "window": win_idx,
                "virtual_time": mock.virtual_time,
                "bandwidth_mbps": raw_bw / 1e6,
                "compression_ratio": result.compression_ratio,
                "avg_bits": result.avg_precision_bits,
                "total_bytes_mb": result.total_bytes / 1e6,
                "transfer_time_sec": transfer_time,
                "running_ppl": math.exp(total_loss / max(total_tokens, 1)),
            })

            # Cleanup
            del dyn_cache, recon_obj, cont_out, logits, targets, abkt_kv, reconstructed_kv
            torch.cuda.empty_cache()

            if (win_idx + 1) % 10 == 0 or win_idx == 0:
                running_ppl = math.exp(total_loss / max(total_tokens, 1))
                elapsed = time.time() - t_start
                print(f"    [{strategy}] window {win_idx + 1}/{len(windows)}: "
                      f"PPL={running_ppl:.4f}  xfer={total_transfer_time:.1f}s  "
                      f"({elapsed:.1f}s elapsed)")

        ppl = math.exp(total_loss / max(total_tokens, 1))
        avg_bytes = total_bytes_sum / len(windows)
        avg_fp16 = total_fp16_sum / len(windows)
        compression_ratio = avg_fp16 / max(avg_bytes, 1)
        avg_bits = 16.0 / max(compression_ratio, 0.01)

        results[strategy] = {
            "strategy": strategy,
            "ppl": ppl,
            "total_transfer_time": total_transfer_time,
            "avg_transfer_time": total_transfer_time / len(windows),
            "avg_compression": compression_ratio,
            "avg_bits": avg_bits,
            "timeout_count": timeout_count,
            "state_transitions": state_transitions,
            "num_windows": len(windows),
            "timeline": timeline,
        }

        print(f"  [{strategy}] DONE: PPL={ppl:.4f}, "
              f"transfer={total_transfer_time:.1f}s, "
              f"compress={results[strategy]['avg_compression']:.2f}x")

    return results


def print_scenario_table(scenario_name: str, results: Dict[str, dict]):
    """Print formatted comparison table for one scenario."""
    info = SCENARIOS[scenario_name]
    print(f"\n{'=' * 80}")
    print(f"  Scenario: {scenario_name} — {info['desc']}")
    print(f"{'=' * 80}")
    header = (f"{'Strategy':<18} {'PPL':>10} {'Transfer(s)':>12} "
              f"{'Compress':>10} {'AvgBits':>8} {'Timeouts':>9}")
    print(header)
    print("-" * len(header))

    for strategy in ["fp16", "uniform_int8", "uniform_int4", "uniform_int2", "abkt"]:
        if strategy not in results:
            continue
        r = results[strategy]
        marker = " *" if strategy == "abkt" else ""
        print(f"{strategy + marker:<18} {r['ppl']:>10.4f} "
              f"{r['total_transfer_time']:>11.1f} "
              f"{r['avg_compression']:>9.2f}x "
              f"{r['avg_bits']:>7.1f} "
              f"{r['timeout_count']:>9}")
    print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="ABKT Network Fluctuation Evaluation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--scenario", nargs="+", default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Network scenarios to test")
    parser.add_argument("--strategy", nargs="+", default=None,
                        help=f"Strategies to evaluate (default: all)")
    parser.add_argument("--budget-ratio", type=float, default=0.5,
                        help="Budget ratio for ABKT (default: 0.5)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit windows for fast testing")
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.scenario is None:
        args.scenario = list(SCENARIOS.keys())
    if args.strategy is None:
        args.strategy = ALL_STRATEGIES

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).to(args.device).eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    print(f"Loading dataset: wikitext/wikitext-2-raw-v1")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = "\n\n".join(ex["text"] for ex in ds if ex["text"].strip())
    all_ids = tokenizer.encode(all_text)

    ctx_len = args.context_len
    stride = args.stride
    windows = []
    for i in range(0, len(all_ids) - ctx_len + 1, stride):
        windows.append(all_ids[i:i + ctx_len])
    if args.max_windows:
        windows = windows[:args.max_windows]
    print(f"Windows: {len(windows)}, context={ctx_len}, stride={stride}")

    all_results = {}

    for scenario_name in args.scenario:
        print(f"\n{'#' * 60}")
        print(f"  Evaluating scenario: {scenario_name}")
        print(f"{'#' * 60}")

        scenario_results = evaluate_scenario(
            model, windows, scenario_name,
            strategies=args.strategy,
            budget_ratio=args.budget_ratio,
            context_len=ctx_len,
            stride=stride,
            seed=args.seed,
            device=args.device,
        )

        print_scenario_table(scenario_name, scenario_results)
        all_results[scenario_name] = scenario_results

    # Save JSON
    if args.output:
        # Strip timeline from saved results to reduce file size
        save_data = {}
        for scen, strats in all_results.items():
            save_data[scen] = {}
            for strat, r in strats.items():
                r_copy = dict(r)
                r_copy["timeline_summary"] = {
                    "num_points": len(r["timeline"]),
                    "bandwidth_range_mbps": (
                        min(t["bandwidth_mbps"] for t in r["timeline"]),
                        max(t["bandwidth_mbps"] for t in r["timeline"]),
                    ),
                    "compression_range": (
                        min(t["compression_ratio"] for t in r["timeline"]),
                        max(t["compression_ratio"] for t in r["timeline"]),
                    ),
                }
                save_data[scen][strat] = r_copy

        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
