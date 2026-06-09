#!/usr/bin/env python3
"""Offline ABKT pipeline simulator under simulated network conditions.

Exercises the full decision pipeline (importance scoring, precision allocation,
quantization) under synthetic bandwidth profiles without real network hardware.

Usage:
    python tools/simulate_network.py --model /ssd/models/opt-2.7b-safetensors
    python tools/simulate_network.py --model /ssd/models/opt-2.7b-safetensors --scenario step
    python tools/simulate_network.py --model /ssd/models/opt-2.7b-safetensors --max-windows 3
"""

from __future__ import annotations

import argparse
import json
import time
import sys
import os
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from backend.precision_allocator import PrecisionAllocator, Precision
from backend.adaptive_quant import AdaptiveQuantizer
from backend.token_importance import TokenImportanceEvaluator
from backend.mock_network import (
    MockNetworkProbeClient, NetworkProfile,
    step_scenario, ramp_scenario, oscillate_scenario,
)
from pd_inference.kv_cache import KVCache


SCENARIOS = {
    "stable_high": lambda: NetworkProfile([(0, 100e6, 2.0), (600, 100e6, 2.0)]),
    "stable_low": lambda: NetworkProfile([(0, 10e6, 5.0), (600, 10e6, 5.0)]),
    "mid_drop": lambda: step_scenario(100e6, 15e6, 10.0),
    "gradual": lambda: ramp_scenario(100e6, 10e6, 30.0),
    "oscillate": lambda: oscillate_scenario(80e6, 15e6, 5.0, 10),
}


def simulate(
    model,
    windows: List[List[int]],
    scenario_name: str,
    budget_ratio: float = 0.5,
    device: str = "cuda",
) -> dict:
    """Run ABKT pipeline under simulated network for one scenario."""
    scenario = SCENARIOS[scenario_name]()
    mock = MockNetworkProbeClient(scenario)
    mock.start()

    evaluator = TokenImportanceEvaluator()
    allocator = PrecisionAllocator()
    quantizer = AdaptiveQuantizer()

    timeline = []
    state_transitions = 0
    prev_state = None
    timeout_count = 0

    for win_idx, window_ids in enumerate(windows):
        # Advance virtual time (simulate 2s between transfers)
        mock.advance_time(2.0)

        # Probe network
        mock.probe_now()
        snapshot = mock.get_snapshot()

        # Track state transitions
        if prev_state is not None and snapshot.state != prev_state:
            state_transitions += 1
        prev_state = snapshot.state

        # Extract KV cache
        context_input = torch.tensor([window_ids[:512]], device=device)
        with torch.no_grad():
            context_out = model(context_input, use_cache=True)
        orig_kv = context_out.past_key_values
        kv_cache_obj = KVCache.from_dynamic_cache(orig_kv)
        abkt_kv = kv_cache_obj.to_abkt_dict()
        num_layers = kv_cache_obj.num_layers
        seq_len = kv_cache_obj.seq_len
        del orig_kv, context_out

        # Compute importance
        importance = evaluator.compute(abkt_kv, num_layers, seq_len)

        # Allocate precision
        total_fp16 = PrecisionAllocator._total_bytes(abkt_kv, Precision.FP16)
        budget = min(snapshot.budget_bytes, total_fp16 * budget_ratio)
        result = allocator.allocate(
            importance, abkt_kv, budget,
            num_layers_total=num_layers,
        )

        # Quantize
        quantized_kv, metadata = quantizer.quantize(abkt_kv, result.precision_map)

        # Simulate transfer time
        transfer_time = result.total_bytes / max(snapshot.bandwidth_bps, 1.0)
        if transfer_time > 60.0:
            timeout_count += 1

        # Record transfer for EWMA calibration
        mock.record_transfer(
            int(result.total_bytes), transfer_time, result.compression_ratio
        )

        timeline.append({
            "window": win_idx,
            "virtual_time": mock.virtual_time,
            "state": snapshot.state.value,
            "bandwidth_mbps": snapshot.bandwidth_bps / 1e6,
            "budget_mb": snapshot.budget_bytes / 1e6,
            "compression_ratio": result.compression_ratio,
            "avg_bits": result.avg_precision_bits,
            "total_bytes_mb": result.total_bytes / 1e6,
            "transfer_time_sec": transfer_time,
            "feasible": result.feasible,
        })

        del abkt_kv, quantized_kv, metadata
        torch.cuda.empty_cache()

    return {
        "scenario": scenario_name,
        "num_windows": len(windows),
        "state_transitions": state_transitions,
        "timeout_count": timeout_count,
        "avg_compression": sum(t["compression_ratio"] for t in timeline) / max(len(timeline), 1),
        "avg_bits": sum(t["avg_bits"] for t in timeline) / max(len(timeline), 1),
        "timeline": timeline,
    }


def main():
    parser = argparse.ArgumentParser(description="ABKT Network Simulation")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--scenario", default="mid_drop",
                        choices=list(SCENARIOS.keys()),
                        help="Network scenario to simulate")
    parser.add_argument("--max-windows", type=int, default=5)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Save results as JSON")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    ).to(args.device).eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = "\n\n".join(ex["text"] for ex in ds if ex["text"].strip())
    all_ids = tokenizer.encode(all_text)

    windows = []
    for i in range(0, len(all_ids) - 2048 + 1, 512):
        windows.append(all_ids[i:i + 2048])
    if args.max_windows:
        windows = windows[:args.max_windows]

    print(f"Simulating: {args.scenario} ({len(windows)} windows)")
    result = simulate(model, windows, args.scenario,
                      budget_ratio=args.budget_ratio, device=args.device)

    print(f"\n{'=' * 60}")
    print(f"  Scenario: {result['scenario']}")
    print(f"  State transitions: {result['state_transitions']}")
    print(f"  Timeouts: {result['timeout_count']}")
    print(f"  Avg compression: {result['avg_compression']:.2f}x")
    print(f"  Avg bits: {result['avg_bits']:.1f}")
    print(f"{'=' * 60}")

    for t in result["timeline"]:
        print(f"  win={t['window']} t={t['virtual_time']:.1f}s "
              f"state={t['state']} bw={t['bandwidth_mbps']:.1f}MB/s "
              f"compress={t['compression_ratio']:.2f}x "
              f"bits={t['avg_bits']:.1f} "
              f"xfer={t['transfer_time_sec']:.2f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
