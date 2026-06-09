#!/usr/bin/env python3
"""Network scenario test runner for ABKT.

Runs multiple simulated network scenarios and produces comparison metrics.
Tests that the ABKT pipeline adapts correctly to different network conditions.

Usage:
    python tools/test_network_scenarios.py --model /ssd/models/opt-2.7b-safetensors
    python tools/test_network_scenarios.py --model /ssd/models/opt-2.7b-safetensors --max-windows 3
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.simulate_network import SCENARIOS, simulate, main as sim_main


def run_all_scenarios(model_path: str, max_windows: int = 3,
                      budget_ratio: float = 0.5, device: str = "cuda") -> list:
    """Run all scenarios and collect results."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(device).eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    all_text = "\n\n".join(ex["text"] for ex in ds if ex["text"].strip())
    all_ids = tokenizer.encode(all_text)

    windows = []
    for i in range(0, len(all_ids) - 2048 + 1, 512):
        windows.append(all_ids[i:i + 2048])
    if max_windows:
        windows = windows[:max_windows]

    results = []
    for scenario_name in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"  Scenario: {scenario_name}")
        print(f"{'=' * 60}")
        try:
            result = simulate(model, windows, scenario_name,
                              budget_ratio=budget_ratio, device=device)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "scenario": scenario_name,
                "error": str(e),
                "num_windows": 0,
            })
        torch.cuda.empty_cache()

    return results


def validate_results(results: list) -> dict:
    """Validate scenario results against expected behavior."""
    validations = {}

    for r in results:
        name = r.get("scenario", "unknown")
        checks = []

        if r.get("error"):
            checks.append(("no_error", False, r["error"]))
        else:
            checks.append(("no_error", True, ""))

            # Stable high: should have high compression, no timeouts
            if name == "stable_high":
                checks.append(("high_compression",
                               r["avg_compression"] >= 1.5,
                               f"avg_compression={r['avg_compression']:.2f}"))
                checks.append(("no_timeouts",
                               r["timeout_count"] == 0,
                               f"timeouts={r['timeout_count']}"))

            # Stable low: should have low compression, high bits
            if name == "stable_low":
                checks.append(("low_compression",
                               r["avg_compression"] <= 5.0,
                               f"avg_compression={r['avg_compression']:.2f}"))

            # Mid drop: should have state transitions
            if name == "mid_drop":
                checks.append(("state_transition",
                               r["state_transitions"] >= 1,
                               f"transitions={r['state_transitions']}"))

            # Oscillate: should have multiple state transitions
            if name == "oscillate":
                checks.append(("multiple_transitions",
                               r["state_transitions"] >= 2,
                               f"transitions={r['state_transitions']}"))

        validations[name] = checks

    return validations


def main():
    parser = argparse.ArgumentParser(description="ABKT Network Scenario Tests")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--output", default="results_scenarios.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = run_all_scenarios(
        args.model,
        max_windows=args.max_windows,
        budget_ratio=args.budget_ratio,
        device=args.device,
    )

    # Validate
    validations = validate_results(results)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"{'Scenario':<20} {'Compress':>10} {'Bits':>8} {'Transitions':>12} {'Timeouts':>10} {'Pass':>6}")
    print("-" * 70)
    for r in results:
        name = r.get("scenario", "?")
        if r.get("error"):
            print(f"{name:<20} {'ERROR':>10} {'':>8} {'':>12} {'':>10} {'FAIL':>6}")
        else:
            checks = validations.get(name, [])
            passed = all(c[1] for c in checks)
            print(f"{name:<20} {r['avg_compression']:>9.2f}x {r['avg_bits']:>7.1f} "
                  f"{r['state_transitions']:>12} {r['timeout_count']:>10} "
                  f"{'PASS' if passed else 'FAIL':>6}")
    print(f"{'=' * 70}")

    # Detail per scenario
    for name, checks in validations.items():
        failed = [c for c in checks if not c[1]]
        if failed:
            print(f"\n  {name} FAILURES:")
            for _, ok, detail in failed:
                print(f"    - {detail}")

    # Save
    output = {
        "results": results,
        "validations": {k: [(c[0], c[1], c[2]) for c in v]
                        for k, v in validations.items()},
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Exit code
    all_passed = all(all(c[1] for c in checks) for checks in validations.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
