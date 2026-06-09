#!/usr/bin/env python3
"""Multi-model evaluation runner for ABKT.

Runs eval_ppl.py across multiple models and collects results into a
single JSON with combined Pareto plots.

Usage:
    # Create a models config (tools/models.yaml):
    #   models:
    #     - name: opt-2.7b
    #       path: /ssd/models/opt-2.7b-safetensors
    #     - name: qwen2.5-3b
    #       path: /ssd/models/qwen2.5-3b

    python tools/run_multi_model.py --config tools/models.yaml
    python tools/run_multi_model.py --config tools/models.yaml --max-windows 5
    python tools/run_multi_model.py --config tools/models.yaml --strategy abkt uniform_int4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import List


def run_eval(
    model_name: str,
    model_path: str,
    strategies: List[str],
    max_windows: int = None,
    budget_ratio: float = 0.5,
    context_len: int = 2048,
    stride: int = 512,
) -> List[dict]:
    """Run eval_ppl.py for a single model and return results."""
    script = Path(__file__).parent / "eval_ppl.py"
    cmd = [
        sys.executable, str(script),
        "--model", model_path,
        "--strategy", *strategies,
        "--context-len", str(context_len),
        "--stride", str(stride),
        "--budget-ratio", str(budget_ratio),
    ]
    if max_windows:
        cmd.extend(["--max-windows", str(max_windows)])

    output_file = f"results_{model_name}.json"
    cmd.extend(["--output", output_file])

    print(f"\n{'=' * 60}")
    print(f"  Running: {model_name}")
    print(f"  Model: {model_path}")
    print(f"  Strategies: {', '.join(strategies)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: eval_ppl.py failed for {model_name} (rc={result.returncode})")
        return []

    if os.path.exists(output_file):
        with open(output_file) as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(description="ABKT Multi-Model Evaluation")
    parser.add_argument("--config", required=True, help="YAML config with model list")
    parser.add_argument("--strategy", nargs="+", default=None,
                        help="Strategies to evaluate (default: all)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Limit windows per model")
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--context-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--output", default="results_multi_model.json",
                        help="Combined output JSON")
    parser.add_argument("--plot", default="pareto_multi_model.png",
                        help="Combined Pareto plot")
    args = parser.parse_args()

    # Load model config
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML required. Install with: pip install pyyaml")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    models = config.get("models", [])
    if not models:
        print("No models found in config.")
        sys.exit(1)

    all_results = []
    for model_cfg in models:
        name = model_cfg["name"]
        path = model_cfg["path"]
        results = run_eval(
            name, path,
            strategies=args.strategy or [
                "fp16", "abkt", "uniform_int8", "uniform_int4", "uniform_int2", "random",
            ],
            max_windows=args.max_windows,
            budget_ratio=args.budget_ratio,
            context_len=args.context_len,
            stride=args.stride,
        )
        for r in results:
            r["model"] = name
        all_results.extend(results)

    # Save combined results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {args.output}")

    # Generate combined Pareto plot
    if len(all_results) >= 2:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 7))
            model_names = sorted(set(r["model"] for r in all_results))
            markers = ["o", "s", "^", "D", "v", "P"]
            colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22"]

            for mi, model_name in enumerate(model_names):
                model_results = [r for r in all_results if r["model"] == model_name]
                for r in model_results:
                    ax.scatter(
                        r["compression_ratio"], r["ppl"],
                        s=120, c=colors[mi % len(colors)],
                        marker=markers[mi % len(markers)],
                        edgecolors="black", linewidths=0.5, zorder=5,
                    )
                    ax.annotate(
                        f"{r['strategy']}",
                        (r["compression_ratio"], r["ppl"]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=7, color=colors[mi % len(colors)],
                    )

                # Per-model Pareto frontier
                sorted_r = sorted(model_results, key=lambda p: p["compression_ratio"])
                frontier = []
                best_ppl = float("inf")
                for p in sorted_r:
                    if p["ppl"] < best_ppl:
                        frontier.append(p)
                        best_ppl = p["ppl"]
                if len(frontier) >= 2:
                    fx = [p["compression_ratio"] for p in frontier]
                    fy = [p["ppl"] for p in frontier]
                    ax.plot(fx, fy, "--", color=colors[mi % len(colors)],
                            alpha=0.5, linewidth=1.5, label=model_name)

            ax.set_xlabel("Compression Ratio (higher is better)", fontsize=12)
            ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
            ax.set_title("ABKT Multi-Model: Quality vs Compression", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.plot, dpi=150, bbox_inches="tight")
            print(f"Combined plot saved to {args.plot}")
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")


if __name__ == "__main__":
    main()
