#!/usr/bin/env python3
"""Long-sequence PPL plotter for ABKT evaluation results.

Reads JSON output from eval_ppl.py --long-seq and generates a line plot
showing PPL vs context length for each strategy.

Usage:
    python tools/plot_longseq.py results_longseq.json
    python tools/plot_longseq.py results_longseq.json --output longseq.png
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_longseq(results: List[dict], output: str, title: str = None):
    """Generate PPL vs context length plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "fp16": "#2ecc71",
        "abkt": "#e74c3c",
        "uniform_int8": "#3498db",
        "uniform_int4": "#9b59b6",
        "uniform_int2": "#e67e22",
        "random": "#95a5a6",
    }

    # Group by strategy
    strategies = {}
    for r in results:
        s = r["strategy"]
        strategies.setdefault(s, []).append(r)

    for strategy, data in strategies.items():
        data.sort(key=lambda x: x.get("context_len", 2048))
        x = [d.get("context_len", 2048) for d in data]
        y = [d["ppl"] for d in data]
        color = colors.get(strategy, "#1abc9c")
        ax.plot(x, y, "o-", color=color, linewidth=2, markersize=8,
                label=strategy, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("ABKT: PPL vs Context Length", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="ABKT Long-Sequence PPL Plotter")
    parser.add_argument("input", help="JSON results from eval_ppl.py --long-seq")
    parser.add_argument("--output", default="longseq_ppl.png", help="Output image")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    if not results:
        print("No results found.")
        sys.exit(1)

    plot_longseq(results, args.output, args.title)


if __name__ == "__main__":
    main()
