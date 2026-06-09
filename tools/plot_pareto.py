#!/usr/bin/env python3
"""Pareto frontier plotter for ABKT evaluation results.

Reads JSON output from eval_ppl.py and generates a scatter plot with
Pareto frontier showing quality (PPL) vs compression ratio tradeoff.

Usage:
    python tools/plot_pareto.py results.json
    python tools/plot_pareto.py results.json --output pareto.png
    python tools/plot_pareto.py results.json --title "OPT-2.7B WikiText-2"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def pareto_frontier(points: List[dict]) -> List[dict]:
    """Find Pareto-optimal points (minimize PPL, maximize compression).

    A point is Pareto-optimal if no other point has both lower PPL
    and higher compression ratio.
    """
    sorted_pts = sorted(points, key=lambda p: p["compression_ratio"])
    frontier = []
    best_ppl = float("inf")
    for p in sorted_pts:
        if p["ppl"] < best_ppl:
            frontier.append(p)
            best_ppl = p["ppl"]
    return frontier


def plot_pareto(results: List[dict], output: str, title: str = None):
    """Generate Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map for strategies
    colors = {
        "fp16": "#2ecc71",
        "abkt": "#e74c3c",
        "uniform_int8": "#3498db",
        "uniform_int4": "#9b59b6",
        "uniform_int2": "#e67e22",
        "random": "#95a5a6",
    }
    ablation_color = "#1abc9c"

    # Plot all points
    for r in results:
        strategy = r["strategy"]
        color = colors.get(strategy, ablation_color)
        marker = "D" if strategy.startswith("abkt_") else "o"
        ax.scatter(r["compression_ratio"], r["ppl"],
                   s=120, c=color, marker=marker, edgecolors="black",
                   linewidths=0.5, zorder=5)
        ax.annotate(strategy, (r["compression_ratio"], r["ppl"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, color=color, fontweight="bold")

    # Plot Pareto frontier
    frontier = pareto_frontier(results)
    if len(frontier) >= 2:
        fx = [p["compression_ratio"] for p in frontier]
        fy = [p["ppl"] for p in frontier]
        ax.plot(fx, fy, "r--", alpha=0.5, linewidth=1.5, label="Pareto frontier")
        ax.legend(fontsize=10)

    ax.set_xlabel("Compression Ratio (higher is better)", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("ABKT KV Cache Compression: Quality vs Compression", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="ABKT Pareto Frontier Plotter")
    parser.add_argument("input", help="JSON results from eval_ppl.py")
    parser.add_argument("--output", default="pareto_frontier.png",
                        help="Output image path")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    if not results:
        print("No results found in input file.")
        sys.exit(1)

    plot_pareto(results, args.output, args.title)


if __name__ == "__main__":
    main()
