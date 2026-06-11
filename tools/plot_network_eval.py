#!/usr/bin/env python3
"""Generate comparison plots from network fluctuation evaluation results.

Usage:
    python tools/plot_network_eval.py results_network.json
    python tools/plot_network_eval.py results_network.json --output network_eval.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


STRATEGY_COLORS = {
    "fp16": "#2196F3",
    "uniform_int8": "#4CAF50",
    "uniform_int4": "#FF9800",
    "uniform_int2": "#F44336",
    "abkt": "#9C27B0",
}
STRATEGY_LABELS = {
    "fp16": "FP16 (no compression)",
    "uniform_int8": "Uniform INT8",
    "uniform_int4": "Uniform INT4",
    "uniform_int2": "Uniform INT2",
    "abkt": "ABKT (adaptive)",
}


def plot_ppl_vs_transfer(data: dict, ax: plt.Axes, scenario: str):
    """Scatter plot: x=transfer time, y=PPL for each strategy."""
    results = data[scenario]
    for strategy in ["fp16", "uniform_int8", "uniform_int4", "uniform_int2", "abkt"]:
        if strategy not in results:
            continue
        r = results[strategy]
        color = STRATEGY_COLORS.get(strategy, "gray")
        label = STRATEGY_LABELS.get(strategy, strategy)
        marker = "*" if strategy == "abkt" else "o"
        size = 200 if strategy == "abkt" else 100
        ax.scatter(r["total_transfer_time"], r["ppl"],
                   c=color, marker=marker, s=size, label=label,
                   zorder=5 if strategy == "abkt" else 3,
                   edgecolors="black" if strategy == "abkt" else "none",
                   linewidths=1.5)
    ax.set_xlabel("Total Transfer Time (s)")
    ax.set_ylabel("Perplexity (PPL)")
    ax.set_title(f"Scenario: {scenario}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_compression_timeline(data: dict, ax: plt.Axes, scenario: str):
    """Line plot: compression ratio over virtual time for each strategy."""
    results = data[scenario]
    for strategy in ["uniform_int8", "uniform_int4", "abkt"]:
        if strategy not in results:
            continue
        timeline = results[strategy].get("timeline", [])
        if not timeline:
            continue
        times = [t["virtual_time"] for t in timeline]
        compressions = [t["compression_ratio"] for t in timeline]
        color = STRATEGY_COLORS.get(strategy, "gray")
        label = STRATEGY_LABELS.get(strategy, strategy)
        lw = 2.5 if strategy == "abkt" else 1.5
        ax.plot(times, compressions, color=color, label=label, linewidth=lw)

    # Overlay bandwidth as background
    ref_timeline = None
    for s in results:
        if results[s].get("timeline"):
            ref_timeline = results[s]["timeline"]
            break
    if ref_timeline:
        times = [t["virtual_time"] for t in ref_timeline]
        bw = [t["bandwidth_mbps"] for t in ref_timeline]
        ax2 = ax.twinx()
        ax2.fill_between(times, bw, alpha=0.1, color="gray", label="Bandwidth")
        ax2.set_ylabel("Bandwidth (MB/s)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

    ax.set_xlabel("Virtual Time (s)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title(f"Compression Adaptation: {scenario}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_summary_table(data: dict, ax: plt.Axes, scenario: str):
    """Render results as a table figure."""
    ax.axis("off")
    results = data[scenario]

    strategies = ["fp16", "uniform_int8", "uniform_int4", "uniform_int2", "abkt"]
    rows = []
    for s in strategies:
        if s not in results:
            continue
        r = results[s]
        rows.append([
            STRATEGY_LABELS.get(s, s),
            f"{r['ppl']:.2f}",
            f"{r['total_transfer_time']:.1f}",
            f"{r['avg_compression']:.2f}x",
            f"{r.get('avg_bits', 0):.1f}",
            str(r.get('timeout_count', 0)),
        ])

    col_labels = ["Strategy", "PPL", "Transfer(s)", "Compress", "AvgBits", "Timeouts"]
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Highlight ABKT row
    for i, s in enumerate(strategies):
        if s not in results:
            continue
        if s == "abkt":
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor("#E8D5F5")
        table[i + 1, 0].set_text_props(
            fontweight="bold" if s == "abkt" else "normal"
        )

    ax.set_title(f"Results: {scenario}", fontweight="bold", pad=20)


def main():
    parser = argparse.ArgumentParser(description="Plot ABKT Network Evaluation")
    parser.add_argument("input", help="Results JSON from eval_network.py")
    parser.add_argument("--output", default="network_eval.png",
                        help="Output image path")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    scenarios = list(data.keys())
    n_scenarios = len(scenarios)

    # Layout: 3 columns per scenario (scatter, timeline, table)
    fig = plt.figure(figsize=(7 * 3, 5 * n_scenarios))
    gs = gridspec.GridSpec(n_scenarios, 3, figure=fig, hspace=0.4, wspace=0.3)

    for i, scenario in enumerate(scenarios):
        # Scatter: PPL vs Transfer Time
        ax1 = fig.add_subplot(gs[i, 0])
        plot_ppl_vs_transfer(data, ax1, scenario)

        # Timeline: Compression ratio over time
        ax2 = fig.add_subplot(gs[i, 1])
        plot_compression_timeline(data, ax2, scenario)

        # Table
        ax3 = fig.add_subplot(gs[i, 2])
        plot_summary_table(data, ax3, scenario)

    fig.suptitle("ABKT Network Fluctuation Evaluation", fontsize=16, fontweight="bold", y=1.01)
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
