#!/usr/bin/env python3
"""Compact tradeoff scatter: computation vs. communication wallclock, annotated with accuracy.

Usage:
    python -m analysis.sweep.sweep_tradeoff exp2/adasparse_lorav2_fullgrid
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.sweep.sweep_scatter import (
    DISPLAY_NAMES, HP_SHORT, HP_VISUAL, MARKERS,
    collect_experiments, detect_method,
)


def plot_tradeoff(records, method, output_path):
    n1, n2 = HP_VISUAL[method]

    vals1 = sorted(set(r["hp"][n1] for r in records), key=lambda x: float(x))
    vals2 = sorted(set(r["hp"][n2] for r in records), key=lambda x: float(x))

    colors = plt.cm.tab10.colors
    color_map = {v: colors[i % len(colors)] for i, v in enumerate(vals1)}
    marker_map = {v: MARKERS[i % len(MARKERS)] for i, v in enumerate(vals2)}

    fig, ax = plt.subplots(figsize=(10, 7))
    display = DISPLAY_NAMES.get(method, method)
    ax.set_title(f"{display} — Computation vs. Communication Tradeoff",
                 fontsize=13, fontweight="bold")

    for r in records:
        c = color_map[r["hp"][n1]]
        m = marker_map[r["hp"][n2]]
        x_val = r["computing_wallclock"]
        y_val = r["upload_wallclock"] + r["download_wallclock"]
        acc = r["val_acc"]
        ax.scatter(x_val, y_val, c=[c], marker=m, s=120,
                   edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(f"{acc:.3f}", (x_val, y_val), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, alpha=0.85, fontweight="bold")

    ax.set_xlabel("Computation Wallclock (min)", fontsize=11)
    ax.set_ylabel("Communication Wallclock (min)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compact tradeoff scatter: computation vs. communication, annotated with accuracy")
    parser.add_argument("experiment_dir", help="Parent directory with strategy_* subdirs")
    args = parser.parse_args()

    parent = args.experiment_dir
    if not os.path.isdir(parent):
        print(f"ERROR: {parent} is not a directory", file=sys.stderr)
        return 1

    method = detect_method(parent)
    if method is None:
        print(f"ERROR: cannot detect method from {parent}", file=sys.stderr)
        return 1
    print(f"Detected method: {DISPLAY_NAMES.get(method, method)}")

    records = collect_experiments(parent, method)
    if not records:
        print("ERROR: no valid experiments found", file=sys.stderr)
        return 1
    print(f"Found {len(records)} experiments")

    output_path = os.path.join(parent, "sweep_tradeoff.png")
    plot_tradeoff(records, method, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
