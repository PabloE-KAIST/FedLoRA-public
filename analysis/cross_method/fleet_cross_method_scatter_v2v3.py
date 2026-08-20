#!/usr/bin/env python3
"""Cross-method scatter plots: AdaSparse-LoRA v2 vs v3 only (gamma=0.5).

Usage:
    python -m analysis.cross_method.fleet_cross_method_scatter_v2v3
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.sweep.sweep_scatter import parse_eval_results, parse_system_summary

FLEET_DIRS = {
    "AdaSparse-LoRA v2":  "exp_golden/archive/distributed_adasparse_lorav2",
    "AdaSparse-LoRA v3":  "exp_golden/archive/distributed_adasparse_lorav3",
}

METHOD_MARKERS = {
    "AdaSparse-LoRA v2":  "o",
    "AdaSparse-LoRA v3":  "P",
}

METHOD_COLORS = {
    "AdaSparse-LoRA v2":  "#ff7f0e",
    "AdaSparse-LoRA v3":  "#9467bd",
}


def collect_all():
    all_records = {}
    for method, parent in FLEET_DIRS.items():
        if not os.path.isdir(parent):
            print(f"  SKIP dir not found: {parent}", file=sys.stderr)
            continue
        records = []
        for entry in sorted(os.listdir(parent)):
            full = os.path.join(parent, entry)
            if not os.path.isdir(full) or not entry.startswith("strategy_"):
                continue
            acc = parse_eval_results(os.path.join(full, "eval_results.log"))
            if acc is None:
                acc = parse_eval_results(os.path.join(full, "eval_results.raw"))
            if acc is None:
                continue
            metrics = parse_system_summary(os.path.join(full, "system_metrics.log"))
            if metrics is None:
                continue
            records.append({"val_acc": acc, "dirname": entry, **metrics})
        if records:
            all_records[method] = records
            print(f"  {method}: {len(records)} experiments")
    return all_records


def plot_cross_method(all_records, output_path, title_suffix=""):
    subplot_configs = [
        ("total_communicated_gb", "Total Communicated (GB)"),
        ("comm_wallclock", "Communication Wallclock (min)"),
        ("computing_wallclock", "Computing Wallclock (min)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Fleet Results — Accuracy vs. Cost by Method{title_suffix}",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (x_key, x_label) in zip(axes, subplot_configs):
        for method, records in all_records.items():
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            xs, ys = [], []
            for r in records:
                if x_key == "comm_wallclock":
                    x_val = r["upload_wallclock"] + r["download_wallclock"]
                elif x_key == "total_communicated_gb":
                    x_val = r["total_communicated_mb"] / 1024
                else:
                    x_val = r[x_key]
                xs.append(x_val)
                ys.append(r["val_acc"])
            ax.scatter(xs, ys, c=color, marker=marker, s=100,
                       edgecolors="black", linewidth=0.5, zorder=3,
                       label=method, alpha=0.85)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Final Val Accuracy", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper right", fontsize=9,
                   framealpha=0.9, handletextpad=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_method_tradeoff(all_records, output_path, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(
        f"Fleet Results — Computation vs. Communication Tradeoff{title_suffix}",
        fontsize=13, fontweight="bold",
    )

    method_accs = {}
    for method, records in all_records.items():
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        accs = []
        for r in records:
            x_val = r["computing_wallclock"]
            y_val = r["upload_wallclock"] + r["download_wallclock"]
            acc = r["val_acc"]
            accs.append(acc)
            ax.scatter(x_val, y_val, c=color, marker=marker, s=100,
                       edgecolors="black", linewidth=0.5, zorder=3, alpha=0.85)
            ax.annotate(f"{acc:.3f}", (x_val, y_val), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, alpha=0.85, fontweight="bold")
        method_accs[method] = accs

    handles = []
    for method in all_records:
        accs = method_accs.get(method, [])
        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            label = f"{method} (acc {mean:.3f}±{std:.3f}, n={len(accs)})"
        else:
            label = method
        h = plt.Line2D([0], [0], marker=METHOD_MARKERS[method], color="w",
                        markerfacecolor=METHOD_COLORS[method],
                        markeredgecolor="black", markersize=10, label=label)
        handles.append(h)
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)

    ax.set_xlabel("Computation Wallclock (min)", fontsize=11)
    ax.set_ylabel("Communication Wallclock (min)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("Collecting fleet experiments (v2 & v3)...")
    all_records = collect_all()
    if not all_records:
        print("ERROR: no valid experiments found", file=sys.stderr)
        return 1

    os.makedirs("0_results", exist_ok=True)
    suffix = " (v2 & v3)"
    plot_cross_method(all_records,
                      "0_results/fleet_cross_method_scatter_v2v3.png",
                      title_suffix=suffix)
    plot_cross_method_tradeoff(all_records,
                               "0_results/fleet_cross_method_tradeoff_v2v3.png",
                               title_suffix=suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
