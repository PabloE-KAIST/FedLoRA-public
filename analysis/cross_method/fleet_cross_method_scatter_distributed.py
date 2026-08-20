#!/usr/bin/env python3
"""Cross-method scatter + tradeoff plots for distributed fleet experiments.

Reads from exp_distributed/<method>/<task>__strategy_* directories.

Usage:
    python -m analysis.cross_method.fleet_cross_method_scatter_distributed --task rte --output-dir 0_results/rte
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.sweep.sweep_scatter import parse_eval_results, parse_system_summary

METHOD_DIRS = {
    "FedIT":              ["exp_distributed/golden/fedit", "exp_distributed/fedit"],
    "HetLoRA":            ["exp_distributed/golden/hetlora", "exp_distributed/hetlora"],
    "FAH-QLoRA":          ["exp_distributed/golden/fahqlora", "exp_distributed/fahqlora"],
    "AdaSparse-LoRA v2":  ["exp_distributed/golden/adasparse_lorav2", "exp_distributed/adasparse_lorav2"],
    "AdaSparse-LoRA v3":  ["exp_distributed/golden/adasparse_lorav3", "exp_distributed/adasparse_lorav3"],
}

METHOD_MARKERS = {
    "FedIT":              "D",
    "HetLoRA":            "^",
    "FAH-QLoRA":          "s",
    "AdaSparse-LoRA v2":  "o",
    "AdaSparse-LoRA v3":  "P",
}

METHOD_COLORS = {
    "FedIT":              "#d62728",
    "HetLoRA":            "#2ca02c",
    "FAH-QLoRA":          "#1f77b4",
    "AdaSparse-LoRA v2":  "#ff7f0e",
    "AdaSparse-LoRA v3":  "#9467bd",
}

TASK_METRIC = {
    "rte": "val_acc",
    "mrpc": "val_f1",
    "stsb": "val_pearson",
    "cola": "val_mcc",
    "sst2": "val_acc",
    "qnli": "val_acc",
    "qqp": "val_f1",
    "mnli": "val_acc",
}


def parse_eval_metric(filepath, metric_key="val_acc"):
    """Parse the Final round's weighted-avg metric from eval_results.log.

    Falls back to the last server Round N weighted_avg if the Final summary
    has NaN (caused by a now-fixed best-round tracker bug).
    """
    import ast
    import math
    if not os.path.isfile(filepath):
        return None
    last_round_val = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if obj.get("Role", "").startswith("Server") and \
                    isinstance(obj.get("Round"), int):
                v = obj.get("Results_weighted_avg", {}).get(metric_key)
                if v is not None:
                    try:
                        if not math.isnan(v):
                            last_round_val = v
                    except (TypeError, ValueError):
                        last_round_val = v
            if obj.get("Round") == "Final":
                try:
                    v = obj["Results_raw"]["client_summarized_weighted_avg"][metric_key]
                    if v is not None and not math.isnan(v):
                        return v
                except (KeyError, TypeError, ValueError):
                    pass
    return last_round_val


def collect_all(task, metric_key):
    all_records = {}
    for method, parents in METHOD_DIRS.items():
        if isinstance(parents, str):
            parents = [parents]
        records = []
        seen_dirs = set()
        prefix = f"{task}__strategy_"
        for parent in parents:
            if not os.path.isdir(parent):
                continue
            for entry in sorted(os.listdir(parent)):
                full = os.path.join(parent, entry)
                if not os.path.isdir(full) or not entry.startswith(prefix):
                    continue
                if entry in seen_dirs:
                    continue
                seen_dirs.add(entry)
                acc = parse_eval_metric(os.path.join(full, "eval_results.log"), metric_key)
                if acc is None:
                    acc = parse_eval_metric(os.path.join(full, "eval_results.raw"), metric_key)
                if acc is None:
                    print(f"  SKIP (no Final {metric_key}): {entry}", file=sys.stderr)
                    continue
                metrics = parse_system_summary(os.path.join(full, "system_metrics.log"))
                if metrics is None:
                    print(f"  SKIP (no system summary): {entry}", file=sys.stderr)
                    continue
                records.append({
                    "val_acc": acc,
                    "dirname": entry,
                    **metrics,
                })
        if records:
            all_records[method] = records
            print(f"  {method}: {len(records)} experiments")
    return all_records


def plot_cross_method_scatter(all_records, output_path, task, metric_label):
    subplot_configs = [
        ("total_communicated_mb", "Total Communicated (MB)"),
        ("comm_wallclock", "Communication Wallclock (min)"),
        ("computing_wallclock", "Computing Wallclock (min)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"Fleet Results [{task.upper()}] — {metric_label} vs. Cost by Method",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, (x_key, x_label) in zip(axes, subplot_configs):
        for method, records in all_records.items():
            color = METHOD_COLORS[method]
            marker = METHOD_MARKERS[method]
            xs, ys = [], []
            for r in records:
                if x_key == "comm_wallclock":
                    x_val = r["upload_wallclock"] + r["download_wallclock"]
                else:
                    x_val = r[x_key]
                xs.append(x_val)
                ys.append(r["val_acc"])
            ax.scatter(xs, ys, c=color, marker=marker, s=100,
                       edgecolors="black", linewidth=0.5, zorder=3,
                       label=method, alpha=0.85)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="best", fontsize=9,
                   framealpha=0.9, handletextpad=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_method_tradeoff(all_records, output_path, task, metric_label):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title(f"Fleet Results [{task.upper()}] — Computation vs. Communication Tradeoff",
                 fontsize=13, fontweight="bold")

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
            label = f"{method} ({metric_label.split()[-1].lower()} {mean:.3f}±{std:.3f}, n={len(accs)})"
        else:
            label = method
        h = plt.Line2D([0], [0], marker=METHOD_MARKERS[method], color="w",
                        markerfacecolor=METHOD_COLORS[method],
                        markeredgecolor="black", markersize=10,
                        label=label)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="GLUE task name (rte, mrpc, etc.)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: 0_results/<task>)")
    args = parser.parse_args()

    task = args.task.lower()
    metric_key = TASK_METRIC.get(task, "val_acc")
    metric_labels = {
        "val_f1": "Final Val F1",
        "val_acc": "Final Val Accuracy",
        "val_pearson": "Final Val Pearson",
        "val_mcc": "Final Val MCC",
    }
    metric_label = metric_labels.get(metric_key, f"Final {metric_key}")
    output_dir = args.output_dir or f"0_results/{task}"

    print(f"Collecting fleet experiments for {task.upper()} (metric: {metric_key})...")
    all_records = collect_all(task, metric_key)
    if not all_records:
        print("ERROR: no valid experiments found", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    plot_cross_method_scatter(
        all_records,
        os.path.join(output_dir, "fleet_cross_method_scatter.png"),
        task, metric_label,
    )
    plot_cross_method_tradeoff(
        all_records,
        os.path.join(output_dir, "fleet_cross_method_tradeoff.png"),
        task, metric_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
