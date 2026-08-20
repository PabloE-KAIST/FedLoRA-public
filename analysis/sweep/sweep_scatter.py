#!/usr/bin/env python3
"""Sweep scatter plot: accuracy vs. cost tradeoffs across hyperparameter combos.

Usage:
    python3 analysis/sweep_scatter.py exp2/hetlora
    python3 analysis/sweep_scatter.py exp2/adasparse_lorav2_fullgrid
"""

import argparse
import ast
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (hp_names, regex_pattern) — order matters: try longer patterns first
# Each method can have multiple patterns (new format with windows, old without)
METHOD_PATTERNS = {
    "hetlora": [
        (("regularizer", "decay"),
         r"regularizer_([0-9e.\-]+)__decay_([0-9.]+)"),
    ],
    "adasparse_lorav2": [
        (("regularizer", "gamma", "ul", "dl"),
         r"regularizer_([0-9e.\-]+)__gamma_([0-9.]+)__ul_(\d+)__dl_(\d+)"),
        (("regularizer", "gamma"),
         r"regularizer_([0-9e.\-]+)__gamma_([0-9.]+)"),
    ],
    "adasparse_lorav3": [
        (("regularizer", "gamma", "ul", "dl"),
         r"regularizer_([0-9e.\-]+)__gamma_([0-9.]+)__ul_(\d+)__dl_(\d+)"),
        (("regularizer", "gamma"),
         r"regularizer_([0-9e.\-]+)__gamma_([0-9.]+)"),
    ],
    "fah_qlora": [
        (("initr", "lambda"),
         r"initr_([0-9]+)__lambda_([0-9e.\-]+)"),
    ],
}

# Which two HP dimensions to use for color and marker encoding
HP_VISUAL = {
    "hetlora":          ("regularizer", "decay"),
    "adasparse_lorav2": ("gamma", "regularizer"),
    "adasparse_lorav3": ("gamma", "regularizer"),
    "fah_qlora":        ("initr", "lambda"),
}

HP_SHORT = {
    "regularizer": "rw", "decay": "d", "gamma": "γ",
    "initr": "r", "lambda": "λ", "ul": "ul", "dl": "dl",
}

DISPLAY_NAMES = {
    "hetlora": "HetLoRA",
    "adasparse_lorav2": "AdaSparse-LoRA v2",
    "adasparse_lorav3": "AdaSparse-LoRA v3",
    "fah_qlora": "FAH-QLoRA",
}

MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<", ">", "p"]


def detect_method(parent_dir):
    name = os.path.basename(os.path.normpath(parent_dir)).lower()
    for prefix in ("distributed_", ""):
        stripped = name.replace(prefix, "", 1) if name.startswith(prefix) else name
        for key in METHOD_PATTERNS:
            if key in stripped:
                return key
    for entry in os.listdir(parent_dir):
        if entry.startswith("strategy_"):
            for key, patterns in METHOD_PATTERNS.items():
                for _, pat in patterns:
                    if re.search(pat, entry):
                        return key
    return None


def parse_hyperparams(dirname, method):
    for hp_names, pattern in METHOD_PATTERNS[method]:
        m = re.search(pattern, dirname)
        if m:
            return {hp_names[i]: m.group(i + 1) for i in range(len(hp_names))}
    return None


def parse_eval_results(filepath):
    if not os.path.isfile(filepath):
        return None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            if obj.get("Round") == "Final":
                try:
                    return obj["Results_raw"]["client_summarized_weighted_avg"]["val_acc"]
                except KeyError:
                    return None
    return None


def parse_system_summary(filepath):
    if not os.path.isfile(filepath):
        return None
    last_line = None
    with open(filepath) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                last_line = stripped
    if not last_line:
        return None
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        return None
    if "fl_endtime_minutes" not in data:
        return None
    try:
        wc = data["wallclock_time_minutes"]
        comm = data["total_communication_megabytes"]
        return {
            "total_communicated_mb": comm["total_communicated_megabytes"],
            "upload_wallclock": wc["total_uploading_time"],
            "download_wallclock": wc["total_downloading_time"],
            "computing_wallclock": wc["total_computing_time"],
        }
    except KeyError:
        return None


def collect_experiments(parent_dir, method):
    raw = []
    for entry in sorted(os.listdir(parent_dir)):
        full = os.path.join(parent_dir, entry)
        if not os.path.isdir(full) or not entry.startswith("strategy_"):
            continue
        hp = parse_hyperparams(entry, method)
        if hp is None:
            print(f"  SKIP (no HP match): {entry}", file=sys.stderr)
            continue
        acc = parse_eval_results(os.path.join(full, "eval_results.log"))
        if acc is None:
            acc = parse_eval_results(os.path.join(full, "eval_results.raw"))
        if acc is None:
            print(f"  SKIP (no Final acc): {entry}", file=sys.stderr)
            continue
        metrics = parse_system_summary(os.path.join(full, "system_metrics.log"))
        if metrics is None:
            print(f"  SKIP (no system summary): {entry}", file=sys.stderr)
            continue
        ts_match = re.search(r"__(\d{14})$", entry)
        ts = ts_match.group(1) if ts_match else "0"
        hp_key = tuple(hp[k] for k in sorted(hp.keys()))
        raw.append({
            "hp": hp,
            "hp_key": hp_key,
            "val_acc": acc,
            "timestamp": ts,
            "dirname": entry,
            **metrics,
        })

    best = {}
    for r in raw:
        key = r["hp_key"]
        if key not in best or r["timestamp"] > best[key]["timestamp"]:
            best[key] = r
    records = sorted(best.values(), key=lambda r: r["hp_key"])
    return records


def make_label(hp, method):
    parts = []
    for k in sorted(hp.keys()):
        parts.append(f"{HP_SHORT.get(k, k)}={hp[k]}")
    return ", ".join(parts)


def plot_sweep_scatter(records, method, output_path):
    n1, n2 = HP_VISUAL[method]

    vals1 = sorted(set(r["hp"][n1] for r in records), key=lambda x: float(x))
    vals2 = sorted(set(r["hp"][n2] for r in records), key=lambda x: float(x))

    colors = plt.cm.tab10.colors
    color_map = {v: colors[i % len(colors)] for i, v in enumerate(vals1)}
    marker_map = {v: MARKERS[i % len(MARKERS)] for i, v in enumerate(vals2)}

    subplot_configs = [
        ("total_communicated_mb", "Total Communicated (MB)"),
        ("comm_wallclock", "Communication Wallclock (min)"),
        ("computing_wallclock", "Computing Wallclock (min)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    display = DISPLAY_NAMES.get(method, method)
    fig.suptitle(f"{display} — Accuracy vs. Cost Tradeoffs", fontsize=14, fontweight="bold", y=1.02)

    for ax, (x_key, x_label) in zip(axes, subplot_configs):
        for r in records:
            c = color_map[r["hp"][n1]]
            m = marker_map[r["hp"][n2]]
            if x_key == "comm_wallclock":
                x_val = r["upload_wallclock"] + r["download_wallclock"]
            else:
                x_val = r[x_key]
            y_val = r["val_acc"]
            ax.scatter(x_val, y_val, c=[c], marker=m, s=120, edgecolors="black", linewidth=0.5, zorder=3)
            label = make_label(r["hp"], method)
            ax.annotate(label, (x_val, y_val), textcoords="offset points",
                        xytext=(5, 5), fontsize=6.5, alpha=0.85)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Final Val Accuracy", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    color_handles = []
    for v in vals1:
        h = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[v],
                        markeredgecolor="black", markersize=10, label=f"{HP_SHORT[n1]}={v}")
        color_handles.append(h)
    marker_handles = []
    for v in vals2:
        h = plt.Line2D([0], [0], marker=marker_map[v], color="w", markerfacecolor="white",
                        markeredgecolor="black", markersize=10, markeredgewidth=1.5,
                        label=f"{HP_SHORT[n2]}={v}")
        marker_handles.append(h)

    all_handles = color_handles + [plt.Line2D([], [], linestyle="none")] + marker_handles
    axes[0].legend(handles=all_handles, loc="lower left",
                   fontsize=8, framealpha=0.9, handletextpad=0.5)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sweep scatter: accuracy vs. cost tradeoffs")
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

    output_path = os.path.join(parent, "sweep_scatter.png")
    plot_sweep_scatter(records, method, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
