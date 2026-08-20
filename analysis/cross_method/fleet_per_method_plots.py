#!/usr/bin/env python3
"""Per-method fleet plots: scatter, tradeoff, and accuracy vs total wallclock.

Visually encodes regularizer weight per data point (marker shape) and
other hyperparameters (color by decay/gamma, size by UL window).

Usage:
    python -m analysis.cross_method.fleet_per_method_plots \
        --method hetlora --task rte --exp-dir exp_distributed \
        --output-dir 0_results/second_mainExperiment_artifacts/rte/hetlora
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from analysis.cross_method.fleet_cross_method_final import parse_eval_metric, TASK_METRIC
from analysis.sweep.sweep_scatter import parse_system_summary

METHOD_DIR_MAP = {
    "hetlora": "hetlora",
    "adasparse_lorav2": "adasparse_lorav2",
    "adasparse_lorav3": "adasparse_lorav3",
    "fahqlora": "fahqlora",
    "fedit": "fedit",
}

DISPLAY_NAMES = {
    "hetlora": "HetLoRA",
    "adasparse_lorav2": "AdaSparse-LoRA v2",
    "adasparse_lorav3": "AdaSparse-LoRA v3",
    "fahqlora": "FAH-QLoRA",
    "fedit": "FedIT",
}

LEGACY_SST2_DIRS = {
    "fedit": "exp_golden/archive/distributed_fedit_r64",
    "hetlora": "exp_golden/archive/distributed_hetlora",
    "fahqlora": "exp_golden/archive/distributed_fah_qlora",
    "adasparse_lorav2": "exp_golden/archive/distributed_adasparse_lorav2",
    "adasparse_lorav3": "exp_golden/archive/distributed_adasparse_lorav3",
}

# Shape: more polygon edges = higher value
# rw: 3 values → triangle(3), square(4), pentagon(5)
RW_SHAPES = {
    "1e-2": ("v", "rw=0.01"),
    "5e-3": ("^", "rw=0.005"),
    "5e-2": ("s", "rw=0.05"),
    "1e-1": ("p", "rw=0.1"),
}

# init_rank: 2 values → triangle(3), square(4)
INITR_SHAPES = {
    "16": ("v", "init_rank=16"),
    "32": ("^", "init_rank=32"),
    "64": ("s", "init_rank=64"),
}

# Shade: darker = more aggressive pruning / higher variation
GAMMA_DECAY_SHADES = {
    "0.5":  "#a8d8ea",
    "0.50": "#a8d8ea",
    "0.6":  "#57a0d3",
    "0.60": "#57a0d3",
    "0.65": "#2b7bba",
    "0.8":  "#0a3055",
    "0.80": "#0a3055",
    "0.95": "#041a30",
}

LAMBDA_SHADES = {
    "1":  "#a8d8ea",
    "5":  "#2b7bba",
    "10": "#0a3055",
}

MARKER_SIZE = 170

UL_FILL_STYLE = {
    "230": {"facecolor": "color", "hatch": None, "label": "UL=230 (filled)"},
    "460": {"facecolor": "none",  "hatch": None, "label": "UL=460 (hollow)"},
    "690": {"facecolor": "none",  "hatch": "///", "label": "UL=690 (hatched)"},
    "870": {"facecolor": "none",  "hatch": "xxx", "label": "UL=870 (cross-hatched)"},
}

HP_PATTERNS = {
    "hetlora": r"regularizer_([0-9e.\-]+)__decay_([0-9.]+)",
    "adasparse_lorav2": r"regularizer_([0-9e.\-]+)__gamma_([0-9.]+)__ul_(\d+)__dl_(\d+)",
    "adasparse_lorav3": r"regularizer_([0-9e.\-]+)__gamma_([0-9.]+)__ul_(\d+)__dl_(\d+)",
    "fahqlora": r"initr_([0-9]+)__lambda_([0-9e.\-]+)",
}


def parse_hps(dirname, method):
    pattern = HP_PATTERNS.get(method)
    if not pattern:
        return {}
    m = re.search(pattern, dirname)
    if not m:
        return {}
    groups = m.groups()
    if method == "hetlora":
        return {"rw": groups[0], "decay": groups[1]}
    elif method in ("adasparse_lorav2", "adasparse_lorav3"):
        return {"rw": groups[0], "gamma": groups[1], "ul": groups[2], "dl": groups[3]}
    elif method == "fahqlora":
        return {"initr": groups[0], "lambda": groups[1]}
    return {}


def collect_method_data(method, task, exp_dir, fedit_golden_only=False):
    method_dir_name = METHOD_DIR_MAP.get(method, method)
    search_dirs = [
        os.path.join(exp_dir, "golden", method_dir_name),
        os.path.join(exp_dir, method_dir_name),
    ]
    if task == "sst2" and method in LEGACY_SST2_DIRS:
        search_dirs.append(LEGACY_SST2_DIRS[method])
    metric_key = TASK_METRIC.get(task, "val_acc")
    prefix = f"{task}__strategy_"
    legacy_prefix = "strategy_"

    # FedIT runs at rank 64 and is rescaled to rank 200; its non-golden reruns
    # make the operating point drift between regenerations. When requested, pin
    # FedIT to the golden run only for a stable, table-consistent point.
    if method == "fedit" and fedit_golden_only:
        golden_dir = os.path.join(exp_dir, "golden", method_dir_name)
        has_golden = os.path.isdir(golden_dir) and any(
            e.startswith(prefix) for e in os.listdir(golden_dir))
        if has_golden:
            search_dirs = [golden_dir]
            non_golden = []
        else:
            # No golden FedIT for this task (e.g. SST-2): fall back to non-golden.
            non_golden = [p for p in search_dirs if "golden" not in p]
    else:
        non_golden = [p for p in search_dirs if "golden" not in p]
    has_valid_non_golden = False
    for p in non_golden:
        if not os.path.isdir(p):
            continue
        for e in os.listdir(p):
            if e.startswith(prefix) and os.path.isdir(os.path.join(p, e)):
                if parse_eval_metric(os.path.join(p, e, "eval_results.log"), metric_key) is not None:
                    has_valid_non_golden = True
                    break
        if has_valid_non_golden:
            break
    if has_valid_non_golden:
        search_dirs = non_golden

    seen_configs = {}
    for parent in search_dirs:
        if not os.path.isdir(parent):
            continue
        is_legacy = "exp_golden/archive" in parent
        for entry in sorted(os.listdir(parent)):
            full = os.path.join(parent, entry)
            if not os.path.isdir(full):
                continue
            if is_legacy:
                if not entry.startswith(legacy_prefix):
                    continue
            else:
                if not entry.startswith(prefix):
                    continue
            config_key = re.sub(r'__\d{14}$', '', entry)
            seen_configs[config_key] = (parent, entry)

    records = []
    for config_key, (parent, entry) in sorted(seen_configs.items()):
        full = os.path.join(parent, entry)
        acc = parse_eval_metric(os.path.join(full, "eval_results.log"), metric_key)
        if acc is None:
            continue
        metrics = parse_system_summary(os.path.join(full, "system_metrics.log"))
        if metrics is None:
            continue
        hps = parse_hps(entry, method)
        records.append({"val_acc": acc, "dirname": entry, **metrics, **hps})

    return records


def get_visual_props(record, method):
    """Return (marker, facecolor, edgecolor, size, hatch) for a data point.

    v2/v3: color=rw, shape=gamma, fill/hatch=UL.
    HetLoRA: color=rw, shape=decay (no UL dimension).
    FAH-QLoRA/FedIT: legacy encoding.
    """
    if method in ("adasparse_lorav2", "adasparse_lorav3", "hetlora"):
        rw = record.get("rw", "")
        marker = RW_SHAPES.get(rw, ("D", f"rw={rw}"))[0]

        dg = record.get("gamma") or record.get("decay") or ""
        color = GAMMA_DECAY_SHADES.get(dg, "#888888")

        ul = record.get("ul", "")
        style = UL_FILL_STYLE.get(ul, {"facecolor": "color", "hatch": None})
        facecolor = color if style["facecolor"] == "color" else "none"
        hatch = style.get("hatch")

        return marker, facecolor, color, MARKER_SIZE, hatch
    elif method == "fahqlora":
        initr = record.get("initr", "")
        marker = INITR_SHAPES.get(initr, ("D", f"ir={initr}"))[0]

        lam = record.get("lambda", "")
        color = LAMBDA_SHADES.get(lam, "#888888")

        return marker, color, color, MARKER_SIZE, None
    else:
        return "D", "#888888", "#888888", MARKER_SIZE, None


SHADE_OVERRIDES = {
    ("hetlora", "decay", "0.65"): {"rw": "5e-2"},
}


def mark_best_per_shade(records, method, tolerance=0.01):
    """Mark the best-accuracy experiment per gamma/decay/lambda value.
    Within tolerance of best accuracy, prefer lowest total wallclock.
    SHADE_OVERRIDES can force a specific hp for a (method, key, value) triple."""
    if method in ("adasparse_lorav2", "adasparse_lorav3", "hetlora"):
        key = "gamma" if "gamma" in records[0] else "decay"
    elif method == "fahqlora":
        key = "lambda"
    else:
        for r in records:
            r["is_shade_best"] = True
        return

    groups = {}
    for r in records:
        v = r.get(key, "")
        groups.setdefault(v, []).append(r)

    for r in records:
        r["is_shade_best"] = False

    for v, group in groups.items():
        override = SHADE_OVERRIDES.get((method, key, v))
        candidates = group
        if override:
            filtered = [r for r in group if all(r.get(k) == val for k, val in override.items())]
            if filtered:
                candidates = filtered
        best_acc = max(r["val_acc"] for r in candidates)
        within = [r for r in candidates if abs(r["val_acc"] - best_acc) <= tolerance]
        best = min(within, key=lambda r: (
            r["computing_wallclock"] + r["upload_wallclock"] + r["download_wallclock"]))
        best["is_shade_best"] = True


def _scatter_point(ax, x, y, marker, facecolor, edgecolor, size, highlighted,
                   hatch=None):
    lw = 2.5 if highlighted else 1.5
    s = size * 1.3 if highlighted else size
    ec = "black" if highlighted else edgecolor
    ax.scatter(x, y, c=facecolor, marker=marker, s=s,
               edgecolors=ec, linewidth=lw,
               zorder=4 if highlighted else 3, alpha=0.9,
               **({"hatch": hatch} if hatch else {}))


def plot_scatter(records, output_path, task, method, metric_label,
                 y_window=0.20, y_pad=0.01, no_title=False):
    configs = [
        ("total_communicated_gb", "Total Communicated (GB)",
         lambda r: r["total_communicated_mb"] / 1024),
        ("comm_wallclock", "Communication Wallclock (h)",
         lambda r: (r["upload_wallclock"] + r["download_wallclock"]) / 60),
        ("computing_wallclock", "Computing Wallclock (h)",
         lambda r: r["computing_wallclock"] / 60),
    ]

    display = DISPLAY_NAMES.get(method, method)
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    if not no_title:
        fig.suptitle(f"Fleet Sweep [{task.upper()}] — {display}",
                     fontsize=14, fontweight="bold", y=1.02)

    for ax, (_, x_label, x_fn) in zip(axes, configs):
        for r in records:
            marker, facecolor, edgecolor, size, hatch = get_visual_props(r, method)
            _scatter_point(ax, x_fn(r), r["val_acc"],
                           marker, facecolor, edgecolor, size, r.get("is_shade_best", False),
                           hatch=hatch)
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(metric_label, fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.locator_params(nbins=6)
        all_ys = [r["val_acc"] for r in records]
        if all_ys:
            yt = max(all_ys) + y_pad
            yb = yt - y_window
            if min(all_ys) - y_pad < yb:
                yb = min(all_ys) - y_pad
                yt = yb + max(y_window, max(all_ys) - min(all_ys) + 2 * y_pad)
            ax.set_ylim(yb, yt)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    _add_legend(axes[0], records, method)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tradeoff(records, output_path, task, method, metric_label,
                  no_title=False):
    display = DISPLAY_NAMES.get(method, method)
    fig, ax = plt.subplots(figsize=(10, 7))
    if not no_title:
        ax.set_title(f"Fleet Sweep [{task.upper()}] — {display} Computation vs. Communication",
                     fontsize=14, fontweight="bold")

    for r in records:
        x_val = r["computing_wallclock"] / 60
        y_val = (r["upload_wallclock"] + r["download_wallclock"]) / 60
        marker, facecolor, edgecolor, size, hatch = get_visual_props(r, method)
        _scatter_point(ax, x_val, y_val,
                       marker, facecolor, edgecolor, size, r.get("is_shade_best", False),
                       hatch=hatch)
        ax.annotate(f"{r['val_acc']:.3f}", (x_val, y_val), textcoords="offset points",
                    xytext=(5, 5), fontsize=14, alpha=0.85)

    _add_legend(ax, records, method)
    ax.set_xlabel("Computation Wallclock (h)", fontsize=20)
    ax.set_ylabel("Communication Wallclock (h)", fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)
    ax.locator_params(nbins=6)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_acc_vs_wallclock(records, output_path, task, method, metric_label,
                          y_window=0.20, y_pad=0.01, no_title=False):
    display = DISPLAY_NAMES.get(method, method)
    fig, ax = plt.subplots(figsize=(10, 7))
    if not no_title:
        ax.set_title(f"Fleet Sweep [{task.upper()}] — {display} {metric_label} vs. Total Wallclock",
                     fontsize=14, fontweight="bold")

    for r in records:
        total_wc = (r["computing_wallclock"] + r["upload_wallclock"]
                    + r["download_wallclock"]) / 60
        marker, facecolor, edgecolor, size, hatch = get_visual_props(r, method)
        _scatter_point(ax, total_wc, r["val_acc"],
                       marker, facecolor, edgecolor, size, r.get("is_shade_best", False),
                       hatch=hatch)

    _add_legend(ax, records, method)
    ax.set_xlabel("Total Wallclock (compute + comm, hours)", fontsize=20)
    ax.set_ylabel(metric_label, fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)
    ax.locator_params(nbins=6)
    all_ys = [r["val_acc"] for r in records]
    if all_ys:
        yt = max(all_ys) + y_pad
        yb = yt - y_window
        if min(all_ys) - y_pad < yb:
            yb = min(all_ys) - y_pad
            yt = yb + max(y_window, max(all_ys) - min(all_ys) + 2 * y_pad)
        ax.set_ylim(yb, yt)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _add_legend(ax, records, method):
    handles = []

    if method in ("adasparse_lorav2", "adasparse_lorav3", "hetlora"):
        # Shape = rw
        rw_values = sorted(set(r.get("rw", "") for r in records))
        for rw in rw_values:
            shape_info = RW_SHAPES.get(rw, ("D", f"rw={rw}"))
            h = plt.Line2D([0], [0], marker=shape_info[0], color="w",
                            markerfacecolor="#888888", markeredgecolor="black",
                            markersize=12, label=shape_info[1])
            handles.append(h)

        # Shade = gamma/decay (light → dark)
        dg_key = "decay" if "decay" in records[0] else "gamma" if "gamma" in records[0] else None
        if dg_key:
            # The pruning-intensity hyperparameter is named "gamma" internally but is
            # published as eta (η) in the method; show η in the legend for v2/v3.
            dg_label = r"$\eta$" if dg_key == "gamma" else dg_key
            dg_values = sorted(set(r.get(dg_key, "") for r in records))
            for dg in dg_values:
                shade = GAMMA_DECAY_SHADES.get(dg, "#888888")
                h = plt.Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=shade, markeredgecolor=shade,
                                markersize=12, label=f"{dg_label}={dg}")
                handles.append(h)

        # Fill = UL (only for v2/v3)
        ul_values = sorted(set(r.get("ul", "") for r in records if r.get("ul")))
        if ul_values:
            mid_shade = "#2185c5"
            for ul in ul_values:
                style = UL_FILL_STYLE.get(ul, {"facecolor": "color", "hatch": None,
                                                "label": f"UL={ul}"})
                fc = mid_shade if style["facecolor"] == "color" else "white"
                h = mpatches.Patch(facecolor=fc, edgecolor=mid_shade,
                                   linewidth=1.5, hatch=style.get("hatch"),
                                   label=style["label"])
                handles.append(h)
    elif method == "fahqlora":
        # Shape = init_rank
        initr_values = sorted(set(r.get("initr", "") for r in records))
        for ir in initr_values:
            shape_info = INITR_SHAPES.get(ir, ("D", f"ir={ir}"))
            h = plt.Line2D([0], [0], marker=shape_info[0], color="w",
                            markerfacecolor="#888888", markeredgecolor="black",
                            markersize=12, label=shape_info[1])
            handles.append(h)

        # Shade = lambda (light → dark)
        lam_values = sorted(set(r.get("lambda", "") for r in records), key=lambda x: int(x) if x.isdigit() else 0)
        for lam in lam_values:
            shade = LAMBDA_SHADES.get(lam, "#888888")
            h = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=shade, markeredgecolor=shade,
                            markersize=12, label=f"λ={lam}")
            handles.append(h)

    if handles:
        ax.legend(handles=handles, loc="best", fontsize=16, framealpha=0.9,
                  handletextpad=0.3, ncol=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                        choices=list(METHOD_DIR_MAP.keys()))
    parser.add_argument("--task", required=True)
    parser.add_argument("--exp-dir", default="exp_distributed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-only", action="store_true",
                        help="Plot only selected (best-per-shade) experiments, no highlighting")
    parser.add_argument("--y-window", type=float, default=0.20,
                        help="Fixed y-axis range size for accuracy plots (default: 0.20)")
    parser.add_argument("--y-pad", type=float, default=0.01,
                        help="Padding above max / below min within the window (default: 0.01)")
    parser.add_argument("--display-name", default=None,
                        help="Override the display name for this method in plot titles/legends")
    parser.add_argument("--exclude-shade", nargs=2, action="append", default=[],
                        metavar=("KEY", "VALUE"),
                        help="Exclude records matching key/value (e.g. decay 0.95)")
    parser.add_argument("--override-selection", nargs=1, action="append", default=[],
                        metavar="DIRNAME_PATTERN",
                        help="Force-select a record by dirname substring, deselecting others in its shade")
    parser.add_argument("--add-selection", nargs=1, action="append", default=[],
                        metavar="DIRNAME_PATTERN",
                        help="Add a record to the selection without deselecting others")
    parser.add_argument("--no-title", action="store_true",
                        help="Hide plot titles")
    parser.add_argument("--fedit-golden-only", action="store_true",
                        help="Pin FedIT to its golden run only (stable, table-consistent)")
    args = parser.parse_args()

    task = args.task.lower()
    method = args.method
    if args.display_name:
        DISPLAY_NAMES[method] = args.display_name
    metric_key = TASK_METRIC.get(task, "val_acc")
    metric_labels = {
        "val_f1": "Final Val F1",
        "val_acc": "Final Val Accuracy",
        "val_pearson": "Final Val Pearson",
        "val_mcc": "Final Val MCC",
    }
    metric_label = metric_labels.get(metric_key, f"Final {metric_key}")

    print(f"Collecting {DISPLAY_NAMES.get(method, method)} experiments for {task.upper()}...")
    records = collect_method_data(method, task, args.exp_dir,
                                  fedit_golden_only=args.fedit_golden_only)
    if not records:
        print(f"No valid experiments found for {method}/{task}", file=sys.stderr)
        return 1
    print(f"  Found {len(records)} experiments")

    for key, val in args.exclude_shade:
        before = len(records)
        records = [r for r in records if f"{key}_{val}" not in r["dirname"]]
        if len(records) < before:
            print(f"  Excluded {before - len(records)} records matching {key}_{val}")

    mark_best_per_shade(records, method)

    shade_key = None
    if method in ("adasparse_lorav2", "adasparse_lorav3", "hetlora"):
        shade_key = "gamma" if records and "gamma" in records[0] else "decay"
    elif method == "fahqlora":
        shade_key = "lambda"

    for (pattern,) in args.override_selection:
        match = [r for r in records if pattern in r["dirname"]]
        if not match:
            print(f"  WARNING: no record matching '{pattern}'")
            continue
        forced = match[0]
        if shade_key:
            shade_val = forced.get(shade_key, "")
            for r in records:
                if r.get(shade_key, "") == shade_val:
                    r["is_shade_best"] = False
        forced["is_shade_best"] = True
        print(f"  Override: selected '{pattern}' (acc={forced['val_acc']:.4f})")

    for (pattern,) in args.add_selection:
        match = [r for r in records if pattern in r["dirname"]]
        if not match:
            print(f"  WARNING: no record matching '{pattern}'")
            continue
        match[0]["is_shade_best"] = True
        print(f"  Added: '{pattern}' (acc={match[0]['val_acc']:.4f})")

    if args.selection_only:
        records = [r for r in records if r.get("is_shade_best")]
        if not records:
            print(f"No selected experiments for {method}/{task}", file=sys.stderr)
            return 1
        for r in records:
            r["is_shade_best"] = False

    os.makedirs(args.output_dir, exist_ok=True)
    plot_scatter(records, os.path.join(args.output_dir, "fleet_scatter.png"),
                 task, method, metric_label,
                 y_window=args.y_window, y_pad=args.y_pad,
                 no_title=args.no_title)
    plot_tradeoff(records, os.path.join(args.output_dir, "fleet_tradeoff.png"),
                  task, method, metric_label,
                  no_title=args.no_title)
    plot_acc_vs_wallclock(records,
                         os.path.join(args.output_dir, "accuracy_vs_summed_wallclock.png"),
                         task, method, metric_label,
                         y_window=args.y_window, y_pad=args.y_pad,
                         no_title=args.no_title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
