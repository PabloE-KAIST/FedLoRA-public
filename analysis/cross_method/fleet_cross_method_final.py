#!/usr/bin/env python3
"""Cross-method scatter + tradeoff plots for final thesis figures.

Based on fleet_cross_method_scatter_distributed.py with modifications:
  - Minutes → hours on time axes
  - Axis breaks for communication outliers (HetLoRA with least pruning)
  - FedIT accuracy kept but comm/computation re-estimated at rank=200
  - Reads from both golden/ and method dirs

Usage:
    python -m analysis.cross_method.fleet_cross_method_final --task rte --output-dir 0_results/final/rte
    python -m analysis.cross_method.fleet_cross_method_final --task cola --output-dir 0_results/final/cola
"""

import argparse
import os
import re
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from analysis.sweep.sweep_scatter import parse_eval_results, parse_system_summary

METHOD_DIRS = {
    "FedIT":              ["exp_distributed/golden/fedit", "exp_distributed/fedit"],
    "HetLoRA":            ["exp_distributed/golden/hetlora", "exp_distributed/hetlora"],
    "FAH-QLoRA":          ["exp_distributed/golden/fahqlora", "exp_distributed/fahqlora"],
    "AdaSparse-LoRA v2":  ["exp_distributed/golden/adasparse_lorav2", "exp_distributed/adasparse_lorav2"],
    "AdaSparse-LoRA v3":  ["exp_distributed/golden/adasparse_lorav3", "exp_distributed/adasparse_lorav3"],
}

VALID_RW_VALUES = ["5e-3", "5e-2", "1e-1"]

# Per-task rw filters: only include experiments with the golden rw for each task.
# Tasks not listed here accept any rw. Values can be a string or list of strings.
TASK_RW_FILTER = {
    "rte":  {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "mrpc": {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "stsb": {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "cola": {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "cola2": {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "sst2": {"HetLoRA": VALID_RW_VALUES + ["1e-2"], "AdaSparse-LoRA v2": VALID_RW_VALUES + ["1e-2"], "AdaSparse-LoRA v3": VALID_RW_VALUES + ["1e-2"]},
    "qnli": {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "mnli": {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
    "qqp":  {"HetLoRA": VALID_RW_VALUES, "AdaSparse-LoRA v2": VALID_RW_VALUES, "AdaSparse-LoRA v3": VALID_RW_VALUES},
}

# Task aliases: map variant names to the base GLUE task for directory prefix and metric
TASK_ALIAS = {"cola2": "cola"}

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
    "cola2": "val_mcc",
    "sst2": "val_acc",
    "qnli": "val_acc",
    "mnli": "val_acc",
    "qqp": "val_f1",
}

FEDIT_RANK_SCALE = 200 / 64
# SST2's r=64 compute is artificially low (12 clients, thin partitions).
# Peer-task FedIT r=200 is 2.1-3.3x the most expensive other method;
# SST2 other-method max is ~116min, so apply median ratio (2.65x) ≈ 307min.
FEDIT_COMPUTE_OVERRIDE_MIN = {"sst2": 307.0}

LEGACY_SST2_DIRS = {
    "FedIT":              "exp_golden/archive/distributed_fedit_r64",
    "HetLoRA":            "exp_golden/archive/distributed_hetlora",
    "FAH-QLoRA":          "exp_golden/archive/distributed_fah_qlora",
    "AdaSparse-LoRA v2":  "exp_golden/archive/distributed_adasparse_lorav2",
    "AdaSparse-LoRA v3":  "exp_golden/archive/distributed_adasparse_lorav3",
}


def parse_eval_metric(filepath, metric_key="val_acc"):
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
    dir_task = TASK_ALIAS.get(task, task)
    all_records = {}
    for method, parents in METHOD_DIRS.items():
        if isinstance(parents, str):
            parents = [parents]
        parents = list(parents)
        # Add legacy SST-2 dirs (no task prefix in directory names)
        if dir_task == "sst2" and method in LEGACY_SST2_DIRS:
            parents.append(LEGACY_SST2_DIRS[method])
        records = []
        seen_configs = {}  # config_key -> entry (keep latest by timestamp)
        prefix = f"{dir_task}__strategy_"
        legacy_prefix = "strategy_"
        # Skip golden dirs when non-golden has valid (parseable) data
        non_golden = [p for p in parents if "golden" not in p]
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
            parents = non_golden
        for parent in parents:
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
                # Filter by rw if specified for this task+method
                rw_filter = TASK_RW_FILTER.get(task, {}).get(method)
                if rw_filter:
                    if isinstance(rw_filter, str):
                        rw_filter = [rw_filter]
                    if not any(f"regularizer_{rv}" in entry for rv in rw_filter):
                        continue
                # Deduplicate: strip timestamp (last 14 chars) to get config key
                config_key = re.sub(r'__\d{14}$', '', entry)
                if config_key in seen_configs:
                    pass
                seen_configs[config_key] = (parent, entry)

        for config_key, (parent, entry) in sorted(seen_configs.items()):
                full = os.path.join(parent, entry)
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

    # Re-estimate FedIT at rank=200 using rate-aware scaling
    if "FedIT" in all_records:
        # Find the highest-volume non-FedIT experiment for rate reference
        ref_rate = None
        for method, records in all_records.items():
            if method == "FedIT":
                continue
            for r in records:
                comm_wc = r["upload_wallclock"] + r["download_wallclock"]
                if comm_wc > 0:
                    rate = r["total_communicated_mb"] / comm_wc
                    if ref_rate is None or r["total_communicated_mb"] > ref_rate[0]:
                        ref_rate = (r["total_communicated_mb"], comm_wc, rate)

        for r in all_records["FedIT"]:
            r["total_communicated_mb"] *= FEDIT_RANK_SCALE
            if task in FEDIT_COMPUTE_OVERRIDE_MIN:
                r["computing_wallclock"] = FEDIT_COMPUTE_OVERRIDE_MIN[task]
            else:
                r["computing_wallclock"] *= FEDIT_RANK_SCALE

            # Comm wallclock: use the slower of (linear scaling, reference rate)
            # so FedIT is always >= the highest-comm baseline
            linear_wc = (r["upload_wallclock"] + r["download_wallclock"]) * FEDIT_RANK_SCALE
            if ref_rate:
                rate_based_wc = r["total_communicated_mb"] / ref_rate[2]
                comm_wc = max(linear_wc, rate_based_wc)
            else:
                comm_wc = linear_wc
            # Split proportionally between upload and download
            orig_total = r["upload_wallclock"] + r["download_wallclock"]
            if orig_total > 0:
                r["upload_wallclock"] = comm_wc * (r["upload_wallclock"] / orig_total)
                r["download_wallclock"] = comm_wc * (r["download_wallclock"] / orig_total)
            else:
                r["upload_wallclock"] = comm_wc / 2
                r["download_wallclock"] = comm_wc / 2

        fedit_r = all_records["FedIT"][0]
        fedit_wc = fedit_r["upload_wallclock"] + fedit_r["download_wallclock"]
        print(f"  FedIT: re-estimated at rank=200 (comm={fedit_r['total_communicated_mb']/1024:.1f}GB, "
              f"comm_wc={fedit_wc/60:.1f}h, compute_wc={fedit_r['computing_wallclock']/60:.1f}h)")

    return all_records


def _total_wallclock(r):
    return (r["computing_wallclock"] + r["upload_wallclock"] + r["download_wallclock"]) / 60


CROSS_METHOD_OVERRIDES = {
    ("HetLoRA", "decay", "0.65"): {"hp": "regularizer", "val": "5e-2"},
}


def _select_best_per_shade(records, shade_key, method, tolerance=0.01):
    """Find best experiment per shade group. Within tolerance of best accuracy,
    prefer lowest total wallclock.
    CROSS_METHOD_OVERRIDES can force a specific hp for a (method, key, value) triple."""
    groups = {}
    for r in records:
        m = re.search(rf'{shade_key}_([0-9e.\-]+)', r["dirname"])
        shade_val = m.group(1) if m else ""
        groups.setdefault(shade_val, []).append(r)

    selected = set()
    for shade_val, group in groups.items():
        override = CROSS_METHOD_OVERRIDES.get((method, shade_key, shade_val))
        candidates = group
        if override:
            hp, val = override["hp"], override["val"]
            filtered = [r for r in group if f"{hp}_{val}" in r["dirname"]]
            if filtered:
                candidates = filtered
        best_acc = max(r["val_acc"] for r in candidates)
        within = [r for r in candidates if abs(r["val_acc"] - best_acc) <= tolerance]
        best = min(within, key=lambda r: _total_wallclock(r))
        selected.add(id(best))

    return selected


def select_best_configs(all_records, tolerance=0.01):
    """Mark the best config per shade group for highlighting in cross-method plots.

    Same rule as per-method plots: best accuracy per gamma/decay/lambda shade,
    with wallclock tiebreaker within tolerance.
    """
    for method, records in all_records.items():
        if not records:
            continue

        key = SHADE_KEYS_BY_METHOD.get(method)
        if key:
            selected_ids = _select_best_per_shade(records, key, method, tolerance)
            for r in records:
                r["is_selected"] = (id(r) in selected_ids)
        else:
            # FedIT: single config, always selected
            for r in records:
                r["is_selected"] = True


def find_outlier_threshold(all_records, key_fn, percentile=95):
    """Find a break (break_lo, break_hi) isolating a few high outliers (e.g. the
    full-rank FedIT reference) from the main cluster of points.

    break_lo hugs the top of the main cluster with a small headroom so the main
    panel spends its space on the cluster rather than empty range; break_hi sits
    just above the largest outlier. Returns (None, None) when there is no clear
    outlier group (a single contiguous spread), in which case no break is drawn.
    """
    all_vals = []
    for records in all_records.values():
        for r in records:
            all_vals.append(key_fn(r))
    if len(all_vals) < 3:
        return None, None
    vals = sorted(all_vals)
    max_val = max(vals)
    total_range = max_val - vals[0]
    if total_range == 0:
        return None, None

    # Largest gap between consecutive values; points above it are outlier candidates.
    max_gap, gap_idx = 0.0, -1
    for i in range(len(vals) - 1):
        gap = vals[i + 1] - vals[i]
        if gap > max_gap:
            max_gap, gap_idx = gap, i

    n_above = len(vals) - gap_idx - 1
    # Require the gap to dominate the range and only a small minority above it;
    # otherwise the data is a single spread and a break would mislead.
    if gap_idx < 0 or max_gap <= 0.4 * total_range or n_above > max(1, int(len(vals) * 0.3)):
        return None, None

    # Hug the top of the main cluster: small headroom above the cluster max, so the
    # main panel is filled by the cluster instead of empty space up to the outlier.
    cluster_max = vals[gap_idx]
    cluster_span = cluster_max - vals[0]
    margin = max(0.15 * cluster_span, 0.10 * cluster_max)
    return cluster_max + margin, max_val * 1.05


def make_broken_axis(fig, pos, orientation="x"):
    """Create a pair of axes with a broken axis for outliers.
    Returns (ax_main, ax_break) where ax_break shows the outlier region."""
    if orientation == "x":
        gs = GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05,
                      left=pos[0], right=pos[0]+pos[2],
                      bottom=pos[1], top=pos[1]+pos[3])
        ax_main = fig.add_subplot(gs[0, 0])
        ax_break = fig.add_subplot(gs[0, 1])
        ax_break.yaxis.set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_break.spines['left'].set_visible(False)
        ax_break.tick_params(left=False)
    return ax_main, ax_break


def plot_on_axis_pair(ax_main, ax_break, xs, ys, sels, break_lo, break_hi,
                      color, marker, label, method):
    """Plot data on a broken axis pair, splitting points by threshold.
    Selected points get thick black outline."""
    xs_main, ys_main, sel_main = [], [], []
    xs_break, ys_break, sel_break = [], [], []
    for x, y, s in zip(xs, ys, sels):
        if x >= break_lo:
            xs_break.append(x); ys_break.append(y); sel_break.append(s)
        if x <= break_lo:
            xs_main.append(x); ys_main.append(y); sel_main.append(s)

    for ax, xv, yv, sv in [(ax_main, xs_main, ys_main, sel_main),
                            (ax_break, xs_break, ys_break, sel_break)]:
        if not xv:
            continue
        # Plot non-selected
        xn = [x for x, s in zip(xv, sv) if not s]
        yn = [y for y, s in zip(yv, sv) if not s]
        if xn:
            ax.scatter(xn, yn, c=color, marker=marker, s=150,
                       edgecolors="black", linewidth=0.5, zorder=3,
                       label=label if ax is ax_main else None, alpha=0.85)
        # Plot selected with thick outline
        xs_sel = [x for x, s in zip(xv, sv) if s]
        ys_sel = [y for y, s in zip(yv, sv) if s]
        if xs_sel:
            ax.scatter(xs_sel, ys_sel, c=color, marker=marker, s=240,
                       edgecolors="black", linewidth=2.5, zorder=4,
                       label=label if ax is ax_main and not xn else None, alpha=1.0)


def add_break_marks(ax_main, ax_break):
    """Add diagonal break marks between the two axes."""
    d = 0.015
    kwargs = dict(transform=ax_main.transAxes, color='k', clip_on=False, linewidth=1)
    ax_main.plot((1-d, 1+d), (-d, +d), **kwargs)
    ax_main.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    kwargs.update(transform=ax_break.transAxes)
    ax_break.plot((-d*4, +d*4), (-d, +d), **kwargs)
    ax_break.plot((-d*4, +d*4), (1-d, 1+d), **kwargs)


def _get_x_val(r, x_key):
    if x_key == "comm_wallclock":
        return (r["upload_wallclock"] + r["download_wallclock"]) / 60
    elif x_key == "computing_wallclock":
        return r["computing_wallclock"] / 60
    elif x_key == "total_communicated_gb":
        return r["total_communicated_mb"] / 1024
    return r[x_key]


def _scatter_on_axes(axes, breaks, all_records, x_key):
    """Scatter data across multiple axes separated by break thresholds.
    axes has len(breaks)+1 elements. Segment j covers: breaks[j-1]..breaks[j]."""
    for method, records in all_records.items():
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        for r in records:
            x_val = _get_x_val(r, x_key)
            y_val = r["val_acc"]
            is_sel = r.get("is_selected", False)
            sz = 240 if is_sel else 150
            lw = 2.5 if is_sel else 0.5
            al = 1.0 if is_sel else 0.85
            zz = 4 if is_sel else 3
            for j, ax in enumerate(axes):
                lo = breaks[j - 1] if j > 0 else -float("inf")
                hi = breaks[j] if j < len(breaks) else float("inf")
                if lo <= x_val <= hi:
                    ax.scatter(x_val, y_val, c=color, marker=marker, s=sz,
                               edgecolors="black", linewidth=lw, zorder=zz, alpha=al)


def plot_cross_method_scatter(all_records, output_path, task, metric_label,
                              break_comm_hours=None, break_compute_wc_hours=None,
                              selection_only=False,
                              y_window=0.20, y_pad=0.01, no_title=False,
                              no_legend=False):
    subplot_configs = [
        ("total_communicated_gb", "Total Communicated (GB)"),
        ("comm_wallclock", "Communication Wallclock (h)"),
        ("computing_wallclock", "Computing Wallclock (h)"),
    ]

    # Determine break points per axis
    axis_breaks = {}
    for x_key, _ in subplot_configs:
        key_fn = lambda r, k=x_key: _get_x_val(r, k)
        if x_key == "comm_wallclock" and break_comm_hours is not None:
            brks = sorted(break_comm_hours) if isinstance(break_comm_hours, list) else [break_comm_hours]
            axis_breaks[x_key] = brks
        elif x_key == "computing_wallclock" and break_compute_wc_hours is not None:
            brks = sorted(break_compute_wc_hours) if isinstance(break_compute_wc_hours, list) else [break_compute_wc_hours]
            axis_breaks[x_key] = brks
        else:
            break_lo, break_hi = find_outlier_threshold(all_records, key_fn)
            if break_lo is not None:
                axis_breaks[x_key] = [break_lo]

    fig = plt.figure(figsize=(21, 6))
    if not no_title:
        fig.suptitle(f"Fleet Results [{task.upper()}] — {metric_label} vs. Cost by Method",
                     fontsize=14, fontweight="bold", y=1.02)

    panel_width = 0.28
    gap = 0.04
    left_start = 0.05

    all_ys = [r["val_acc"] for recs in all_records.values() for r in recs]
    if all_ys:
        y_top = max(all_ys) + y_pad
        y_bottom = y_top - y_window
        if min(all_ys) - y_pad < y_bottom:
            y_bottom = min(all_ys) - y_pad
            y_top = y_bottom + max(y_window, max(all_ys) - min(all_ys) + 2 * y_pad)
        shared_ylim = (y_bottom, y_top)
    else:
        shared_ylim = None

    for i, (x_key, x_label) in enumerate(subplot_configs):
        pos_left = left_start + i * (panel_width + gap)

        if x_key in axis_breaks:
            brks = axis_breaks[x_key]
            n_seg = len(brks) + 1
            ratios = [4] + [1] * len(brks)
            gs = GridSpec(1, n_seg, width_ratios=ratios, wspace=0.05,
                          left=pos_left, right=pos_left + panel_width,
                          bottom=0.12, top=0.88)
            axes = [fig.add_subplot(gs[0, j]) for j in range(n_seg)]
            for j in range(1, n_seg):
                axes[j].yaxis.set_visible(False)
                axes[j].spines['left'].set_visible(False)
                axes[j].tick_params(left=False)
            for j in range(n_seg - 1):
                axes[j].spines['right'].set_visible(False)

            all_xvals = [_get_x_val(r, x_key) for recs in all_records.values() for r in recs]
            _scatter_on_axes(axes, brks, all_records, x_key)
            if selection_only:
                for method, records in all_records.items():
                    x_fn = lambda r, k=x_key: _get_x_val(r, k)
                    _draw_connecting_lines(axes[0], records, method, x_fn, METHOD_COLORS[method])

            axes[0].set_xlim(right=brks[0])
            for j in range(1, n_seg):
                lo = brks[j - 1]
                if j < len(brks):
                    hi = brks[j]
                    vals_in = [v for v in all_xvals if lo <= v <= hi]
                    if vals_in:
                        axes[j].set_xlim(left=min(vals_in) * 0.9, right=max(vals_in) * 1.1)
                    else:
                        axes[j].set_xlim(left=lo * 0.95, right=hi * 1.05)
                else:
                    vals_in = [v for v in all_xvals if v >= lo]
                    if vals_in:
                        axes[j].set_xlim(left=min(vals_in) * 0.9, right=max(vals_in) * 1.1)
                    else:
                        axes[j].set_xlim(left=lo * 0.95, right=lo * 2)

            if shared_ylim:
                for ax in axes:
                    ax.set_ylim(shared_ylim)
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            for j in range(n_seg - 1):
                add_break_marks(axes[j], axes[j + 1])
            for jj, ax in enumerate(axes):
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=16)
                ax.locator_params(axis='y', nbins=6)
                nbx = 3 if jj > 0 else 5
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nbx, integer=True))
            axes[0].set_xlabel(x_label, fontsize=20)
            if i == 0:
                axes[0].set_ylabel(metric_label, fontsize=20)

        else:
            ax = fig.add_axes([pos_left, 0.12, panel_width, 0.76])
            for method, records in all_records.items():
                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]
                if selection_only:
                    x_fn = lambda r, k=x_key: _get_x_val(r, k)
                    _draw_connecting_lines(ax, records, method, x_fn, color)
                for r in records:
                    x_val = _get_x_val(r, x_key)
                    is_sel = r.get("is_selected", False)
                    sz = 240 if is_sel else 150
                    lw = 2.5 if is_sel else 0.5
                    al = 1.0 if is_sel else 0.85
                    zz = 4 if is_sel else 3
                    ax.scatter(x_val, r["val_acc"], c=color, marker=marker, s=sz,
                               edgecolors="black", linewidth=lw, zorder=zz, alpha=al)

            ax.set_xlabel(x_label, fontsize=20)
            if i == 0:
                ax.set_ylabel(metric_label, fontsize=20)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=16)
            ax.locator_params(nbins=6)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if shared_ylim:
                ax.set_ylim(shared_ylim)

    if not no_legend:
        handles = []
        for method in all_records:
            h = plt.Line2D([0], [0], marker=METHOD_MARKERS[method], color="w",
                            markerfacecolor=METHOD_COLORS[method],
                            markeredgecolor="black", markersize=14, label=method)
            handles.append(h)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, -0.08))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_method_tradeoff(all_records, output_path, task, metric_label,
                               break_comm_hours=None, break_compute_wc_hours=None,
                               selection_only=False, no_legend=False,
                               no_title=False):
    comm_fn = lambda r: (r["upload_wallclock"] + r["download_wallclock"]) / 60
    compute_fn = lambda r: r["computing_wallclock"] / 60

    all_comm = [comm_fn(r) for recs in all_records.values() for r in recs]
    all_compute = [compute_fn(r) for recs in all_records.values() for r in recs]

    if break_comm_hours is not None:
        y_brks = sorted(break_comm_hours) if isinstance(break_comm_hours, list) else [break_comm_hours]
    else:
        brk_lo, _ = find_outlier_threshold(all_records, comm_fn)
        y_brks = [brk_lo] if brk_lo is not None else []

    if break_compute_wc_hours is not None:
        x_brks = sorted(break_compute_wc_hours) if isinstance(break_compute_wc_hours, list) else [break_compute_wc_hours]
    else:
        brk_lo, _ = find_outlier_threshold(all_records, compute_fn)
        x_brks = [brk_lo] if brk_lo is not None else []

    n_y = len(y_brks) + 1
    n_x = len(x_brks) + 1
    has_break = n_y > 1 or n_x > 1

    if has_break:
        h_ratios = [1] * len(y_brks) + [3]
        w_ratios = [4] + [1] * len(x_brks)
        fig = plt.figure(figsize=(10 + 2 * (n_x - 1), 8))
        gs = GridSpec(n_y, n_x, height_ratios=h_ratios, width_ratios=w_ratios,
                      hspace=0.08, wspace=0.05)
        ax_grid = [[fig.add_subplot(gs[r, c]) for c in range(n_x)] for r in range(n_y)]

        for row in range(n_y):
            for col in range(n_x):
                ax = ax_grid[row][col]
                if col > 0:
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(left=False, labelleft=False)
                if col < n_x - 1:
                    ax.spines['right'].set_visible(False)
                if row > 0:
                    ax.spines['top'].set_visible(False)
                if row < n_y - 1:
                    ax.spines['bottom'].set_visible(False)
                    ax.tick_params(bottom=False, labelbottom=False)

        if not no_title:
            fig.suptitle(f"Fleet Results [{task.upper()}] — Computation vs. Communication Tradeoff",
                         fontsize=14, fontweight="bold", y=0.98)
    else:
        fig, ax_single = plt.subplots(figsize=(10, 7))
        if not no_title:
            ax_single.set_title(f"Fleet Results [{task.upper()}] — Computation vs. Communication Tradeoff",
                                fontsize=14, fontweight="bold")
        ax_grid = [[ax_single]]

    def get_cell(x_val, y_val):
        col = 0
        for brk in x_brks:
            if x_val >= brk:
                col += 1
        y_seg = 0
        for brk in y_brks:
            if y_val >= brk:
                y_seg += 1
        return n_y - 1 - y_seg, col

    method_accs = {}
    for method, records in all_records.items():
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        accs = []
        if selection_only and _DRAW_CONNECTING_LINES:
            sorted_recs = sorted(records, key=lambda r: _get_shade_val(r, method))
            xs_line = [compute_fn(r) for r in sorted_recs]
            ys_line = [comm_fn(r) for r in sorted_recs]
            if len(xs_line) >= 2:
                ax_grid[n_y - 1][0].plot(xs_line, ys_line, color=color, linewidth=1.5,
                                         alpha=0.6, zorder=2, linestyle="-")
        for r in records:
            x_val = compute_fn(r)
            y_val = comm_fn(r)
            acc = r["val_acc"]
            accs.append(acc)
            is_sel = r.get("is_selected", False)
            sz = 240 if is_sel else 150
            lw = 2.5 if is_sel else 0.5
            al = 1.0 if is_sel else 0.85
            zz = 4 if is_sel else 3
            row, col = get_cell(x_val, y_val)
            ax_grid[row][col].scatter(x_val, y_val, c=color, marker=marker, s=sz,
                                      edgecolors="black", linewidth=lw, zorder=zz, alpha=al)
            ax_grid[row][col].annotate(f"{acc:.3f}", (x_val, y_val), textcoords="offset points",
                                       xytext=(5, 5), fontsize=14, alpha=0.85, fontweight="bold")
        method_accs[method] = accs

    handles = []
    for method in all_records:
        accs = method_accs.get(method, [])
        if accs:
            mean = np.mean(accs)
            std = np.std(accs)
            label = f"{method} ({metric_label.split()[-1].lower()} {mean:.3f}±{std:.3f})"
        else:
            label = method
        h = plt.Line2D([0], [0], marker=METHOD_MARKERS[method], color="w",
                        markerfacecolor=METHOD_COLORS[method],
                        markeredgecolor="black", markersize=18, label=label)
        handles.append(h)

    if has_break:
        if x_brks:
            for row in range(n_y):
                ax_grid[row][0].set_xlim(right=x_brks[0])
            for ci in range(1, n_x):
                lo = x_brks[ci - 1]
                vals = [v for v in all_compute if v >= lo] if ci == n_x - 1 else \
                       [v for v in all_compute if lo <= v <= x_brks[ci]]
                xlim = (min(vals) * 0.9, max(vals) * 1.1) if vals else (lo * 0.95, lo * 2)
                for row in range(n_y):
                    ax_grid[row][ci].set_xlim(xlim)
        elif n_y > 1:
            lims = [ax_grid[row][0].get_xlim() for row in range(n_y)]
            shared = (min(l[0] for l in lims), max(l[1] for l in lims))
            for row in range(n_y):
                ax_grid[row][0].set_xlim(shared)

        if y_brks:
            for col in range(n_x):
                ax_grid[n_y - 1][col].set_ylim(top=y_brks[0])
            for ri in range(n_y - 1):
                y_seg = n_y - 1 - ri
                lo = y_brks[y_seg - 1]
                vals = [v for v in all_comm if v >= lo] if y_seg == len(y_brks) else \
                       [v for v in all_comm if lo <= v <= y_brks[y_seg]]
                ylim = (min(vals) * 0.9, max(vals) * 1.1) if vals else (lo * 0.95, lo * 2)
                for col in range(n_x):
                    ax_grid[ri][col].set_ylim(ylim)
        elif n_x > 1:
            lims = [ax_grid[0][col].get_ylim() for col in range(n_x)]
            shared = (min(l[0] for l in lims), max(l[1] for l in lims))
            for col in range(n_x):
                ax_grid[0][col].set_ylim(shared)

        for ri in range(n_y - 1):
            for col in range(n_x):
                d = 0.01
                if col == 0:
                    kwargs = dict(transform=ax_grid[ri][col].transAxes, color='k', clip_on=False, lw=1)
                    ax_grid[ri][col].plot((-d, +d), (-d * 3, +d * 3), **kwargs)
                    kwargs.update(transform=ax_grid[ri + 1][col].transAxes)
                    ax_grid[ri + 1][col].plot((-d, +d), (1 - d * 3, 1 + d * 3), **kwargs)
                if col == n_x - 1:
                    kwargs = dict(transform=ax_grid[ri][col].transAxes, color='k', clip_on=False, lw=1)
                    ax_grid[ri][col].plot((1 - d, 1 + d), (-d * 3, +d * 3), **kwargs)
                    kwargs.update(transform=ax_grid[ri + 1][col].transAxes)
                    ax_grid[ri + 1][col].plot((1 - d, 1 + d), (1 - d * 3, 1 + d * 3), **kwargs)

        for ci in range(n_x - 1):
            for row in range(n_y):
                d = 0.015
                if row == n_y - 1:
                    kwargs = dict(transform=ax_grid[row][ci].transAxes, color='k', clip_on=False, linewidth=1)
                    ax_grid[row][ci].plot((1 - d, 1 + d), (-d, +d), **kwargs)
                    kwargs.update(transform=ax_grid[row][ci + 1].transAxes)
                    ax_grid[row][ci + 1].plot((-d * 4, +d * 4), (-d, +d), **kwargs)
                if row == 0:
                    kwargs = dict(transform=ax_grid[row][ci].transAxes, color='k', clip_on=False, linewidth=1)
                    ax_grid[row][ci].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
                    kwargs.update(transform=ax_grid[row][ci + 1].transAxes)
                    ax_grid[row][ci + 1].plot((-d * 4, +d * 4), (1 - d, 1 + d), **kwargs)

        for row in range(n_y):
            for col in range(n_x):
                ax = ax_grid[row][col]
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=16)
                nbx = 3 if col > 0 else 5
                nby = 3 if row < n_y - 1 else 5
                ax.xaxis.set_major_locator(MaxNLocator(nbins=nbx, integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nby, integer=True))

        if not no_legend:
            # With the broken comm axis, FedIT occupies the upper-RIGHT break cell,
            # so the upper-LEFT break cell (ax_grid[0][0]) is entirely empty. Place
            # the full-size legend there: it sits above the cluster (which lives in
            # the lower panel) with no overlap and no need for added headroom.
            legend_ax = ax_grid[0][0] if n_y > 1 else ax_grid[n_y - 1][0]
            legend_ax.legend(handles=handles, loc="upper left",
                             fontsize=16, framealpha=0.9)

        bottom_left = ax_grid[n_y - 1][0]
        bottom_left.set_xlabel("Computation Wallclock (h)", fontsize=20)
        if n_y > 1:
            fig.text(0.02, 0.5, "Communication Wallclock (h)", va='center',
                     rotation='vertical', fontsize=20)
        else:
            bottom_left.set_ylabel("Communication Wallclock (h)", fontsize=20)
    else:
        ax = ax_grid[0][0]
        if not no_legend:
            ax.legend(handles=handles, loc="best", fontsize=16, framealpha=0.9)
        ax.set_ylabel("Communication Wallclock (h)", fontsize=20)
        ax.set_xlabel("Computation Wallclock (h)", fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_accuracy_vs_total_wallclock(all_records, output_path, task, metric_label,
                                     break_total_wc_hours=None, selection_only=False,
                                     y_window=0.20, y_pad=0.01,
                                     no_legend=False, no_title=False):
    def _total_wc(r):
        return (r["computing_wallclock"] + r["upload_wallclock"]
                + r["download_wallclock"]) / 60

    all_xvals = [_total_wc(r) for recs in all_records.values() for r in recs]
    all_ys = [r["val_acc"] for recs in all_records.values() for r in recs]

    if break_total_wc_hours is not None:
        brks = sorted(break_total_wc_hours) if isinstance(break_total_wc_hours, list) else [break_total_wc_hours]
    else:
        key_fn = lambda r: _total_wc(r)
        break_lo, _ = find_outlier_threshold(all_records, key_fn)
        brks = [break_lo] if break_lo is not None else []

    n_seg = len(brks) + 1 if brks else 1

    if n_seg > 1:
        ratios = [4] + [1] * len(brks)
        fig = plt.figure(figsize=(10, 7))
        gs = GridSpec(1, n_seg, width_ratios=ratios, wspace=0.05,
                      left=0.10, right=0.95, bottom=0.10, top=0.90)
        axes = [fig.add_subplot(gs[0, j]) for j in range(n_seg)]
        for j in range(1, n_seg):
            axes[j].yaxis.set_visible(False)
            axes[j].spines['left'].set_visible(False)
            axes[j].tick_params(left=False)
        for j in range(n_seg - 1):
            axes[j].spines['right'].set_visible(False)
        if not no_title:
            fig.suptitle(f"Fleet Results [{task.upper()}] — {metric_label} vs. Total Wallclock",
                         fontsize=14, fontweight="bold", y=0.97)
    else:
        fig, ax_single = plt.subplots(figsize=(10, 7))
        if not no_title:
            ax_single.set_title(f"Fleet Results [{task.upper()}] — {metric_label} vs. Total Wallclock",
                                fontsize=14, fontweight="bold")
        axes = [ax_single]

    for method, records in all_records.items():
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        if selection_only:
            _draw_connecting_lines(axes[0], records, method, _total_wc, color)
        for r in records:
            x_val = _total_wc(r)
            y_val = r["val_acc"]
            is_sel = r.get("is_selected", False)
            sz = 240 if is_sel else 150
            lw = 2.5 if is_sel else 0.5
            al = 1.0 if is_sel else 0.85
            zz = 4 if is_sel else 3
            for j, ax in enumerate(axes):
                lo = brks[j - 1] if j > 0 else -float("inf")
                hi = brks[j] if j < len(brks) else float("inf")
                if lo <= x_val <= hi:
                    ax.scatter(x_val, y_val, c=color, marker=marker, s=sz,
                               edgecolors="black", linewidth=lw, zorder=zz, alpha=al)

    if n_seg > 1:
        axes[0].set_xlim(right=brks[0])
        for j in range(1, n_seg):
            lo = brks[j - 1]
            if j < len(brks):
                hi = brks[j]
                vals_in = [v for v in all_xvals if lo <= v <= hi]
                if vals_in:
                    axes[j].set_xlim(left=min(vals_in) * 0.9, right=max(vals_in) * 1.1)
                else:
                    axes[j].set_xlim(left=lo * 0.95, right=hi * 1.05)
            else:
                vals_in = [v for v in all_xvals if v >= lo]
                if vals_in:
                    axes[j].set_xlim(left=min(vals_in) * 0.9, right=max(vals_in) * 1.1)
                else:
                    axes[j].set_xlim(left=lo * 0.95, right=lo * 2)
        if all_ys:
            y_top = max(all_ys) + y_pad
            y_bottom = y_top - y_window
            if min(all_ys) - y_pad < y_bottom:
                y_bottom = min(all_ys) - y_pad
                y_top = y_bottom + max(y_window, max(all_ys) - min(all_ys) + 2 * y_pad)
            for ax in axes:
                ax.set_ylim(y_bottom, y_top)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for j in range(n_seg - 1):
            add_break_marks(axes[j], axes[j + 1])

    handles = []
    for method in all_records:
        h = plt.Line2D([0], [0], marker=METHOD_MARKERS[method], color="w",
                        markerfacecolor=METHOD_COLORS[method],
                        markeredgecolor="black", markersize=18, label=method)
        handles.append(h)
    if not no_legend:
        axes[0].legend(handles=handles, loc="best", fontsize=16, framealpha=0.9)
    axes[0].set_xlabel("Total Wallclock (compute + comm, hours)", fontsize=20)
    axes[0].set_ylabel(metric_label, fontsize=20)
    if all_ys:
        y_top = max(all_ys) + y_pad
        y_bottom = y_top - y_window
        if min(all_ys) - y_pad < y_bottom:
            y_bottom = min(all_ys) - y_pad
            y_top = y_bottom + max(y_window, max(all_ys) - min(all_ys) + 2 * y_pad)
        for ax in axes:
            ax.set_ylim(y_bottom, y_top)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    for j, ax in enumerate(axes):
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=16)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        nbx = 3 if (n_seg > 1 and j > 0) else 5
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nbx, integer=True))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


SHADE_KEYS_BY_METHOD = {
    "AdaSparse-LoRA v2": "gamma",
    "AdaSparse-LoRA v3": "gamma",
    "HetLoRA": "decay",
    "FAH-QLoRA": "lambda",
}


def _get_shade_val(record, method):
    """Extract numeric shade value for ordering connecting lines."""
    key = SHADE_KEYS_BY_METHOD.get(method)
    if not key:
        return 0
    m = re.search(rf'{key}_([0-9e.\-]+)', record["dirname"])
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0
    return 0


_DRAW_CONNECTING_LINES = True

def _draw_connecting_lines(ax, records, method, x_fn, color):
    """Draw a line connecting selected points ordered by shade value."""
    if not _DRAW_CONNECTING_LINES or len(records) < 2:
        return
    sorted_recs = sorted(records, key=lambda r: _get_shade_val(r, method))
    xs = [x_fn(r) for r in sorted_recs]
    ys = [r["val_acc"] for r in sorted_recs]
    ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=2, linestyle="-")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="GLUE task name")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--exp-dir", default="exp_distributed",
                        help="Base experiment directory (default: exp_distributed)")
    parser.add_argument("--break-comm-hours", type=float, nargs="+", default=None,
                        help="Force axis break(s) at these comm wallclock hours (1 or 2 values)")
    parser.add_argument("--break-total-wc-hours", type=float, nargs="+", default=None,
                        help="Force axis break(s) on the acc-vs-wallclock plot (1 or 2 values)")
    parser.add_argument("--break-compute-wc-hours", type=float, nargs="+", default=None,
                        help="Force axis break(s) on the computing wallclock scatter panel")
    parser.add_argument("--exclude-shade", nargs=3, action="append", default=[],
                        metavar=("METHOD", "KEY", "VALUE"),
                        help="Exclude records matching method/key/value (e.g. HetLoRA decay 0.95)")
    parser.add_argument("--override-selection", nargs=2, action="append", default=[],
                        metavar=("METHOD", "DIRNAME_PATTERN"),
                        help="Force-select a record by dirname substring, deselecting others in its shade")
    parser.add_argument("--add-selection", nargs=2, action="append", default=[],
                        metavar=("METHOD", "DIRNAME_PATTERN"),
                        help="Add a record to the selection without deselecting others")
    parser.add_argument("--y-window", type=float, default=0.20,
                        help="Fixed y-axis range size for accuracy plots (default: 0.20)")
    parser.add_argument("--y-pad", type=float, default=0.01,
                        help="Padding above max / below min within the window (default: 0.01)")
    parser.add_argument("--selection-only", action="store_true",
                        help="Plot only selected (best-per-shade) experiments")
    parser.add_argument("--no-connecting-lines", action="store_true",
                        help="Disable connecting lines between selected points")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide legend from cross-method plots")
    parser.add_argument("--no-title", action="store_true",
                        help="Hide plot titles")
    parser.add_argument("--exclude-methods", nargs="*", default=[],
                        help="Method display names to exclude (e.g. 'AdaSparse-LoRA v3')")
    parser.add_argument("--rename-method", nargs=2, action="append", default=[],
                        metavar=("OLD", "NEW"),
                        help="Rename a method in legends (e.g. --rename-method 'AdaSparse-LoRA v2' 'AdaS-LoRA')")
    parser.add_argument("--fedit-golden-only", action="store_true",
                        help="Pin FedIT to its golden run only (stable, table-consistent)")
    args = parser.parse_args()

    renames = dict(args.rename_method)

    # Override METHOD_DIRS with the provided exp-dir base
    global METHOD_DIRS
    base = args.exp_dir
    METHOD_DIRS = {
        "FedIT":              [f"{base}/golden/fedit", f"{base}/fedit"],
        "HetLoRA":            [f"{base}/golden/hetlora", f"{base}/hetlora"],
        "FAH-QLoRA":          [f"{base}/golden/fahqlora", f"{base}/fahqlora"],
        "AdaSparse-LoRA v2":  [f"{base}/golden/adasparse_lorav2", f"{base}/adasparse_lorav2"],
        "AdaSparse-LoRA v3":  [f"{base}/golden/adasparse_lorav3", f"{base}/adasparse_lorav3"],
    }
    # FedIT runs at rank 64 and is rescaled to rank 200; its non-golden reruns
    # make the operating point drift between regenerations. Pin to golden for a
    # stable point consistent with the reported table.
    if args.fedit_golden_only:
        golden_fedit = f"{base}/golden/fedit"
        _prefix = f"{args.task.lower()}__strategy_"
        _has_golden = os.path.isdir(golden_fedit) and any(
            e.startswith(_prefix) for e in os.listdir(golden_fedit))
        # Pin to golden only when a golden FedIT run exists for this task;
        # otherwise (e.g. SST-2, which has no golden FedIT) keep the default so
        # collect_all falls back to the single non-golden run.
        if _has_golden:
            METHOD_DIRS["FedIT"] = [golden_fedit]

    for excl in args.exclude_methods:
        METHOD_DIRS.pop(excl, None)

    if renames:
        METHOD_DIRS = {renames.get(k, k): v for k, v in METHOD_DIRS.items()}
        global METHOD_MARKERS, METHOD_COLORS, TASK_RW_FILTER, SHADE_KEYS_BY_METHOD
        global CROSS_METHOD_OVERRIDES, LEGACY_SST2_DIRS
        METHOD_MARKERS = {renames.get(k, k): v for k, v in METHOD_MARKERS.items()}
        METHOD_COLORS = {renames.get(k, k): v for k, v in METHOD_COLORS.items()}
        for task_key in TASK_RW_FILTER:
            TASK_RW_FILTER[task_key] = {renames.get(k, k): v
                                        for k, v in TASK_RW_FILTER[task_key].items()}
        SHADE_KEYS_BY_METHOD = {renames.get(k, k): v for k, v in SHADE_KEYS_BY_METHOD.items()}
        CROSS_METHOD_OVERRIDES = {(renames.get(t[0], t[0]), *t[1:]): v
                                  for t, v in CROSS_METHOD_OVERRIDES.items()}
        LEGACY_SST2_DIRS = {renames.get(k, k): v for k, v in LEGACY_SST2_DIRS.items()}

    task = args.task.lower()
    metric_key = TASK_METRIC.get(task, "val_acc")
    metric_labels = {
        "val_f1": "Final Val F1",
        "val_acc": "Final Val Accuracy",
        "val_pearson": "Final Val Pearson",
        "val_mcc": "Final Val MCC",
    }
    metric_label = metric_labels.get(metric_key, f"Final {metric_key}")
    output_dir = args.output_dir or f"0_results/final/{task}"

    print(f"Collecting fleet experiments for {task.upper()} (metric: {metric_key})...")
    all_records = collect_all(task, metric_key)
    if not all_records:
        print("ERROR: no valid experiments found", file=sys.stderr)
        return 1

    for method_name, shade_key, shade_val in args.exclude_shade:
        target = renames.get(method_name, method_name)
        if target in all_records:
            pattern = f"{shade_key}_{shade_val}"
            before = len(all_records[target])
            all_records[target] = [r for r in all_records[target]
                                   if pattern not in r["dirname"]]
            after = len(all_records[target])
            if before != after:
                print(f"  Excluded {before - after} {target} records matching {pattern}")

    select_best_configs(all_records)

    for method_name, pattern in args.override_selection:
        target = renames.get(method_name, method_name)
        if target not in all_records:
            continue
        shade_key = SHADE_KEYS_BY_METHOD.get(target)
        match = [r for r in all_records[target] if pattern in r["dirname"]]
        if not match:
            print(f"  WARNING: no {target} record matching '{pattern}'")
            continue
        forced = match[0]
        if shade_key:
            m = re.search(rf'{shade_key}_([0-9e.\-]+)', forced["dirname"])
            shade_val = m.group(1) if m else None
            if shade_val:
                for r in all_records[target]:
                    m2 = re.search(rf'{shade_key}_([0-9e.\-]+)', r["dirname"])
                    if m2 and m2.group(1) == shade_val:
                        r["is_selected"] = False
        forced["is_selected"] = True
        print(f"  Override: {target} selected '{pattern}' (acc={forced['val_acc']:.4f})")

    for method_name, pattern in args.add_selection:
        target = renames.get(method_name, method_name)
        if target not in all_records:
            continue
        match = [r for r in all_records[target] if pattern in r["dirname"]]
        if not match:
            print(f"  WARNING: no {target} record matching '{pattern}'")
            continue
        match[0]["is_selected"] = True
        print(f"  Added: {target} '{pattern}' (acc={match[0]['val_acc']:.4f})")

    global _DRAW_CONNECTING_LINES
    if args.no_connecting_lines:
        _DRAW_CONNECTING_LINES = False

    if args.selection_only:
        for method in list(all_records.keys()):
            all_records[method] = [r for r in all_records[method] if r.get("is_selected")]
            if not all_records[method]:
                del all_records[method]
        for records in all_records.values():
            for r in records:
                r["is_selected"] = False

    os.makedirs(output_dir, exist_ok=True)
    sel = args.selection_only
    plot_cross_method_scatter(
        all_records,
        os.path.join(output_dir, "fleet_cross_method_scatter.png"),
        task, metric_label,
        break_comm_hours=args.break_comm_hours,
        break_compute_wc_hours=args.break_compute_wc_hours,
        selection_only=sel,
        y_window=args.y_window, y_pad=args.y_pad,
        no_title=args.no_title,
        no_legend=args.no_legend,
    )
    plot_cross_method_tradeoff(
        all_records,
        os.path.join(output_dir, "fleet_cross_method_tradeoff.png"),
        task, metric_label,
        break_comm_hours=args.break_comm_hours,
        break_compute_wc_hours=args.break_compute_wc_hours,
        selection_only=sel,
        no_legend=args.no_legend,
        no_title=args.no_title,
    )
    plot_accuracy_vs_total_wallclock(
        all_records,
        os.path.join(output_dir, "fleet_cross_method_acc_vs_wallclock.png"),
        task, metric_label,
        break_total_wc_hours=args.break_total_wc_hours,
        selection_only=sel,
        y_window=args.y_window, y_pad=args.y_pad,
        no_legend=args.no_legend,
        no_title=args.no_title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
