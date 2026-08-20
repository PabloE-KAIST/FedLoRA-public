#!/usr/bin/env python3
"""Cross-method scatter plots with FedIT r=200 estimate and axis breaks.

Generates two pairs of plots:
  - Without AdaSparse-LoRA v3
  - With AdaSparse-LoRA v3

FedIT r=200 is a synthetic point estimated from r=64 actuals scaled by 200/64.

Usage:
    python -m analysis.cross_method.fleet_cross_method_scatter_r200
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from analysis.sweep.sweep_scatter import parse_eval_results, parse_system_summary

# --- FedIT r=200 synthetic data -----------------------------------------------
# Estimated from r=64 actuals. Communication scales linearly (200/64 = 3.125x).
# Computing uses α=0.15 adapter fraction: compute_r200 = 16.78 * (0.85 + 0.15*3.125).
FEDIT_R200 = {
    "val_acc": 0.9610,
    "total_communicated_mb": 144014.0,  # MB; converted to GB at plot time
    "upload_wallclock": 524.3,
    "download_wallclock": 471.9,
    "computing_wallclock": 22.1,
    "dirname": "fedit_r200_estimate",
}

FLEET_DIRS = {
    "HetLoRA":            "exp_golden/archive/distributed_hetlora",
    "FAH-QLoRA":          "exp_golden/archive/distributed_fah_qlora",
    "AdaSparse-LoRA v2":  "exp_golden/archive/distributed_adasparse_lorav2",
    "AdaSparse-LoRA v3":  "exp_golden/archive/distributed_adasparse_lorav3",
}

METHOD_MARKERS = {
    "FedIT (r=200)":      "D",
    "HetLoRA":            "^",
    "FAH-QLoRA":          "s",
    "AdaSparse-LoRA v2":  "o",
    "AdaSparse-LoRA v3":  "P",
}

METHOD_COLORS = {
    "FedIT (r=200)":      "#d62728",
    "HetLoRA":            "#2ca02c",
    "FAH-QLoRA":          "#1f77b4",
    "AdaSparse-LoRA v2":  "#ff7f0e",
    "AdaSparse-LoRA v3":  "#9467bd",
}


def collect_all():
    all_records = {"FedIT (r=200)": [FEDIT_R200]}
    for method, parent in FLEET_DIRS.items():
        if not os.path.isdir(parent):
            print(f"  SKIP dir not found: {parent}", file=sys.stderr)
            continue
        records = []
        for entry in sorted(os.listdir(parent)):
            full = os.path.join(parent, entry)
            if not os.path.isdir(full) or not entry.startswith("strategy_"):
                continue
            if method in ("AdaSparse-LoRA v2", "AdaSparse-LoRA v3") and "gamma_0.5" not in entry:
                continue
            if method == "HetLoRA" and "decay_0.95" in entry:
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


def _draw_break_marks(ax, pos, axis="x", d=0.015, angle=45):
    """Draw diagonal break marks on an axis at the given position (in axes coords)."""
    kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=0.8)
    dx = d * np.cos(np.radians(angle))
    dy = d * np.sin(np.radians(angle))
    if axis == "x":
        ax.plot((pos - dx, pos + dx), (-dy, +dy), **kwargs)
        ax.plot((pos - dx, pos + dx), (1 - dy, 1 + dy), **kwargs)
    else:
        ax.plot((-dx, +dx), (pos - dy, pos + dy), **kwargs)
        ax.plot((1 - dx, 1 + dx), (pos - dy, pos + dy), **kwargs)


def _make_broken_axis_pair(fig, gs_slot, break_frac=0.75, width_ratio=3):
    """Create a pair of axes sharing y-axis with a broken x-axis.

    Returns (ax_left, ax_right) where ax_left covers the main data cluster
    and ax_right covers the outlier region.
    """
    import matplotlib.gridspec as mgs
    inner = mgs.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_slot,
        width_ratios=[width_ratio, 1], wspace=0.04,
    )
    ax_l = fig.add_subplot(inner[0])
    ax_r = fig.add_subplot(inner[1], sharey=ax_l)
    ax_r.tick_params(labelleft=False)
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(left=False)
    return ax_l, ax_r


def _plot_on_broken_pair(ax_l, ax_r, xs, ys, xlim_l, xlim_r, **scatter_kw):
    """Scatter data across a broken axis pair."""
    ax_l.scatter(xs, ys, **scatter_kw)
    ax_r.scatter(xs, ys, **scatter_kw)
    ax_l.set_xlim(xlim_l)
    ax_r.set_xlim(xlim_r)


def plot_cross_method(all_records, output_path, title_suffix=""):
    import matplotlib.gridspec as mgs

    subplot_configs = [
        ("total_communicated_gb", "Total Communicated (GB)", True),
        ("comm_wallclock", "Communication Wallclock (min)", True),
        ("computing_wallclock", "Computing Wallclock (min)", False),
    ]

    fig = plt.figure(figsize=(24, 6.5))
    fig.suptitle(f"Fleet Results — Accuracy vs. Cost by Method{title_suffix}",
                 fontsize=14, fontweight="bold", y=1.02)
    outer = mgs.GridSpec(1, 3, figure=fig, wspace=0.28)

    all_axes = []

    for col_idx, (x_key, x_label, needs_break) in enumerate(subplot_configs):
        # Collect all x values to determine ranges
        all_xs = []
        for method, records in all_records.items():
            for r in records:
                if x_key == "comm_wallclock":
                    x_val = r["upload_wallclock"] + r["download_wallclock"]
                elif x_key == "total_communicated_gb":
                    x_val = r["total_communicated_mb"] / 1024
                else:
                    x_val = r[x_key]
                all_xs.append((x_val, method))

        xs_sorted = sorted(all_xs, key=lambda t: t[0])
        max_val = xs_sorted[-1][0]
        second_max = xs_sorted[-2][0]

        if needs_break and max_val > second_max * 1.8:
            ax_l, ax_r = _make_broken_axis_pair(fig, outer[col_idx])

            cluster_max = second_max
            pad_l = (cluster_max - xs_sorted[0][0]) * 0.08
            pad_r = max_val * 0.04
            xlim_l = (xs_sorted[0][0] - pad_l, cluster_max + pad_l * 3)
            xlim_r = (max_val - pad_r * 8, max_val + pad_r * 3)

            for method, records in all_records.items():
                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]
                mxs, mys = [], []
                for r_rec in records:
                    if x_key == "comm_wallclock":
                        x_val = r_rec["upload_wallclock"] + r_rec["download_wallclock"]
                    elif x_key == "total_communicated_gb":
                        x_val = r_rec["total_communicated_mb"] / 1024
                    else:
                        x_val = r_rec[x_key]
                    mxs.append(x_val)
                    mys.append(r_rec["val_acc"])
                _plot_on_broken_pair(
                    ax_l, ax_r, mxs, mys, xlim_l, xlim_r,
                    c=color, marker=marker, s=100,
                    edgecolors="black", linewidth=0.5, zorder=3,
                    label=method, alpha=0.85,
                )

            ax_l.set_xlabel(x_label, fontsize=11)
            ax_l.xaxis.set_label_coords(0.6, -0.10, transform=ax_l.transAxes)
            ax_l.set_ylabel("Final Val Accuracy", fontsize=11)
            ax_l.grid(True, alpha=0.3)
            ax_r.grid(True, alpha=0.3)
            ax_l.tick_params(labelsize=9)
            ax_r.tick_params(labelsize=9)

            _draw_break_marks(ax_l, 1.0, axis="x")
            _draw_break_marks(ax_r, 0.0, axis="x")

            all_axes.append(ax_l)
        else:
            ax = fig.add_subplot(outer[col_idx])
            for method, records in all_records.items():
                color = METHOD_COLORS[method]
                marker = METHOD_MARKERS[method]
                mxs, mys = [], []
                for r_rec in records:
                    if x_key == "comm_wallclock":
                        x_val = r_rec["upload_wallclock"] + r_rec["download_wallclock"]
                    elif x_key == "total_communicated_gb":
                        x_val = r_rec["total_communicated_mb"] / 1024
                    else:
                        x_val = r_rec[x_key]
                    mxs.append(x_val)
                    mys.append(r_rec["val_acc"])
                ax.scatter(mxs, mys, c=color, marker=marker, s=100,
                           edgecolors="black", linewidth=0.5, zorder=3,
                           label=method, alpha=0.85)
            ax.set_xlabel(x_label, fontsize=11)
            ax.set_ylabel("Final Val Accuracy", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
            all_axes.append(ax)

    handles, labels = all_axes[0].get_legend_handles_labels()
    all_axes[0].legend(handles, labels, loc="upper right", fontsize=9,
                       framealpha=0.9, handletextpad=0.5)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_method_tradeoff(all_records, output_path, title_suffix=""):
    import matplotlib.gridspec as mgs

    all_compute = []
    all_comm = []
    for method, records in all_records.items():
        for r in records:
            all_compute.append(r["computing_wallclock"])
            all_comm.append(r["upload_wallclock"] + r["download_wallclock"])

    compute_sorted = sorted(all_compute)
    comm_sorted = sorted(all_comm)
    max_compute = compute_sorted[-1]
    second_max_compute = compute_sorted[-2]
    min_compute = compute_sorted[0]
    second_min_compute = compute_sorted[1]
    max_comm = comm_sorted[-1]
    second_max_comm = comm_sorted[-2]

    x_break_high = max_compute > second_max_compute * 1.8
    x_break_low = second_min_compute > min_compute * 1.8
    x_break = x_break_high or x_break_low
    y_break = max_comm > second_max_comm * 1.8

    fig = plt.figure(figsize=(11, 8))
    fig.suptitle(
        f"Fleet Results — Computation vs. Communication Tradeoff{title_suffix}",
        fontsize=13, fontweight="bold",
    )

    if x_break and y_break:
        w_ratios = [1, 3] if x_break_low else [3, 1]
        gs = mgs.GridSpec(2, 2, figure=fig,
                          height_ratios=[1, 3], width_ratios=w_ratios,
                          hspace=0.06, wspace=0.04)
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_br = fig.add_subplot(gs[1, 1], sharey=ax_bl)
        ax_tl = fig.add_subplot(gs[0, 0], sharex=ax_bl)
        ax_tr = fig.add_subplot(gs[0, 1], sharex=ax_br, sharey=ax_tl)

        ax_tl.tick_params(labelbottom=False)
        ax_tr.tick_params(labelbottom=False, labelleft=False)
        ax_br.tick_params(labelleft=False)

        ax_bl.spines["right"].set_visible(False)
        ax_bl.spines["top"].set_visible(False)
        ax_br.spines["left"].set_visible(False)
        ax_br.spines["top"].set_visible(False)
        ax_tl.spines["right"].set_visible(False)
        ax_tl.spines["bottom"].set_visible(False)
        ax_tr.spines["left"].set_visible(False)
        ax_tr.spines["bottom"].set_visible(False)

        ax_br.tick_params(left=False)
        ax_tr.tick_params(left=False)

        all_panels = [ax_bl, ax_br, ax_tl, ax_tr]

        if x_break_low:
            cluster_x_min = second_min_compute
            cluster_x_max = max_compute
            outlier_x = min_compute
            pad_xr = (cluster_x_max - cluster_x_min) * 0.08
            pad_xl = outlier_x * 0.15
            xlim_l = (outlier_x - pad_xl, outlier_x + pad_xl * 2)
            xlim_r = (cluster_x_min - pad_xr, cluster_x_max + pad_xr)
        else:
            pad_xl = (second_max_compute - min_compute) * 0.08
            pad_xr = max_compute * 0.04
            xlim_l = (min_compute - pad_xl, second_max_compute + pad_xl * 3)
            xlim_r = (max_compute - pad_xr * 8, max_compute + pad_xr * 3)

        min_comm = min(all_comm)
        pad_yb = (second_max_comm - min_comm) * 0.06
        pad_yt = max_comm * 0.03
        ylim_bot = (min_comm - pad_yb * 2, second_max_comm + pad_yb * 5)
        ylim_top = (max_comm - pad_yt * 6, max_comm + pad_yt * 4)

        ax_bl.set_xlim(xlim_l); ax_tl.set_xlim(xlim_l)
        ax_br.set_xlim(xlim_r); ax_tr.set_xlim(xlim_r)
        ax_bl.set_ylim(ylim_bot); ax_br.set_ylim(ylim_bot)
        ax_tl.set_ylim(ylim_top); ax_tr.set_ylim(ylim_top)

        _draw_break_marks(ax_bl, 1.0, axis="x")
        _draw_break_marks(ax_br, 0.0, axis="x")
        _draw_break_marks(ax_tl, 1.0, axis="x")
        _draw_break_marks(ax_tr, 0.0, axis="x")
        _draw_break_marks(ax_tl, 0.0, axis="y")
        _draw_break_marks(ax_bl, 1.0, axis="y")
        _draw_break_marks(ax_tr, 0.0, axis="y")
        _draw_break_marks(ax_br, 1.0, axis="y")

        legend_ax = ax_tr if x_break_low else ax_tl
    elif y_break:
        gs = mgs.GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.06)
        ax_top = fig.add_subplot(gs[0])
        ax_bl = fig.add_subplot(gs[1], sharex=ax_top)
        ax_top.tick_params(labelbottom=False)
        ax_top.spines["bottom"].set_visible(False)
        ax_bl.spines["top"].set_visible(False)

        min_comm = min(all_comm)
        pad_bot = (second_max_comm - min_comm) * 0.06
        ylim_bot = (min_comm - pad_bot * 2, second_max_comm + pad_bot * 5)
        pad_top = max_comm * 0.03
        ylim_top = (max_comm - pad_top * 6, max_comm + pad_top * 4)
        ax_bl.set_ylim(ylim_bot)
        ax_top.set_ylim(ylim_top)
        _draw_break_marks(ax_top, 0.0, axis="y")
        _draw_break_marks(ax_bl, 1.0, axis="y")

        all_panels = [ax_bl, ax_top]
        legend_ax = ax_top
    else:
        ax_bl = fig.add_subplot(111)
        all_panels = [ax_bl]
        legend_ax = ax_bl

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
            for ax in all_panels:
                ax.scatter(x_val, y_val, c=color, marker=marker, s=100,
                           edgecolors="black", linewidth=0.5, zorder=3, alpha=0.85)
                ax.annotate(f"{acc:.3f}", (x_val, y_val),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=7, alpha=0.85, fontweight="bold")
        method_accs[method] = accs

    for ax in all_panels:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    ax_bl.set_xlabel("Computation Wallclock (min)", fontsize=11)
    ax_bl.set_ylabel("Communication Wallclock (min)", fontsize=11)

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
    legend_ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    print("Collecting fleet experiments across methods...")
    all_records = collect_all()
    if not all_records:
        print("ERROR: no valid experiments found", file=sys.stderr)
        return 1

    os.makedirs("0_results", exist_ok=True)

    # Without v3
    records_no_v3 = {k: v for k, v in all_records.items() if k != "AdaSparse-LoRA v3"}
    plot_cross_method(records_no_v3,
                      "0_results/fleet_cross_method_scatter_r200_no_v3.png",
                      title_suffix=" (FedIT r=200 est., v2 γ=0.5, excl. v3)")
    plot_cross_method_tradeoff(records_no_v3,
                               "0_results/fleet_cross_method_tradeoff_r200_no_v3.png",
                               title_suffix=" (FedIT r=200 est., v2 γ=0.5, excl. v3)")

    # With v3
    plot_cross_method(all_records,
                      "0_results/fleet_cross_method_scatter_r200.png",
                      title_suffix=" (FedIT r=200 est., v2 γ=0.5)")
    plot_cross_method_tradeoff(all_records,
                               "0_results/fleet_cross_method_tradeoff_r200.png",
                               title_suffix=" (FedIT r=200 est., v2 γ=0.5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
