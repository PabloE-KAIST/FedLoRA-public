#!/usr/bin/env python3
"""Unified rank evolution comparison — v2 with split communication panels.

Layout (GridSpec 2×4):
  Top-left  (0, 0:2):  HetLoRA rank evolution
  Top-right (0, 2:4):  FAH-QLoRA rank evolution
  Bot-left  (1, 0:2):  AdaS-LoRA compute rank
  Bot-mid   (1, 2:3):  AdaS-LoRA upload rank
  Bot-right (1, 3:4):  AdaS-LoRA download rank

Usage:
    python -m analysis.cross_method.unified_rank_evolution_v2 \
        --hetlora exp_distributed/golden/hetlora/.../unified_log.log \
        --fahqlora exp_distributed/golden/fahqlora/.../unified_log.log \
        --adaslora exp_distributed/golden/adasparse_lorav2/.../unified_log.log \
        --output 0_results/archive/unified_rank_evolution_v2_mrpc.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from single_run.rank_evolution_hetlora import parse_hetlora_ranks_single_run
from single_run.rank_evolution_fahqlora import parse_fah_ranks
from single_run.rank_evolution_adasparsev2 import parse_adasparse_ranks

CLIENT_COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#ff7f00",
    5: "#984ea3",
    6: "#a65628",
}
AVG_COLOR = "black"
AVG_STYLE = {"color": AVG_COLOR, "linewidth": 3, "linestyle": "--",
             "marker": "D", "markersize": 6, "zorder": 10}


def _plot_client_lines(ax, rounds, client_ranks, clients):
    for cid in clients:
        ys = [client_ranks[r].get(cid, np.nan) for r in rounds]
        ax.plot(rounds, ys, color=CLIENT_COLORS.get(cid, "gray"),
                linewidth=1.8, marker="o", markersize=4, alpha=0.8, zorder=3)


def _plot_avg_line(ax, rounds, avg_ranks):
    ys = [avg_ranks.get(r, np.nan) for r in rounds]
    ax.plot(rounds, ys, **AVG_STYLE)


def _style_ax(ax, ylabel="Rank", compact=False):
    ax.set_xlabel("Training Round", fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(labelsize=16)
    nbins_x = 6 if compact else 10
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=nbins_x))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=-0.5)


def plot_unified(hetlora_log, fahqlora_log, adaslora_log, output_path,
                 y_max=None, task_label=None):
    het_avg, het_client, het_meta = parse_hetlora_ranks_single_run(hetlora_log)
    fah_avg, fah_client = parse_fah_ranks(fahqlora_log)
    (ada_avg_compute, ada_client_compute,
     ada_avg_upload, ada_client_upload,
     ada_avg_download, ada_client_download) = parse_adasparse_ranks(adaslora_log)

    het_rounds = sorted(het_client.keys())
    het_clients = sorted(het_meta["clients"])
    fah_rounds = sorted(fah_client.keys())
    fah_clients = sorted({cid for r in fah_client.values() for cid in r})
    ada_rounds = sorted(ada_client_compute.keys())
    ada_clients = sorted({cid for r in ada_client_compute.values() for cid in r})

    if y_max is None:
        all_ranks = []
        for r in het_rounds:
            all_ranks.extend(het_client[r].values())
        for r in fah_rounds:
            all_ranks.extend(fah_client[r].values())
        for r in ada_rounds:
            all_ranks.extend(ada_client_compute[r].values())
            all_ranks.extend(ada_client_upload.get(r, {}).values())
            all_ranks.extend(ada_client_download.get(r, {}).values())
        y_max = max(all_ranks) * 1.08 if all_ranks else 220

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.30, wspace=0.25,
                           bottom=0.16, top=0.95,
                           left=0.06, right=0.98)

    # --- Top-left: HetLoRA ---
    ax_het = fig.add_subplot(gs[0, 0:2])
    ax_het.set_title("HetLoRA (decay=0.8)", fontsize=18, fontweight="bold")
    client_r_max = het_meta.get("client_init_ranks", {})
    _plot_client_lines(ax_het, het_rounds, het_client, het_clients)
    _plot_avg_line(ax_het, het_rounds, het_avg)
    _style_ax(ax_het)
    ax_het.set_ylim(bottom=0, top=y_max)

    # --- Top-right: FAH-QLoRA ---
    ax_fah = fig.add_subplot(gs[0, 2:4])
    ax_fah.set_title("FAH-QLoRA (init_r=64, λ=1)", fontsize=18, fontweight="bold")
    _plot_client_lines(ax_fah, fah_rounds, fah_client, fah_clients)
    _plot_avg_line(ax_fah, fah_rounds, fah_avg)
    _style_ax(ax_fah)
    ax_fah.set_ylim(bottom=0, top=y_max)

    # --- Bottom-left: AdaS-LoRA compute rank ---
    ax_comp = fig.add_subplot(gs[1, 0:2])
    ax_comp.set_title("AdaS-LoRA — Compute Rank (γ=0.8)", fontsize=18, fontweight="bold")
    _plot_client_lines(ax_comp, ada_rounds, ada_client_compute, ada_clients)
    _plot_avg_line(ax_comp, ada_rounds, ada_avg_compute)
    _style_ax(ax_comp)
    ax_comp.set_ylim(bottom=0, top=y_max)

    # --- Bottom-mid: AdaS-LoRA upload rank ---
    ax_up = fig.add_subplot(gs[1, 2])
    ax_up.set_title("AdaS-LoRA — Upload Rank", fontsize=18, fontweight="bold")
    for cid in ada_clients:
        ys = [ada_client_upload.get(r, {}).get(cid, np.nan) for r in ada_rounds]
        ax_up.plot(ada_rounds, ys, color=CLIENT_COLORS.get(cid, "gray"),
                   linewidth=1.8, marker="^", markersize=4, alpha=0.8, zorder=3)
    up_avg = [ada_avg_upload.get(r, np.nan) for r in ada_rounds]
    ax_up.plot(ada_rounds, up_avg, color=AVG_COLOR, linewidth=3,
               linestyle="--", marker="^", markersize=6, zorder=10)
    _style_ax(ax_up, compact=True)
    ax_up.set_ylim(bottom=0, top=y_max)

    # --- Bottom-right: AdaS-LoRA download rank ---
    ax_dl = fig.add_subplot(gs[1, 3])
    ax_dl.set_title("AdaS-LoRA — Download Rank", fontsize=18, fontweight="bold")
    for cid in ada_clients:
        ys = [ada_client_download.get(r, {}).get(cid, np.nan) for r in ada_rounds]
        ax_dl.plot(ada_rounds, ys, color=CLIENT_COLORS.get(cid, "gray"),
                   linewidth=1.8, marker="v", markersize=4, alpha=0.8, zorder=3)
    dl_avg = [ada_avg_download.get(r, np.nan) for r in ada_rounds]
    ax_dl.plot(ada_rounds, dl_avg, color=AVG_COLOR, linewidth=3,
               linestyle="--", marker="v", markersize=6, zorder=10)
    _style_ax(ax_dl, compact=True)
    ax_dl.set_ylim(bottom=0, top=y_max)

    # --- Shared legend with r_max per client ---
    shared_handles = []
    for cid in sorted(CLIENT_COLORS.keys()):
        r_max = client_r_max.get(cid)
        lbl = f"Client {cid} (r_max={r_max})" if r_max else f"Client {cid}"
        shared_handles.append(Line2D([0], [0], color=CLIENT_COLORS[cid], linewidth=2,
                                     marker="o", markersize=6, label=lbl))
    shared_handles.append(Line2D([0], [0], label="Avg Rank", **AVG_STYLE))
    shared_handles.append(Line2D([0], [0], color="gray", linewidth=2, marker="^",
                                 markersize=6, label="Upload"))
    shared_handles.append(Line2D([0], [0], color="gray", linewidth=2, marker="v",
                                 markersize=6, label="Download"))

    fig.legend(handles=shared_handles, loc="lower center", ncol=5, fontsize=14,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.005))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified rank evolution comparison plot (v2 — split comm panels)")
    parser.add_argument("--hetlora", required=True, help="HetLoRA unified_log.log path")
    parser.add_argument("--fahqlora", required=True, help="FAH-QLoRA unified_log.log path")
    parser.add_argument("--adaslora", required=True, help="AdaSparse-LoRA v2 unified_log.log path")
    parser.add_argument("--output", default="0_results/archive/unified_rank_evolution_v2.png")
    parser.add_argument("--y-max", type=float, default=None,
                        help="Shared y-axis maximum (auto-detected if omitted)")
    parser.add_argument("--task", default=None, help="Task label for the title")
    args = parser.parse_args()

    plot_unified(args.hetlora, args.fahqlora, args.adaslora, args.output,
                 y_max=args.y_max, task_label=args.task)


if __name__ == "__main__":
    main()
