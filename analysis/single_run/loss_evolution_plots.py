#!/usr/bin/env python3
"""
Combined training and validation loss evolution visualization.

Method-agnostic utility that produces two separate figures from one exp_print.log file:

  1. validation_loss_evolution.png
  2. train_loss_evolution.png

It parses FederatedScope log payloads via ast.literal_eval so it works across
methods as long as the log contains Results_raw client entries and, when
available, server Results_weighted_avg entries.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


class TeeStdout:
    """Duplicate console output to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def prepare_run_output_dir(input_path, subdir_name):
    """Create a sibling subfolder next to the input file for all generated outputs."""
    output_dir = Path(input_path).resolve().parent / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _extract_dict_payload(line: str) -> Optional[dict]:
    """Extract and parse the dict payload from a log line, if present."""
    idx = line.find("{")
    if idx < 0:
        return None
    payload = line[idx:].strip()
    last = payload.rfind("}")
    if last >= 0:
        payload = payload[: last + 1]
    try:
        obj = ast.literal_eval(payload)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_loss_series(
    log_path: str,
    *,
    client_metric_key: str,
    client_loss_key: str,
    client_total_key: str,
    server_metric_key: Optional[str] = None,
) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
    """
    Parse per-client loss evolution and a global weighted average per round.

    Returns:
      global_weighted_avg: round -> weighted average metric
      client_losses: round -> {client_id -> metric}
    """
    client_losses: Dict[int, Dict[int, float]] = defaultdict(dict)
    round_loss_sum: Dict[int, float] = defaultdict(float)
    round_total_sum: Dict[int, float] = defaultdict(float)
    server_weighted_avg: Dict[int, float] = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            d = _extract_dict_payload(line)
            if not d:
                continue

            role = d.get("Role", "")
            rnd = d.get("Round", None)
            try:
                rnd = int(rnd)
            except Exception:
                continue

            if isinstance(role, str) and role.startswith("Client #"):
                try:
                    cid = int(role.split("#", 1)[1])
                except Exception:
                    continue

                results = d.get("Results_raw", None)
                if not isinstance(results, dict):
                    continue

                if client_metric_key in results:
                    try:
                        client_losses[rnd][cid] = float(results[client_metric_key])
                    except Exception:
                        pass

                if client_loss_key in results and client_total_key in results:
                    try:
                        round_loss_sum[rnd] += float(results[client_loss_key])
                        round_total_sum[rnd] += float(results[client_total_key])
                    except Exception:
                        pass

            elif role == "Server #" and server_metric_key is not None:
                results = d.get("Results_weighted_avg", None)
                if isinstance(results, dict) and server_metric_key in results:
                    try:
                        server_weighted_avg[rnd] = float(results[server_metric_key])
                    except Exception:
                        pass

    global_weighted_avg: Dict[int, float] = {}
    all_rounds: Set[int] = set(client_losses.keys()) | set(round_loss_sum.keys()) | set(round_total_sum.keys()) | set(server_weighted_avg.keys())

    for rnd in sorted(all_rounds):
        if rnd in server_weighted_avg:
            global_weighted_avg[rnd] = server_weighted_avg[rnd]
        elif round_total_sum.get(rnd, 0.0) > 0:
            global_weighted_avg[rnd] = round_loss_sum[rnd] / round_total_sum[rnd]
        else:
            vals = list(client_losses.get(rnd, {}).values())
            if vals:
                global_weighted_avg[rnd] = float(np.mean(vals))

    return global_weighted_avg, dict(client_losses)


def plot_loss_evolution(
    global_losses: Dict[int, float],
    client_losses: Dict[int, Dict[int, float]],
    *,
    output_path: Optional[str],
    title: str,
    y_label: str,
    global_label: str,
    only_global: bool = False,
    max_clients_in_legend: int = 20,
):
    if not global_losses and not client_losses:
        raise RuntimeError("No loss data found.")

    rounds = sorted(set(global_losses.keys()) | set(client_losses.keys()))
    if not rounds:
        raise RuntimeError("No rounds found.")

    all_clients = sorted({cid for rd in client_losses.values() for cid in rd.keys()})

    fig, ax = plt.subplots(figsize=(12, 7))

    if not only_global and all_clients:
        cmap = plt.colormaps.get_cmap("tab10")
        for idx, cid in enumerate(all_clients):
            xs, ys = [], []
            for r in rounds:
                v = client_losses.get(r, {}).get(cid, None)
                if v is not None:
                    xs.append(r)
                    ys.append(v)
            if ys:
                ax.plot(
                    xs, ys,
                    marker="o", markersize=3, linewidth=1.2,
                    color=cmap(idx % 10), alpha=0.6,
                    label=f"Client {cid}",
                )

    if global_losses:
        xs = sorted(global_losses.keys())
        ys = [global_losses[r] for r in xs]
        ax.plot(
            xs, ys,
            marker="s", markersize=5, linewidth=2.8,
            color="black", linestyle="--",
            label=global_label,
        )

    ax.set_xlabel("Training Round", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if len(rounds) > 25:
        step = max(1, len(rounds) // 10)
        xt = rounds[::step]
        if rounds[-1] not in xt:
            xt = xt + [rounds[-1]]
        ax.set_xticks(xt)
    else:
        ax.set_xticks(rounds)

    handles, labels = ax.get_legend_handles_labels()
    if not only_global and len(labels) > (1 + max_clients_in_legend):
        kept_h, kept_l = [], []
        for i in range(min(max_clients_in_legend, len(handles) - 1)):
            kept_h.append(handles[i])
            kept_l.append(labels[i])
        kept_h.append(handles[-1])
        kept_l.append(labels[-1])
        handles, labels = kept_h, kept_l

    ax.legend(
        handles, labels,
        loc="upper left", bbox_to_anchor=(1.02, 1),
        fontsize=9, framealpha=0.9,
    )

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss evolution from one exp_print.log file."
    )
    parser.add_argument("--log_file", required=True, help="Path to exp_print.log")
    parser.add_argument("--train_output", default=None, help="Output path for training-loss plot")
    parser.add_argument("--val_output", default=None, help="Output path for validation-loss plot")
    parser.add_argument("--train_title", default="Training Loss Evolution", help="Training plot title")
    parser.add_argument("--val_title", default="Validation Loss Evolution", help="Validation plot title")
    parser.add_argument("--only_global", action="store_true", help="Plot only global weighted averages")
    parser.add_argument("--max_clients_in_legend", type=int, default=20, help="Maximum number of clients to show in the legend")
    parser.add_argument("--no_train", action="store_true", help="Skip the training-loss plot")
    parser.add_argument("--no_val", action="store_true", help="Skip the validation-loss plot")

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    if args.no_train and args.no_val:
        print("Error: Both --no_train and --no_val were set. Nothing to plot.")
        sys.exit(1)

    output_dir = prepare_run_output_dir(log_path, 'loss_evolution')
    console_log_path = output_dir / "loss_summary.txt"

    original_stdout = sys.stdout
    console_log_file = open(console_log_path, "w", encoding="utf-8")
    sys.stdout = TeeStdout(original_stdout, console_log_file)

    try:
        print(f"Parsing: {log_path}")
        print(f"Output directory: {output_dir}")

        train_output = args.train_output or str(output_dir / "train_loss_evolution.png")
        val_output = args.val_output or str(output_dir / "validation_loss_evolution.png")

        if not args.no_train:
            train_global, train_clients = parse_loss_series(
                str(log_path),
                client_metric_key="train_avg_loss",
                client_loss_key="train_loss",
                client_total_key="train_total",
                server_metric_key=None,
            )
            if train_global or train_clients:
                print(f"Found training-loss data for {len(sorted(set(train_global.keys()) | set(train_clients.keys())))} rounds")
                plot_loss_evolution(
                    train_global,
                    train_clients,
                    output_path=train_output,
                    title=args.train_title,
                    y_label="Training Average Loss",
                    global_label="Global Weighted Avg",
                    only_global=args.only_global,
                    max_clients_in_legend=args.max_clients_in_legend,
                )
            else:
                print("No training loss data found.")

        if not args.no_val:
            val_global, val_clients = parse_loss_series(
                str(log_path),
                client_metric_key="val_avg_loss",
                client_loss_key="val_loss",
                client_total_key="val_total",
                server_metric_key="val_avg_loss",
            )
            if val_global or val_clients:
                print(f"Found validation-loss data for {len(sorted(set(val_global.keys()) | set(val_clients.keys())))} rounds")
                plot_loss_evolution(
                    val_global,
                    val_clients,
                    output_path=val_output,
                    title=args.val_title,
                    y_label="Validation Average Loss",
                    global_label="Server Weighted Avg",
                    only_global=args.only_global,
                    max_clients_in_legend=args.max_clients_in_legend,
                )
            else:
                print("No validation loss data found.")

        print(f"Console log saved to: {console_log_path}")
    finally:
        sys.stdout = original_stdout
        console_log_file.close()

if __name__ == "__main__":
    main()
