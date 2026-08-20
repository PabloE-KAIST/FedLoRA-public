#!/usr/bin/env python3
"""
AdaSparse-LoRAv3 Rank Evolution Visualization

This script parses an exp_print.log file from AdaSparse-LoRAv3 runs and visualizes
how component counts evolve over training rounds using two subplots:
  - Left: compute/survivor component evolution per client
  - Right: upload and download component evolution per client

Usage:
    python rank_evolution_adasparsev3.py --log_file <path_to_exp_print.log>
    python rank_evolution_adasparsev3.py  # Interactive file selection
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import matplotlib.pyplot as plt


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


def compute_cumulative_used_ranks(
    client_compute: Dict[int, Dict[int, int]],
    client_upload: Dict[int, Dict[int, int]],
    client_download: Dict[int, Dict[int, int]],
) -> Tuple[int, int, int]:
    """Compute cumulative used component counts across all rounds and clients."""
    total_compute = sum(sum(client_map.values()) for client_map in client_compute.values())
    total_upload = sum(sum(client_map.values()) for client_map in client_upload.values())
    total_download = sum(sum(client_map.values()) for client_map in client_download.values())
    return total_compute, total_upload, total_download


def parse_adasparse_v3_ranks(log_path: str) -> Tuple[
    Dict[int, float], Dict[int, Dict[int, int]],
    Dict[int, float], Dict[int, Dict[int, int]],
    Dict[int, float], Dict[int, Dict[int, int]],
]:
    """
    Parse AdaSparse-LoRAv3 component-count data from an exp_print.log file.

    Returns:
        avg_compute: Dict mapping round -> average compute/survivor component count
        client_compute: Dict mapping round -> {client_id -> compute/survivor component count}
        avg_upload: Dict mapping round -> average upload component count
        client_upload: Dict mapping round -> {client_id -> upload component count}
        avg_download: Dict mapping round -> average download component count
        client_download: Dict mapping round -> {client_id -> download component count}
    """
    avg_compute = {}
    avg_upload = {}
    avg_download = {}
    client_compute = defaultdict(dict)
    client_upload = defaultdict(dict)
    client_download = defaultdict(dict)

    round_begin_pattern = re.compile(
        r"Starting(?: a new)? training(?: round)? \(Round #(\d+)\)"
    )
    server_round_start_pattern = re.compile(
        r"Server round (\d+) start:"
    )
    client_init_pattern = re.compile(
        r"Client (\d+): Initialized v3 layer state with \d+ layers, .* total (\d+) ComponentIDs"
    )
    bootstrap_downlink_pattern = re.compile(
        r"Client (\d+) bootstrap downlink \(round (\d+)\): .* n_components=(\d+)"
    )
    round_start_pattern = re.compile(
        r"Client (\d+) round start: n_layers=\d+, n_survivor_components=(\d+), "
        r"prior_upload_count=(\d+), prior_download_count=(\d+)"
    )
    upload_prepared_pattern = re.compile(
        r"Client (\d+): Upload prepared - "
        r"survivor_components=(\d+), upload_components=(\d+), sample_size="
    )

    init_components = {}
    current_round = None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            init_match = client_init_pattern.search(line)
            if init_match:
                cid = int(init_match.group(1))
                total_components = int(init_match.group(2))
                init_components[cid] = total_components
                continue

            round_match = round_begin_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
                continue

            server_round_match = server_round_start_pattern.search(line)
            if server_round_match:
                current_round = int(server_round_match.group(1))
                continue

            bootstrap_match = bootstrap_downlink_pattern.search(line)
            if bootstrap_match:
                cid = int(bootstrap_match.group(1))
                rnd = int(bootstrap_match.group(2))
                n_components = int(bootstrap_match.group(3))
                client_download[rnd][cid] = n_components
                continue

            if current_round is None:
                continue

            start_match = round_start_pattern.search(line)
            if start_match:
                cid = int(start_match.group(1))
                survivor_count = int(start_match.group(2))
                prior_download_count = int(start_match.group(4))
                client_compute[current_round][cid] = survivor_count
                client_download[current_round][cid] = prior_download_count
                continue

            upload_match = upload_prepared_pattern.search(line)
            if upload_match:
                cid = int(upload_match.group(1))
                upload_count = int(upload_match.group(3))
                client_upload[current_round][cid] = upload_count
                continue

    # Fallback bootstrap state if round-0 client round-start lines are missing
    if init_components and 0 not in client_compute:
        for cid, count in init_components.items():
            client_compute[0][cid] = count
            client_download[0][cid] = client_download[0].get(cid, count)
            client_upload[0][cid] = 0

    all_rounds = sorted(set(client_compute.keys()) | set(client_upload.keys()) | set(client_download.keys()))
    for r in all_rounds:
        if client_compute.get(r):
            vals = list(client_compute[r].values())
            avg_compute[r] = float(sum(vals)) / len(vals)
        if client_upload.get(r):
            vals = list(client_upload[r].values())
            avg_upload[r] = float(sum(vals)) / len(vals)
        if client_download.get(r):
            vals = list(client_download[r].values())
            avg_download[r] = float(sum(vals)) / len(vals)

    return (
        dict(avg_compute), dict(client_compute),
        dict(avg_upload), dict(client_upload),
        dict(avg_download), dict(client_download),
    )


def plot_rank_evolution(
    avg_compute: Dict[int, float],
    client_compute: Dict[int, Dict[int, int]],
    avg_upload: Dict[int, float],
    client_upload: Dict[int, Dict[int, int]],
    avg_download: Dict[int, float],
    client_download: Dict[int, Dict[int, int]],
    output_path: str = None,
    title: str = "AdaSparse-LoRAv3 Rank Evolution"
):
    """Create a two-panel plot with explicit per-client upload/download legend."""
    if not avg_compute and not avg_upload and not avg_download:
        print("Error: No component-count data found in log file.")
        return

    rounds = sorted(set(avg_compute.keys()) | set(avg_upload.keys()) | set(avg_download.keys()))

    all_clients = sorted(set().union(
        *[set(d.keys()) for d in client_compute.values()] if client_compute else [set()],
        *[set(d.keys()) for d in client_upload.values()] if client_upload else [set()],
        *[set(d.keys()) for d in client_download.values()] if client_download else [set()],
    ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7), sharex=True, sharey=True)

    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(all_clients))]

    # Left subplot: compute rank/component evolution per client
    for idx, client_id in enumerate(all_clients):
        client_rounds, client_vals = [], []
        for r in rounds:
            if client_id in client_compute.get(r, {}):
                client_rounds.append(r)
                client_vals.append(client_compute[r][client_id])

        if client_vals:
            ax1.plot(
                client_rounds, client_vals,
                marker='o', markersize=4, linewidth=1.5,
                color=colors[idx], alpha=0.75,
                label=f'Client {client_id}'
            )

    avg_compute_rounds = [r for r in rounds if r in avg_compute]
    avg_compute_vals = [avg_compute[r] for r in avg_compute_rounds]
    ax1.plot(
        avg_compute_rounds, avg_compute_vals,
        marker='s', markersize=6, linewidth=3,
        color='black', linestyle='--',
        label='Global Avg Compute'
    )

    ax1.set_xlabel('Training Round', fontsize=12)
    ax1.set_ylabel('Component Count', fontsize=12)
    ax1.set_title('Compute / Survivor Components', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Right subplot: explicit upload and download per client
    for idx, client_id in enumerate(all_clients):
        up_rounds, up_vals = [], []
        down_rounds, down_vals = [], []

        for r in rounds:
            if client_id in client_upload.get(r, {}):
                up_rounds.append(r)
                up_vals.append(client_upload[r][client_id])
            if client_id in client_download.get(r, {}):
                down_rounds.append(r)
                down_vals.append(client_download[r][client_id])

        if up_vals:
            ax2.plot(
                up_rounds, up_vals,
                marker='o', markersize=4, linewidth=1.5,
                color=colors[idx], alpha=0.75,
                label=f'Client {client_id} upload'
            )
        if down_vals:
            ax2.plot(
                down_rounds, down_vals,
                marker='x', markersize=4, linewidth=1.5,
                linestyle='--', color=colors[idx], alpha=0.75,
                label=f'Client {client_id} download'
            )

    avg_upload_rounds = [r for r in rounds if r in avg_upload]
    avg_upload_vals = [avg_upload[r] for r in avg_upload_rounds]
    avg_download_rounds = [r for r in rounds if r in avg_download]
    avg_download_vals = [avg_download[r] for r in avg_download_rounds]

    ax2.plot(
        avg_upload_rounds, avg_upload_vals,
        marker='s', markersize=6, linewidth=3,
        color='black', linestyle='-',
        label='Global Avg Upload'
    )
    ax2.plot(
        avg_download_rounds, avg_download_vals,
        marker='d', markersize=6, linewidth=3,
        color='black', linestyle='--',
        label='Global Avg Download'
    )

    ax2.set_xlabel('Training Round', fontsize=12)
    ax2.set_title('Communication Components', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    tick_interval = max(1, len(rounds) // 20)
    if len(rounds) > 20:
        tick_interval = 10
    xticks = [r for r in rounds if r % tick_interval == 0 or r == rounds[-1]]
    for ax in (ax1, ax2):
        ax.set_xticks(xticks)
        ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)

    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, framealpha=0.9, ncol=1)

    total_compute, total_upload, total_download = compute_cumulative_used_ranks(
        client_compute, client_upload, client_download
    )

    footer_text = (
        f"Cumulative used components  |  Compute: {total_compute}  |  "
        f"Upload: {total_upload}  |  Download: {total_download}"
    )
    fig.text(0.5, 0.01, footer_text, ha='center', va='bottom', fontsize=11)

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    plt.show()


def select_log_file() -> str:
    """Interactive file selection using tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        file_path = filedialog.askopenfilename(
            title="Select exp_print.log file",
            filetypes=[("Log files", "*.log"), ("All files", "*.*")],
            initialdir=Path(__file__).parent / "exp"
        )
        root.destroy()
        return file_path
    except ImportError:
        return input("Enter path to exp_print.log: ").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AdaSparse-LoRAv3 component evolution from exp_print.log"
    )
    parser.add_argument(
        '--log_file', nargs='?', default=None,
        help='Path to exp_print.log file (interactive selection if not provided)'
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output path for saving the plot (optional)'
    )
    parser.add_argument(
        '-t', '--title', default='AdaSparse-LoRAv3 Rank Evolution',
        help='Plot title'
    )
    args = parser.parse_args()

    log_path = args.log_file
    if log_path is None:
        log_path = select_log_file()

    if not log_path or not Path(log_path).exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    output_dir = prepare_run_output_dir(log_path, 'rank_evolution_adasparsev3')
    console_log_path = output_dir / "console_output.txt"

    original_stdout = sys.stdout
    console_log_file = open(console_log_path, "w", encoding="utf-8")
    sys.stdout = TeeStdout(original_stdout, console_log_file)

    try:
        print(f"Parsing: {log_path}")

        (
            avg_compute, client_compute,
            avg_upload, client_upload,
            avg_download, client_download,
        ) = parse_adasparse_v3_ranks(log_path)

        if not avg_compute and not avg_upload and not avg_download:
            print("Error: No AdaSparse-LoRAv3 component data found in the log file.")
            print("Make sure this is an exp_print.log from an AdaSparse-LoRAv3 run.")
            sys.exit(1)

        all_rounds = sorted(set(avg_compute.keys()) | set(avg_upload.keys()) | set(avg_download.keys()))
        all_clients = sorted(set().union(
            *[set(d.keys()) for d in client_compute.values()] if client_compute else [set()],
            *[set(d.keys()) for d in client_upload.values()] if client_upload else [set()],
            *[set(d.keys()) for d in client_download.values()] if client_download else [set()],
        ))
        print(f"Found {len(all_rounds)} rounds with component data")
        print(f"Clients: {all_clients}")
        print(f"Output directory: {output_dir}")

        output_path = args.output or str(output_dir / "rank_evolution.png")

        plot_rank_evolution(
            avg_compute, client_compute,
            avg_upload, client_upload,
            avg_download, client_download,
            output_path, args.title,
        )
        print(f"Console log saved to: {console_log_path}")
    finally:
        sys.stdout = original_stdout
        console_log_file.close()


if __name__ == '__main__':
    main()
