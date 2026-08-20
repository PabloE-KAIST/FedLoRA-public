#!/usr/bin/env python3
"""
FAH-QLoRA Rank Evolution Visualization

This script parses an exp_print.log file from FAH-QLoRA runs and visualizes
how the global average rank and each client's rank evolve over training rounds.

Usage:
    python plot_fah_rank_evolution.py <path_to_exp_print.log>
    python plot_fah_rank_evolution.py  # Interactive file selection
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

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


def parse_fah_ranks(log_path: str) -> Tuple[Dict[int, float], Dict[int, Dict[int, int]]]:
    """
    Parse FAH rank data from an exp_print.log file.
    
    Returns:
        avg_ranks: Dict mapping round -> average rank
        client_ranks: Dict mapping round -> {client_id -> rank}
    """
    avg_ranks = {}
    client_ranks = defaultdict(dict)
    
    # Pattern for "Round X complete" lines
    # Example: [FAH] Round 1 complete: avg_rank=17.0, client_ranks={'c5': 17, 'c6': 17, ...}
    round_complete_pattern = re.compile(
        r'Round (\d+) complete: avg_rank=([\d.]+), client_ranks=\{([^}]+)\}'
    )
    
    # Pattern for Stage 2 output (backup)
    # Example: [FAH] Round 1 Stage 2 (cvxpy): ranks=[17, 17, 10, 25, 16, 34, 9, 8], avg=17.0
    stage2_pattern = re.compile(
        r'Round (\d+) Stage 2.*ranks=\[([^\]]+)\], avg=([\d.]+)'
    )
    
    # Pattern for initial warmup
    init_pattern = re.compile(
        r'Initialized FAH-QLoRA with homogeneous rank=(\d+) for warmup'
    )
    
    # Pattern for client count
    client_setup_pattern = re.compile(r'Client (\d+): FAH-QLoRA enabled')
    
    init_rank = None
    client_ids = set()
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for init rank
            init_match = init_pattern.search(line)
            if init_match:
                init_rank = int(init_match.group(1))
            
            # Collect client IDs
            client_setup_match = client_setup_pattern.search(line)
            if client_setup_match:
                client_ids.add(int(client_setup_match.group(1)))
            
            # Parse round complete line
            round_match = round_complete_pattern.search(line)
            if round_match:
                round_num = int(round_match.group(1))
                avg_rank = float(round_match.group(2))
                ranks_str = round_match.group(3)
                
                avg_ranks[round_num] = avg_rank
                
                # Parse client ranks from string like "'c5': 17, 'c6': 17"
                client_rank_pattern = re.compile(r"'c(\d+)':\s*(\d+)")
                for match in client_rank_pattern.finditer(ranks_str):
                    cid = int(match.group(1))
                    rank = int(match.group(2))
                    client_ranks[round_num][cid] = rank
                continue
            
            # Fallback: parse Stage 2 line
            stage2_match = stage2_pattern.search(line)
            if stage2_match:
                stage2_round = int(stage2_match.group(1))
                # Only use Stage 2 data if we don't already have data for this round
                if stage2_round not in avg_ranks:
                    ranks_list = [int(x.strip()) for x in stage2_match.group(2).split(',')]
                    avg_rank = float(stage2_match.group(3))
                    
                    avg_ranks[stage2_round] = avg_rank
                    # Assign to sorted client IDs
                    sorted_clients = sorted(client_ids) if client_ids else list(range(1, len(ranks_list) + 1))
                    for i, rank in enumerate(ranks_list):
                        if i < len(sorted_clients):
                            client_ranks[stage2_round][sorted_clients[i]] = rank
    
    # Add round 0 (warmup) if we have init_rank
    if init_rank is not None and 0 not in avg_ranks:
        avg_ranks[0] = float(init_rank)
        for cid in client_ids:
            client_ranks[0][cid] = init_rank
    
    return dict(avg_ranks), dict(client_ranks)


def plot_rank_evolution(
    avg_ranks: Dict[int, float],
    client_ranks: Dict[int, Dict[int, int]],
    output_path: str = None,
    title: str = "FAH-QLoRA Rank Evolution"
):
    """
    Create a plot showing rank evolution over training rounds.
    
    Args:
        avg_ranks: Dict mapping round -> average rank
        client_ranks: Dict mapping round -> {client_id -> rank}
        output_path: If provided, save plot to this path
        title: Plot title
    """
    if not avg_ranks:
        print("Error: No rank data found in log file.")
        return
    
    # Get sorted rounds and all client IDs
    rounds = sorted(avg_ranks.keys())
    all_clients = set()
    for round_data in client_ranks.values():
        all_clients.update(round_data.keys())
    all_clients = sorted(all_clients)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette for clients
    cmap = plt.colormaps.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(all_clients))]
    
    # Plot each client's rank evolution
    for idx, client_id in enumerate(all_clients):
        client_round_ranks = []
        client_rounds = []
        for r in rounds:
            if client_id in client_ranks.get(r, {}):
                client_rounds.append(r)
                client_round_ranks.append(client_ranks[r][client_id])
        
        if client_round_ranks:
            ax.plot(
                client_rounds, client_round_ranks,
                marker='o', markersize=4, linewidth=1.5,
                color=colors[idx], alpha=0.7,
                label=f'Client {client_id}'
            )
    
    # Plot global average rank (thicker line)
    avg_round_ranks = [avg_ranks[r] for r in rounds]
    ax.plot(
        rounds, avg_round_ranks,
        marker='s', markersize=6, linewidth=3,
        color='black', linestyle='--',
        label='Global Avg Rank'
    )
    
    # Formatting
    ax.set_xlabel('Training Round', fontsize=12)
    ax.set_ylabel('Rank', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Only show every 10th round on X axis for readability
    tick_interval = max(1, len(rounds) // 20)  # Aim for ~20 ticks max
    if len(rounds) > 20:
        tick_interval = 10
    xticks = [r for r in rounds if r % tick_interval == 0 or r == rounds[-1]]
    ax.set_xticks(xticks)
    ax.set_xlim(min(rounds) - 0.5, max(rounds) + 0.5)
    
    # Legend outside plot
    ax.legend(
        loc='upper left', bbox_to_anchor=(1.02, 1),
        fontsize=9, framealpha=0.9
    )
    
    plt.tight_layout()
    
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
        root.withdraw()  # Hide main window
        
        file_path = filedialog.askopenfilename(
            title="Select exp_print.log file",
            filetypes=[
                ("Log files", "*.log"),
                ("All files", "*.*")
            ],
            initialdir=Path(__file__).parent / "exp"
        )
        root.destroy()
        return file_path
    except ImportError:
        # Fallback to manual input
        return input("Enter path to exp_print.log: ").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FAH-QLoRA rank evolution from exp_print.log"
    )
    parser.add_argument(
        '--log_file', nargs='?', default=None,
        help="Path to exp_print.log file (interactive selection if not provided)"
    )
    parser.add_argument(
        '-o', '--output', default=None,
        help="Output path for saving the plot (optional)"
    )
    parser.add_argument(
        '-t', '--title', default="FAH-QLoRA Rank Evolution",
        help="Plot title"
    )

    args = parser.parse_args()

    log_path = args.log_file
    if log_path is None:
        log_path = select_log_file()

    if not log_path or not Path(log_path).exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    output_dir = prepare_run_output_dir(log_path, 'rank_evolution_fahqlora')
    console_log_path = output_dir / "console_output.txt"

    original_stdout = sys.stdout
    console_log_file = open(console_log_path, "w", encoding="utf-8")
    sys.stdout = TeeStdout(original_stdout, console_log_file)

    try:
        print(f"Parsing: {log_path}")

        avg_ranks, client_ranks = parse_fah_ranks(log_path)

        if not avg_ranks:
            print("Error: No FAH rank data found in the log file.")
            print("Make sure this is an exp_print.log from a FAH-QLoRA run.")
            sys.exit(1)

        print(f"Found {len(avg_ranks)} rounds with rank data")
        print(f"Clients: {sorted(set().union(*[set(d.keys()) for d in client_ranks.values()]))}")
        print(f"Output directory: {output_dir}")

        output_path = args.output or str(output_dir / "rank_evolution.png")

        plot_rank_evolution(avg_ranks, client_ranks, output_path, args.title)
        print(f"Console log saved to: {console_log_path}")
    finally:
        sys.stdout = original_stdout
        console_log_file.close()

if __name__ == "__main__":
    main()

