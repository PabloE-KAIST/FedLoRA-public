#!/usr/bin/env python3
"""
Plot bandwidth history from federated learning experiments.

This script reads the bandwidth history file generated during training and creates:
1. Timeseries plots per client (uplink and downlink)
2. A combined per-client figure when the number of clients is below 20
3. Timeseries plots grouped by client class

Supports both:
- New CSV format (bandwidth_history.csv) from RoundBandwidthManager
- Legacy text format (fah_bandwidth_history.txt) from FAH-QLoRA
"""

import argparse
import csv
import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path


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


def parse_csv_bandwidth_file(filepath):
    """
    Parse the new CSV bandwidth history format from RoundBandwidthManager.

    Returns:
        dict: {
            'mode': str,
            'source': str,
            'clients': {
                client_id: {
                    'class': str,
                    'history': [(round, dl_mbps, ul_mbps), ...]
                }
            }
        }
    """
    result = {
        'mode': 'unknown',
        'source': 'unknown',
        'clients': {}
    }

    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            client_id = int(row['client_id'])
            round_idx = int(row['round'])
            dl_mbps = float(row['download_mbps'])
            ul_mbps = float(row['upload_mbps'])
            trace_class = row.get('trace_class', '') or 'N/A'

            if result['mode'] == 'unknown':
                result['mode'] = row.get('mode', 'unknown')
                result['source'] = row.get('source', 'unknown')

            if client_id not in result['clients']:
                result['clients'][client_id] = {
                    'class': trace_class,
                    'history': []
                }

            result['clients'][client_id]['history'].append(
                (round_idx, dl_mbps, ul_mbps)
            )

    return result


def parse_txt_bandwidth_file(filepath):
    """
    Parse the legacy text bandwidth history format from FAH-QLoRA.

    Returns:
        dict: {
            'mode': str,
            'clients': {
                client_id: {
                    'class': str,
                    'history': [(round, dl_mbps, ul_mbps), ...]
                }
            }
        }
    """
    result = {
        'mode': 'unknown',
        'source': 'legacy',
        'clients': {}
    }

    current_client = None
    current_class = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('Bandwidth mode:'):
                result['mode'] = line.split(':', 1)[1].strip()

            match = re.match(r'Client (\d+) \(Class: (.+)\)', line)
            if match:
                current_client = int(match.group(1))
                current_class = match.group(2)
                result['clients'][current_client] = {
                    'class': current_class,
                    'history': []
                }
                continue

            if current_client is not None and line and not line.startswith('-') and not line.startswith('Round'):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        round_idx = int(parts[0])
                        dl_mbps = float(parts[1])
                        ul_mbps = float(parts[2])
                        result['clients'][current_client]['history'].append(
                            (round_idx, dl_mbps, ul_mbps)
                        )
                    except (ValueError, IndexError):
                        continue

    return result


def parse_bandwidth_file(filepath):
    """
    Parse bandwidth history file, auto-detecting format.

    Returns:
        dict with mode, source, and client data
    """
    if filepath.endswith('.csv'):
        return parse_csv_bandwidth_file(filepath)
    else:
        return parse_txt_bandwidth_file(filepath)


def plot_per_client(data, output_dir, generate_individual=False):
    """Create the combined all-clients figure and optionally per-client plots."""
    os.makedirs(output_dir, exist_ok=True)

    sorted_clients = sorted(data['clients'].items())

    if generate_individual:
        for client_id, client_data in sorted_clients:
            history = client_data['history']
            if not history:
                continue

            rounds, dl_values, ul_values = zip(*history)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(
                f'Client {client_id} Bandwidth History (Class: {client_data["class"]})',
                fontsize=14,
                fontweight='bold'
            )

            ax1.plot(rounds, dl_values, 'b-', linewidth=1.5, label='Downlink')
            ax1.set_xlabel('Round', fontsize=11)
            ax1.set_ylabel('Downlink Bandwidth (Mbps)', fontsize=11)
            ax1.set_title('Downlink Bandwidth Over Time', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(rounds, ul_values, 'r-', linewidth=1.5, label='Uplink')
            ax2.set_xlabel('Round', fontsize=11)
            ax2.set_ylabel('Uplink Bandwidth (Mbps)', fontsize=11)
            ax2.set_title('Uplink Bandwidth Over Time', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            if len(rounds) <= 10:
                ax1.set_xticks(rounds)
                ax2.set_xticks(rounds)
            else:
                step = max(1, len(rounds) // 4)
                ax1.set_xticks(rounds[::step])
                ax2.set_xticks(rounds[::step])

            plt.tight_layout()

            output_file = os.path.join(output_dir, f'client_{client_id}_bandwidth.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved plot for client {client_id}: {output_file}")
    else:
        print("Skipping individual per-client bandwidth plots (enable with --per-client-plots).")

    if len(sorted_clients) < 20:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        fig.suptitle('Bandwidth History for All Clients', fontsize=14, fontweight='bold')

        colors = plt.cm.tab20(np.linspace(0, 1, max(len(sorted_clients), 1)))

        for idx, (client_id, client_data) in enumerate(sorted_clients):
            history = client_data['history']
            if not history:
                continue

            rounds, dl_values, ul_values = zip(*history)
            label = f'Client {client_id}'
            color = colors[idx]

            ax1.plot(rounds, dl_values, linewidth=1.5, label=label, color=color)
            ax2.plot(rounds, ul_values, linewidth=1.5, label=label, color=color)

        ax1.set_ylabel('Downlink Bandwidth (Mbps)', fontsize=11)
        ax1.set_title('Downlink Bandwidth Over Time', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', ncol=2, fontsize=9)

        ax2.set_xlabel('Round', fontsize=11)
        ax2.set_ylabel('Uplink Bandwidth (Mbps)', fontsize=11)
        ax2.set_title('Uplink Bandwidth Over Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', ncol=2, fontsize=9)
        
        all_rounds = sorted({r for _, client_data in sorted_clients for r, _, _ in client_data['history']})

        if len(all_rounds) <= 10:
            ax1.set_xticks(all_rounds)
            ax2.set_xticks(all_rounds)
        else:
            step = max(1, len(all_rounds) // 4)
            ax1.set_xticks(all_rounds[::step])
            ax2.set_xticks(all_rounds[::step])


        plt.tight_layout()
        combined_output = os.path.join(output_dir, 'all_clients_bandwidth.png')
        plt.savefig(combined_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved combined all-clients plot: {combined_output}")
    else:
        print(
            f"Skipping combined all-clients plot because client count is {len(sorted_clients)} (>= 20)."
        )


def plot_by_class(data, output_dir):
    """Create timeseries plots grouped by client class."""
    os.makedirs(output_dir, exist_ok=True)

    class_data = defaultdict(lambda: {'dl': defaultdict(list), 'ul': defaultdict(list)})

    for client_id, client_data in data['clients'].items():
        class_name = client_data['class']
        history = client_data['history']

        for round_idx, dl_mbps, ul_mbps in history:
            class_data[class_name]['dl'][round_idx].append(dl_mbps)
            class_data[class_name]['ul'][round_idx].append(ul_mbps)

    fig, (ax_dl, ax_ul) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Bandwidth by Client Class', fontsize=14, fontweight='bold')
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_data)))

    for idx, (class_name, class_values) in enumerate(class_data.items()):
        color = colors[idx]

        dl_rounds = sorted(class_values['dl'].keys())
        dl_means = [np.mean(class_values['dl'][r]) for r in dl_rounds]
        dl_stds = [np.std(class_values['dl'][r]) for r in dl_rounds]

        ax_dl.plot(
            dl_rounds,
            dl_means,
            'o-',
            linewidth=2,
            markersize=4,
            label=f'{class_name} (mean)',
            color=color
        )
        ax_dl.fill_between(
            dl_rounds,
            [m - s for m, s in zip(dl_means, dl_stds)],
            [m + s for m, s in zip(dl_means, dl_stds)],
            alpha=0.2,
            color=color,
        )

        ul_rounds = sorted(class_values['ul'].keys())
        ul_means = [np.mean(class_values['ul'][r]) for r in ul_rounds]
        ul_stds = [np.std(class_values['ul'][r]) for r in ul_rounds]

        ax_ul.plot(
            ul_rounds,
            ul_means,
            'o-',
            linewidth=2,
            markersize=4,
            label=f'{class_name} (mean)',
            color=color
        )
        ax_ul.fill_between(
            ul_rounds,
            [m - s for m, s in zip(ul_means, ul_stds)],
            [m + s for m, s in zip(ul_means, ul_stds)],
            alpha=0.2,
            color=color,
        )

        all_rounds = sorted({
            r
            for class_values in class_data.values()
            for r in class_values['dl'].keys()
        })

        if len(all_rounds) <= 10:
            ax_dl.set_xticks(all_rounds)
            ax_ul.set_xticks(all_rounds)
        else:
            step = max(1, len(all_rounds) // 4)
            ax_dl.set_xticks(all_rounds[::step])
            ax_ul.set_xticks(all_rounds[::step])
    

    ax_dl.set_ylabel('Downlink Bandwidth (Mbps)', fontsize=12)
    ax_dl.set_title('Downlink Bandwidth by Client Class', fontsize=12)
    ax_dl.grid(True, alpha=0.3)
    ax_dl.legend(loc='best')

    ax_ul.set_xlabel('Round', fontsize=12)
    ax_ul.set_ylabel('Uplink Bandwidth (Mbps)', fontsize=12)
    ax_ul.set_title('Uplink Bandwidth by Client Class', fontsize=12)
    ax_ul.grid(True, alpha=0.3)
    ax_ul.legend(loc='best')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'bandwidth_by_class.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined bandwidth-by-class plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot bandwidth history from federated learning experiments'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to bandwidth history file (CSV or legacy TXT)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: a bandwidth_history subfolder next to the input file)'
    )
    parser.add_argument(
        '--per-client-plots',
        action='store_true',
        help='Also generate individual client bandwidth figures under per_client_plots/'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = prepare_run_output_dir(args.input_file, 'bandwidth_history')

    console_log_path = output_dir / 'bandwidth_summary.txt'

    original_stdout = sys.stdout
    console_log_file = open(console_log_path, 'w', encoding='utf-8')
    
    sys.stdout = TeeStdout(original_stdout, console_log_file)

    try:
        print(f"Parsing bandwidth history from: {args.input_file}")
        print(f"Output directory: {output_dir}")

        data = parse_bandwidth_file(args.input_file)

        if not data['clients']:
            print("Error: No client data found in file")
            return 1

        print(f"Found {len(data['clients'])} clients")
        print(f"Bandwidth mode: {data['mode']}")

        print(f"\nGenerating plots in: {output_dir}")

        per_client_dir = output_dir / 'per_client_plots'
        plot_per_client(
            data,
            str(per_client_dir),
            generate_individual=args.per_client_plots,
        )
        plot_by_class(data, str(output_dir))

        print("\nPlotting complete!")
        print(f"Console log saved to: {console_log_path}")
        return 0
    finally:
        sys.stdout = original_stdout
        console_log_file.close()

if __name__ == '__main__':
    exit(main())