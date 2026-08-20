#!/usr/bin/env python3
"""
Script to visualize system_metrics.log files from federated learning experiments.

Usage:
    python plot_system_metrics.py <path_to_system_metrics.log>

The script will generate and save plots in a "system_metrics" folder inside
that same experiment directory.
"""

import json
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Color palette
COLORS = plt.cm.tab20.colors


def _safe_float(value):
    """Return float(value) when possible, otherwise None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_total_communicated(summary_data: dict):
    if not summary_data:
        return None
    comm = summary_data.get('total_communication_megabytes', {})
    if isinstance(comm, dict):
        return _safe_float(comm.get('total_communicated_megabytes'))
    return None


def _summary_total_uploaded(summary_data: dict):
    if not summary_data:
        return None
    comm = summary_data.get('total_communication_megabytes', {})
    if isinstance(comm, dict):
        return _safe_float(comm.get('total_uploaded_megabytes'))
    return None


def _summary_total_downloaded(summary_data: dict):
    if not summary_data:
        return None
    comm = summary_data.get('total_communication_megabytes', {})
    if isinstance(comm, dict):
        return _safe_float(comm.get('total_downloaded_megabytes'))
    return None


def _summary_wallclock_time(summary_data: dict, key: str):
    if not summary_data:
        return None
    wallclock_time = summary_data.get('wallclock_time_minutes', {})
    if isinstance(wallclock_time, dict):
        return _safe_float(wallclock_time.get(key))
    return None


def _summary_aggregate_client_time(summary_data: dict, key: str):
    if not summary_data:
        return None
    aggregate_time = summary_data.get('aggregate_client_time_minutes', {})
    if isinstance(aggregate_time, dict):
        return _safe_float(aggregate_time.get(key))
    return None


def load_system_metrics(filepath: str):
    """
    Load and parse the system_metrics.log file.

    Returns:
        clients_data: List of client metrics (excluding first placeholder and last summary)
        summary_data: The summary data from the last line
    """
    clients_data = []
    summary_data = None

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse line {i+1}: {e}")
            continue

        if not isinstance(data, dict):
            print(f"Warning: Line {i+1} did not parse as a dictionary, skipping")
            continue

        # First line is typically placeholder (id=0 with all zeros)
        if i == 0 and data.get('id') == 0 and 'fl_endtime_minutes' not in data:
            continue

        # Final summary entry
        if 'fl_endtime_minutes' in data:
            summary_data = data
            continue

        clients_data.append(data)

    return clients_data, summary_data


def plot_total_time_breakdown(clients_data: list[dict], output_dir: str):
    """Plot stacked bar chart of total time breakdown per client."""
    client_ids = [d['id'] for d in clients_data]

    computing_time = [d['total_time_minutes']['total_computing_time'] for d in clients_data]
    uploading_time = [d['total_time_minutes']['total_uploading_time'] for d in clients_data]
    downloading_time = [d['total_time_minutes']['total_downloading_time'] for d in clients_data]

    x = np.arange(len(client_ids))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x, computing_time, width, label='Computing', color=COLORS[0])
    ax.bar(x, uploading_time, width, bottom=computing_time, label='Uploading', color=COLORS[2])
    ax.bar(
        x,
        downloading_time,
        width,
        bottom=[c + u for c, u in zip(computing_time, uploading_time)],
        label='Downloading',
        color=COLORS[4],
    )

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Time (minutes)')
    ax.set_title('Total Training Time Breakdown per Client')
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.legend(loc='upper right')

    for i, (c, u, d) in enumerate(zip(computing_time, uploading_time, downloading_time)):
        total = c + u + d
        ax.annotate(f'{total:.1f}', xy=(i, total), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'total_training_time_breakdown.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    print('  Saved: total_training_time_breakdown.png')


def plot_total_communication(clients_data: list[dict], output_dir: str):
    """Plot uploaded and downloaded totals per client."""
    client_ids = [d['id'] for d in clients_data]

    uploaded = [d['total_communication_megabytes']['total_uploaded_megabytes'] for d in clients_data]
    downloaded = [d['total_communication_megabytes']['total_downloaded_megabytes'] for d in clients_data]

    x = np.arange(len(client_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width / 2, uploaded, width, label='Uploaded', color=COLORS[6])
    ax.bar(x + width / 2, downloaded, width, label='Downloaded', color=COLORS[8])

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Data (MB)')
    ax.set_title('Total Uploaded+Downloaded Amount')
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'total_communicated_amount.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    print('  Saved: total_communicated_amount.png')


def plot_memory_breakdown(clients_data: list[dict], output_dir: str):
    """Plot total memory breakdown per client and per-round component evolution."""
    client_ids = [d['id'] for d in clients_data]

    frozen_model = [d['total_memory_megabytes'].get('final_memory_frozenModel', 0) for d in clients_data]
    lora_weights = [d['total_memory_megabytes'].get('final_memory_LoRAWeights', 0) for d in clients_data]
    activations = [d['total_memory_megabytes'].get('final_memory_Activations', 0) for d in clients_data]
    optimizer = [d['total_memory_megabytes'].get('final_memory_OptimizerStates', 0) for d in clients_data]
    gradients = [d['total_memory_megabytes'].get('final_memory_Gradients', 0) for d in clients_data]

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Memory Breakdown and Per-Round Component Evolution', fontsize=14)

    ax = axes[0, 0]
    x = np.arange(len(client_ids))
    width = 0.6

    bottom = np.zeros(len(client_ids))
    components = [
        (frozen_model, 'Frozen Model', COLORS[0]),
        (lora_weights, 'LoRA Weights', COLORS[2]),
        (activations, 'Activations', COLORS[4]),
        (optimizer, 'Optimizer States', COLORS[6]),
        (gradients, 'Gradients', COLORS[8]),
    ]

    for values, label, color in components:
        ax.bar(x, values, width, bottom=bottom, label=label, color=color)
        bottom += np.array(values, dtype=float)

    ax.set_xlabel('Client ID')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Breakdown per Client at Last Training Round')
    ax.set_xticks(x)
    ax.set_xticklabels(client_ids)
    ax.legend(loc='upper right', fontsize=9)

    per_round_panels = [
        ('final_memory_LoRAWeights', 'LoRA Weights', axes[0, 1]),
        ('final_memory_frozenModel', 'Frozen Model', axes[1, 0]),
        ('final_memory_OptimizerStates', 'Optimizer States', axes[1, 1]),
        ('final_memory_Activations', 'Activations', axes[2, 0]),
        ('final_memory_Gradients', 'Gradients', axes[2, 1]),
    ]

    for key, title, ax in per_round_panels:
        for i, client in enumerate(clients_data):
            per_round = client.get('per_round_memory_megabytes', {})
            if not per_round:
                continue

            rounds = sorted(per_round.keys(), key=lambda x: int(x.replace('round', '')))
            round_nums = [int(r.replace('round', '')) for r in rounds]
            values = []
            for r in rounds:
                rd = per_round.get(r, {})
                values.append(rd.get(key, np.nan) if isinstance(rd, dict) else np.nan)

            ax.plot(
                round_nums,
                values,
                marker='o',
                markersize=3,
                linewidth=1.5,
                label=f'Client {client["id"]}',
                color=COLORS[i % len(COLORS)],
                alpha=0.75,
            )

        ax.set_xlabel('Round')
        ax.set_ylabel('Memory (MB)')
        ax.set_title(f'Per-Round {title} Memory')

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(6, len(handles)), fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'total_memory_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: total_memory_breakdown.png')


def plot_per_round_time(clients_data: list[dict], output_dir: str):
    """Plot per-round training time for all clients."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    time_components = [
        ('training_time', 'Training Time', axes[0, 0]),
        ('computing_time', 'Computing Time', axes[0, 1]),
        ('uploading_time', 'Uploading Time', axes[1, 0]),
        ('downloading_time', 'Downloading Time', axes[1, 1]),
    ]

    for component, title, ax in time_components:
        for i, client in enumerate(clients_data):
            per_round = client.get('per_round_time_minutes', {})
            if not per_round:
                continue

            rounds = sorted(per_round.keys(), key=lambda x: int(x.replace('round', '')))
            round_nums = [int(r.replace('round', '')) for r in rounds]
            values = [per_round[r].get(component, 0) for r in rounds]

            ax.plot(
                round_nums,
                values,
                marker='o',
                markersize=3,
                label=f'Client {client["id"]}',
                color=COLORS[i % len(COLORS)],
                alpha=0.7,
            )

        ax.set_xlabel('Round')
        ax.set_ylabel('Time (minutes)')
        ax.set_title(f'Per-Round {title}')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'per-round_training_time.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    print('  Saved: per-round_training_time.png')


def plot_per_round_communication(clients_data: list[dict], output_dir: str):
    """Plot per-round communication for all clients."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    comm_components = [
        ('communicated_megabytes', 'Uploaded+Downloaded Amount', axes[0]),
        ('uploaded_megabytes', 'Uploaded', axes[1]),
        ('downloaded_megabytes', 'Downloaded', axes[2]),
    ]

    for component, title, ax in comm_components:
        for i, client in enumerate(clients_data):
            per_round = client.get('per_round_communication_megabytes', {})
            if not per_round:
                continue

            rounds = sorted(per_round.keys(), key=lambda x: int(x.replace('round', '')))
            round_nums = [int(r.replace('round', '')) for r in rounds]
            values = [per_round[r].get(component, 0) for r in rounds]

            ax.plot(
                round_nums,
                values,
                marker='o',
                markersize=3,
                label=f'Client {client["id"]}',
                color=COLORS[i % len(COLORS)],
                alpha=0.7,
            )

        ax.set_xlabel('Round')
        ax.set_ylabel('Data (MB)')
        ax.set_title(f'Per-Round {title}')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'per-round_communicated_amount.png'),
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    print('  Saved: per-round_communicated_amount.png')


def plot_server_totals(summary_data: dict, output_dir: str):
    """Plot server summary with estimated wallclock and aggregate communication totals."""
    if not summary_data:
        print('  Skipped: server_totals.png (no summary data)')
        return

    total_training = _summary_wallclock_time(summary_data, 'total_training_time')
    total_computing = _summary_wallclock_time(summary_data, 'total_computing_time')
    total_uploading = _summary_wallclock_time(summary_data, 'total_uploading_time')
    total_downloading = _summary_wallclock_time(summary_data, 'total_downloading_time')
    fl_endtime = _safe_float(summary_data.get('fl_endtime_minutes'))
    total_uploaded = _summary_total_uploaded(summary_data)
    total_downloaded = _summary_total_downloaded(summary_data)
    total_communicated = _summary_total_communicated(summary_data)

    time_labels = ['Computing', 'Uploading', 'Downloading']
    time_values = [
        total_computing or 0.0,
        total_uploading or 0.0,
        total_downloading or 0.0,
    ]

    comm_labels = ['Uploaded', 'Downloaded']
    comm_values = [
        total_uploaded or 0.0,
        total_downloaded or 0.0,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(time_labels, time_values, color=[COLORS[0], COLORS[2], COLORS[4]])
    axes[0].set_ylabel('Time (minutes)')
    axes[0].set_title('Estimated Wallclock Time Breakdown')
    if total_training is not None:
        axes[0].annotate(
            f'Estimated wallclock total: {total_training:.2f} min',
            xy=(0.98, 0.95),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
    if fl_endtime is not None:
        axes[0].annotate(
            f'FL end-to-end wallclock: {fl_endtime:.2f} min',
            xy=(0.98, 0.84),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
        )

    axes[1].bar(comm_labels, comm_values, color=[COLORS[6], COLORS[8]])
    axes[1].set_ylabel('Data (MB)')
    axes[1].set_title('Aggregate Uploaded/Downloaded Amount')
    if total_communicated is not None:
        axes[1].annotate(
            f'Total communicated: {total_communicated:.2f} MB',
            xy=(0.98, 0.95),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'server_totals.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: server_totals.png')


def plot_client_comparison_heatmap(clients_data: list[dict], output_dir: str):
    """Create a heatmap comparing key metrics across clients."""
    if len(clients_data) < 2:
        print('  Skipped: client_comparison_heatmap.png (need at least 2 clients)')
        return

    client_ids = [d['id'] for d in clients_data]

    metrics = {
        'Total Training (min)': [d['total_time_minutes']['total_training_time'] for d in clients_data],
        'Computing (min)': [d['total_time_minutes']['total_computing_time'] for d in clients_data],
        'Uploading (min)': [d['total_time_minutes']['total_uploading_time'] for d in clients_data],
        'Downloading (min)': [d['total_time_minutes']['total_downloading_time'] for d in clients_data],
        'Total Comm (MB)': [d['total_communication_megabytes']['total_communicated_megabytes'] for d in clients_data],
        'Final Memory (MB)': [d['total_memory_megabytes']['final_memory'] for d in clients_data],
    }

    data = []
    metric_names = list(metrics.keys())
    for name in metric_names:
        values = np.array(metrics[name], dtype=float)
        if values.max() > 0:
            normalized = values / values.max()
        else:
            normalized = np.zeros_like(values)
        data.append(normalized)

    data = np.array(data)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(client_ids)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels([f'Client {c}' for c in client_ids])
    ax.set_yticklabels(metric_names)

    for i in range(len(metric_names)):
        for j in range(len(client_ids)):
            original_value = metrics[metric_names[i]][j]
            ax.text(
                j,
                i,
                f'{original_value:.1f}',
                ha='center',
                va='center',
                fontsize=7,
                color='black' if data[i, j] < 0.7 else 'white',
            )

    ax.set_title('Client Metrics Comparison (Color: Zero-Based Normalized, Text: Actual Values)')
    plt.colorbar(im, ax=ax, label='Normalized Value')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'client_comparison_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: client_comparison_heatmap.png')


def generate_summary_text(clients_data: list[dict], summary_data: dict, output_dir: str):
    """Generate a text summary of the experiment."""
    lines = []
    lines.append('=' * 60)
    lines.append('FEDERATED LEARNING EXPERIMENT - SYSTEM METRICS SUMMARY')
    lines.append('=' * 60)
    lines.append('')

    if summary_data:
        fl_time = _safe_float(summary_data.get('fl_endtime_minutes'))
        wallclock_training = _summary_wallclock_time(summary_data, 'total_training_time')
        wallclock_computing = _summary_wallclock_time(summary_data, 'total_computing_time')
        wallclock_uploading = _summary_wallclock_time(summary_data, 'total_uploading_time')
        wallclock_downloading = _summary_wallclock_time(summary_data, 'total_downloading_time')

        aggregate_training = _summary_aggregate_client_time(summary_data, 'total_training_time')
        aggregate_computing = _summary_aggregate_client_time(summary_data, 'total_computing_time')
        aggregate_uploading = _summary_aggregate_client_time(summary_data, 'total_uploading_time')
        aggregate_downloading = _summary_aggregate_client_time(summary_data, 'total_downloading_time')

        total_comm = _summary_total_communicated(summary_data)
        total_uploaded = _summary_total_uploaded(summary_data)
        total_downloaded = _summary_total_downloaded(summary_data)

        lines.append('EXPERIMENT WALLCLOCK')
        lines.append('-' * 40)
        lines.append(
            f'FL end-to-end wallclock: {fl_time:.2f} minutes'
            if fl_time is not None else
            'FL end-to-end wallclock: N/A'
        )
        lines.append('')

        lines.append('ESTIMATED WALLCLOCK BREAKDOWN FROM PER-ROUND CLIENT MAXIMA')
        lines.append('-' * 40)
        lines.append(
            f'Estimated training wallclock:    {wallclock_training:.2f} minutes'
            if wallclock_training is not None else
            'Estimated training wallclock:    N/A'
        )
        lines.append(
            f'Estimated computing wallclock:   {wallclock_computing:.2f} minutes'
            if wallclock_computing is not None else
            'Estimated computing wallclock:   N/A'
        )
        lines.append(
            f'Estimated uploading wallclock:   {wallclock_uploading:.2f} minutes'
            if wallclock_uploading is not None else
            'Estimated uploading wallclock:   N/A'
        )
        lines.append(
            f'Estimated downloading wallclock: {wallclock_downloading:.2f} minutes'
            if wallclock_downloading is not None else
            'Estimated downloading wallclock: N/A'
        )
        lines.append('')

        lines.append('AGGREGATE CLIENT TOTALS')
        lines.append('-' * 40)
        lines.append(
            f'Summed client training time:     {aggregate_training:.2f} minutes'
            if aggregate_training is not None else
            'Summed client training time:     N/A'
        )
        lines.append(
            f'Summed client computing time:    {aggregate_computing:.2f} minutes'
            if aggregate_computing is not None else
            'Summed client computing time:    N/A'
        )
        lines.append(
            f'Summed client uploading time:    {aggregate_uploading:.2f} minutes'
            if aggregate_uploading is not None else
            'Summed client uploading time:    N/A'
        )
        lines.append(
            f'Summed client downloading time:  {aggregate_downloading:.2f} minutes'
            if aggregate_downloading is not None else
            'Summed client downloading time:  N/A'
        )
        lines.append(
            f'Total communicated:              {total_comm:.2f} MB'
            if total_comm is not None else
            'Total communicated:              N/A'
        )
        lines.append(
            f'Total uploaded:                  {total_uploaded:.2f} MB'
            if total_uploaded is not None else
            'Total uploaded:                  N/A'
        )
        lines.append(
            f'Total downloaded:                {total_downloaded:.2f} MB'
            if total_downloaded is not None else
            'Total downloaded:                N/A'
        )
        lines.append('')

    lines.append(f'Number of Clients: {len(clients_data)}')
    lines.append('')

    total_times = [d['total_time_minutes']['total_training_time'] for d in clients_data]
    total_comm_values = [d['total_communication_megabytes']['total_communicated_megabytes'] for d in clients_data]
    total_mem = [d['total_memory_megabytes']['final_memory'] for d in clients_data]

    lines.append('CLIENT STATISTICS')
    lines.append('-' * 40)
    lines.append(
        f'Training Time (min):  Mean={np.mean(total_times):.2f}, Std={np.std(total_times):.2f}, '
        f'Min={np.min(total_times):.2f}, Max={np.max(total_times):.2f}'
    )
    lines.append(
        f'Communication (MB):   Mean={np.mean(total_comm_values):.2f}, Std={np.std(total_comm_values):.2f}, '
        f'Min={np.min(total_comm_values):.2f}, Max={np.max(total_comm_values):.2f}'
    )
    lines.append(
        f'Final Memory (MB):    Mean={np.mean(total_mem):.2f}, Std={np.std(total_mem):.2f}, '
        f'Min={np.min(total_mem):.2f}, Max={np.max(total_mem):.2f}'
    )
    lines.append('')

    lines.append('PER-CLIENT DETAILS')
    lines.append('-' * 40)
    for client in clients_data:
        cid = client['id']
        tt = client['total_time_minutes']['total_training_time']
        tc = client['total_communication_megabytes']['total_communicated_megabytes']
        tm = client['total_memory_megabytes']['final_memory']
        lines.append(f'Client {cid:2d}: Time={tt:6.2f} min, Comm={tc:7.2f} MB, Memory={tm:8.2f} MB')

    lines.append('')
    lines.append('=' * 60)

    summary_text = "\n".join(lines)

    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        f.write(summary_text)

    print('  Saved: metrics_summary.txt')
    print('')
    print(summary_text)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize system_metrics.log files from federated learning experiments.'
    )
    parser.add_argument('filepath', type=str, help='Path to the system_metrics.log file')

    args = parser.parse_args()

    filepath = args.filepath

    if not os.path.exists(filepath):
        print(f'Error: File not found: {filepath}')
        return 1

    base_dir = os.path.dirname(os.path.abspath(filepath))
    output_dir = os.path.join(base_dir, 'system_metrics')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Loading: {filepath}')
    print(f'Output directory: {output_dir}')
    print('')

    clients_data, summary_data = load_system_metrics(filepath)

    if not clients_data:
        print('Error: No client data found in the file.')
        return 1

    print(f'Found {len(clients_data)} clients')
    print('')
    print('Generating plots...')

    plot_total_time_breakdown(clients_data, output_dir)
    plot_total_communication(clients_data, output_dir)
    plot_memory_breakdown(clients_data, output_dir)
    plot_per_round_time(clients_data, output_dir)
    plot_per_round_communication(clients_data, output_dir)
    plot_server_totals(summary_data, output_dir)
    plot_client_comparison_heatmap(clients_data, output_dir)

    print('')
    print('Generating summary...')
    generate_summary_text(clients_data, summary_data, output_dir)

    print('')
    print('Done!')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())