#!/usr/bin/env python3
"""
Merge per-client system_metrics files from distributed worker containers
into a single system_metrics.log that the analysis pipeline can consume.

Usage:
    python merge_system_metrics.py <exp_dir> [--fl-endtime MINUTES]

Reads:  <exp_dir>/worker_metrics/system_metrics_client_*.log
Writes: <exp_dir>/system_metrics.log
"""
import argparse
import glob
import json
import os
import sys

DECIMAL_PLACES = 4


def _round(val):
    return round(val, DECIMAL_PLACES)


def load_client_metrics(metrics_dir):
    """Load all per-client system_metrics files."""
    pattern = os.path.join(metrics_dir, "system_metrics_client_*.log")
    files = sorted(glob.glob(pattern))
    if not files:
        return []

    client_metrics = []
    for fpath in files:
        last_valid = None
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                cid = data.get("id")
                if cid is None or cid == 0 or cid in ("sys_avg", "sys_std"):
                    continue
                last_valid = data
        if last_valid is not None:
            client_metrics.append(last_valid)
    return client_metrics


def compute_server_aggregate(client_metrics, fl_endtime_minutes=None):
    """Compute the server (id=0) aggregate entry from per-client metrics."""
    aggregate_training_time = 0.0
    aggregate_computing_time = 0.0
    aggregate_uploading_time = 0.0
    aggregate_downloading_time = 0.0
    total_communicated_mb = 0.0
    total_uploaded_mb = 0.0
    total_downloaded_mb = 0.0

    for cm in client_metrics:
        time_metrics = cm.get("total_time_minutes", {})
        aggregate_training_time += float(time_metrics.get("total_training_time", 0.0))
        aggregate_computing_time += float(time_metrics.get("total_computing_time", 0.0))
        aggregate_uploading_time += float(time_metrics.get("total_uploading_time", 0.0))
        aggregate_downloading_time += float(time_metrics.get("total_downloading_time", 0.0))

        comm_metrics = cm.get("total_communication_megabytes", {})
        total_communicated_mb += float(comm_metrics.get("total_communicated_megabytes", 0.0))
        total_uploaded_mb += float(comm_metrics.get("total_uploaded_megabytes", 0.0))
        total_downloaded_mb += float(comm_metrics.get("total_downloaded_megabytes", 0.0))

    # Estimated wallclock from per-round client maxima
    all_rounds = set()
    for cm in client_metrics:
        all_rounds.update(cm.get("per_round_time_minutes", {}).keys())

    def _round_sort_key(round_key):
        try:
            return int(str(round_key).replace("round", ""))
        except Exception:
            return str(round_key)

    per_round_wallclock_time = {}
    wallclock_compute_total = 0.0
    wallclock_upload_total = 0.0
    wallclock_download_total = 0.0

    for round_key in sorted(all_rounds, key=_round_sort_key):
        max_compute = 0.0
        max_upload = 0.0
        max_download = 0.0

        for cm in client_metrics:
            round_time = cm.get("per_round_time_minutes", {}).get(round_key, {})
            max_compute = max(max_compute, float(round_time.get("computing_time", 0.0)))
            max_upload = max(max_upload, float(round_time.get("uploading_time", 0.0)))
            max_download = max(max_download, float(round_time.get("downloading_time", 0.0)))

        round_training = max_compute + max_upload + max_download
        per_round_wallclock_time[round_key] = {
            "training_time": _round(round_training),
            "computing_time": _round(max_compute),
            "uploading_time": _round(max_upload),
            "downloading_time": _round(max_download),
        }

        wallclock_compute_total += max_compute
        wallclock_upload_total += max_upload
        wallclock_download_total += max_download

    wallclock_training_total = (
        wallclock_compute_total + wallclock_upload_total + wallclock_download_total
    )

    if fl_endtime_minutes is None:
        fl_endtime_minutes = max(
            (float(cm.get("total_time_minutes", {}).get("total_training_time", 0.0))
             for cm in client_metrics),
            default=0.0,
        )

    return {
        "id": 0,
        "fl_endtime_minutes": _round(fl_endtime_minutes),
        "wallclock_time_minutes": {
            "total_training_time": _round(wallclock_training_total),
            "total_computing_time": _round(wallclock_compute_total),
            "total_uploading_time": _round(wallclock_upload_total),
            "total_downloading_time": _round(wallclock_download_total),
        },
        "per_round_wallclock_time_minutes": per_round_wallclock_time,
        "aggregate_client_time_minutes": {
            "total_training_time": _round(aggregate_training_time),
            "total_computing_time": _round(aggregate_computing_time),
            "total_uploading_time": _round(aggregate_uploading_time),
            "total_downloading_time": _round(aggregate_downloading_time),
        },
        "total_communication_megabytes": {
            "total_communicated_megabytes": _round(total_communicated_mb),
            "total_uploaded_megabytes": _round(total_uploaded_mb),
            "total_downloaded_megabytes": _round(total_downloaded_mb),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Merge distributed system metrics")
    parser.add_argument("exp_dir", help="Experiment output directory")
    parser.add_argument("--fl-endtime", type=float, default=None,
                        help="Observed FL end time in minutes (optional)")
    args = parser.parse_args()

    metrics_dir = os.path.join(args.exp_dir, "worker_metrics")
    if not os.path.isdir(metrics_dir):
        print(f"No worker_metrics directory found at {metrics_dir}", file=sys.stderr)
        sys.exit(1)

    client_metrics = load_client_metrics(metrics_dir)
    if not client_metrics:
        print("No client metrics found to merge.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded metrics from {len(client_metrics)} clients")

    server_entry = compute_server_aggregate(client_metrics, args.fl_endtime)

    outfile = os.path.join(args.exp_dir, "system_metrics.log")
    with open(outfile, "w") as f:
        for cm in sorted(client_metrics, key=lambda x: x.get("id", 0)):
            f.write(json.dumps(cm) + "\n")
        f.write(json.dumps(server_entry) + "\n")

    print(f"Merged system_metrics.log written to {outfile}")
    print(f"  Clients: {len(client_metrics)}")
    wc = server_entry["wallclock_time_minutes"]
    comm = server_entry["total_communication_megabytes"]
    print(f"  Wallclock training: {wc['total_training_time']:.4f} min")
    print(f"  Total communicated: {comm['total_communicated_megabytes']:.4f} MB")


if __name__ == "__main__":
    main()
