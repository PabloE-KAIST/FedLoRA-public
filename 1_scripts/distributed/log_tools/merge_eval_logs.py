#!/usr/bin/env python3
"""
Merge per-client evaluation results from distributed worker logs into
exp_print.log so that analysis scripts (loss_evolution_plots.py, etc.)
can see per-client train/val metrics.

In distributed mode, exp_print.log only contains server-side aggregates
(Results_weighted_avg). Worker logs contain per-client Results_raw for
both train and val. This script extracts those entries and appends them
to exp_print.log.

Usage:
    python merge_eval_logs.py <exp_dir>

Reads:  <exp_dir>/worker_logs/client_*.log
Writes: <exp_dir>/exp_print.log (appends client entries)
"""
import ast
import glob
import os
import re
import sys

RESULTS_RAW_RE = re.compile(r"\{.*'Role':\s*'Client\s*#\d+'.*'Results_raw':\s*\{.*\}.*\}")


def extract_client_entries(worker_logs_dir):
    """Extract unique (client_id, round, metric_type) entries from worker logs."""
    pattern = os.path.join(worker_logs_dir, "client_*.log")
    files = sorted(glob.glob(pattern))
    if not files:
        return []

    seen = set()
    entries = []

    for fpath in files:
        with open(fpath, encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Results_raw" not in line:
                    continue

                idx = line.find("{")
                if idx < 0:
                    continue
                payload = line[idx:].strip()
                last = payload.rfind("}")
                if last >= 0:
                    payload = payload[:last + 1]

                try:
                    d = ast.literal_eval(payload)
                except Exception:
                    continue

                if not isinstance(d, dict):
                    continue

                role = d.get("Role", "")
                rnd = d.get("Round", None)
                results = d.get("Results_raw", None)

                if not (isinstance(role, str) and role.startswith("Client #")
                        and rnd is not None and isinstance(results, dict)):
                    continue

                try:
                    cid = int(role.split("#", 1)[1])
                    rnd = int(rnd)
                except (ValueError, IndexError):
                    continue

                metric_type = "train" if "train_loss" in results else "val" if "val_loss" in results else None
                if metric_type is None:
                    continue

                key = (cid, rnd, metric_type)
                if key in seen:
                    continue
                seen.add(key)

                entries.append((cid, rnd, metric_type, line.rstrip()))

    entries.sort(key=lambda x: (x[1], x[0], x[2]))
    return entries


def get_existing_client_entries(exp_print_path):
    """Check which client entries already exist in exp_print.log."""
    existing = set()
    if not os.path.exists(exp_print_path):
        return existing

    with open(exp_print_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "Results_raw" not in line or "Client #" not in line:
                continue

            idx = line.find("{")
            if idx < 0:
                continue
            payload = line[idx:].strip()
            last = payload.rfind("}")
            if last >= 0:
                payload = payload[:last + 1]

            try:
                d = ast.literal_eval(payload)
            except Exception:
                continue

            if not isinstance(d, dict):
                continue

            role = d.get("Role", "")
            rnd = d.get("Round", None)
            results = d.get("Results_raw", None)

            if not (isinstance(role, str) and role.startswith("Client #")
                    and rnd is not None and isinstance(results, dict)):
                continue

            try:
                cid = int(role.split("#", 1)[1])
                rnd = int(rnd)
            except (ValueError, IndexError):
                continue

            metric_type = "train" if "train_loss" in results else "val" if "val_loss" in results else None
            if metric_type:
                existing.add((cid, rnd, metric_type))

    return existing


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <exp_dir>", file=sys.stderr)
        sys.exit(1)

    exp_dir = sys.argv[1]
    worker_logs_dir = os.path.join(exp_dir, "worker_logs")
    exp_print_path = os.path.join(exp_dir, "exp_print.log")

    if not os.path.isdir(worker_logs_dir):
        print(f"No worker_logs directory at {worker_logs_dir}", file=sys.stderr)
        sys.exit(1)

    entries = extract_client_entries(worker_logs_dir)
    if not entries:
        print("No client Results_raw entries found in worker logs.")
        return

    existing = get_existing_client_entries(exp_print_path)
    new_entries = [e for e in entries if (e[0], e[1], e[2]) not in existing]

    if not new_entries:
        print(f"All {len(entries)} client entries already present in exp_print.log.")
        return

    with open(exp_print_path, "a") as f:
        for cid, rnd, metric_type, line in new_entries:
            f.write(line + "\n")

    train_count = sum(1 for e in new_entries if e[2] == "train")
    val_count = sum(1 for e in new_entries if e[2] == "val")
    clients = sorted(set(e[0] for e in new_entries))
    rounds = sorted(set(e[1] for e in new_entries))

    print(f"Appended {len(new_entries)} client entries to {exp_print_path}")
    print(f"  Train: {train_count}, Val: {val_count}")
    print(f"  Clients: {clients}")
    print(f"  Rounds: {rounds}")


if __name__ == "__main__":
    main()
