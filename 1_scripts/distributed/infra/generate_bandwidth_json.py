#!/usr/bin/env python3
"""Convert a 4G network trace CSV into AegisGov bandwidth_limits JSON files.

Reads UL_bitrate and DL_bitrate from the trace, applies scaling factors,
aggregates into time bins, and outputs JSON files.

Modes:
  Default: produces two files (bandwidth_limits_ul.json, bandwidth_limits_dl.json)
  --per-device: additionally produces per-device scaled UL profiles and a
    scaling_factors.json that the bandwidth_manager uses for per-client UL
    differentiation.

Usage:
    # Base profiles only
    python generate_bandwidth_json.py \\
        --trace data/4Gnetwork_trace/static/B_2018.01.27_13.58.28.csv \\
        --ul-scale 5 --dl-scale 12 --bin-size 10 \\
        --outdir 1_scripts/distributed/infra/generated_bandwidths

    # With per-device UL differentiation
    python generate_bandwidth_json.py \\
        --trace data/4Gnetwork_trace/static/B_2018.01.27_13.58.28.csv \\
        --ul-scale 5 --dl-scale 12 --bin-size 10 \\
        --per-device --clients 12 --bw-seed 42 \\
        --outdir 1_scripts/distributed/infra/generated_bandwidths
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
from pathlib import Path


def read_trace(trace_path: str):
    """Read DL_bitrate and UL_bitrate columns (kbit/s) from the trace CSV."""
    dl_values = []
    ul_values = []
    with open(trace_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dl_values.append(int(row["DL_bitrate"]))
            ul_values.append(int(row["UL_bitrate"]))
    return dl_values, ul_values


def aggregate_bins(values_kbps: list[int], scale_factor: float,
                   bin_size: int) -> list[dict]:
    """Scale values to Mbps and aggregate into time bins.

    Returns a list of {time: int, mbps: float} dicts in AegisGov format.
    """
    limits = []
    for bin_start in range(0, len(values_kbps), bin_size):
        bin_end = min(bin_start + bin_size, len(values_kbps))
        chunk = values_kbps[bin_start:bin_end]
        avg_kbps = statistics.mean(chunk)
        mbps = round(avg_kbps * scale_factor / 1000.0, 2)
        limits.append({"time": bin_start, "mbps": mbps})
    return limits


def generate_scaling_factors(n_clients: int, seed: int) -> list[float]:
    """Generate deterministic per-device scaling factors.

    Factors are drawn from a log-normal distribution, then normalized so
    they sum to n_clients (average = 1.0).  This produces realistic
    heterogeneity: most devices near the mean, a few significantly
    faster or slower.
    """
    rng = random.Random(seed)
    sigma = 0.45
    raw = [math.exp(rng.gauss(0, sigma)) for _ in range(n_clients)]
    total = sum(raw)
    factors = [round(r * n_clients / total, 4) for r in raw]
    # Fix rounding drift
    factors[-1] = round(n_clients - sum(factors[:-1]), 4)
    return factors


def scale_limits(base_limits: list[dict], factor: float) -> list[dict]:
    """Apply a multiplicative factor to each entry's mbps value."""
    return [{"time": e["time"], "mbps": round(e["mbps"] * factor, 2)}
            for e in base_limits]


def write_json(limits: list[dict], path: str):
    out = {"bandwidth_limits": limits}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Written {path} ({len(limits)} thresholds)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", required=True,
                        help="Path to 4G network trace CSV")
    parser.add_argument("--ul-scale", type=float, default=5.0,
                        help="UL scale factor (default: 5)")
    parser.add_argument("--dl-scale", type=float, default=12.0,
                        help="DL scale factor (default: 12)")
    parser.add_argument("--bin-size", type=int, default=10,
                        help="Seconds per time bin for aggregation (default: 10)")
    parser.add_argument("--outdir", default="1_scripts/distributed/infra/generated_bandwidths",
                        help="Output directory for JSON files")
    parser.add_argument("--per-device", action="store_true",
                        help="Generate per-device scaled UL profiles")
    parser.add_argument("--clients", type=int, default=12,
                        help="Number of clients for per-device mode (default: 12)")
    parser.add_argument("--bw-seed", type=int, default=42,
                        help="RNG seed for per-device scaling factors (default: 42)")
    args = parser.parse_args()

    dl_values, ul_values = read_trace(args.trace)
    print(f"Trace: {args.trace}")
    print(f"  Samples: {len(dl_values)}")
    print(f"  DL raw: avg={statistics.mean(dl_values)/1000:.1f} Mbps, "
          f"range=[{min(dl_values)/1000:.1f}, {max(dl_values)/1000:.1f}]")
    print(f"  UL raw: avg={statistics.mean(ul_values)/1000:.2f} Mbps, "
          f"range=[{min(ul_values)/1000:.2f}, {max(ul_values)/1000:.2f}]")

    ul_limits = aggregate_bins(ul_values, args.ul_scale, args.bin_size)
    dl_limits = aggregate_bins(dl_values, args.dl_scale, args.bin_size)

    ul_mbps = [l["mbps"] for l in ul_limits]
    dl_mbps = [l["mbps"] for l in dl_limits]
    print(f"\nAfter scaling (UL×{args.ul_scale}, DL×{args.dl_scale}, "
          f"bin={args.bin_size}s):")
    print(f"  UL: avg={statistics.mean(ul_mbps):.2f} Mbps, "
          f"range=[{min(ul_mbps):.2f}, {max(ul_mbps):.2f}]")
    print(f"  DL: avg={statistics.mean(dl_mbps):.1f} Mbps, "
          f"range=[{min(dl_mbps):.1f}, {max(dl_mbps):.1f}]")
    print(f"  DL/UL ratio: {statistics.mean(dl_mbps)/statistics.mean(ul_mbps):.1f}×")

    ul_path = os.path.join(args.outdir, "bandwidth_limits_ul.json")
    dl_path = os.path.join(args.outdir, "bandwidth_limits_dl.json")
    write_json(ul_limits, ul_path)
    write_json(dl_limits, dl_path)

    if args.per_device:
        n = args.clients
        factors = generate_scaling_factors(n, args.bw_seed)
        print(f"\nPer-device UL scaling (seed={args.bw_seed}, {n} clients):")
        print(f"  Factors: {factors}")
        print(f"  Sum: {sum(factors):.4f} (target: {n})")
        print(f"  Range: [{min(factors):.4f}, {max(factors):.4f}]")

        scaling_info = {}
        for i, factor in enumerate(factors):
            client_id = i + 1
            scaled = scale_limits(ul_limits, factor)
            scaled_mbps = [e["mbps"] for e in scaled]

            # AegisGov-format file: bandwidth_limits{N}.json
            aegis_path = os.path.join(args.outdir, f"bandwidth_limits{client_id}.json")
            write_json(scaled, aegis_path)

            scaling_info[str(client_id)] = {
                "factor": factor,
                "bandwidth_setting_id": client_id,
                "avg_ul_mbps": round(statistics.mean(scaled_mbps), 2),
            }
            print(f"  Client {client_id}: factor={factor:.4f}, "
                  f"avg_UL={statistics.mean(scaled_mbps):.2f} Mbps")

        # Write scaling factors JSON for bandwidth_manager
        sf_path = os.path.join(args.outdir, "bandwidth_scaling_factors.json")
        os.makedirs(os.path.dirname(sf_path) or ".", exist_ok=True)
        with open(sf_path, "w") as f:
            json.dump({
                "seed": args.bw_seed,
                "n_clients": n,
                "clients": scaling_info,
            }, f, indent=2)
        print(f"\n  Written {sf_path}")

    print(f"\nDone. Deploy UL JSON to all devices, DL JSON to the server.")


if __name__ == "__main__":
    main()
