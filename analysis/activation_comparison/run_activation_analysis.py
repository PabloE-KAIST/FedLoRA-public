"""Orchestrate all LoRA activation comparison plots.

Usage:
    python -m analysis.activation_comparison.run_activation_analysis \
        --task rte \
        --ckpt-dir ckpt/activation_analysis/rte \
        --output-dir 0_results/activation_analysis/rte

    python -m analysis.activation_comparison.run_activation_analysis \
        --task rte \
        --ckpt-dir ckpt/activation_analysis/rte \
        --output-dir 0_results/activation_analysis/rte \
        --rename-method adasparse_lorav2 'AdaS-LoRA-C' \
        --rename-method adasparse_lorav3 'AdaS-LoRA-L' \
        --no-legend
"""
import argparse
import csv
import os
import sys

from .load_checkpoints import load_all_methods, METHODS
from .lora_activation_metrics import compute_all_metrics
from .plot_svd_spectrum import plot_aggregated_spectrum, plot_representative_layers
from .plot_subspace_similarity import plot_pairwise_heatmap, plot_distance_to_reference
from .plot_norm_distribution import plot_frobenius_heatmap, plot_frobenius_by_module_type
from .plot_component_scores import plot_spearman_correlation, plot_retention_topk


def parse_args():
    p = argparse.ArgumentParser(description="LoRA activation comparison analysis")
    p.add_argument("--task", required=True, help="GLUE task name")
    p.add_argument("--ckpt-dir", required=True, help="Directory with method checkpoints")
    p.add_argument("--output-dir", required=True, help="Output directory for plots")
    p.add_argument("--reference", default="FedIT", help="Reference method name")
    p.add_argument("--rename-method", nargs=2, action="append", default=[],
                    metavar=("FROM", "TO"), help="Rename method for display")
    p.add_argument("--exclude-methods", nargs="+", default=[], help="Methods to exclude")
    p.add_argument("--no-legend", action="store_true")
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--k-values", nargs="+", type=int, default=[10, 20, 40],
                    help="Subspace dimensions for Grassmann distance")
    return p.parse_args()


def main():
    args = parse_args()

    renames = dict(args.rename_method)
    methods = [m for m in METHODS if m not in args.exclude_methods]

    print(f"=== LoRA Activation Comparison: {args.task} ===")
    print(f"  Checkpoint dir: {args.ckpt_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print()

    all_data = load_all_methods(args.ckpt_dir, methods=methods, renames=renames)

    if not all_data:
        print("ERROR: No checkpoints loaded. Exiting.")
        sys.exit(1)

    ref = args.reference
    if ref not in all_data:
        for old, new in renames.items():
            if new == ref and old in all_data:
                break
        else:
            print(f"ERROR: Reference method '{ref}' not found. Available: {list(all_data.keys())}")
            sys.exit(1)

    print(f"\nLoaded {len(all_data)} methods: {list(all_data.keys())}")
    for name, data in all_data.items():
        n_layers = len(data["pairs"])
        ranks = [min(ab["A"].shape[0], ab["B"].shape[1]) for ab in data["pairs"].values()]
        print(f"  {name}: {n_layers} LoRA layers, ranks {min(ranks)}-{max(ranks)}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = args.output_dir

    print("\n--- SVD Spectrum ---")
    plot_aggregated_spectrum(
        all_data, os.path.join(out, "svd_spectrum_aggregated.png"),
        no_legend=args.no_legend, no_title=args.no_title,
    )
    plot_representative_layers(
        all_data, os.path.join(out, "svd_spectrum_representative.png"),
        no_legend=args.no_legend, no_title=args.no_title,
    )

    print("\n--- Subspace Similarity ---")
    plot_pairwise_heatmap(
        all_data, os.path.join(out, "subspace_distance_heatmap.png"),
        k=args.k_values[0], no_legend=args.no_legend, no_title=args.no_title,
    )
    plot_distance_to_reference(
        all_data, os.path.join(out, "subspace_distance_vs_fedit.png"),
        reference=ref, k_values=args.k_values,
        no_legend=args.no_legend, no_title=args.no_title,
    )

    print("\n--- Per-Layer Capacity ---")
    plot_frobenius_heatmap(
        all_data, os.path.join(out, "frobenius_per_layer.png"),
        no_legend=args.no_legend, no_title=args.no_title,
    )
    plot_frobenius_by_module_type(
        all_data, os.path.join(out, "frobenius_by_module_type.png"),
        no_legend=args.no_legend, no_title=args.no_title,
    )

    print("\n--- Component Scores ---")
    plot_spearman_correlation(
        all_data, os.path.join(out, "component_score_correlation.png"),
        reference=ref, no_legend=args.no_legend, no_title=args.no_title,
    )
    plot_retention_topk(
        all_data, os.path.join(out, "component_retention_topk.png"),
        reference=ref, k_values=[10, 20, 40],
        no_legend=args.no_legend, no_title=args.no_title,
    )

    print("\n--- Summary Metrics ---")
    metrics = compute_all_metrics(all_data, reference=ref, k_values=args.k_values)
    csv_path = os.path.join(out, "summary_table.csv")
    if metrics:
        fieldnames = ["method"] + sorted(next(iter(metrics.values())).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for name, m in metrics.items():
                row = {"method": name}
                row.update({k: f"{v:.4f}" for k, v in m.items()})
                writer.writerow(row)
        print(f"  Saved: {csv_path}")

    print(f"\n=== Done: {len(os.listdir(out))} files in {out} ===")


if __name__ == "__main__":
    main()
