"""Component importance score comparison: Spearman correlation and top-k retention."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from .lora_activation_metrics import (
    component_scores_from_pairs,
    component_rank_correlation,
    component_retention_fraction,
)

METHOD_COLORS = {
    "FedIT": "#d62728",
    "HetLoRA": "#2ca02c",
    "FAH-QLoRA": "#1f77b4",
    "AdaS-LoRA-C": "#ff7f0e",
    "AdaS-LoRA-L": "#9467bd",
}

METHOD_ORDER = ["FedIT", "HetLoRA", "FAH-QLoRA", "AdaS-LoRA-C", "AdaS-LoRA-L"]


def plot_spearman_correlation(
    all_data: Dict[str, dict],
    output_path: str,
    reference: str = "FedIT",
    no_legend: bool = False,
    no_title: bool = False,
):
    ref_scores = component_scores_from_pairs(all_data[reference]["pairs"])
    others = [n for n in METHOD_ORDER if n in all_data and n != reference]
    others += [n for n in all_data if n not in others and n != reference]

    correlations = []
    for name in others:
        scores = component_scores_from_pairs(all_data[name]["pairs"])
        rho = component_rank_correlation(ref_scores, scores)
        correlations.append(rho)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [METHOD_COLORS.get(n, "#888888") for n in others]
    bars = ax.bar(range(len(others)), correlations, color=colors,
                  edgecolor="black", linewidth=0.8)

    for bar, val in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(others)))
    ax.set_xticklabels(others, fontsize=11)
    ax.set_ylabel(f"Spearman ρ vs {reference}", fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    if not no_title:
        ax.set_title(f"Component Importance Rank Correlation with {reference}", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_retention_topk(
    all_data: Dict[str, dict],
    output_path: str,
    reference: str = "FedIT",
    k_values: List[int] = [10, 20, 40],
    no_legend: bool = False,
    no_title: bool = False,
):
    ref_scores = component_scores_from_pairs(all_data[reference]["pairs"])
    others = [n for n in METHOD_ORDER if n in all_data and n != reference]
    others += [n for n in all_data if n not in others and n != reference]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(others))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        fractions = []
        for name in others:
            scores = component_scores_from_pairs(all_data[name]["pairs"])
            frac = component_retention_fraction(ref_scores, scores, top_k=k)
            fractions.append(frac)

        offset = (i - len(k_values) / 2 + 0.5) * width
        colors = [METHOD_COLORS.get(n, "#888888") for n in others]
        ax.bar(x + offset, fractions, width, label=f"top-{k}",
               color=colors, alpha=0.6 + 0.15 * i, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(others, fontsize=11)
    ax.set_ylabel(f"Fraction of {reference}'s Top-k Retained", fontsize=12)
    ax.set_ylim(0, 1.05)
    if not no_title:
        ax.set_title(f"Top-k Component Retention vs {reference}", fontsize=14)
    if not no_legend:
        ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
