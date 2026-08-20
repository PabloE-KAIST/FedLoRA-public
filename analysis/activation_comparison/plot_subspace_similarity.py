"""Subspace similarity heatmap and distance-to-FedIT bar chart."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from .lora_activation_metrics import pairwise_grassmann

METHOD_COLORS = {
    "FedIT": "#d62728",
    "HetLoRA": "#2ca02c",
    "FAH-QLoRA": "#1f77b4",
    "AdaS-LoRA-C": "#ff7f0e",
    "AdaS-LoRA-L": "#9467bd",
}

METHOD_ORDER = ["FedIT", "HetLoRA", "FAH-QLoRA", "AdaS-LoRA-C", "AdaS-LoRA-L"]


def plot_pairwise_heatmap(
    all_data: Dict[str, dict],
    output_path: str,
    k: int = 10,
    no_legend: bool = False,
    no_title: bool = False,
):
    all_dw = {name: data["delta_w"] for name, data in all_data.items()}
    ordered = [n for n in METHOD_ORDER if n in all_dw]
    ordered += [n for n in all_dw if n not in ordered]
    ordered_dw = {n: all_dw[n] for n in ordered}

    D, names = pairwise_grassmann(ordered_dw, k=k)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(D, cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Grassmann Distance", fontsize=11)

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(names, fontsize=10)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{D[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if D[i, j] > D.max() * 0.6 else "black")

    if not no_title:
        ax.set_title(f"Pairwise Subspace Distance (k={k})", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_distance_to_reference(
    all_data: Dict[str, dict],
    output_path: str,
    reference: str = "FedIT",
    k_values: List[int] = [10, 20, 40],
    no_legend: bool = False,
    no_title: bool = False,
):
    all_dw = {name: data["delta_w"] for name, data in all_data.items()}
    others = [n for n in METHOD_ORDER if n in all_dw and n != reference]
    others += [n for n in all_dw if n not in others and n != reference]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(others))
    width = 0.8 / len(k_values)

    for i, k in enumerate(k_values):
        dists = []
        for name in others:
            D, names = pairwise_grassmann(
                {reference: all_dw[reference], name: all_dw[name]}, k=k
            )
            ref_idx = names.index(reference)
            m_idx = names.index(name)
            dists.append(D[ref_idx, m_idx])

        offset = (i - len(k_values) / 2 + 0.5) * width
        colors = [METHOD_COLORS.get(n, "#888888") for n in others]
        bars = ax.bar(x + offset, dists, width, label=f"k={k}",
                      color=colors, alpha=0.7 + 0.1 * i, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(others, fontsize=11)
    ax.set_ylabel(f"Grassmann Distance to {reference}", fontsize=12)
    if not no_title:
        ax.set_title(f"Subspace Distance to {reference}", fontsize=14)
    if not no_legend:
        ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
