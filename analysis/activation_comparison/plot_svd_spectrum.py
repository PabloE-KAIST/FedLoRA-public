"""SVD spectrum comparison across methods."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Dict, List, Optional

from .load_checkpoints import short_layer_name, layer_sort_key
from .lora_activation_metrics import svd_spectrum_per_layer, aggregated_svd_spectrum

METHOD_COLORS = {
    "FedIT": "#d62728",
    "HetLoRA": "#2ca02c",
    "FAH-QLoRA": "#1f77b4",
    "AdaS-LoRA-C": "#ff7f0e",
    "AdaS-LoRA-L": "#9467bd",
}

METHOD_ORDER = ["FedIT", "HetLoRA", "FAH-QLoRA", "AdaS-LoRA-C", "AdaS-LoRA-L"]


def _ordered_methods(names):
    ordered = [n for n in METHOD_ORDER if n in names]
    ordered += [n for n in names if n not in ordered]
    return ordered


def plot_aggregated_spectrum(
    all_data: Dict[str, dict],
    output_path: str,
    max_components: int = 64,
    no_legend: bool = False,
    no_title: bool = False,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    for name in _ordered_methods(all_data.keys()):
        data = all_data[name]
        spectra = svd_spectrum_per_layer(data["delta_w"])
        agg = aggregated_svd_spectrum(spectra, max_components=max_components)
        nonzero = np.sum(agg > 1e-10)
        x = np.arange(1, nonzero + 1)
        color = METHOD_COLORS.get(name, None)
        ax.semilogy(x, agg[:nonzero], label=name, color=color, linewidth=2)

    ax.set_xlabel("Singular Value Index", fontsize=13)
    ax.set_ylabel("Singular Value (log scale)", fontsize=13)
    if not no_title:
        ax.set_title("Aggregated SVD Spectrum of ΔW = B·A", fontsize=14)
    if not no_legend:
        ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_representative_layers(
    all_data: Dict[str, dict],
    output_path: str,
    n_panels: int = 4,
    no_legend: bool = False,
    no_title: bool = False,
):
    ref_name = "FedIT" if "FedIT" in all_data else list(all_data.keys())[0]
    ref_layers = list(all_data[ref_name]["delta_w"].keys())
    ref_layers_sorted = sorted(ref_layers, key=layer_sort_key)

    n = len(ref_layers_sorted)
    indices = [int(i * n / (n_panels + 1)) for i in range(1, n_panels + 1)]
    selected = [ref_layers_sorted[i] for i in indices]

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, layer_key in zip(axes, selected):
        for name in _ordered_methods(all_data.keys()):
            data = all_data[name]
            if layer_key not in data["delta_w"]:
                continue
            dw = data["delta_w"][layer_key]
            _, S, _ = np.linalg.svd(dw.float().cpu().numpy(), full_matrices=False)
            nonzero = np.sum(S > 1e-10)
            x = np.arange(1, nonzero + 1)
            color = METHOD_COLORS.get(name, None)
            ax.semilogy(x, S[:nonzero], label=name, color=color, linewidth=1.5)

        ax.set_xlabel("SV Index", fontsize=11)
        short = short_layer_name(layer_key)
        ax.set_title(short, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Singular Value", fontsize=11)
    if not no_legend:
        axes[-1].legend(fontsize=9, loc="upper right")
    if not no_title:
        fig.suptitle("SVD Spectrum per Layer", fontsize=14, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
