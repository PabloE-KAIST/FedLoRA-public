"""Per-layer Frobenius norm heatmap and module-type breakdown."""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Dict, Optional

from .load_checkpoints import short_layer_name, layer_sort_key
from .lora_activation_metrics import frobenius_per_layer

METHOD_ORDER = ["FedIT", "HetLoRA", "FAH-QLoRA", "AdaS-LoRA-C", "AdaS-LoRA-L"]

MODULE_TYPES = {
    "attention.self.in_proj": "Attn In-Proj",
    "attention.output.dense": "Attn Output",
    "intermediate.dense": "Intermediate",
    "output.dense": "Output",
}


def _module_type(base_name: str) -> str:
    for pattern, label in MODULE_TYPES.items():
        if pattern in base_name:
            return label
    return "Other"


def _encoder_layer_idx(base_name: str) -> int:
    m = re.search(r"layer\.(\d+)\.", base_name)
    return int(m.group(1)) if m else -1


def plot_frobenius_heatmap(
    all_data: Dict[str, dict],
    output_path: str,
    no_legend: bool = False,
    no_title: bool = False,
):
    ref_name = "FedIT" if "FedIT" in all_data else list(all_data.keys())[0]
    layers = sorted(all_data[ref_name]["delta_w"].keys(), key=layer_sort_key)
    methods = [n for n in METHOD_ORDER if n in all_data]
    methods += [n for n in all_data if n not in methods]

    matrix = np.zeros((len(methods), len(layers)))
    for i, name in enumerate(methods):
        frob = frobenius_per_layer(all_data[name]["delta_w"])
        for j, layer in enumerate(layers):
            matrix[i, j] = frob.get(layer, 0.0)

    row_maxes = matrix.max(axis=1, keepdims=True)
    row_maxes[row_maxes < 1e-12] = 1.0
    matrix_norm = matrix / row_maxes

    fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.5), len(methods) * 0.8 + 1.5))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Normalized ||ΔW||_F", fontsize=11)

    short_labels = [short_layer_name(l) for l in layers]
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=8, ha="center")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)

    if not no_title:
        ax.set_title("Per-Layer Adapter Capacity (||ΔW||_F)", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_frobenius_by_module_type(
    all_data: Dict[str, dict],
    output_path: str,
    no_legend: bool = False,
    no_title: bool = False,
):
    ref_name = "FedIT" if "FedIT" in all_data else list(all_data.keys())[0]
    layers = list(all_data[ref_name]["delta_w"].keys())

    module_labels = sorted(set(_module_type(l) for l in layers))
    methods = [n for n in METHOD_ORDER if n in all_data]
    methods += [n for n in all_data if n not in methods]

    from .plot_svd_spectrum import METHOD_COLORS

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(module_labels))
    width = 0.8 / len(methods)

    for i, name in enumerate(methods):
        frob = frobenius_per_layer(all_data[name]["delta_w"])
        means = []
        for mt in module_labels:
            vals = [frob.get(l, 0.0) for l in layers if _module_type(l) == mt]
            means.append(np.mean(vals) if vals else 0.0)

        offset = (i - len(methods) / 2 + 0.5) * width
        color = METHOD_COLORS.get(name, None)
        ax.bar(x + offset, means, width, label=name, color=color,
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(module_labels, fontsize=11)
    ax.set_ylabel("Mean ||ΔW||_F", fontsize=12)
    if not no_title:
        ax.set_title("Adapter Capacity by Module Type", fontsize=14)
    if not no_legend:
        ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
