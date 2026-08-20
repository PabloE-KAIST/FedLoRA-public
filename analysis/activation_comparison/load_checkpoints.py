"""Load FL checkpoints and extract LoRA A/B pairs per layer."""
import os
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional


METHODS = ["fedit", "hetlora", "fahqlora", "adasparse_lorav2", "adasparse_lorav3"]

DISPLAY_NAMES = {
    "fedit": "FedIT",
    "hetlora": "HetLoRA",
    "fahqlora": "FAH-QLoRA",
    "adasparse_lorav2": "AdaS-LoRA-C",
    "adasparse_lorav3": "AdaS-LoRA-L",
}


def _find_checkpoint(ckpt_dir: str, method: str) -> Optional[str]:
    """Find the final checkpoint for a method in the ckpt directory."""
    candidates = [
        os.path.join(ckpt_dir, f"final_{method}.ckpt"),
        os.path.join(ckpt_dir, f"{method}.ckpt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    pattern_prefix = f"final_{method}"
    if os.path.isdir(ckpt_dir):
        for fn in os.listdir(ckpt_dir):
            if fn.startswith(pattern_prefix) and fn.endswith(".ckpt"):
                return os.path.join(ckpt_dir, fn)
    return None


def extract_lora_pairs_from_state_dict(
    state_dict: dict,
) -> OrderedDict:
    """Extract ordered LoRA A/B pairs from a state dict.

    Returns:
        OrderedDict mapping base_name -> {"A": tensor, "B": tensor}
        Ordered by layer index for consistent cross-method comparison.
    """
    pairs = {}

    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if "lora_A" in key and "lora_B" not in key:
            base = key.split("lora_A")[0].rstrip(".")
            if base.startswith("model."):
                base = base[len("model."):]
            pairs.setdefault(base, {"A": None, "B": None})
            pairs[base]["A"] = tensor
        elif "lora_B" in key:
            base = key.split("lora_B")[0].rstrip(".")
            if base.startswith("model."):
                base = base[len("model."):]
            pairs.setdefault(base, {"A": None, "B": None})
            pairs[base]["B"] = tensor

    complete = OrderedDict()
    for base in sorted(pairs.keys()):
        if pairs[base]["A"] is not None and pairs[base]["B"] is not None:
            complete[base] = pairs[base]

    return complete


def compute_delta_w(pairs: OrderedDict) -> OrderedDict:
    """Compute ΔW = B @ A for each LoRA layer."""
    result = OrderedDict()
    for base, ab in pairs.items():
        A = ab["A"].float()
        B = ab["B"].float()
        result[base] = B @ A
    return result


def load_all_methods(
    ckpt_dir: str,
    methods: Optional[List[str]] = None,
    renames: Optional[Dict[str, str]] = None,
) -> Dict[str, dict]:
    """Load checkpoints for all methods.

    Returns:
        Dict mapping display_name -> {
            "pairs": OrderedDict of A/B pairs,
            "delta_w": OrderedDict of ΔW matrices,
            "state_dict": full state dict,
        }
    """
    if methods is None:
        methods = METHODS
    if renames is None:
        renames = {}

    result = {}
    for method in methods:
        path = _find_checkpoint(ckpt_dir, method)
        if path is None:
            print(f"[WARN] No checkpoint found for {method} in {ckpt_dir}")
            continue

        print(f"Loading {method}: {path}")
        sd = torch.load(path, map_location="cpu")

        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]

        pairs = extract_lora_pairs_from_state_dict(sd)
        if not pairs:
            print(f"[WARN] No LoRA pairs found in {method} checkpoint")
            continue

        display = renames.get(method, DISPLAY_NAMES.get(method, method))
        result[display] = {
            "pairs": pairs,
            "delta_w": compute_delta_w(pairs),
            "state_dict": sd,
        }

    return result


def short_layer_name(base_name: str) -> str:
    """Shorten a LoRA layer base name for plot labels.

    Example:
        'base_model.model.deberta.encoder.layer.3.attention.self.in_proj'
        -> 'L3.attn.in_proj'
    """
    import re
    m = re.search(r"layer\.(\d+)\.", base_name)
    layer_idx = m.group(1) if m else "?"

    if "attention.self.in_proj" in base_name:
        module = "attn.in_proj"
    elif "attention.output.dense" in base_name:
        module = "attn.out"
    elif "intermediate.dense" in base_name:
        module = "inter"
    elif "output.dense" in base_name:
        module = "out"
    else:
        parts = base_name.split(".")
        module = parts[-1] if parts else base_name

    return f"L{layer_idx}.{module}"


def layer_sort_key(base_name: str) -> Tuple[int, int]:
    """Sort key for LoRA layers: (encoder_layer_idx, module_type_order)."""
    import re
    m = re.search(r"layer\.(\d+)\.", base_name)
    layer_idx = int(m.group(1)) if m else 999

    module_order = 3
    if "attention.self.in_proj" in base_name:
        module_order = 0
    elif "attention.output.dense" in base_name:
        module_order = 1
    elif "intermediate.dense" in base_name:
        module_order = 2
    elif "output.dense" in base_name:
        module_order = 3

    return (layer_idx, module_order)
