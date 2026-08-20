"""
Client configuration generator for HeteroLoRA.

This module provides functions to generate client-specific LoRA configurations
with different ranks, alpha values, and target modules based on various
distribution strategies.
"""
import os
import json
import random
import numpy as np
from scipy.stats import norm
import logging

logger = logging.getLogger(__name__)


def get_client_lora_config(config_types, num_clients, strategy='random',
                          seed=42, default_alpha=16, default_dropout=0.05,
                          **kwargs):
    """
    Generate client-specific LoRA configurations based on distribution strategy.
    
    Args:
        config_types: Dict mapping type names to their LoRA configurations
            Format: {
                'Type_0': {'q_proj': 8, 'v_proj': 8, ...},
                'Type_1': {'q_proj': 16, 'v_proj': 16, ...},
                ...
            }
        num_clients: Total number of clients
        strategy: Distribution strategy
            - 'homo': All clients use the same configuration (returns None)
            - 'random': Random assignment of types
            - 'heavy_tail': Heavy-tailed distribution (80% Type_0, 10% Type_1, etc.)
            - 'heavy_tail_strong': Strong heavy-tailed distribution
            - 'normal': Normal distribution over types
        seed: Random seed for reproducibility
        default_alpha: Default LoRA alpha value
        default_dropout: Default LoRA dropout rate
    
    Returns:
        Dict mapping client IDs to their configurations:
        {
            'alpha': default_alpha,
            'lora_dropout': default_dropout,
            'Client_0': {'q_proj': 8, 'v_proj': 8, ...},
            'Client_1': {'q_proj': 16, 'v_proj': 16, ...},
            ...
        }
    """
    if strategy == 'homo':
        return None
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    config_local = {
        'alpha': default_alpha,
        'lora_dropout': default_dropout
    }
    
    if strategy == 'random':
        # Random assignment of types
        # Note: FederatedScope uses 1-indexed client IDs, so we generate Client_1, Client_2, etc.
        type_names = list(config_types.keys())
        for i in range(1, num_clients + 1):
            type_name = random.choice(type_names)
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'heavy_tail':
        # Heavy-tailed distribution: 80% Type_0, 10% Type_1, 5% Type_2, 5% Type_3
        # Note: FederatedScope uses 1-indexed client IDs
        type_names = sorted(config_types.keys())
        for i in range(1, num_clients + 1):
            rand_num = random.random()
            if rand_num < 0.80:
                type_name = type_names[0] if len(type_names) > 0 else 'Type_0'
            elif rand_num < 0.90:
                type_name = type_names[1] if len(type_names) > 1 else type_names[0]
            elif rand_num < 0.95:
                type_name = type_names[2] if len(type_names) > 2 else type_names[0]
            else:
                type_name = type_names[3] if len(type_names) > 3 else type_names[0]
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'heavy_tail_strong':
        # Strong heavy-tailed distribution toward small ranks:
        # 90% Type_0 (smallest), 7% Type_1, 2% Type_2, 1% Type_3 (largest)
        # Note: FederatedScope uses 1-indexed client IDs
        type_names = sorted(config_types.keys())
        for i in range(1, num_clients + 1):
            rand_num = random.random()
            if rand_num < 0.90:
                type_name = type_names[0] if len(type_names) > 0 else 'Type_0'
            elif rand_num < 0.97:
                type_name = type_names[1] if len(type_names) > 1 else type_names[0]
            elif rand_num < 0.99:
                type_name = type_names[2] if len(type_names) > 2 else type_names[0]
            else:
                type_name = type_names[3] if len(type_names) > 3 else type_names[0]
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'normal':
        # Normal distribution over types
        # Note: FederatedScope uses 1-indexed client IDs
        type_names = sorted(config_types.keys())
        positions = np.array(list(range(len(type_names))))
        mu = len(type_names) / 2.0
        sigma = len(type_names) / 3.0
        probabilities = norm.pdf(positions, mu, sigma)
        probabilities /= probabilities.sum()
        
        for i in range(1, num_clients + 1):
            selected_idx = np.random.choice(positions, p=probabilities)
            type_name = type_names[int(selected_idx)]
            config_local['Client_' + str(i)] = config_types[type_name].copy()
    
    elif strategy == 'custom':
        # Exact type mixes matching the physical fleet composition.
        # Uses seed controlled shuffle so which client id gets which type is randomized but reproducible.
        type_names = sorted(config_types.keys())

        if len(type_names) < 4:
            raise ValueError("custom strategy expects at least 4 types: Type_0..Type_3")

        custom_mixes = {
            6:  (2, 2, 1, 1),
            12: (3, 4, 3, 2),
        }
        if num_clients not in custom_mixes:
            raise ValueError(
                f"custom strategy supports num_clients in {sorted(custom_mixes)}, got {num_clients}"
            )
        mix = custom_mixes[num_clients]
        type_list = []
        for t, count in zip(type_names[:4], mix):
            type_list.extend([t] * count)
        random.shuffle(type_list)

        for i in range(1, num_clients + 1):
            type_name = type_list[i - 1]
            config_local['Client_' + str(i)] = config_types[type_name].copy()

    elif strategy == 'distributed_fleet':
        config_local = _build_distributed_fleet_config(
            config_types, num_clients, default_alpha, default_dropout,
            manifest_path=kwargs.get('manifest_path'),
            device_class_rank_map=kwargs.get('device_class_rank_map'),
        )

    else:
        raise ValueError(f"Unknown distribution strategy: {strategy}")
    
    return config_local



DISTRIBUTED_FLEET_RANK_MAP = {
    'agxorin':  200,   # fp32, 28% at r=200 (65893 MB total)
    'agxavier': 150,   # fp32, 61% at r=150 (32517 MB total)
    'x86':      120,   # fp32+grad_ckpt, 30% at r=120 (11708 MB total)
    'orinnx':   64,    # bf16, 66% at r=64 (16416 MB total)
}

# Qwen2.5-1.5B fleet rank map (Milestone 3), finalized by the G0 OOM smoke.
# Finding: for LoRA the base model + activations dominate memory, NOT the rank
# (r=64 fits every class). The binding constraint is the device, not the rank:
# x86/x86-worker (11GB, Turing) canNOT safely fit Qwen-1.5B (fp32 92-96%, bf16
# emulated) -> x86 is EXCLUDED from the Qwen fleet; Dolly/CodeAlpaca run Jetson-only.
# Validated: orinnx fp32 r64 seq512 = 11.26/16.4GB; agxorin/agxavier have ample RAM.
# Selected when env FEDLORA_FLEET_RANK_MAP=qwen (set by the LLM fleet queues).
DISTRIBUTED_FLEET_RANK_MAP_QWEN = {
    'agxorin':  32,    # Ampere, 64GB
    'agxavier': 24,    # Volta, 32GB
    'x86':      16,    # UNUSED (x86 excluded from Qwen fleet); kept for safety
    'orinnx':   16,    # Ampere, 16GB -- halved to cut AdaS stage-1 component memory (OOM headroom)
}


def _build_distributed_fleet_config(config_types, num_clients, default_alpha,
                                    default_dropout, manifest_path=None,
                                    device_class_rank_map=None):
    """
    Build per-client LoRA configs from the fleet manifest.

    Each client's rank is determined by its budget_class (falling back to
    device_class) mapped through ``device_class_rank_map`` (or the module-level
    ``DISTRIBUTED_FLEET_RANK_MAP`` default).

    Args:
        manifest_path: Path to client_manifest.json.
        device_class_rank_map: Optional override mapping budget_class -> rank.
    """
    if device_class_rank_map is not None:
        rank_map = device_class_rank_map
    elif os.environ.get('FEDLORA_FLEET_RANK_MAP', '').lower() == 'qwen':
        rank_map = DISTRIBUTED_FLEET_RANK_MAP_QWEN
    else:
        rank_map = DISTRIBUTED_FLEET_RANK_MAP

    if manifest_path is None:
        raise ValueError(
            "distributed_fleet strategy requires manifest_path kwarg"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)
    clients = manifest.get('clients', [])

    if len(clients) != num_clients:
        raise ValueError(
            f"Manifest has {len(clients)} clients but config expects "
            f"{num_clients}"
        )

    first_type = next(iter(config_types.values()))
    module_names = list(first_type.keys())

    config_local = {
        'alpha': default_alpha,
        'lora_dropout': default_dropout,
    }

    for entry in clients:
        cid = entry['client_id']
        budget_cls = entry.get('budget_class', entry['device_class'])
        rank = rank_map.get(budget_cls)
        if rank is None:
            raise ValueError(
                f"No rank mapping for budget_class='{budget_cls}' "
                f"(client {cid}). Known classes: {list(rank_map.keys())}"
            )
        config_local[f'Client_{cid}'] = {m: rank for m in module_names}

    logger.info(
        "distributed_fleet: assigned ranks %s",
        {e['client_id']: rank_map.get(
            e.get('budget_class', e['device_class']))
         for e in clients}
    )
    return config_local


def get_client_rank_caps(config_local, reduction='min'):
    """
    Derive immutable per-client scalar rank caps from a heterogeneous
    client configuration dictionary.

    Args:
        config_local: Client configuration dict from get_client_lora_config
            or a manually supplied config_local.
        reduction: How to reduce module-wise ranks to a single scalar cap.
            - 'min': safe scalar cap for schedulers that use one rank per client
            - 'max': largest module rank per client

    Returns:
        Dict mapping integer client IDs to integer caps, e.g.:
        {1: 64, 2: 120, 3: 200}
    """
    if config_local is None:
        return {}

    if reduction not in {'min', 'max'}:
        raise ValueError("reduction must be either 'min' or 'max'")

    client_rank_caps = {}

    for client_key, client_config in config_local.items():
        client_key_str = str(client_key)
        if not client_key_str.startswith('Client_'):
            continue
        if not hasattr(client_config, 'items'):
            continue

        numeric_ranks = [
            int(v) for _, v in client_config.items()
            if isinstance(v, (int, float))
        ]
        if not numeric_ranks:
            continue

        try:
            client_id = int(client_key_str.split('_')[-1])
        except ValueError:
            continue

        client_rank_caps[client_id] = (
            min(numeric_ranks) if reduction == 'min' else max(numeric_ranks)
        )

    return client_rank_caps

def get_default_client_types():
    """
    Get default client type configurations similar to FlexLoRA.
    
    Returns:
        Dict mapping type names to their LoRA rank configurations
    """
    return {
        'Type_0': {  # small
            'q_proj': 64, 'v_proj': 64, 'k_proj': 64, 'o_proj': 64,
            'gate_proj': 64, 'down_proj': 64, 'up_proj': 64
        },
        'Type_1': {  # medium-low (was previously Type_3)
            'q_proj': 120, 'v_proj': 120, 'k_proj': 120, 'o_proj': 120,
            'gate_proj': 120, 'down_proj': 120, 'up_proj': 120
        },
        'Type_2': {  # medium — matches agxavier budget (Option B)
            'q_proj': 150, 'v_proj': 150, 'k_proj': 150, 'o_proj': 150,
            'gate_proj': 150, 'down_proj': 150, 'up_proj': 150
        },
        'Type_3': {  # large (was previously Type_1)
            'q_proj': 200, 'v_proj': 200, 'k_proj': 200, 'o_proj': 200,
            'gate_proj': 200, 'down_proj': 200, 'up_proj': 200
        }
    }


def get_client_config_with_alpha(config_types, num_clients, strategy='random',
                                 seed=42, alpha_config=None, default_alpha=16,
                                 default_dropout=0.05):
    """
    Generate client configurations with client-specific alpha values.
    
    Args:
        config_types: Dict mapping type names to their LoRA configurations
        num_clients: Total number of clients
        strategy: Distribution strategy (same as get_client_lora_config)
        seed: Random seed
        alpha_config: Dict mapping client IDs or types to alpha values
            Format: {'Client_0': 16, 'Client_1': 32, ...} or
                    {'Type_0': 16, 'Type_1': 32, ...}
        default_alpha: Default alpha if not specified per client
        default_dropout: Default dropout rate
    
    Returns:
        Dict with client configurations including per-client alpha values
    """
    config_local = get_client_lora_config(
        config_types, num_clients, strategy, seed, default_alpha, default_dropout
    )
    
    if config_local is None:
        return None
    
    # Add client-specific alpha values if provided
    if alpha_config:
        for client_id in config_local.keys():
            if client_id.startswith('Client_'):
                # Check if alpha is specified for this client
                if client_id in alpha_config:
                    config_local[client_id + '_alpha'] = alpha_config[client_id]
                # Check if alpha is specified for the client's type
                elif 'Type_' in str(config_local[client_id]):
                    # Try to infer type from config and get alpha
                    for type_name, alpha in alpha_config.items():
                        if type_name.startswith('Type_'):
                            # This is a simplified check - in practice you'd need
                            # to track which type each client got
                            pass
    
    return config_local


def get_client_target_modules(config_local, default_target_modules=None):
    """
    Get target modules for each client based on their configuration.
    
    Args:
        config_local: Client configuration dict from get_client_lora_config
        default_target_modules: Default list of target modules if not specified
    
    Returns:
        Dict mapping client IDs to their target module lists
    """
    if config_local is None:
        return None
    
    if default_target_modules is None:
        default_target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj',
                                 'gate_proj', 'down_proj', 'up_proj']
    
    target_modules_dict = {}
    for client_id, client_config in config_local.items():
        if client_id.startswith('Client_'):
            # Extract target modules from the client's rank configuration
            # (modules that have ranks defined)
            target_modules = list(client_config.keys())
            target_modules_dict[client_id] = target_modules
    
    return target_modules_dict
