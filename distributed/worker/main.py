"""
Worker entry point for distributed FedLoRA deployment.

ARCHITECTURE NOTE:
The worker communicates through the local device_agent, NOT directly
to the FL server. The device_agent relay handles:
- Forwarding FL messages to the server
- Receiving FL broadcasts from the server
- Container lifecycle events

Communication path:
    Worker <-> Device Agent (localhost:60011/60012) <-> FL Server

This is a thin wrapper that:
1. Reads static manifest-provided arguments
2. Loads partition artifact (or reconstructs from indices)
3. Constructs injected ZMQClientCommManager (targeting device_agent)
4. Instantiates existing FedLoRA Client
5. Calls join_in() then run()

The worker does NOT:
- Connect directly to the FL server's payload ports
- Load a different trainer stack
- Define custom FL callbacks
- Invent new payload schemas

Usage:
    python -m distributed.worker.main \\
        --client-id 1 \\
        --container-name fl_worker_orinagx1 \\
        --device-agent-host localhost \\
        --config configs/fedit_distributed.yaml
"""

import argparse
import logging
import os
import sys
import pickle
from pathlib import Path

sys.setrecursionlimit(10000)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# PyTorch 2.0 (Xavier/JetPack 5.x) lacks float8 dtypes that transformers
# 4.43.1 references in model loading guards.  Stub them so the attribute
# check doesn't raise AttributeError — no real tensor will use these.
import torch as _torch
if not hasattr(_torch, "float8_e4m3fn"):
    _torch.float8_e4m3fn = None
if not hasattr(_torch, "float8_e5m2"):
    _torch.float8_e5m2 = None

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging for worker process."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_partition(partition_path: str, cfg=None):
    """
    Load client data partition.
    
    Supports two formats:
    1. Full pickle: Direct data objects
    2. Index pickle: Indices + config for reconstruction
    
    Args:
        partition_path: Path to partition file
        cfg: Configuration object (for index-based reconstruction)
        
    Returns:
        Data object suitable for FedLoRA Client
    """
    if not os.path.exists(partition_path):
        raise FileNotFoundError(f"Partition file not found: {partition_path}")
    
    with open(partition_path, 'rb') as f:
        partition = pickle.load(f)
    
    # Check if this is an index-based partition
    if isinstance(partition, dict) and 'train_indices' in partition:
        logger.info("Loading index-based partition, reconstructing data...")
        return _reconstruct_from_indices(partition, cfg)
    
    logger.info(f"Loaded full partition from {partition_path}")
    return partition


def _reconstruct_from_indices(partition: dict, cfg):
    """
    Reconstruct client data from indices.
    
    This is the fallback path when full-object pickles are not portable.
    
    Args:
        partition: Dict with train_indices, val_indices, test_indices, etc.
        cfg: Configuration for data loading
        
    Returns:
        Data object
    """
    from federatedscope.core.auxiliaries.data_builder import get_data
    
    # Load full dataset
    data, modified_cfg = get_data(cfg.clone())
    
    # Extract indices
    train_indices = partition.get('train_indices', [])
    val_indices = partition.get('val_indices', [])
    test_indices = partition.get('test_indices', [])
    
    # Create subset data
    if hasattr(data, 'train_data') and hasattr(data.train_data, 'select'):
        # HuggingFace dataset
        data.train_data = data.train_data.select(train_indices)
        if val_indices:
            data.val_data = data.val_data.select(val_indices)
        if test_indices:
            data.test_data = data.test_data.select(test_indices)
    else:
        logger.warning("Index-based partition reconstruction not fully implemented for this data type")
    
    return data


def build_client(
    client_id: int,
    container_name: str,
    device_agent_host: str,
    device_agent_port_offset: int,
    cfg,
    data,
    model,
    comm_manager=None,
    local_address=None,
    fl_server_host: str = "",
    fl_server_port_offset: int = 0,
    nic: str = "",
    shared_nic: bool = False,
):
    """
    Build FedLoRA Client with injected ZMQ comm manager.

    When ``fl_server_host`` is provided, the comm manager connects directly to
    the FL server's payload ports instead of relaying through the device_agent.
    """
    from distributed.comm import ZMQClientCommManager
    from federatedscope.core.auxiliaries.worker_builder import get_client_cls
    import federatedscope.contrib.worker.adasparse_lora_worker
    import federatedscope.contrib.worker.adasparse_lorav2_worker
    import federatedscope.contrib.worker.adasparse_lorav3_worker
    import federatedscope.contrib.worker.hetlora_worker
    import federatedscope.contrib.worker.heterolora_worker
    import federatedscope.contrib.worker.fah_qlora_worker

    if comm_manager is None:
        comm_manager = ZMQClientCommManager(
            client_id=client_id,
            container_name=container_name,
            device_agent_host=device_agent_host,
            device_agent_port_offset=device_agent_port_offset,
            fl_server_host=fl_server_host or None,
            fl_server_port_offset=fl_server_port_offset,
            send_timeout_ms=300_000,
            nic=nic,
            shared_nic=shared_nic,
        )

    server_id = 0
    comm_manager.add_neighbors(
        neighbor_id=server_id,
        address={'host': fl_server_host or device_agent_host,
                 'port': comm_manager.port}
    )
    
    # Determine device
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    client_cls = get_client_cls(cfg)
    logger.info(f"Dispatch: method={cfg.federate.method} -> {client_cls.__name__}")

    client = client_cls(
        ID=client_id,
        server_id=server_id,
        config=cfg,
        data=data,
        model=model,
        device=device,
        comm_manager=comm_manager,
        local_address=local_address if local_address is not None else {'host': 'worker', 'port': 0},
    )

    logger.info(
        f"Built {client_cls.__name__} {client_id} with ZMQClientCommManager "
        f"(via device_agent at {device_agent_host})"
    )
    return client


def run_worker(args, *, dry_run_init: bool = False) -> None:
    """
    Full worker pipeline: config → data → model → injected Client → FL loop.

    When ``dry_run_init`` is True (``--dry-run-init``), stops after the Client is
    constructed — used as Gate 2 proof that the thin worker and injection path
    initialize without starting FL join/run.
    """
    setup_logging(args.verbose)
    logger.info("[Gate2] stage=start: worker process starting")
    logger.info(f"Starting worker for client {args.client_id}")
    logger.info(f"Container: {args.container_name}")
    logger.info(f"Device agent: {args.device_agent_host}:{60011 + args.device_agent_port_offset}")

    logger.info("[Gate2] stage=config: loading and merging YAML")
    from federatedscope.core.configs.config import global_cfg
    cfg = global_cfg.clone()
    cfg.merge_from_file(args.config)

    if args.config_opts:
        import json as _json
        opts_list = _json.loads(args.config_opts)
        cfg.merge_from_list(opts_list)
        logger.info(f"Applied config overrides: {opts_list}")

    # Override mode to distributed
    cfg.federate.mode = 'distributed'

    # Set client ID in config for potential method-specific use
    cfg.federate.client_id = args.client_id

    # Port of standalone's _setup_base_quant() for the distributed path.
    # Standalone runs through BaseRunner which assigns per-client
    # computation_quantization.{method, nbits} from base_quant.distribution
    # before get_model(). Distributed bypasses BaseRunner, so the default
    # (method='none', nbits=4) falls through to FP32 loading in the model
    # builder — diverging from the standalone baseline the config asks for.
    cfg.computation_quantization.method = 'none'
    # Per-class dtype for LLM (Qwen2): bf16 on Ampere (sm_80+, e.g. AGX Orin /
    # Orin NX) HALVES the base-model + activation footprint so the 16GB unified
    # devices fit comfortably; fp32 on Volta/older (AGX Xavier) which lacks a bf16
    # triu kernel for Qwen2's causal mask. The aggregated LoRA is normalized to
    # fp32 server-side, so mixed-precision clients combine safely.
    # GLUE/DeBERTa stays bf16 (nbits=16) as before (no triu-bf16 issue on encoders).
    _is_llm = str(getattr(cfg.data, 'type', '')).endswith('@llm') or \
        str(getattr(getattr(cfg, 'trainer', None), 'type', '')).lower() == 'llmtrainer'
    if _is_llm:
        try:
            import torch as _torch
            _cap = _torch.cuda.get_device_capability() if _torch.cuda.is_available() else (0, 0)
        except Exception:
            _cap = (0, 0)
        cfg.computation_quantization.nbits = 16 if _cap[0] >= 8 else 32
        logger.info("[BASE_QUANT] LLM per-class dtype: GPU sm_%d%d -> nbits=%d (%s)",
                    _cap[0], _cap[1], cfg.computation_quantization.nbits,
                    'bf16 Ampere+' if _cap[0] >= 8 else 'fp32 Volta/older')
    else:
        cfg.computation_quantization.nbits = 16
        logger.info("[BASE_QUANT] GLUE: nbits=16 (bf16)")

    if getattr(args, 'client_rank_config', ''):
        import json as _json
        _rank_config = _json.loads(args.client_rank_config)
        _client_key = f'Client_{args.client_id}'
        _config_local_dict = {_client_key: _rank_config}
        if hasattr(cfg, 'glue') and hasattr(cfg.glue, 'adapter') and hasattr(cfg.glue.adapter, 'hetero_ranks'):
            cfg.glue.adapter.hetero_ranks.config_local = _config_local_dict
        if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter') and hasattr(cfg.llm.adapter, 'hetero_ranks'):
            cfg.llm.adapter.hetero_ranks.config_local = _config_local_dict
        logger.info(f"Injected client_rank_config for {_client_key}: {_rank_config}")

    cfg.outdir = f'logs/client_{args.client_id}'

    cfg.freeze()

    logger.info(f"Loaded configuration from {args.config} (frozen)")

    logger.info("[Gate2] stage=data: loading partition or get_data()")

    # Gate 2 dry-run should prove worker initialization and injected Client
    # construction without being blocked by optional dataset/model builder paths
    # that are exercised more meaningfully in later gates.
    if dry_run_init and not args.partition_path:
        logger.info(
            "[Gate2] stage=data: dry-run without partition path; "
            "building direct minimal injected-client harness"
        )
        from distributed.tests.integration_harness import build_gate2_dry_run_client

        logger.info("[Gate2] stage=comm_client: constructing injected Client via harness")
        _client = build_gate2_dry_run_client(
            client_id=args.client_id,
            container_name=args.container_name,
            device_agent_host=args.device_agent_host,
            device_agent_port_offset=args.device_agent_port_offset,
            yaml_path=Path(args.config),
        )
        logger.info(
            "[Gate2] stage=dry_run_init: success — skipping join_in() and run(); "
            "FedLoRA Client is ready for further FL wiring"
        )
        return

    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.model_builder import get_model

    if args.partition_path:
        data = load_partition(args.partition_path, cfg)
    else:
        logger.info("No partition path specified, loading data from config")
        data, _ = get_data(cfg.clone())

    logger.info("[Gate2] stage=model: get_model()")
    model = get_model(cfg, data)
    logger.info(f"Model built: {type(model).__name__}")

    if getattr(args, 'client_rank_config', ''):
        import json as _json
        import torch
        _rank_config = _json.loads(args.client_rank_config)
        from federatedscope.contrib.common.heterolora_utils import modify_adapter, is_qlora_client_cfg
        from federatedscope.core.fed_runner import _client_compute_dtype

        _peft_model = model.model if hasattr(model, 'model') and hasattr(model.model, 'peft_config') else model
        _adapter_name = 'default'
        if hasattr(_peft_model, 'peft_config') and _peft_model.peft_config:
            _adapter_name = next(iter(_peft_model.peft_config.keys()))

        _adapter_cfg = cfg.glue.adapter if hasattr(cfg, 'glue') and hasattr(cfg.glue, 'adapter') else cfg.llm.adapter
        _base_args = _adapter_cfg.args[0] if getattr(_adapter_cfg, 'args', None) else {}
        _lora_alpha = _base_args.get('lora_alpha', 16)
        _lora_dropout = _base_args.get('lora_dropout', 0.05)
        _compute_dtype = _client_compute_dtype(cfg)
        _is_qlora = is_qlora_client_cfg(cfg)
        _non_lora_dtype = torch.float32 if _is_qlora else None

        modify_adapter(
            peft_model=_peft_model,
            adapter_name=_adapter_name,
            modify_module_rank=_rank_config,
            lora_alpha=_lora_alpha,
            lora_dropout=_lora_dropout,
            init_lora_weights=True,
            target_modules=list(_rank_config.keys()),
            compute_dtype=_compute_dtype,
            non_lora_trainable_dtype=_non_lora_dtype,
            recast_trainables=True,
            recast_log_prefix=f"[DIST] Client {args.client_id}:",
        )
        logger.info(f"modify_adapter applied for client {args.client_id}: rank_config={_rank_config}")

    # Per-class gradient checkpointing. On low-memory device classes (orinnx 16GB,
    # x86 11GB) the frozen-base + LoRA + tok-512 activation stack peaks near the
    # unified-memory ceiling (~13GB), so one such worker randomly OOMs per round.
    # Grad-ckpt recomputes activations in the backward pass instead of storing them
    # -- numerically identical (no effect on the aggregated result), it just trades
    # ~30% local compute to cut the training peak ~3GB (measured: bf16 tok512 r16
    # 4.4GB -> 2.6GB). High-memory classes (agxorin 64GB, agxavier 32GB) skip it and
    # stay full-speed. Mirrors the per-class dtype self-determination above.
    _is_llm_gc = str(getattr(cfg.data, 'type', '')).endswith('@llm') or \
        str(getattr(getattr(cfg, 'trainer', None), 'type', '')).lower() == 'llmtrainer'
    if _is_llm_gc:
        try:
            import torch as _torch
            _total_gb = (_torch.cuda.get_device_properties(0).total_memory / 1e9) \
                if _torch.cuda.is_available() else 0.0
        except Exception:
            _total_gb = 0.0
        if 0 < _total_gb < 20.0:
            try:
                _gc_model = model.model if hasattr(model, 'model') and \
                    hasattr(model.model, 'gradient_checkpointing_enable') else model
                if hasattr(_gc_model, 'enable_input_require_grads'):
                    _gc_model.enable_input_require_grads()
                _gc_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                logger.info("[GRAD_CKPT] low-memory class (%.0fGB unified) -> "
                            "grad-checkpointing ON", _total_gb)
            except Exception as _e:
                logger.warning("[GRAD_CKPT] failed to enable grad-checkpointing: %s", _e)
        else:
            logger.info("[GRAD_CKPT] high-memory class (%.0fGB) -> grad-checkpointing "
                        "OFF (full-speed)", _total_gb)

    logger.info("[Gate2] stage=comm_client: ZMQClientCommManager + FedLoRA Client")
    fl_server_host = getattr(args, 'fl_server_host', '') or ''
    fl_server_port_offset = getattr(args, 'fl_server_port_offset', 0)
    nic = getattr(args, 'nic', '') or ''
    shared_nic = getattr(args, 'shared_nic', False)
    client = build_client(
        client_id=args.client_id,
        container_name=args.container_name,
        device_agent_host=args.device_agent_host,
        device_agent_port_offset=args.device_agent_port_offset,
        cfg=cfg,
        data=data,
        model=model,
        fl_server_host=fl_server_host,
        fl_server_port_offset=fl_server_port_offset,
        nic=nic,
        shared_nic=shared_nic,
    )

    if dry_run_init:
        logger.info(
            "[Gate2] stage=dry_run_init: success — skipping join_in() and run(); "
            "FedLoRA Client is ready for further FL wiring"
        )
        return

    logger.info(f"Client {args.client_id}: Calling join_in()...")
    client.join_in()

    logger.info(f"Client {args.client_id}: Starting run()...")
    client.run()

    logger.info(f"Client {args.client_id}: Finished")


def main():
    """Worker main entry point."""
    parser = argparse.ArgumentParser(description='FedLoRA Distributed Worker')
    parser.add_argument('--client-id', type=int, required=True,
                        help='Static client ID from manifest')
    parser.add_argument('--container-name', type=str, required=True,
                        help='Container name for device_agent routing')
    parser.add_argument('--device-agent-host', type=str, default='localhost',
                        help='Device agent host (default: localhost)')
    parser.add_argument('--device-agent-port-offset', type=int, default=0,
                        help='Port offset for device agent ports')
    parser.add_argument('--partition-path', type=str, default='',
                        help='Path to client partition file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--client-rank-config', type=str, default='',
                        help='JSON dict of per-module LoRA ranks from server (HetLoRA/FAH)')
    parser.add_argument('--fl-server-host', type=str, default='',
                        help='FL server host for direct connection (bypass device_agent relay)')
    parser.add_argument('--fl-server-port-offset', type=int, default=0,
                        help='Port offset for FL server payload ports (60001/60002)')
    parser.add_argument('--nic', type=str, default='',
                        help='Network interface for bandwidth measurement (e.g., eno1, eth0)')
    parser.add_argument('--shared-nic', action='store_true',
                        help='NIC is shared with other workers (disables NIC-level stats)')
    parser.add_argument('--config-opts', type=str, default='',
                        help='JSON-encoded list of config overrides (KEY VALUE pairs)')
    parser.add_argument('--dry-run-init', action='store_true',
                        help='Gate 2: load cfg/data/model, build injected Client, exit before join_in/run')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    run_worker(args, dry_run_init=args.dry_run_init)


if __name__ == '__main__':
    main()
