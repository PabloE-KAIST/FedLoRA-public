"""
Configuration for GLUE benchmark tasks with FAH-QLoRA support.

This module extends the configuration with GLUE-specific options for
sequence classification and regression tasks (SST-2, QNLI, MRPC, STS-B, RTE, etc.)
with DeBERTa and other transformer models.
"""
import logging
from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_glue_cfg(cfg):
    """
    Extend configuration with GLUE-specific options.
    
    This includes:
    - Sequence length and tokenization settings
    - Task-specific configurations
    - Adapter settings mirroring llm.adapter structure
    - FAH-QLoRA configuration for GLUE tasks
    """
    # ---------------------------------------------------------------------- #
    # GLUE related options
    # ---------------------------------------------------------------------- #
    cfg.glue = CN()
    cfg.glue.max_length = 128  # Maximum sequence length for tokenization
    cfg.glue.retry_on_nan_loss = True  # Retry batch if loss is NaN
    cfg.glue.task_name = ''  # Populated automatically from data.type during dataset loading
    cfg.glue.mnli_eval_split = 'matched'  # 'matched' or 'mismatched'

    # Cache directory for models and tokenizers
    cfg.glue.cache_dir = ''
    
    # ---------------------------------------------------------------------- #
    # Adapters for GLUE (mirrors llm.adapter structure for consistency)
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter = CN()
    cfg.glue.adapter.use = False
    cfg.glue.adapter.args = [{}]
    cfg.glue.adapter.mv_to_cpu = False
    
    # HeteroLoRA specific configurations
    cfg.glue.adapter.max_rank = 64  # Maximum rank for zero-padding/truncation
    
    # Client configuration distribution strategy
    # Options: 'homo', 'random', 'heavy_tail', 'heavy_tail_strong', 'normal'
    cfg.glue.adapter.hetero_strategy = 'homo'
    cfg.glue.adapter.manifest_path = ''
    
    # Client-specific configurations (optional, can be auto-generated)
    cfg.glue.adapter.hetero_ranks = CN(new_allowed=True)
    cfg.glue.adapter.hetero_alpha = CN(new_allowed=True)
    cfg.glue.adapter.hetero_target_modules = CN(new_allowed=True)

    # ---------------------------------------------------------------------- #
    # Base Model Quantization (usable with or without FAH-QLoRA)
    # This mirrors cfg.*.adapter.fah.base_quant, but is NOT gated by fah.enabled.
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.base_quant = CN()
    cfg.glue.adapter.base_quant.enabled = False
    cfg.glue.adapter.base_quant.distribution = CN(new_allowed=True)
    cfg.glue.adapter.base_quant.lora_dtype = 'fp32'

    # ---------------------------------------------------------------------- #
    # FAH-QLoRA: Federated Adaptive Heterogeneous QLoRA for GLUE
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.fah = CN()
    cfg.glue.adapter.fah.enabled = False
    cfg.glue.adapter.fah.init_rank = 8
    cfg.glue.adapter.fah.r_min = 2
    cfg.glue.adapter.fah.r_max = 64
    cfg.glue.adapter.fah.lambda_dec = 1
    cfg.glue.adapter.fah.lambda_inc = 1
    cfg.glue.adapter.fah.warmup_rounds = 1
    cfg.glue.adapter.fah.alpha_fraction = 0.3
    cfg.glue.adapter.fah.validation_fraction = 0.1
    cfg.glue.adapter.fah.validation_steps = 10
    
    # ---------------------------------------------------------------------- #
    # Heterogeneous Base Model Quantization (Official FAH-QLoRA feature)
    # Per the paper: "Devices are divided into three groups: one-third utilize
    # 4-bit quantized base models, another third use 8-bit quantized base 
    # models, and the remaining third employ 16-bit quantized base models.
    # The default LoRA data type is FP32."
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.fah.base_quant = CN()
    cfg.glue.adapter.fah.base_quant.enabled = False  # Enable heterogeneous base quantization
    # Distribution of quantization bits across clients
    # Keys are bit-widths (4, 8, 16), values are fractions summing to 1.0
    # Default: 1/3 each as per paper
    cfg.glue.adapter.fah.base_quant.distribution = CN(new_allowed=True)
    # Note: The actual default values {4: 0.333, 8: 0.333, 16: 0.334} are set
    # programmatically in fed_runner.py since CN doesn't support numeric keys
    cfg.glue.adapter.fah.base_quant.lora_dtype = 'fp32'  # LoRA modules dtype: 'fp32' or 'fp16'

    # ---------------------------------------------------------------------- #
    # HetLoRA Complete: Rank self-pruning + Sparsity-weighted aggregation
    # Mirrors llm.adapter.hetlora structure for GLUE tasks
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.hetlora = CN()
    cfg.glue.adapter.hetlora.enabled = False
    
    # Rank bounds
    cfg.glue.adapter.hetlora.rank_min = 4
    cfg.glue.adapter.hetlora.rank_max = 64
    cfg.glue.adapter.hetlora.init_rank = 64
    
    # Pruning configuration
    cfg.glue.adapter.hetlora.pruning = CN()
    cfg.glue.adapter.hetlora.pruning.enabled = True
    cfg.glue.adapter.hetlora.pruning.decay = 0.99     # Decay factor for prune threshold
    cfg.glue.adapter.hetlora.pruning.regularizer_weight = 0.01
    
    # Aggregation configuration
    cfg.glue.adapter.hetlora.aggregation = CN()
    cfg.glue.adapter.hetlora.aggregation.mode = 'sparsity_weighted'
    cfg.glue.adapter.hetlora.aggregation.epsilon = 1e-8

    # ---------------------------------------------------------------------- #
    # AdaSparse-LoRA: Index-aware rank-1 component aggregation
    # Mirrors llm.adapter.adasparse_lora structure for GLUE tasks
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.adasparse_lora = CN()
    cfg.glue.adapter.adasparse_lora.enabled = False
    
    # Rank bounds
    cfg.glue.adapter.adasparse_lora.rank_min = 4
    cfg.glue.adapter.adasparse_lora.rank_max = 64
    cfg.glue.adapter.adasparse_lora.init_rank = 64
    
    # Pruning configuration (component-based, ratio-based)
    cfg.glue.adapter.adasparse_lora.pruning = CN()
    cfg.glue.adapter.adasparse_lora.pruning.enabled = True
    cfg.glue.adapter.adasparse_lora.pruning.gamma = 0.9
    cfg.glue.adapter.adasparse_lora.pruning.regularizer_weight = 0.01
    
    # Aggregation configuration (index-aware)
    cfg.glue.adapter.adasparse_lora.aggregation = CN()
    cfg.glue.adapter.adasparse_lora.aggregation.mode = 'sparsity_weighted' #or sample_size
    cfg.glue.adapter.adasparse_lora.aggregation.epsilon = 1e-8
    
    # ---------------------------------------------------------------------- #
    # AdaSparse-LoRAv2: Two-stage sparse federated LoRA
    # Stage 1: Structural sparsity over survivor set (same as v1)
    # Stage 2: Communication sparsity via residual-aware budgeted selection
    # Mirrors llm.adapter.adasparse_lorav2 structure for GLUE tasks
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.adasparse_lorav2 = CN()
    cfg.glue.adapter.adasparse_lorav2.enabled = False  # Master switch
    
    # Rank bounds
    cfg.glue.adapter.adasparse_lorav2.rank_min = 4
    cfg.glue.adapter.adasparse_lorav2.rank_max = 64
    cfg.glue.adapter.adasparse_lorav2.init_rank = 64
    
    # Stage 1 configuration (structural sparsity over survivor set)
    cfg.glue.adapter.adasparse_lorav2.stage1 = CN()
    cfg.glue.adapter.adasparse_lorav2.stage1.gamma = 0.9
    cfg.glue.adapter.adasparse_lorav2.stage1.regularizer_weight = 0.01
    
    # Stage 2 configuration (communication sparsity over survivors)
    cfg.glue.adapter.adasparse_lorav2.stage2 = CN()
    cfg.glue.adapter.adasparse_lorav2.stage2.enabled = True
    cfg.glue.adapter.adasparse_lorav2.stage2.q_up_bits = 8
    cfg.glue.adapter.adasparse_lorav2.stage2.q_down_bits = 8
    cfg.glue.adapter.adasparse_lorav2.stage2.cmeta_bits = 32
    cfg.glue.adapter.adasparse_lorav2.stage2.uplink_budget_window_s = 1.0
    cfg.glue.adapter.adasparse_lorav2.stage2.downlink_budget_window_s = 1.0
    cfg.glue.adapter.adasparse_lorav2.stage2.selection_rule = 'greedy_ratio'
    cfg.glue.adapter.adasparse_lorav2.stage2.residual_enabled = True   
 
    # Aggregation configuration
    cfg.glue.adapter.adasparse_lorav2.aggregation = CN()
    cfg.glue.adapter.adasparse_lorav2.aggregation.mode = 'sample_size'
    cfg.glue.adapter.adasparse_lorav2.aggregation.epsilon = 1e-8
    
    # ---------------------------------------------------------------------- #
    # AdaSparse-LoRAv3: True layer-aware component identity
    # Key difference from v2: exact layer-aware ComponentID = (layer_key, global_idx)
    # Supports grouped payloads and cross-layer global competition
    # ---------------------------------------------------------------------- #
    cfg.glue.adapter.adasparse_lorav3 = CN()
    cfg.glue.adapter.adasparse_lorav3.enabled = False  # Master switch
    
    # Rank bounds
    # NOTE: rank_min is intentionally a shared/global minimum (not per-layer).
    # This is a deliberate simplification for research clarity. The shared rank_min
    # ensures no layer is pruned below a minimum threshold across the entire model.
    # Future work could introduce per-layer minimum policies if needed, but the
    # current implementation keeps a simpler shared/global rank_min behavior.
    cfg.glue.adapter.adasparse_lorav3.rank_min = 4
    cfg.glue.adapter.adasparse_lorav3.rank_max = 64
    cfg.glue.adapter.adasparse_lorav3.init_rank = 64
    
    # Stage 1 configuration (structural sparsity over survivor set)
    cfg.glue.adapter.adasparse_lorav3.stage1 = CN()
    cfg.glue.adapter.adasparse_lorav3.stage1.gamma = 0.9
    cfg.glue.adapter.adasparse_lorav3.stage1.regularizer_weight = 0.01
    
    # Stage 2 configuration (communication sparsity over survivors)
    cfg.glue.adapter.adasparse_lorav3.stage2 = CN()
    cfg.glue.adapter.adasparse_lorav3.stage2.enabled = True
    cfg.glue.adapter.adasparse_lorav3.stage2.q_up_bits = 8
    cfg.glue.adapter.adasparse_lorav3.stage2.q_down_bits = 8
    cfg.glue.adapter.adasparse_lorav3.stage2.cmeta_bits = 32
    cfg.glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s = 1.0
    cfg.glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s = 1.0
    cfg.glue.adapter.adasparse_lorav3.stage2.selection_rule = 'greedy_ratio'
    cfg.glue.adapter.adasparse_lorav3.stage2.residual_enabled = True
      
    # Aggregation configuration
    cfg.glue.adapter.adasparse_lorav3.aggregation = CN()
    cfg.glue.adapter.adasparse_lorav3.aggregation.mode = 'sample_size'
    cfg.glue.adapter.adasparse_lorav3.aggregation.epsilon = 1e-8
    
    # V3-specific: Layer-aware component identity configuration
    cfg.glue.adapter.adasparse_lorav3.component_identity = 'layer_global_idx'  # ComponentID = (layer_key, global_idx)
    cfg.glue.adapter.adasparse_lorav3.layer_key_mode = 'canonical_exact_path'  # Use exact layer paths
    cfg.glue.adapter.adasparse_lorav3.grouped_payloads = True  # Wire format uses grouped dicts
    
    # Milestone A/B competition flags
    # False = layer-wise selection (Milestone A safety), True = cross-layer global pool
    cfg.glue.adapter.adasparse_lorav3.stage1_global_competition = True
    cfg.glue.adapter.adasparse_lorav3.stage2_global_competition = True


def assert_glue_cfg(cfg):
    """
    Validate GLUE configuration settings.
    """
    if getattr(cfg.glue, 'mnli_eval_split', 'matched') not in ('matched', 'mismatched'):
        raise ValueError(
            f"Invalid glue.mnli_eval_split={cfg.glue.mnli_eval_split}. "
            f"Use 'matched' or 'mismatched'."
        )


register_config('glue', extend_glue_cfg)

