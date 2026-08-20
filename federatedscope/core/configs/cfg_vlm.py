import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config

logger = logging.getLogger(__name__)


def extend_vlm_cfg(cfg):
    cfg.vlm = CN()
    cfg.vlm.tok_len = 512
    cfg.vlm.retry_on_nan_loss = False
    cfg.vlm.image_dir = ''
    cfg.vlm.min_pixels = 200704    # 256*28*28
    cfg.vlm.max_pixels = 1003520   # 1280*28*28

    cfg.vlm.cache = CN()
    cfg.vlm.cache.model = ''

    cfg.vlm.adapter = CN()
    cfg.vlm.adapter.use = False
    cfg.vlm.adapter.args = [{}]
    cfg.vlm.adapter.mv_to_cpu = False

    cfg.vlm.adapter.max_rank = 64
    cfg.vlm.adapter.hetero_strategy = 'homo'
    cfg.vlm.adapter.manifest_path = ''
    cfg.vlm.adapter.hetero_ranks = CN(new_allowed=True)
    cfg.vlm.adapter.hetero_alpha = CN(new_allowed=True)
    cfg.vlm.adapter.hetero_target_modules = CN(new_allowed=True)

    cfg.vlm.adapter.base_quant = CN()
    cfg.vlm.adapter.base_quant.enabled = False
    cfg.vlm.adapter.base_quant.distribution = CN(new_allowed=True)
    cfg.vlm.adapter.base_quant.lora_dtype = 'fp32'

    cfg.vlm.adapter.fah = CN()
    cfg.vlm.adapter.fah.enabled = False
    cfg.vlm.adapter.fah.init_rank = 8
    cfg.vlm.adapter.fah.r_min = 2
    cfg.vlm.adapter.fah.r_max = 64
    cfg.vlm.adapter.fah.lambda_dec = 1
    cfg.vlm.adapter.fah.lambda_inc = 1
    cfg.vlm.adapter.fah.warmup_rounds = 1
    cfg.vlm.adapter.fah.alpha_fraction = 0.3
    cfg.vlm.adapter.fah.validation_fraction = 0.1
    cfg.vlm.adapter.fah.validation_steps = 10
    cfg.vlm.adapter.fah.uplink_min_mbps = 5
    cfg.vlm.adapter.fah.uplink_max_mbps = 20
    cfg.vlm.adapter.fah.downlink_mbps = 50
    cfg.vlm.adapter.fah.network_trace_path = ''
    cfg.vlm.adapter.fah.network_trace_distribution = CN(new_allowed=True)
    cfg.vlm.adapter.fah.bandwidth_mode = 'static'
    cfg.vlm.adapter.fah.base_quant = CN()
    cfg.vlm.adapter.fah.base_quant.enabled = False
    cfg.vlm.adapter.fah.base_quant.distribution = CN(new_allowed=True)
    cfg.vlm.adapter.fah.base_quant.lora_dtype = 'fp32'

    cfg.vlm.adapter.hetlora = CN()
    cfg.vlm.adapter.hetlora.enabled = False
    cfg.vlm.adapter.hetlora.rank_min = 2
    cfg.vlm.adapter.hetlora.rank_max = 64
    cfg.vlm.adapter.hetlora.init_rank = 64
    cfg.vlm.adapter.hetlora.pruning = CN()
    cfg.vlm.adapter.hetlora.pruning.enabled = True
    cfg.vlm.adapter.hetlora.pruning.decay = 0.99
    cfg.vlm.adapter.hetlora.pruning.regularizer_weight = 0.01
    cfg.vlm.adapter.hetlora.aggregation = CN()
    cfg.vlm.adapter.hetlora.aggregation.mode = 'sparsity_weighted'
    cfg.vlm.adapter.hetlora.aggregation.epsilon = 1e-8

    cfg.vlm.adapter.adasparse_lora = CN()
    cfg.vlm.adapter.adasparse_lora.enabled = False
    cfg.vlm.adapter.adasparse_lora.rank_min = 2
    cfg.vlm.adapter.adasparse_lora.rank_max = 64
    cfg.vlm.adapter.adasparse_lora.init_rank = 64
    cfg.vlm.adapter.adasparse_lora.pruning = CN()
    cfg.vlm.adapter.adasparse_lora.pruning.enabled = True
    cfg.vlm.adapter.adasparse_lora.pruning.gamma = 0.9
    cfg.vlm.adapter.adasparse_lora.pruning.regularizer_weight = 0.01
    cfg.vlm.adapter.adasparse_lora.aggregation = CN()
    cfg.vlm.adapter.adasparse_lora.aggregation.mode = 'sparsity_weighted'
    cfg.vlm.adapter.adasparse_lora.aggregation.epsilon = 1e-8

    cfg.vlm.adapter.adasparse_lorav2 = CN()
    cfg.vlm.adapter.adasparse_lorav2.enabled = False
    cfg.vlm.adapter.adasparse_lorav2.rank_min = 2
    cfg.vlm.adapter.adasparse_lorav2.rank_max = 64
    cfg.vlm.adapter.adasparse_lorav2.init_rank = 64
    cfg.vlm.adapter.adasparse_lorav2.stage1 = CN()
    cfg.vlm.adapter.adasparse_lorav2.stage1.gamma = 0.9
    cfg.vlm.adapter.adasparse_lorav2.stage1.regularizer_weight = 0.01
    cfg.vlm.adapter.adasparse_lorav2.stage2 = CN()
    cfg.vlm.adapter.adasparse_lorav2.stage2.enabled = True
    cfg.vlm.adapter.adasparse_lorav2.stage2.q_up_bits = 8
    cfg.vlm.adapter.adasparse_lorav2.stage2.q_down_bits = 8
    cfg.vlm.adapter.adasparse_lorav2.stage2.cmeta_bits = 32
    cfg.vlm.adapter.adasparse_lorav2.stage2.uplink_budget_window_s = 1.0
    cfg.vlm.adapter.adasparse_lorav2.stage2.downlink_budget_window_s = 1.0
    cfg.vlm.adapter.adasparse_lorav2.stage2.selection_rule = 'greedy_ratio'
    cfg.vlm.adapter.adasparse_lorav2.stage2.residual_enabled = True
    cfg.vlm.adapter.adasparse_lorav2.bandwidth = CN()
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.enabled = True
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.mode = 'static'
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.network_trace_path = ''
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.network_trace_distribution = CN(new_allowed=True)
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.uplink_min_mbps = 5
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.uplink_max_mbps = 20
    cfg.vlm.adapter.adasparse_lorav2.bandwidth.downlink_mbps = 50
    cfg.vlm.adapter.adasparse_lorav2.aggregation = CN()
    cfg.vlm.adapter.adasparse_lorav2.aggregation.mode = 'sample_size'
    cfg.vlm.adapter.adasparse_lorav2.aggregation.epsilon = 1e-8

    cfg.vlm.adapter.adasparse_lorav3 = CN()
    cfg.vlm.adapter.adasparse_lorav3.enabled = False
    cfg.vlm.adapter.adasparse_lorav3.rank_min = 4
    cfg.vlm.adapter.adasparse_lorav3.rank_max = 64
    cfg.vlm.adapter.adasparse_lorav3.init_rank = 64
    cfg.vlm.adapter.adasparse_lorav3.stage1 = CN()
    cfg.vlm.adapter.adasparse_lorav3.stage1.gamma = 0.9
    cfg.vlm.adapter.adasparse_lorav3.stage1.regularizer_weight = 0.01
    cfg.vlm.adapter.adasparse_lorav3.stage2 = CN()
    cfg.vlm.adapter.adasparse_lorav3.stage2.enabled = True
    cfg.vlm.adapter.adasparse_lorav3.stage2.q_up_bits = 8
    cfg.vlm.adapter.adasparse_lorav3.stage2.q_down_bits = 8
    cfg.vlm.adapter.adasparse_lorav3.stage2.cmeta_bits = 32
    cfg.vlm.adapter.adasparse_lorav3.stage2.uplink_budget_window_s = 1.0
    cfg.vlm.adapter.adasparse_lorav3.stage2.downlink_budget_window_s = 1.0
    cfg.vlm.adapter.adasparse_lorav3.stage2.selection_rule = 'greedy_ratio'
    cfg.vlm.adapter.adasparse_lorav3.stage2.residual_enabled = True
    cfg.vlm.adapter.adasparse_lorav3.aggregation = CN()
    cfg.vlm.adapter.adasparse_lorav3.aggregation.mode = 'sample_size'
    cfg.vlm.adapter.adasparse_lorav3.aggregation.epsilon = 1e-8
    cfg.vlm.adapter.adasparse_lorav3.component_identity = 'layer_global_idx'
    cfg.vlm.adapter.adasparse_lorav3.layer_key_mode = 'canonical_exact_path'
    cfg.vlm.adapter.adasparse_lorav3.grouped_payloads = True
    cfg.vlm.adapter.adasparse_lorav3.stage1_global_competition = True
    cfg.vlm.adapter.adasparse_lorav3.stage2_global_competition = True


def assert_vlm_cfg(cfg):
    pass


register_config("vlm", extend_vlm_cfg)
