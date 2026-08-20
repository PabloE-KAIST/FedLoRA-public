"""Shared utilities for FederatedScope method refactor."""

from .config_resolver import (
    normalize_method_name,
    is_glue_task,
    is_vlm_task,
    get_adapter_root,
    get_adapter_args_list,
    get_hetero_ranks_cfg,
    get_hetlora_cfg,
    get_adasparse_cfg,
    get_adasparse_v2_cfg,
    get_adasparse_v3_cfg,
    get_fah_cfg,
    get_active_hetero_config_local,
    get_effective_target_modules,
    get_effective_max_rank,
)
from .payload_utils import (
    is_wrapped_payload,
    extract_model_para,
    copy_non_model_fields,
    build_wrapped_model_payload,
    get_payload_field,
)
from .rank_utils import (
    resolve_client_key,
    get_client_config,
    infer_rank_from_client_rank_config,
    logical_rank_from_indices,
    indices_from_rank,
    normalize_indices,
    validate_nonempty_indices,
)
from .bandwidth_manager import RoundBandwidthManager
from .network_trace_loader import (
    NetworkTraceLoader,
    assign_clients_to_classes,
    create_trace_loaders,
)
from .client_config_generator import (
    get_client_lora_config,
    get_client_rank_caps,
    get_default_client_types,
    get_client_config_with_alpha,
    get_client_target_modules,
)
from .heterolora_utils import (
    fah_resolve_client_compute_dtype,
    fah_cast_trainable_params_for_quantization,
    is_qlora_client_cfg,
    modify_adapter,
    distribute_weight_fast,
    load_weight_local,
    truncate_lora_weights,
    update_hetero_ranks_config,
    compute_lora_size,
    iter_lora_pairs,
    tail_penalty,
    tail_score,
    compute_effective_lora_norm_sq,
    compute_client_sparsity_weight,
    compute_sparsity_weight_from_state_dict,
    truncate_client_lora_to_rank,
    get_current_lora_rank,
    get_current_lora_rank_from_state_dict,
)
from .adasparse_lora_utils import (
    compute_component_scores,
    compute_component_scores_from_state_dict,
    compute_lowset_and_score,
    slice_lora_by_indices,
    reorder_lora_by_keep_positions,
    slice_update_by_keep_positions,
    validate_indices,
    validate_indices_match_rank,
    get_component_norm,
    compute_per_component_norms_from_state_dict,
    distribute_weights_by_indices,
    expand_client_update_to_global,
    compute_model_update_from_snapshot,
    apply_residual_to_update,
    compute_stage2_upload_scores,
    compute_stage2_downlink_scores,
    compute_component_upload_cost,
    compute_component_downlink_cost,
    greedy_select_by_score_cost_ratio,
    slice_model_update_by_indices,
    apply_sparse_update_to_model,
    update_residual_buffers_after_upload,
    prune_residual_buffers,
    validate_upload_subset,
    validate_download_subset,
    compute_residual_norm_summary,
)
from .adasparse_lorav3_utils import (
    # Layer key helpers
    canonicalize_lora_layer_key,
    strip_rank_suffix,
    get_canonical_lora_param_key,
    infer_layer_keys_from_state_dict,
    get_lora_keys_for_layer,
    get_lora_keys_for_layer_from_full_keys,
    # ComponentID helpers
    flatten_grouped_indices_by_layer,
    group_component_ids_by_layer,
    normalize_indices_to_grouped,
    normalize_grouped_indices_payload,  # deprecated, use normalize_indices_to_grouped
    validate_grouped_indices,
    # Layer-aware scoring
    compute_component_scores_grouped,
    compute_component_scores_grouped_from_state_dict,
    compute_lowset_grouped,
    # Stage 2 scoring (grouped)
    compute_stage2_upload_scores_grouped,
    compute_stage2_downlink_scores_grouped,
    # Layer-aware cost helpers
    compute_component_upload_cost_grouped,
    compute_component_downlink_cost_grouped,
    # Selection helpers (grouped)
    greedy_select_by_score_cost_ratio_grouped,
    # Distribution/slicing helpers (grouped)
    distribute_weights_by_layer_indices,
    slice_model_update_by_component_ids,
    apply_sparse_update_to_model_grouped,
    # Residual buffer helpers (grouped)
    update_residual_buffers_after_upload_grouped,
    apply_residual_to_update_grouped,
    prune_residual_buffers_grouped,
    # Model update computation
    compute_model_update_from_snapshot_grouped,
    # Validation helpers
    validate_upload_subset_grouped,
    validate_download_subset_grouped,
    compute_residual_norm_summary_grouped,
    validate_v3_integration_state,
)