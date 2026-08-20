"""Method-specific AdaSparse-LoRAv3 client.

This client implements true layer-aware AdaSparse-LoRA with:
- ComponentID = (layer_key, global_idx) internal representation
- Grouped survivor/upload/download indices per layer
- Layer-aware Stage 1 structural pruning
- Layer-aware Stage 2 communication sparsity
- ComponentID-keyed residual buffers

Key differences from v2:
- v2: flat integer component index shared across all layers
- v3: exact layer-specific ComponentID for all operations
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple

import federatedscope.contrib.common as fs_common
from federatedscope.contrib.worker.base_refactor_client import BaseRefactorClient
from federatedscope.contrib.common.adasparse_lorav3_utils import (
    ComponentID,
    canonicalize_lora_layer_key,
    infer_layer_keys_from_state_dict,
    get_lora_keys_for_layer,
    flatten_grouped_indices_by_layer,
    group_component_ids_by_layer,
    normalize_indices_to_grouped,
    compute_component_scores_grouped,
    compute_lowset_grouped,
    compute_stage1_adjusted_scores_grouped,
    compute_stage2_upload_scores_grouped,
    compute_component_upload_cost_grouped,
    greedy_select_by_score_cost_ratio_grouped,
    slice_model_update_by_component_ids,
    compute_model_update_from_snapshot_grouped,
    apply_residual_to_update_grouped,
    update_residual_buffers_after_upload_grouped,
    compute_residual_norm_summary_grouped,
    prune_residual_buffers_grouped,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) #logger.setLevel(logging.DEBUG)


class AdaSparseLoRAv3Client(BaseRefactorClient):
    """AdaSparse-LoRAv3 client with true layer-aware component identity."""
    
    METHOD_NAME = 'adasparse_lorav3'

    def _init_adasparse_lorav3(self):
        """
        Initialize AdaSparse-LoRAv3 attributes for layer-aware federated LoRA.
        
        V3 maintains layer-grouped state:
        - survivors_by_layer: Dict[layer_key, List[int]] - structural survivors per layer
        - upload_components_last: List[ComponentID] - uploaded components last round
        - download_components_last: List[ComponentID] - downloaded components last round
        - residual_buffers: Dict[ComponentID, Dict[str, Tensor]] - keyed by (layer_key, global_idx)
        """
        method = getattr(self._cfg.federate, 'method', '').lower()
    
        self.adasparse_v3_enabled = (
            method == 'adasparse_lorav3' and
            hasattr(self._cfg, 'llm') and
            hasattr(self._cfg.llm.adapter, 'adasparse_lorav3') and
            getattr(self._cfg.llm.adapter.adasparse_lorav3, 'enabled', False)
        )
    
        # Also check glue adapter config
        if not self.adasparse_v3_enabled:
            self.adasparse_v3_enabled = (
                method == 'adasparse_lorav3' and
                hasattr(self._cfg, 'glue') and
                hasattr(self._cfg.glue.adapter, 'adasparse_lorav3') and
                getattr(self._cfg.glue.adapter.adasparse_lorav3, 'enabled', False)
            )
    
        if not self.adasparse_v3_enabled:
            # Initialize empty state for disabled v3
            self._init_disabled_v3_state()
            return

        # Get adasparse_lorav2 config
        is_glue_task = (
            hasattr(self._cfg, 'data') and
            hasattr(self._cfg.data, 'type') and
            '@glue' in self._cfg.data.type.lower()
        )

        # Get adasparse_lorav3 config
        v3_cfg = fs_common.get_adasparse_v3_cfg(self._cfg)
        if v3_cfg is None:
            self._init_disabled_v3_state()
            return
    
        # Store config values
        self.adasparse_v3_rank_min = getattr(v3_cfg, 'rank_min', 2)
        self.adasparse_v3_rank_max = getattr(v3_cfg, 'rank_max', 64)
        self.adasparse_v3_init_rank = getattr(v3_cfg, 'init_rank', 64)
    
        # Stage 1 config (structural pruning)
        stage1_cfg = getattr(v3_cfg, 'stage1', None)
        if stage1_cfg:
            self.adasparse_v3_gamma = getattr(stage1_cfg, 'gamma', 0.9)
            self.adasparse_v3_reg_weight = getattr(stage1_cfg, 'regularizer_weight', 0.01)
        else:
            self.adasparse_v3_gamma = 0.9
            self.adasparse_v3_reg_weight = 0.01
    
        # Stage 2 config (communication sparsity)
        stage2_cfg = getattr(v3_cfg, 'stage2', None)
        if stage2_cfg:
            self.adasparse_v3_stage2_enabled = getattr(stage2_cfg, 'enabled', True)
            self.adasparse_v3_q_up_bits = getattr(stage2_cfg, 'q_up_bits', 8)
            self.adasparse_v3_q_down_bits = getattr(stage2_cfg, 'q_down_bits', 8)
            self.adasparse_v3_cmeta_bits = getattr(stage2_cfg, 'cmeta_bits', 32)
            self.adasparse_v3_uplink_window = getattr(stage2_cfg, 'uplink_budget_window_s', 1.0)
            self.adasparse_v3_downlink_window = getattr(stage2_cfg, 'downlink_budget_window_s', 1.0)
            self.adasparse_v3_selection_rule = getattr(stage2_cfg, 'selection_rule', 'greedy_ratio')
            self.adasparse_v3_residual_enabled = getattr(stage2_cfg, 'residual_enabled', True)
        else:
            self.adasparse_v3_stage2_enabled = True
            self.adasparse_v3_q_up_bits = 8
            self.adasparse_v3_q_down_bits = 8
            self.adasparse_v3_cmeta_bits = 32
            self.adasparse_v3_uplink_window = 1.0
            self.adasparse_v3_downlink_window = 1.0
            self.adasparse_v3_selection_rule = 'greedy_ratio'
            self.adasparse_v3_residual_enabled = True
        
        # V3-specific config
        self.adasparse_v3_stage1_global_competition = getattr(v3_cfg, 'stage1_global_competition', False)
        self.adasparse_v3_stage2_global_competition = getattr(v3_cfg, 'stage2_global_competition', False)
        
        # Stage 1 score-normalization / layer-importance prior
        self.adasparse_v3_stage1_norm_eps = 1e-12
        self.adasparse_v3_stage1_importance_strength = 1.0
        self.adasparse_v3_stage1_importance_beta = 0.2
        self.adasparse_v3_stage1_importance_min = 0.25
        self.adasparse_v3_stage1_importance_max = 1.0
        
        # Layer-importance state (updated after each pruning round)
        self.adasparse_v3_layer_importance: Dict[str, float] = {}
        
        # Debug bookkeeping for adjusted scores
        self.adasparse_v3_layer_medians_last: Dict[str, float] = {}
        self.adasparse_v3_adjusted_scores_before_low: Dict[ComponentID, float] = {}
    
        # Initialize v3 layer-grouped state
        # Will be populated on first model access
        self.adasparse_v3_survivors_by_layer: Dict[str, List[int]] = {}
        self.adasparse_v3_survivor_components: List[ComponentID] = []
        self.adasparse_v3_upload_components_last: Optional[List[ComponentID]] = None
        self.adasparse_v3_download_components_last: Optional[List[ComponentID]] = None
        
        # Pre-round state for Stage 1 pruning
        self.adasparse_v3_low_candidates_before: Optional[List[ComponentID]] = None
        self.adasparse_v3_scores_before_low: Dict[ComponentID, float] = {}
        self.adasparse_v3_survivors_before_stage1: Optional[Dict[str, List[int]]] = None
        
        # Snapshot and residual state for Stage 2
        self.adasparse_v3_pre_round_lora_snapshot: Optional[dict] = None
        self.adasparse_v3_residual_buffers: Dict[ComponentID, Dict[str, torch.Tensor]] = {}
        
        # Bandwidth and budget state
        self.adasparse_v3_bandwidth_info_last = None
        self.adasparse_v3_uplink_budget_last = None
        self.adasparse_v3_downlink_budget_last = None
        
        # Flag to track if we've initialized layer keys
        self.adasparse_v3_layer_keys_initialized = False

        # Mandatory startup log
        if bool(getattr(self._cfg, 'debug', False)):
            logger.debug(
                f"Client {self.ID} startup: "
                f"method=adasparse_lorav3, v3_enabled={self.adasparse_v3_enabled}, "
                f"v2_disabled=True, config_subtree={'glue' if is_glue_task else 'llm'}.adapter.adasparse_lorav3"
            )
        logger.info(
            f"Client {self.ID}: enabled, "
            f"init_rank={self.adasparse_v3_init_rank}, "
            f"rank_bounds=[{self.adasparse_v3_rank_min}, {self.adasparse_v3_rank_max}], "
            f"stage1_gamma={self.adasparse_v3_gamma}, stage1_reg_weight={self.adasparse_v3_reg_weight},"
            f"stage2_enabled={self.adasparse_v3_stage2_enabled}, "
            f"global_competition(stage1/stage2)={self.adasparse_v3_stage1_global_competition}/{self.adasparse_v3_stage2_global_competition}"
        )
    
    def _init_disabled_v3_state(self):
        """Initialize empty state for disabled v3."""
        self.adasparse_v3_survivors_by_layer = {}
        self.adasparse_v3_survivor_components = []
        self.adasparse_v3_upload_components_last = None
        self.adasparse_v3_download_components_last = None
        self.adasparse_v3_low_candidates_before = None
        self.adasparse_v3_scores_before_low = {}
        self.adasparse_v3_survivors_before_stage1 = None
        self.adasparse_v3_pre_round_lora_snapshot = None
        self.adasparse_v3_residual_buffers = {}
        self.adasparse_v3_bandwidth_info_last = None
        self.adasparse_v3_uplink_budget_last = None
        self.adasparse_v3_downlink_budget_last = None
        self.adasparse_v3_layer_keys_initialized = False
        # Stage 1 importance state (disabled defaults)
        self.adasparse_v3_stage1_norm_eps = 1e-12
        self.adasparse_v3_stage1_importance_strength = 1.0
        self.adasparse_v3_stage1_importance_beta = 0.2
        self.adasparse_v3_stage1_importance_min = 0.25
        self.adasparse_v3_stage1_importance_max = 1.0
        self.adasparse_v3_layer_importance = {}
        self.adasparse_v3_layer_medians_last = {}
        self.adasparse_v3_adjusted_scores_before_low = {}

    def _init_layer_keys_from_model(self):
        """
        Initialize layer keys and survivors from the model.
        Called lazily when model is first available.
        
        V3 properly initializes per-layer survivor state using the full rank config,
        NOT collapsing to one scalar. Each layer gets its own initial rank based on
        pattern matching against the rank config.
        """
        if self.adasparse_v3_layer_keys_initialized:
            return
        
        if not hasattr(self, 'trainer') or not hasattr(self.trainer, 'ctx'):
            return
        
        if not hasattr(self.trainer.ctx, 'model'):
            return
        
        try:
            state_dict = self.trainer.ctx.model.state_dict()
            layer_keys = infer_layer_keys_from_state_dict(state_dict)
            
            if not layer_keys:
                logger.warning(f"Client {self.ID}: No LoRA layers found in model")
                return
            
            # Get the full per-layer rank config for this client
            client_rank_config = self._get_client_rank_config_v3()
            default_rank = self.adasparse_v3_init_rank
            
            # Initialize survivors per layer using exact layer-aware config
            rank_summary = {}  # For logging
            for layer_key in layer_keys:
                # Determine rank for this specific layer using pattern matching
                layer_rank = self._get_rank_for_layer_v3(layer_key, client_rank_config, default_rank)
                self.adasparse_v3_survivors_by_layer[layer_key] = list(range(layer_rank))
                
                # Initialize layer importance to 1.0 for new layers (do not reset existing)
                if layer_key not in self.adasparse_v3_layer_importance:
                    self.adasparse_v3_layer_importance[layer_key] = 1.0
                
                # Track for logging
                rank_summary[layer_rank] = rank_summary.get(layer_rank, 0) + 1
            
            # Build flattened ComponentID list
            self.adasparse_v3_survivor_components = flatten_grouped_indices_by_layer(
                self.adasparse_v3_survivors_by_layer
            )
            
            self.adasparse_v3_layer_keys_initialized = True
            
            # Log initialization with per-layer detail
            if bool(getattr(self._cfg, 'debug', False)):
                rank_dist_str = ", ".join(f"r{r}:{c}" for r, c in sorted(rank_summary.items()))
                logger.debug(
                    f"Client {self.ID}: Initialized v3 layer state with "
                    f"{len(layer_keys)} layers, rank_distribution=[{rank_dist_str}], "
                    f"total {len(self.adasparse_v3_survivor_components)} ComponentIDs"
                )
                
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to initialize layer keys: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _get_client_rank_config_v3(self) -> Optional[Dict[str, int]]:
        """
        Get the full per-layer rank config for this client.
        
        Returns the rank config dict mapping patterns to ranks, or None if not available.
        This preserves exact per-layer rank information instead of collapsing to a scalar.
        """
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        if not config_local:
            return None
        
        # Try both 1-indexed and 0-indexed client keys
        client_key_1indexed = f'Client_{self.ID}'
        client_key_0indexed = f'Client_{self.ID - 1}'
        
        if client_key_1indexed in config_local and config_local[client_key_1indexed]:
            return dict(config_local[client_key_1indexed])
        elif client_key_0indexed in config_local and config_local[client_key_0indexed]:
            return dict(config_local[client_key_0indexed])
        
        return None
    
    def _get_rank_for_layer_v3(
        self, 
        layer_key: str, 
        rank_config: Optional[Dict[str, int]], 
        default_rank: int
    ) -> int:
        """
        Determine the initial rank for a specific layer using pattern matching.
        
        This mirrors the pattern-matching logic used in heterolora_utils._get_rank_for_param
        to ensure consistency with the existing rank config format.
        
        Args:
            layer_key: Canonical layer key (e.g., 'model.layers.0.self_attn.q_proj')
            rank_config: Dict mapping patterns to ranks (e.g., {'q_proj': 8, 'v_proj': 16})
            default_rank: Default rank if no pattern matches
            
        Returns:
            The rank for this layer
        """
        if rank_config is None:
            return default_rank
        
        # Use longest matching pattern (same logic as _get_rank_for_param)
        best_len = -1
        best_rank = default_rank
        
        for pattern, rank in rank_config.items():
            if pattern in layer_key and len(pattern) > best_len:
                best_len = len(pattern)
                try:
                    best_rank = int(rank)
                except (TypeError, ValueError):
                    pass
        
        return best_rank

    def _adasparse_v3_log_round_start(self):
        """Log v3 state at the start of each round."""
        if not self.adasparse_v3_enabled:
            return
        
        self._init_layer_keys_from_model()
        
        n_layers = len(self.adasparse_v3_survivors_by_layer)
        n_components = len(self.adasparse_v3_survivor_components)
        n_upload = len(self.adasparse_v3_upload_components_last) if self.adasparse_v3_upload_components_last else 0
        n_download = len(self.adasparse_v3_download_components_last) if self.adasparse_v3_download_components_last else 0
        
        # Validate V3 integration state in debug mode
        if bool(getattr(self._cfg, 'debug', False)):
            from federatedscope.contrib.common.adasparse_lorav3_utils import validate_v3_integration_state
            validate_v3_integration_state(
                self.adasparse_v3_survivors_by_layer,
                self.adasparse_v3_survivor_components,
                self.adasparse_v3_upload_components_last,
                self.adasparse_v3_download_components_last,
                context=f"Client {self.ID} round start",
                raise_on_error=False
            )
        
        logger.info(
            f"Client {self.ID} round start: "
            f"n_layers={n_layers}, n_survivor_components={n_components}, "
            f"prior_upload_count={n_upload}, prior_download_count={n_download}"
        )

    def _adasparse_v3_save_pre_round_lora_snapshot(self):
        """Save a snapshot of current LoRA tensors before local training."""
        if not self.adasparse_v3_enabled:
            return
    
        try:
            model_state_dict = self.trainer.ctx.model.state_dict()
        
            # Extract, clone, and move to CPU for storage (reduces GPU memory pressure)
            lora_snapshot = {}
            sample_device = None
            sample_dtype = None
            for key, tensor in model_state_dict.items():
                if isinstance(tensor, torch.Tensor) and ('lora_A' in key or 'lora_B' in key):
                    lora_snapshot[key] = tensor.clone().cpu()
                    if sample_device is None:
                        sample_device = tensor.device
                        sample_dtype = tensor.dtype
        
            self.adasparse_v3_pre_round_lora_snapshot = lora_snapshot
        
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID}: Saved v3 pre-round LoRA snapshot "
                    f"with {len(lora_snapshot)} tensors on CPU "
                    f"(source model was {sample_device}, {sample_dtype})"
                )
        
        except Exception as e:
            logger.error(f"Client {self.ID}: Failed to save pre-round LoRA snapshot: {e}")
            self.adasparse_v3_pre_round_lora_snapshot = None

    def _adasparse_v3_record_lowset_before(self):
        """
        Record low-set ComponentIDs and per-candidate scores before training.
        
        V3: Scores are computed per ComponentID = (layer_key, global_idx),
        not collapsed across layers. Low-set selection uses adjusted scores
        (median-normalized + layer-importance prior).
        """
        if not self.adasparse_v3_enabled:
            return
        
        self._init_layer_keys_from_model()
    
        try:
            model = self.trainer.ctx.model
            
            # Compute raw per-ComponentID scores
            raw_scores = compute_component_scores_grouped(
                model, self.adasparse_v3_survivors_by_layer
            )
            
            if not raw_scores:
                self.adasparse_v3_low_candidates_before = []
                self.adasparse_v3_scores_before_low = {}
                self.adasparse_v3_adjusted_scores_before_low = {}
                self.adasparse_v3_layer_medians_last = {}
                self.adasparse_v3_survivors_before_stage1 = {
                    k: list(v) for k, v in self.adasparse_v3_survivors_by_layer.items()
                }
                return
            
            # Compute adjusted scores with layer normalization + importance prior
            adjusted_scores, normalized_scores, layer_medians = compute_stage1_adjusted_scores_grouped(
                raw_scores,
                layer_importance=self.adasparse_v3_layer_importance,
                eps=self.adasparse_v3_stage1_norm_eps,
                importance_strength=self.adasparse_v3_stage1_importance_strength,
            )
            
            # Store layer medians for debugging
            self.adasparse_v3_layer_medians_last = dict(layer_medians)
            
            # Compute low-set from ADJUSTED scores
            low_candidates, low_score_sum = compute_lowset_grouped(
                adjusted_scores,
                self.adasparse_v3_gamma,
                self.adasparse_v3_rank_min,
                global_competition=self.adasparse_v3_stage1_global_competition
            )
            
            # Store per-candidate baseline scores from RAW scores for prune comparison
            self.adasparse_v3_low_candidates_before = low_candidates
            self.adasparse_v3_scores_before_low = {
                cid: raw_scores[cid] for cid in low_candidates if cid in raw_scores
            }
            
            # Store adjusted scores for debugging
            self.adasparse_v3_adjusted_scores_before_low = {
                cid: adjusted_scores[cid] for cid in low_candidates if cid in adjusted_scores
            }
            
            self.adasparse_v3_survivors_before_stage1 = {
                k: list(v) for k, v in self.adasparse_v3_survivors_by_layer.items()
            }
            
            # Log low-set info with both raw and adjusted scores
            if low_candidates:
                raw_vals = [raw_scores[cid] for cid in low_candidates if cid in raw_scores]
                adj_vals = [adjusted_scores[cid] for cid in low_candidates if cid in adjusted_scores]
                
                if raw_vals and adj_vals:
                    raw_min, raw_avg, raw_max = min(raw_vals), sum(raw_vals) / len(raw_vals), max(raw_vals)
                    adj_min, adj_avg, adj_max = min(adj_vals), sum(adj_vals) / len(adj_vals), max(adj_vals)
                    
                    if bool(getattr(self._cfg, 'debug', False)):
                        sorted_layers = sorted(self.adasparse_v3_layer_importance.keys())
                        sample_layers = sorted_layers[:5]

                        importance_sample = ", ".join(
                            f"{layer.split('.')[-1]}:{self.adasparse_v3_layer_importance.get(layer, 1.0):.3f}"
                            for layer in sample_layers
                        ) if sample_layers else "none"

                        median_sample = ", ".join(
                            f"{layer.split('.')[-1]}:{layer_medians.get(layer, 0.0):.6f}"
                            for layer in sample_layers
                        ) if sample_layers else "none"

                        median_values = list(layer_medians.values())
                        if median_values:
                            med_min = min(median_values)
                            med_avg = sum(median_values) / len(median_values)
                            med_max = max(median_values)
                            median_summary = f"{med_min:.6f}/{med_avg:.6f}/{med_max:.6f}"
                        else:
                            median_summary = "0.000000/0.000000/0.000000"

                        logger.debug(
                            f"Client {self.ID}: Stage1 v3 low_candidates={len(low_candidates)}, "
                            f"raw(min/avg/max)={raw_min:.4f}/{raw_avg:.4f}/{raw_max:.4f}, "
                            f"adjusted(min/avg/max)={adj_min:.4f}/{adj_avg:.4f}/{adj_max:.4f}, "
                            f"layer_medians(min/avg/max)={median_summary}, "
                            f"importance_sample=[{importance_sample}], "
                            f"median_sample=[{median_sample}]"
                        )
            
            # Set model attributes for trainer regularizer
            setattr(model, "adasparse_v3_low_candidates", low_candidates)
            setattr(model, "adasparse_v3_survivors_by_layer", self.adasparse_v3_survivors_by_layer)
        
        except Exception as e:
            logger.warning(f"Client {self.ID}: Failed to compute Stage1 v3 low-set: {e}")
            self.adasparse_v3_low_candidates_before = None
            self.adasparse_v3_scores_before_low = {}
            self.adasparse_v3_adjusted_scores_before_low = {}
            self.adasparse_v3_layer_medians_last = {}
            self.adasparse_v3_survivors_before_stage1 = None

    def _update_adasparse_v3_layer_importance_from_pruning(
        self,
        low_candidates: List[ComponentID],
        prune_candidates: List[ComponentID]
    ):
        """
        Update layer importance based on per-layer prune rate using EMA.
        
        If a layer produces a higher prune rate, its importance decreases.
        Lower importance means more prune pressure in future rounds.
        
        Args:
            low_candidates: List of low-set ComponentIDs before pruning
            prune_candidates: List of ComponentIDs that were actually pruned
        """
        low_by_layer = group_component_ids_by_layer(low_candidates)
        pruned_by_layer = group_component_ids_by_layer(prune_candidates)
        
        beta = self.adasparse_v3_stage1_importance_beta
        imp_min = self.adasparse_v3_stage1_importance_min
        imp_max = self.adasparse_v3_stage1_importance_max
        
        updated_layers = []
        
        for layer_key in self.adasparse_v3_survivors_by_layer.keys():
            prev_importance = self.adasparse_v3_layer_importance.get(layer_key, 1.0)
            low_count = len(low_by_layer.get(layer_key, []))
            pruned_count = len(pruned_by_layer.get(layer_key, []))
            
            if low_count <= 0:
                new_importance = prev_importance
            else:
                prune_rate = pruned_count / max(1, low_count)
                target_importance = 1.0 - prune_rate
                new_importance = (1 - beta) * prev_importance + beta * target_importance
                new_importance = max(imp_min, min(imp_max, new_importance))
            
            self.adasparse_v3_layer_importance[layer_key] = new_importance
            
            if prev_importance != new_importance:
                updated_layers.append((layer_key, low_count, pruned_count, prev_importance, new_importance))
        
        # Debug log: sample a few layers
        if bool(getattr(self._cfg, 'debug', False)) and updated_layers:
            sample = updated_layers[:3]
            for layer_key, lc, pc, prev, new in sample:
                pr = pc / max(1, lc) if lc > 0 else 0.0
                logger.debug(
                    f"Client {self.ID}: layer_importance '{layer_key}': "
                    f"low={lc}, pruned={pc}, rate={pr:.2f}, "
                    f"importance: {prev:.3f} -> {new:.3f}"
                )

    def _adasparse_v3_stage1_prune(self):
        """
        Perform Stage 1 structural pruning for v3 after training.
        
        Compares per-ComponentID scores before/after training.
        Prunes only components whose score decreased.
        """
        if not self.adasparse_v3_enabled:
            return self.adasparse_v3_survivors_by_layer
        
        if (self.adasparse_v3_low_candidates_before is None or 
            not hasattr(self, 'adasparse_v3_scores_before_low') or
            self.adasparse_v3_survivors_before_stage1 is None):
            logger.info(f"Client {self.ID}: Stage1 v3 no pruning (missing pre-training state)")
            return self.adasparse_v3_survivors_by_layer
    
        try:
            model = self.trainer.ctx.model
            low_candidates = self.adasparse_v3_low_candidates_before
            m = len(low_candidates)
            
            if m == 0:
                logger.info(
                    f"Client {self.ID}: Stage1 v3 no pruning (m=0)"
                )
                return self.adasparse_v3_survivors_by_layer
            
            # Compute scores after training
            scores_after = compute_component_scores_grouped(
                model, self.adasparse_v3_survivors_by_layer
            )
            
            if not scores_after:
                return self.adasparse_v3_survivors_by_layer
            
            # Per-candidate comparison: find candidates whose score decreased
            prune_candidates = []
            
            for cid in low_candidates:
                if cid not in scores_after:
                    continue
                before = self.adasparse_v3_scores_before_low.get(cid, 0.0)
                after = scores_after[cid]
                
                if after < before:
                    prune_candidates.append(cid)
            
            n_decreased = len(prune_candidates)
            
            # Enforce SHARED/GLOBAL rank_min constraint.
            # NOTE: V3 intentionally uses a SHARED/GLOBAL rank_min across all layers,
            # NOT per-layer minima. This is a deliberate simplification for current
            # research purposes. The total survivors across all layers must stay >= rank_min.
            # Future work could introduce layer-wise minimum policies, but this
            # implementation intentionally keeps the simpler shared/global behavior.
            #
            # This applies in BOTH layer-wise and global-competition modes:
            # - stage1_global_competition affects selection mode (layer-wise vs global)
            # - but the minimum floor policy is always shared/global
            total_survivors = sum(len(v) for v in self.adasparse_v3_survivors_by_layer.values())
            max_prunable = max(0, total_survivors - self.adasparse_v3_rank_min)
            if len(prune_candidates) > max_prunable:
                prune_candidates = prune_candidates[:max_prunable]
            
            n_pruned = len(prune_candidates)
            
            if n_pruned == 0:
                # Still update importance even with no actual pruning
                self._update_adasparse_v3_layer_importance_from_pruning(
                    low_candidates,
                    prune_candidates
                )
                
                # Compute importance summary for logging
                imp_values = list(self.adasparse_v3_layer_importance.values())
                if imp_values:
                    imp_min, imp_avg, imp_max = min(imp_values), sum(imp_values) / len(imp_values), max(imp_values)
                    imp_summary = f", importance(min/avg/max)={imp_min:.2f}/{imp_avg:.2f}/{imp_max:.2f}"
                else:
                    imp_summary = ""
                
                logger.info(
                    f"Client {self.ID}: Stage1 v3 candidates={m}, "
                    f"decreased={n_decreased}, pruned=0{imp_summary}"
                )
                return self.adasparse_v3_survivors_by_layer
            
            # Perform pruning: remove pruned ComponentIDs from survivors
            pruned_set = set(prune_candidates)
            
            new_survivors_by_layer = {}
            for layer_key, indices in self.adasparse_v3_survivors_by_layer.items():
                new_indices = [
                    idx for idx in indices
                    if (layer_key, idx) not in pruned_set
                ]
                new_survivors_by_layer[layer_key] = sorted(new_indices)
            
            # Update state
            old_count = len(self.adasparse_v3_survivor_components)
            self.adasparse_v3_survivors_by_layer = new_survivors_by_layer
            self.adasparse_v3_survivor_components = flatten_grouped_indices_by_layer(new_survivors_by_layer)
            new_count = len(self.adasparse_v3_survivor_components)
            
            # Clean up residual buffers for pruned components.
            # Mirror v2 behavior: log each removed
            # residual buffer in debug mode for the exact pruned ComponentID.

            # Replaced the approach below with the one further done to avoid too much clutter
            # if bool(getattr(self._cfg, 'debug', False)):
            #     if hasattr(self, 'adasparse_v3_residual_buffers') and self.adasparse_v3_residual_buffers:
            #         for cid in prune_candidates:
            #             if cid in self.adasparse_v3_residual_buffers:
            #                 logger.debug(
            #                     f"Client {self.ID}: Removed residual buffer for pruned component {cid}"
            #                 )

            if bool(getattr(self._cfg, 'debug', False)):
                residuals = getattr(self, 'adasparse_v3_residual_buffers', {}) or {}
                removed_count = len(set(prune_candidates) & set(residuals.keys()))
                logger.debug(
                    f"Client {self.ID}: Removed residual buffers for pruned component {removed_count}/{len(residuals)} "
                )

            self.adasparse_v3_residual_buffers = prune_residual_buffers_grouped(
                self.adasparse_v3_residual_buffers, prune_candidates
            )
            
            # Update layer importance based on prune rates
            self._update_adasparse_v3_layer_importance_from_pruning(
                low_candidates,
                prune_candidates
            )
            
            # Compute importance summary for logging
            imp_values = list(self.adasparse_v3_layer_importance.values())
            if imp_values:
                imp_min, imp_avg, imp_max = min(imp_values), sum(imp_values) / len(imp_values), max(imp_values)
                imp_summary = f", importance(min/avg/max)={imp_min:.2f}/{imp_avg:.2f}/{imp_max:.2f}"
            else:
                imp_summary = ""
            
            logger.info(
                f"Client {self.ID}: Stage1 v3 candidates={m}, "
                f"decreased={n_decreased}, pruned={n_pruned}, "
                f"total_components: {old_count} -> {new_count}{imp_summary}"
            )
            
            return self.adasparse_v3_survivors_by_layer
        
        except Exception as e:
            logger.warning(f"Client {self.ID}: Stage1 v3 pruning failed: {e}")
            import traceback
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(traceback.format_exc())
            return self.adasparse_v3_survivors_by_layer

    def _sync_adasparse_v3_state_from_message(self,
                                              survivor_indices=None,
                                              download_indices=None,
                                              bandwidth_info=None,
                                              client_rank_config=None):
        """
        Keep AdaSparse-LoRAv3 state aligned with server payloads.
        
        Uses unified V3 normalization for backward compatibility:
        - Grouped format: survivor_indices = {layer_key: [indices]}
        - Legacy flat format: survivor_indices = [flat_list] - expanded to all layers
        """
        if not self.adasparse_v3_enabled:
            return
        
        self._init_layer_keys_from_model()
        
        # Get layer keys for flat->grouped expansion
        layer_keys = list(self.adasparse_v3_survivors_by_layer.keys())
        
        # Update structural survivor set using unified normalization
        if survivor_indices is not None:
            # Use unified V3 normalization (expands flat to all layers)
            normalized_survivors = normalize_indices_to_grouped(
                survivor_indices, layer_keys=layer_keys
            )
            
            if normalized_survivors:
                self.adasparse_v3_survivors_by_layer = {
                    k: list(v) for k, v in normalized_survivors.items()
                }
                self.adasparse_v3_survivor_components = flatten_grouped_indices_by_layer(
                    self.adasparse_v3_survivors_by_layer
                )
                
                logger.info(
                    f"Client {self.ID} received v3 survivor_indices: "
                    f"n_layers={len(self.adasparse_v3_survivors_by_layer)}, "
                    f"count={len(self.adasparse_v3_survivor_components)}"
                )
        
        # Update download indices using unified normalization
        if download_indices is not None:
            # Use unified V3 normalization (expands flat to all layers)
            normalized_download = normalize_indices_to_grouped(
                download_indices, layer_keys=layer_keys
            )
            
            if normalized_download:
                self.adasparse_v3_download_components_last = flatten_grouped_indices_by_layer(
                    normalized_download
                )
            
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID} received v3 download_indices: "
                    f"count={len(self.adasparse_v3_download_components_last)}"
                )
        
        # Update bandwidth info for Stage 2 budgets
        if bandwidth_info is not None:
            self.adasparse_v3_bandwidth_info_last = bandwidth_info
            ul_kbits = bandwidth_info.get('upload_kbits', 0)
            dl_kbits = bandwidth_info.get('download_kbits', 0)
            
            if 'uplink_budget_bits' in bandwidth_info:
                self.adasparse_v3_uplink_budget_last = bandwidth_info['uplink_budget_bits']
            else:
                self.adasparse_v3_uplink_budget_last = ul_kbits * self.adasparse_v3_uplink_window * 1000
            
            if 'downlink_budget_bits' in bandwidth_info:
                self.adasparse_v3_downlink_budget_last = bandwidth_info['downlink_budget_bits']
            else:
                self.adasparse_v3_downlink_budget_last = dl_kbits * self.adasparse_v3_downlink_window * 1000
        
        # Fall back to client_rank_config if survivor_indices not provided
        # V3: Use per-layer pattern matching, not scalar collapse
        if survivor_indices is None and client_rank_config is not None:
            try:
                for layer_key in list(self.adasparse_v3_survivors_by_layer.keys()):
                    # Use pattern matching to get rank for this specific layer
                    layer_rank = self._get_rank_for_layer_v3(
                        layer_key, 
                        client_rank_config, 
                        self.adasparse_v3_init_rank
                    )
                    self.adasparse_v3_survivors_by_layer[layer_key] = list(range(layer_rank))
                self.adasparse_v3_survivor_components = flatten_grouped_indices_by_layer(
                    self.adasparse_v3_survivors_by_layer
                )
            except Exception:
                pass

    def _adasparse_v3_stage2_select_upload(self):
        """
        Perform Stage 2 upload selection for v3.
        
        Returns:
            Tuple of (selected_components, upload_dict, upload_indices_grouped)
        """
        if not self.adasparse_v3_enabled:
            return None, None, None

        if self.adasparse_v3_pre_round_lora_snapshot is None:
            logger.warning(f"Client {self.ID}: v3 Stage 2 skipped (no snapshot)")
            return None, None, None

        # Clean "stage-2 OFF": when stage2.enabled=False, transmit ALL stage-1 survivors
        # through the normal aggregation path -- no residual accounting, no bandwidth
        # budget, no component selection. Without this branch the method early-returns
        # (None, None, None) and the shared upload path raises RuntimeError ("Failed to
        # build native v3 upload payload"), leaving stage-2-off unusable. This is the
        # supported accuracy-only configuration; note that stage-2 ON currently emits a
        # sparse/partial downlink that the generic download canonicalizer cannot scatter
        # (see docs/federation_bug.md). Byte-identical when stage2.enabled=True.
        if not self.adasparse_v3_stage2_enabled:
            try:
                model_state = self.trainer.ctx.model.state_dict()
                delta_dict = compute_model_update_from_snapshot_grouped(
                    model_state,
                    self.adasparse_v3_pre_round_lora_snapshot,
                    self.adasparse_v3_survivors_by_layer,
                    cfg=self._cfg
                )
                # ALL survivors selected; upload_dict is the full survivor-set delta.
                selected_components = list(self.adasparse_v3_survivor_components)
                upload_dict = slice_model_update_by_component_ids(
                    delta_dict,
                    self.adasparse_v3_survivors_by_layer,
                    selected_components
                )
                upload_indices_grouped = group_component_ids_by_layer(selected_components)
                self.adasparse_v3_upload_components_last = selected_components
                # No residual buffers are touched (nothing is withheld => residual == 0).
                logger.info(
                    f"Client {self.ID}: v3 Stage 2 OFF -> uploading all "
                    f"{len(selected_components)} survivors (no budget/residual)"
                )
                return selected_components, upload_dict, upload_indices_grouped
            except Exception as e:
                logger.error(
                    f"Client {self.ID}: v3 stage-2-off upload failed: {e}")
                return None, None, None
        
        try:
            model_state = self.trainer.ctx.model.state_dict()
            
            # Compute model update from snapshot
            delta_dict = compute_model_update_from_snapshot_grouped(
                model_state,
                self.adasparse_v3_pre_round_lora_snapshot,
                self.adasparse_v3_survivors_by_layer,
                cfg=self._cfg
            )
            
            # Apply residuals to get effective update
            if self.adasparse_v3_residual_enabled and self.adasparse_v3_residual_buffers:
                effective_update = apply_residual_to_update_grouped(
                    delta_dict,
                    self.adasparse_v3_residual_buffers,
                    self.adasparse_v3_survivors_by_layer,
                    cfg=self._cfg
                )
            else:
                effective_update = delta_dict
            
            # Compute upload scores and costs
            upload_scores = compute_stage2_upload_scores_grouped(
                effective_update,
                self.adasparse_v3_survivors_by_layer
            )
            
            upload_costs = compute_component_upload_cost_grouped(
                effective_update,
                self.adasparse_v3_survivors_by_layer,
                q_bits=self.adasparse_v3_q_up_bits,
                cmeta_bits=self.adasparse_v3_cmeta_bits
            )
            
            # Select components under budget
            budget = self.adasparse_v3_uplink_budget_last if self.adasparse_v3_uplink_budget_last else float('inf')
            
            selected_components = greedy_select_by_score_cost_ratio_grouped(
                upload_scores,
                upload_costs,
                budget,
                global_competition=self.adasparse_v3_stage2_global_competition
            )
            
            if not selected_components:
                # Select all if budget is infinite or selection fails
                selected_components = self.adasparse_v3_survivor_components
            
            # Slice updates to selected components
            upload_dict = slice_model_update_by_component_ids(
                effective_update,
                self.adasparse_v3_survivors_by_layer,
                selected_components
            )
            
            # Update residual buffers
            self.adasparse_v3_residual_buffers = update_residual_buffers_after_upload_grouped(
                self.adasparse_v3_residual_buffers,
                effective_update,
                selected_components,
                self.adasparse_v3_survivors_by_layer,
                cfg=self._cfg
            )

            # Log residual state summary in the same style as v2.
            residual_summary = compute_residual_norm_summary_grouped(
                self.adasparse_v3_residual_buffers
            )
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(
                    f"Client {self.ID}: residual state after upload: "
                    f"count={residual_summary.get('count', 0)}, "
                    f"total={residual_summary.get('total', 0.0):.6f}, "
                    f"avg={residual_summary.get('avg', 0.0):.6f}, "
                    f"max={residual_summary.get('max', 0.0):.6f}"
                )

            # Store for logging
            self.adasparse_v3_upload_components_last = selected_components

            # Convert to grouped format for wire
            upload_indices_grouped = group_component_ids_by_layer(selected_components)
            
            used_budget = sum(upload_costs.get(cid, 0) for cid in selected_components)
            budget_ratio = used_budget / budget if budget > 0 and budget != float('inf') else 0.0
            
            logger.info(
                f"Client {self.ID}: v3 Stage 2 upload: "
                f"survivors={len(self.adasparse_v3_survivor_components)}, "
                f"selected={len(selected_components)}, "
                f"budget_ratio={budget_ratio*100:.1f}%"
            )
            
            return selected_components, upload_dict, upload_indices_grouped
        
        except Exception as e:
            logger.error(f"Client {self.ID}: v3 Stage 2 upload selection failed: {e}")
            import traceback
            if bool(getattr(self._cfg, 'debug', False)):
                logger.debug(traceback.format_exc())
            return None, None, None
