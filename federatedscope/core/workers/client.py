import copy
import gc
import logging
import sys
import pickle
import time
import random

import torch

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    calculate_time_cost, add_prefix_to_path, get_ds_rank
from federatedscope.core.workers.base_client import BaseClient
import federatedscope.contrib.common as fs_common

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) #logger.setLevel(logging.DEBUG)


class Client(BaseClient):
    """
    The Client class, which describes the behaviors of client in an FL \
    course. The behaviors are described by the handling functions (named as \
    ``callback_funcs_for_xxx``)

    Arguments:
        ID: The unique ID of the client, which is assigned by the server
        when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: The configuration
        data: The data owned by the client
        model: The model maintained locally
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        ss_manager: secret sharing manager
        msg_buffer: dict buffer for storing message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
    """

    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(Client, self).__init__(ID, state, config, model, strategy)

        self.data = data

        # Register message handlers
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        # the unseen_client indicates that whether this client contributes to
        # FL process by training on its local data and uploading the local
        # model update, which is useful for check the participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.is_unseen_client = is_unseen_client

        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int):
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attack_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        # Attack only support the stand alone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)
        self.device = device
        self.debug_mode = bool(getattr(self._cfg, 'debug', False))

        # For client-side evaluation
        self.best_results = dict()
        self.history_results = dict()
        # in local or global training mode, we do use the early stopper.
        # Otherwise, we set patience=0 to deactivate the local early-stopper
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in [
                "local", "global"
            ] else 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # Communication and communication ability
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None:
            self.comp_speed = float(
                kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(
                kwargs['resource_info']['communication'])  # (kbit/s)
        else:
            self.comp_speed = None
            self.comm_bandwidth = None

        if self._cfg.backend == 'torch':
            try:
                self.model_size = sys.getsizeof(pickle.dumps(
                    self.model)) / 1024.0 * 8.  # kbits
            except Exception as error:
                self.model_size = 1.0
                logger.warning(f'{error} in calculate model size.')
        else:
            # TODO: calculate model size for TF Model
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:'
                           f'{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id
        # Support injected comm_manager for distributed ZMQ deployment
        if 'comm_manager' in kwargs and kwargs['comm_manager'] is not None:
            self.comm_manager = kwargs['comm_manager']
            self.local_address = kwargs.get('local_address', None)
            logger.info('Client: Using injected comm_manager')
        elif self.mode == 'standalone':
            comm_queue = kwargs['shared_comm_queue']
            if self._cfg.federate.process_num <= 1:
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            else:
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
            self.local_address = None
        elif self.mode == 'distributed':
            host = kwargs['host']
            port = kwargs['port']
            server_host = kwargs['server_host']
            server_port = kwargs['server_port']
            self.comm_manager = gRPCCommManager(
                host=host,
                port=port,
                client_num=self._cfg.federate.client_num,
                cfg=self._cfg.distribute)
            logger.info('Client: Listen to {}:{}...'.format(host, port))
            self.comm_manager.add_neighbors(neighbor_id=server_id,
                                            address={
                                                'host': server_host,
                                                'port': server_port
                                            })
            self.local_address = {
                'host': self.comm_manager.host,
                'port': self.comm_manager.port
            }
        
        # FAH-QLoRA initialization
        self._init_fah_qloRA()
        
        # Extended metrics tracking initialization
        self._init_extended_metrics_tracking()
        
        # Shared bandwidth tracking initialization
        self._init_bandwidth_tracking()
        
        # HetLoRA initialization (rank self-pruning)
        self._init_hetlora()
        
        # AdaSparse-LoRA initialization
        self._init_adasparse_lora()
        
        # AdaSparse-LoRAv2 initialization
        self._init_adasparse_lorav2()
        
        # AdaSparse-LoRAv3 initialization
        self._init_adasparse_lorav3()

    def _init_adasparse_lora(self):
        """
        Generic AdaSparse-LoRA hook implementation.

        Real AdaSparse-LoRA client behavior lives in
        federatedscope.contrib.worker.methods.adasparse_lora_client.
        The shared client keeps only a disabled implementation so that
        non-AdaSparse-LoRA methods do not carry AdaSparse-LoRA-specific
        rank/index initialization and pruning logic.
        """
        self.adasparse_enabled = False
        self.adasparse_indices_current = None
        self.adasparse_low_positions_before = None
        self.adasparse_scores_before_low = {}
        self.adasparse_indices_before = None
        self.adasparse_score_before = None
        self.adasparse_pruning_enabled = False
        self.adasparse_rank_min = None
        self.adasparse_rank_max = None
        self.adasparse_init_rank = None
        self.adasparse_gamma = None
        self.adasparse_reg_weight = None

    def _adasparse_record_lowset_before(self):
        """Generic AdaSparse-LoRA hook."""
        return

    def _adasparse_prune_and_prepare_upload(self, model_para: dict):
        """Generic AdaSparse-LoRA hook."""
        return model_para, self.adasparse_indices_current

    def _init_adasparse_lorav2(self):
        """
        Generic AdaSparse-LoRAv2 hook implementation.

        Real AdaSparse-LoRAv2 client behavior lives in
        federatedscope.contrib.worker.methods.adasparse_lorav2_client.
        The shared client keeps only a disabled implementation so that
        non-AdaSparse-LoRAv2 methods do not carry v2-specific structural
        pruning, sparse upload bookkeeping, or bandwidth-state initialization.
        """
        self.adasparse_v2_enabled = False
        self.adasparse_v2_survivor_indices_current = None
        self.adasparse_v2_upload_indices_last = None
        self.adasparse_v2_download_indices_last = None
        self.adasparse_v2_indices_before_stage1 = None
        self.adasparse_v2_low_positions_before = None
        self.adasparse_v2_scores_before_low = {}
        self.adasparse_v2_pre_round_lora_snapshot = None
        self.adasparse_v2_residual_buffers = None
        self.adasparse_v2_bandwidth_info_last = None
        self.adasparse_v2_uplink_budget_last = None
        self.adasparse_v2_downlink_budget_last = None
        self.adasparse_v2_rank_min = None
        self.adasparse_v2_rank_max = None
        self.adasparse_v2_init_rank = None
        self.adasparse_v2_gamma = None
        self.adasparse_v2_reg_weight = None
        self.adasparse_v2_stage2_enabled = False
        self.adasparse_v2_q_up_bits = None
        self.adasparse_v2_q_down_bits = None
        self.adasparse_v2_cmeta_bits = None
        self.adasparse_v2_uplink_window = None
        self.adasparse_v2_downlink_window = None
        self.adasparse_v2_selection_rule = None
        self.adasparse_v2_residual_enabled = False
        self.adasparse_v2_bandwidth_enabled = False
        self.adasparse_v2_bandwidth_mode = None

    def _adasparse_v2_log_round_start(self):
        """Generic AdaSparse-LoRAv2 hook."""
        return

    def _adasparse_v2_save_pre_round_lora_snapshot(self):
        """Generic AdaSparse-LoRAv2 hook."""
        return

    def _adasparse_v2_record_lowset_before(self):
        """Generic AdaSparse-LoRAv2 hook."""
        return

    def _adasparse_v2_stage1_prune(self):
        """Generic AdaSparse-LoRAv2 hook."""
        return self.adasparse_v2_survivor_indices_current

    def _init_adasparse_lorav3(self):
        """
        Generic AdaSparse-LoRAv3 hook implementation.

        Real AdaSparse-LoRAv3 client behavior lives in
        federatedscope.contrib.worker.methods.adasparse_lorav3_client.
        The shared client keeps only a disabled implementation so that
        non-AdaSparse-LoRAv3 methods do not carry v3-specific layer-aware
        ComponentID state, grouped wire metadata bookkeeping, or Stage 1/2
        sparse communication state.
        """
        self.adasparse_v3_enabled = False
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
        self.adasparse_v3_rank_min = None
        self.adasparse_v3_rank_max = None
        self.adasparse_v3_init_rank = None
        self.adasparse_v3_gamma = None
        self.adasparse_v3_reg_weight = None
        self.adasparse_v3_stage2_enabled = False
        self.adasparse_v3_q_up_bits = None
        self.adasparse_v3_q_down_bits = None
        self.adasparse_v3_cmeta_bits = None
        self.adasparse_v3_uplink_window = None
        self.adasparse_v3_downlink_window = None
        self.adasparse_v3_selection_rule = None
        self.adasparse_v3_residual_enabled = False
        self.adasparse_v3_stage1_global_competition = False
        self.adasparse_v3_stage2_global_competition = False
        self.adasparse_v3_layer_keys_initialized = False

    def _adasparse_v3_log_round_start(self):
        """Generic AdaSparse-LoRAv3 hook."""
        return

    def _adasparse_v3_save_pre_round_lora_snapshot(self):
        """Generic AdaSparse-LoRAv3 hook."""
        return

    def _adasparse_v3_record_lowset_before(self):
        """Generic AdaSparse-LoRAv3 hook."""
        return

    def _adasparse_v3_stage1_prune(self):
        """Generic AdaSparse-LoRAv3 hook."""
        return self.adasparse_v3_survivors_by_layer

    def _sync_adasparse_v3_state_from_message(self,
                                               survivor_indices=None,
                                               download_indices=None,
                                               bandwidth_info=None,
                                               client_rank_config=None):
        """Generic AdaSparse-LoRAv3 hook."""
        return

    def _init_hetlora(self):
        """
        Generic HetLoRA hook implementation.

        Real HetLoRA behavior lives in
        federatedscope.contrib.worker.methods.hetlora_client.HetLoRAClient.
        The shared client keeps only a disabled implementation so that
        non-HetLoRA methods do not carry HetLoRA-specific state logic.
        """
        self.hetlora_enabled = False
        self.hetlora_current_rank = None
        self.hetlora_tail_score_before = None
        self.hetlora_pruning_enabled = False
        self.hetlora_rank_min = None
        self.hetlora_rank_max = None
        self.hetlora_init_rank = None
        self.hetlora_decay = None
        self.hetlora_reg_weight = None
        self._hetlora_last_rank_config = None

    def _hetlora_record_tail_score_before(self):
        """Generic HetLoRA hook."""
        return

    def _hetlora_prune_and_send_rank(self, model_para: dict, sender, round_idx, timestamp):
        """Generic HetLoRA hook."""
        return model_para

    def _expects_client_specific_hetero_payload(self):
        """Whether this client should refuse plain max-rank heterogeneous payloads."""
        method = fs_common.normalize_method_name(getattr(self._cfg.federate, 'method', ''))
        config_local = fs_common.get_active_hetero_config_local(self._cfg)
        return config_local is not None and method in ['hetlora', 'adasparse_lora']

    def _apply_client_specific_heterolora_payload(self,
                                                content,
                                                client_rank_config_from_msg=None,
                                                context='train'):
        """Apply a distributed-format hetero-LoRA download to the client model.

        Canonicalizes the aggregated global adapter (distributed keys "base.rank") to the
        model's state_dict keys so ``trainer.update`` actually loads it. This is the base
        implementation for every client that does NOT override it (HetLoRA, AdaSparse
        v1/v2/v3): previously it was a no-op stub, so those methods silently dropped the
        global adapter and never federated. Non-distributed / non-LoRA payloads pass
        through unchanged; HeteroLoRA/FAH subclasses override this for rank-resize.
        """
        if context == 'finish':
            return content
        try:
            from federatedscope.contrib.common.heterolora_utils import \
                apply_distributed_lora_download
            strict = bool(getattr(self._cfg.federate,
                                  'assert_download_consumed', True))
            out, n_consumed, n_model = apply_distributed_lora_download(
                content, self.trainer.ctx.model, strict=strict,
                debug=bool(getattr(self._cfg, 'debug', False)),
                is_partial=bool(getattr(self, '_download_is_partial', False)))
            # Stash the LoRA subset actually applied, for the optional
            # tensor-equality-at-round-start assertion after trainer.update.
            self._pending_download_lora = {
                k: v for k, v in out.items()
                if 'lora_A' in k or 'lora_B' in k
            }
            self._last_download_stats = (n_consumed, n_model)
            return out
        except ImportError:
            return content

    def _maybe_assert_download_applied(self):
        """Round-start tensor-equality check: after loading the aggregated global adapter,
        the client model's LoRA params must equal the applied download. Off by default
        (per-round cost); enable with federate.assert_download_tensor_equality=True."""
        import torch
        pending = getattr(self, '_pending_download_lora', None)
        self._pending_download_lora = None
        if not pending:
            return
        if not bool(getattr(self._cfg.federate,
                            'assert_download_tensor_equality', False)):
            return
        sd = self.trainer.ctx.model.state_dict()
        for k, v in pending.items():
            if k not in sd or not isinstance(sd[k], torch.Tensor):
                raise AssertionError(
                    f"[federation] applied download key {k} absent from model after load")
            if not torch.allclose(sd[k].detach().float().cpu(),
                                  torch.as_tensor(v).float().cpu(), atol=1e-4):
                raise AssertionError(
                    f"[federation] model LoRA {k} does not match the applied global "
                    f"adapter after round-start load (tensor-equality assertion failed)")

    def _extract_head_params_for_upload(self):
        """Absolute trainable non-LoRA (task-head: classifier + pooler) params for the
        synchronized head-federation path (AdaS recovery). Uploaded SEPARATELY from the
        LoRA delta so the server can sample-size average + REPLACE the head. Empty dict on
        any failure (head federation then simply no-ops -- caught by the 2-round test)."""
        try:
            from federatedscope.contrib.common.head_federation import head_keys_from_model
            model = self.trainer.ctx.model
            sd = model.state_dict()
            return {k: sd[k].detach().cpu().clone()
                    for k in head_keys_from_model(model) if k in sd}
        except Exception as e:
            logger.warning(f"Client {getattr(self,'ID','?')}: head extract failed: {e}")
            return {}

    def _sync_adasparse_state_from_message(self,
                                           client_rank_config=None,
                                           adasparse_indices=None):
        """Generic AdaSparse-LoRA hook."""
        return

    def _sync_adasparse_v2_state_from_message(self,
                                               survivor_indices=None,
                                               download_indices=None,
                                               bandwidth_info=None,
                                               client_rank_config=None):
        """Generic AdaSparse-LoRAv2 hook."""
        return

    def _init_extended_metrics_tracking(self):
        """Initialize generic extended metrics tracking for all methods."""
        self.extended_metrics_enabled = (
            hasattr(self._cfg, 'monitor') and
            hasattr(self._cfg.monitor, 'system_metrics_mode') and
            self._cfg.monitor.system_metrics_mode == 'extended'
        )
        self.ext_bandwidth_per_round = {}
        self.ext_cuda_baseline = 0
        self.ext_cuda_baseline_allocated = 0
        self.ext_cuda_baseline_reserved = 0
        self._ext_pending_download_bytes = {}
        if self.extended_metrics_enabled:
            logger.info(f"[ExtMetrics] Client {self.ID}: Extended metrics tracking enabled")

    def _compute_extended_payload_bytes(self, payload):
        """Compute payload size in bytes for tensors stored in dict/list payloads."""
        if payload is None:
            return 0
        if isinstance(payload, dict):
            return self._monitor.compute_model_bytes(payload)
        if isinstance(payload, (list, tuple)):
            return sum(self._compute_extended_payload_bytes(item) for item in payload)
        return 0

    def _track_extended_download_metrics(self, round_id, content, is_warmup=False):
        """Track generic per-round download bytes and bandwidth for all methods.

        The download size is always computed from the actual payload received by
        the client. For heterogeneous methods such as HetLoRA and FAH-QLoRA,
        the server already sends client-specific truncated LoRA tensors, so using
        a shared max-rank LoRA size would overestimate downloads and incorrectly
        make all clients appear identical.
        """
        if not self.extended_metrics_enabled:
            return

        ul_bw, dl_bw = self._get_round_bandwidth(round_id)
        self._monitor.track_round_bandwidth(round_id, ul_bw, dl_bw)

        download_bytes = self._compute_extended_payload_bytes(content)
        self._ext_pending_download_bytes[round_id] = download_bytes

    def _track_extended_compute_metrics(self, round_id, compute_seconds, model, optimizer=None):
        """Track generic per-round compute time and memory for all methods."""
        if not self.extended_metrics_enabled:
            return
        self._monitor.track_round_compute_time(round_id, compute_seconds)
        device = self.device if hasattr(self, 'device') else None
        cuda_baseline_allocated = getattr(getattr(self.trainer, 'ctx', None), 'ext_cuda_baseline_allocated', None)
        cuda_baseline_reserved = getattr(getattr(self.trainer, 'ctx', None), 'ext_cuda_baseline_reserved', None)
        if cuda_baseline_allocated is None:
            cuda_baseline_allocated = getattr(self, 'ext_cuda_baseline_allocated', getattr(self, 'ext_cuda_baseline', 0))
        if cuda_baseline_reserved is None:
            cuda_baseline_reserved = getattr(self, 'ext_cuda_baseline_reserved', 0)
        self._monitor.track_round_memory(
            round_id,
            model,
            optimizer,
            device,
            cuda_baseline_allocated=cuda_baseline_allocated,
            cuda_baseline_reserved=cuda_baseline_reserved,
        )
        if self._monitor.sys_trained_params == 0:
            self._monitor.track_trained_parameters(model)

    def _track_extended_upload_metrics(self, round_id, shared_model_para, is_warmup=False):
        """Track generic per-round upload bytes and end-to-end communication timing."""
        if not self.extended_metrics_enabled:
            return
        upload_bytes = self._compute_extended_payload_bytes(shared_model_para)
        download_bytes = self._ext_pending_download_bytes.pop(round_id, 0)

        # Extract OS-level measured bandwidth from comm_manager if available
        measured_ul_kbps = 0.0
        measured_dl_kbps = 0.0
        upload_wall_seconds = 0.0
        download_wall_seconds = 0.0
        bm = getattr(getattr(self, 'comm_manager', None), 'bandwidth_measure', None)
        if bm is not None:
            metrics = bm.get_round_metrics()
            if metrics is not None:
                measured_ul_kbps = metrics.measured_ul_kbps
                measured_dl_kbps = metrics.measured_dl_kbps
                upload_wall_seconds = metrics.upload_wall_seconds
                download_wall_seconds = metrics.download_wall_seconds

        self._monitor.track_round_communication(
            round_id,
            upload_bytes,
            download_bytes,
            is_warmup_round=False,
            measured_ul_kbps=measured_ul_kbps,
            measured_dl_kbps=measured_dl_kbps,
            upload_wall_seconds=upload_wall_seconds,
            download_wall_seconds=download_wall_seconds,
        )

    def _get_measured_bandwidth_for_feedback(self):
        """Extract most recent NIC-measured bandwidth for feedback to server."""
        bm = getattr(getattr(self, 'comm_manager', None), 'bandwidth_measure', None)
        if bm is None:
            return None
        for m in reversed(bm.get_all_metrics()):
            if m.measured_ul_kbps > 0 or m.measured_dl_kbps > 0:
                return {
                    'round': m.round_idx,
                    'measured_ul_kbps': m.measured_ul_kbps,
                    'measured_dl_kbps': m.measured_dl_kbps,
                }
        return None

    def _get_round_bandwidth(self, round_id):
        """Get round bandwidth in kbit/s, preferring synced server-provided values."""
        if hasattr(self, 'ext_bandwidth_per_round') and round_id in self.ext_bandwidth_per_round:
            return self.ext_bandwidth_per_round[round_id]
        if round_id in getattr(self, 'bandwidth_history_by_round', {}):
            info = self.bandwidth_history_by_round[round_id]
            return (
                info.get('upload_kbits', 5000.0),
                info.get('download_kbits', 50000.0),
            )
        if self.bandwidth_info_last is not None:
            return self._get_current_bandwidth_rates()

        comm_cfg = getattr(getattr(self._cfg, 'federate', None), 'communication', None)
        if comm_cfg is not None:
            return (
                float(getattr(comm_cfg, 'uplink_min_mbps', 5.0)) * 1000.0,
                float(getattr(comm_cfg, 'downlink_mbps', 50.0)) * 1000.0,
            )
        return (5000.0, 50000.0)

    def _init_bandwidth_tracking(self):
        """
        Initialize shared bandwidth tracking state.
        
        Sets up per-round bandwidth history tracking for all methods.
        Called during client initialization.
        """
        # Last received bandwidth_info from server
        self.bandwidth_info_last = None
        
        # Per-round bandwidth history: {round_idx: bandwidth_info}
        self.bandwidth_history_by_round = {}
        
        logger.info(f"Client {self.ID}: Bandwidth tracking initialized")

    def _sync_bandwidth_info_from_message(self, bandwidth_info, round_idx=None):
        """
        Store bandwidth_info received from server.
        
        This method:
        - Stores the latest bandwidth_info
        - Records per-round bandwidth history
        - Can be extended by subclasses to pass values to method-specific logic
        
        Args:
            bandwidth_info: Bandwidth info dict from server message
            round_idx: Optional round index (extracted from bandwidth_info if not provided)
        """
        if bandwidth_info is None:
            return
        
        self.bandwidth_info_last = bandwidth_info
        
        # Extract round index
        if round_idx is None:
            round_idx = bandwidth_info.get('round', -1)
        
        # Store in history
        self.bandwidth_history_by_round[round_idx] = bandwidth_info
        
        # Log receipt
        ul_kbits = bandwidth_info.get('upload_kbits', 0)
        dl_kbits = bandwidth_info.get('download_kbits', 0)
        mode = bandwidth_info.get('mode', 'unknown')
        source = bandwidth_info.get('source', 'unknown')
        
        logger.debug(
            f"Client {self.ID} round {round_idx}: "
            f"bandwidth_info synced - "
            f"UL={ul_kbits/1000:.2f}Mbps, DL={dl_kbits/1000:.2f}Mbps, "
            f"mode={mode}, source={source}"
        )

    def _get_current_bandwidth_rates(self):
        """
        Get current upload/download bandwidth rates in kbit/s.
        
        Returns:
            Tuple of (upload_kbits, download_kbits)
        """
        if self.bandwidth_info_last is None:
            # Fallback to defaults
            return (5000.0, 50000.0)  # 5 Mbps UL, 50 Mbps DL
        
        return (
            self.bandwidth_info_last.get('upload_kbits', 5000.0),
            self.bandwidth_info_last.get('download_kbits', 50000.0),
        )

    def _extract_bandwidth_info_from_content(self, content):
        """
        Extract bandwidth_info from message content if present.
        
        Args:
            content: Message content (could be dict or raw model_para)
        
        Returns:
            bandwidth_info dict if present, None otherwise
        """
        if isinstance(content, dict) and 'bandwidth_info' in content:
            return content.get('bandwidth_info')
        return None

    def _prepare_cuda_memory_tracking(self, device=None):
        """Default no-op CUDA memory tracking hook."""
        return

    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def _calculate_model_delta(self, init_model, updated_model):
        if not isinstance(init_model, list):
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            model_delta = copy.deepcopy(init_model[model_index])
            for key in init_model[model_index].keys():
                model_delta[key] = updated_model[model_index][
                    key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        else:
            return model_deltas[0]

    def _fah_resolve_compute_dtype(self):
        """Default compute dtype used by non-FAH clients."""
        return torch.float32

    def _fah_cast_trainable_params(self, dtype: torch.dtype, model=None, log_prefix: str = "[FAH]"):
        """Default no-op dtype recast hook for non-FAH clients."""
        return

    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self):
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def run_standalone(self):
        """
        Run in standalone mode
        """
        self.join_in()
        self.run()

    def callback_funcs_for_model_para(self, message: Message):
            """
            The handling function for receiving model parameters, \
            which triggers the local training process. \
            This handling function is widely used in various FL courses.

            Arguments:
                message: The received message
            """
            if 'ss' in message.msg_type:
                # A fragment of the shared secret
                state, content, timestamp = message.state, message.content, \
                                            message.timestamp
                self.msg_buffer['train'][state].append(content)

                if len(self.msg_buffer['train']
                    [state]) == self._cfg.federate.client_num:
                    # Check whether the received fragments are enough
                    model_list = self.msg_buffer['train'][state]
                    sample_size, first_aggregate_model_para = model_list[0]
                    single_model_case = True
                    if isinstance(first_aggregate_model_para, list):
                        assert isinstance(first_aggregate_model_para[0], dict), \
                            "aggregate_model_para should a list of multiple " \
                            "state_dict for multiple models"
                        single_model_case = False
                    else:
                        assert isinstance(first_aggregate_model_para, dict), \
                            "aggregate_model_para should " \
                            "a state_dict for single model case"
                        first_aggregate_model_para = [first_aggregate_model_para]
                        model_list = [[model] for model in model_list]

                    for sub_model_idx, aggregate_single_model_para in enumerate(
                            first_aggregate_model_para):
                        for key in aggregate_single_model_para:
                            for i in range(1, len(model_list)):
                                aggregate_single_model_para[key] += model_list[i][
                                    sub_model_idx][key]

                    self.comm_manager.send(
                        Message(msg_type='model_para',
                                sender=self.ID,
                                receiver=[self.server_id],
                                state=self.state,
                                timestamp=timestamp,
                                content=(sample_size, first_aggregate_model_para[0]
                                        if single_model_case else
                                        first_aggregate_model_para)))

            else:
                round = message.state
                sender = message.sender
                timestamp = message.timestamp
                content = message.content

                # Signal bandwidth measurement module about new round
                bm = getattr(getattr(self, 'comm_manager', None), 'bandwidth_measure', None)
                if bm is not None:
                    bm.begin_round(round)

                # Handle FAH-wrapped or AdaSparse-LoRA wrapped content
                # (contains model_para, fah_ranks, client_rank_config, bandwidth_info, adasparse_indices)
                # AdaSparse-LoRAv2 also includes: survivor_indices, download_indices
                fah_ranks = None
                client_rank_config_from_msg = None
                bandwidth_info_from_msg = None
                adasparse_indices_from_msg = None
                # V2-specific fields
                survivor_indices_from_msg = None
                download_indices_from_msg = None
                is_partial_downlink_from_msg = False
                
                if isinstance(content, dict) and ('fah_ranks' in content or 'client_rank_config' in content or 
                                                   'bandwidth_info' in content or
                                                   'adasparse_indices' in content or 'survivor_indices' in content or
                                                   'download_indices' in content):
                    fah_ranks = content.get('fah_ranks', None)
                    client_rank_config_from_msg = content.get('client_rank_config', None)
                    bandwidth_info_from_msg = content.get('bandwidth_info', None)
                    adasparse_indices_from_msg = content.get('adasparse_indices', None)
                    # V2-specific fields
                    survivor_indices_from_msg = content.get('survivor_indices', None)
                    download_indices_from_msg = content.get('download_indices', None)
                    # V2 partial downlink flag (Option A masked refresh semantics)
                    is_partial_downlink_from_msg = content.get('is_partial_downlink', False)
                    content = fs_common.extract_model_para(content, default=content)
                    
                    # [AdaSparse-LoRAv2] Update three-state bookkeeping from server
                    if self.adasparse_v2_enabled and survivor_indices_from_msg is not None:
                        self._sync_adasparse_v2_state_from_message(
                            survivor_indices=survivor_indices_from_msg,
                            download_indices=download_indices_from_msg,
                            bandwidth_info=bandwidth_info_from_msg,
                            client_rank_config=client_rank_config_from_msg,
                        )

                    # [AdaSparse-LoRAv3] Update grouped layer-aware state from server
                    # during train. Native v3 wire format uses grouped metadata
                    # plus exact-layer client_rank_config.
                    if self.adasparse_v3_enabled and (
                        survivor_indices_from_msg is not None or
                        download_indices_from_msg is not None or
                        bandwidth_info_from_msg is not None or
                        client_rank_config_from_msg is not None
                    ):
                        self._sync_adasparse_v3_state_from_message(
                            survivor_indices=survivor_indices_from_msg,
                            download_indices=download_indices_from_msg,
                            bandwidth_info=bandwidth_info_from_msg,
                            client_rank_config=client_rank_config_from_msg,
                        )
                    
                    # [AdaSparse-LoRA] Update indices/rank bookkeeping from server
                    if self.adasparse_enabled:
                        self._sync_adasparse_state_from_message(
                            client_rank_config=client_rank_config_from_msg,
                            adasparse_indices=adasparse_indices_from_msg,
                        )
                        if adasparse_indices_from_msg is not None:
                            if self.debug_mode:
                                logger.info(
                                    f"[AdaSparse-LoRA] Client {self.ID} received indices: "
                                    f"{self.adasparse_indices_current[:5]}..., rank={len(self.adasparse_indices_current)}"
                                )
                    
                    # Update FAH rank pairs
                    if fah_ranks is not None:
                        self.fah_current_rank, self.fah_current_hat_rank = fah_ranks
                        if self.debug_mode:
                            logger.debug(
                                f"[FAH] Client {self.ID} received ranks: "
                                f"r={self.fah_current_rank}, r̂={self.fah_current_hat_rank}"
                            )
                    
                    if client_rank_config_from_msg is not None:
                        if self.debug_mode:
                            logger.debug(
                                f"Client {self.ID} received client_rank_config from server"
                            )
                    
                    # [Shared] Sync bandwidth_info through shared helper
                    # This must run even when FAH extended metrics are disabled
                    self._sync_bandwidth_info_from_message(bandwidth_info_from_msg, round)
                    
                    # [ExtMetrics] Also store in legacy ext_bandwidth_per_round for backward compat
                    if bandwidth_info_from_msg is not None:
                        ul_kbits = bandwidth_info_from_msg.get('upload_kbits', 0)
                        dl_kbits = bandwidth_info_from_msg.get('download_kbits', 0)
                        self.ext_bandwidth_per_round[round] = (ul_kbits, dl_kbits)
                        if self.debug_mode:
                            logger.debug(
                                f"[ExtMetrics] Client {self.ID} received bandwidth for round {round}: "
                                f"ul={ul_kbits:.1f}kbit/s, dl={dl_kbits:.1f}kbit/s"
                            )
                    

                # dequantization
                if self._cfg.quantization.method == 'uniform':
                    from federatedscope.core.compression import \
                        symmetric_uniform_dequantization
                    if isinstance(content, list):  # multiple model
                        content = [
                            symmetric_uniform_dequantization(x) for x in content
                        ]
                    else:
                        content = symmetric_uniform_dequantization(content)

                # [ExtMetrics] Track download metrics from the actual received payload
                # before any local post-processing such as heterogeneous loading or
                # sparse masked-refresh application.
                self._track_extended_download_metrics(round, content, is_warmup=False)

                # Handle heterogeneous LoRA loading if needed
                # Check if we have distributed weights (keys with ".rank" suffix)
                has_distributed_format = any(
                    '.' in key and key.split('.')[-1].isdigit() and
                    ('lora_A' in key or 'lora_B' in key)
                    for key in content.keys()
                )
                
                hetero_expected = self._expects_client_specific_hetero_payload()
                method_name = getattr(self._cfg.federate, 'method', '').lower()

                if hetero_expected:
                    if method_name == 'hetlora' and not has_distributed_format:
                        raise RuntimeError(
                            f"[Client {self.ID}] Expected distributed heterogeneous LoRA payload "
                            f"at round {round}, but received plain model_para instead."
                        )
                    if method_name in ['adasparse_lora', 'adasparse-lora'] and adasparse_indices_from_msg is None:
                        raise RuntimeError(
                            f"[Client {self.ID}] Expected AdaSparse client-specific payload "
                            f"at round {round}, but received model_para without adasparse_indices."
                        )

                # [AdaSparse-LoRAv2] Handle partial downlink (Option A masked refresh)
                # This must come BEFORE the normal hetero_lora path to bypass rank-resize
                is_v2_partial_downlink = (
                    self.adasparse_v2_enabled and
                    is_partial_downlink_from_msg and
                    download_indices_from_msg is not None and
                    survivor_indices_from_msg is not None
                )
                
                if is_v2_partial_downlink:
                    # Option A partial downlink: apply sparse masked refresh to existing survivor-sized model
                    # Do NOT resize model to payload subset rank - keep model at full survivor rank
                    from federatedscope.contrib.common.adasparse_lora_utils import apply_sparse_update_to_model
                    from federatedscope.contrib.common.heterolora_utils import load_weight_local
                    
                    # Extract LoRA tensors from distributed-format payload at download_indices rank
                    download_rank = len(download_indices_from_msg)
                    target_modules = getattr(
                        getattr(getattr(self._cfg, 'llm', None), 'adapter', None),
                        'target_modules', None
                    ) or getattr(
                        getattr(getattr(self._cfg, 'glue', None), 'adapter', None),
                        'target_modules', []
                    )
                    download_rank_config = {m: download_rank for m in target_modules}
                    
                    # Load sparse payload at download_indices rank (NOT survivor rank)
                    sparse_lora_dict = load_weight_local(
                        weighted_single_weights=content,
                        model=self.trainer.ctx.model,
                        client_rank_config=download_rank_config,
                        debug=False,
                    )
                    
                    # Get current model state dict (at survivor rank)
                    model_state_dict = self.trainer.ctx.model.state_dict()
                    model_lora_only = {
                        k: v for k, v in model_state_dict.items()
                        if 'lora_A' in k or 'lora_B' in k
                    }
                    
                    # Apply sparse masked refresh: overwrite only download_indices positions
                    refreshed_lora = apply_sparse_update_to_model(
                        model_state_dict=model_lora_only,
                        sparse_update_dict=sparse_lora_dict,
                        download_indices=download_indices_from_msg,
                        survivor_indices=survivor_indices_from_msg,
                    )
                    
                    # Combine refreshed LoRA with non-LoRA parameters
                    non_lora_content = {
                        k: v for k, v in model_state_dict.items()
                        if 'lora_A' not in k and 'lora_B' not in k
                    }
                    content = {**non_lora_content, **refreshed_lora}
                    
                    logger.info(
                        f"[AdaSparse-LoRAv2] Client {self.ID}: Applied partial downlink "
                        f"(masked refresh) for {len(download_indices_from_msg)} of "
                        f"{len(survivor_indices_from_msg)} survivors"
                    )
                
                
                # Flag partial (Stage-2 sparse) downlink so the generic canonicalizer refuses
                # to silently mis-place a subset (needs index-based scatter; see the util).
                self._download_is_partial = bool(is_partial_downlink_from_msg)
                content = self._apply_client_specific_heterolora_payload(
                    content,
                    client_rank_config_from_msg=client_rank_config_from_msg,
                    context='train',
                )

                # When clients share the local model, we must set strict=True to
                # ensure all model params are overwritten and synchronized with the received model
                if self._cfg.federate.process_num > 1:
                    for k, v in content.items():
                        content[k] = v.to(self.device)

                self.trainer.update(content,
                                    strict=self._cfg.federate.share_local_model)
                # Federation guard: verify the aggregated global adapter was actually
                # loaded into the client model this round (off by default; see the flag).
                self._maybe_assert_download_applied()
                # [FAH] After loading parameters from the server, LoRA tensors may be fp32
                # (communication dtype) even when this client computes in fp16/bf16.
                # Recast trainable tensors back to the intended compute dtype to prevent
                # mixed-dtype attention matmuls.
                try:
                    _cd = self._fah_resolve_compute_dtype()
                    self._fah_cast_trainable_params(_cd, log_prefix=f"[FAH] Client {self.ID}:")
                except Exception as _e:
                    logger.warning("[FAH] Client %s: failed to recast trainable params after update: %s", self.ID, _e)
                
                # [HetLoRA] Record tail importance score before training for pruning decision
                self._hetlora_record_tail_score_before()
                
                # [AdaSparse-LoRA] Record low-set positions and score before training
                self._adasparse_record_lowset_before()
                
                # [AdaSparse-LoRAv2] Log round start state, save snapshot, and record low-set for Stage 1
                self._adasparse_v2_log_round_start()
                self._adasparse_v2_save_pre_round_lora_snapshot()
                self._adasparse_v2_record_lowset_before()

                # [AdaSparse-LoRAv3] Log round start state, save snapshot, and
                # record low-set candidates for Stage 1. These hooks are the v3
                # counterpart of the existing v2 bookkeeping and must run before
                # local training so Stage 1/Stage 2 operate on correct pre-round state.
                self._adasparse_v3_log_round_start()
                self._adasparse_v3_save_pre_round_lora_snapshot()
                self._adasparse_v3_record_lowset_before()

                self.state = round
                
                skip_train_isolated_or_global_mode = \
                    self.early_stopper.early_stopped and \
                    self._cfg.federate.method in ["local", "global"]
                if self.is_unseen_client or skip_train_isolated_or_global_mode:
                    # for these cases (1) unseen client (2) isolated_global_mode,
                    # we do not local train and upload local model
                    sample_size, model_para_all, results = \
                        0, self.trainer.get_model_para(), {}
                    if skip_train_isolated_or_global_mode:
                        logger.info(
                            f"[Local/Global mode] Client #{self.ID} has been "
                            f"early stopped, we will skip the local training")
                        self._monitor.local_converged()
                else:
                    if self.early_stopper.early_stopped and \
                            self._monitor.local_convergence_round == 0:
                        logger.info(
                            f"[Normal FL Mode] Client #{self.ID} has been locally "
                            f"early stopped. "
                            f"The next FL update may result in negative effect")
                        self._monitor.local_converged()
                    
                    # [ExtMetrics] Prepare CUDA memory tracking before training
                    self._prepare_cuda_memory_tracking(self.device)
                    
                    # Track training time for FAH profiling
                    train_start_time = time.time()
                    sample_size, model_para_all, results = self.trainer.train()
                    train_end_time = time.time()
                    compute_seconds = train_end_time - train_start_time
                    
                    # [ExtMetrics] Track compute time and memory after training
                    optimizer = getattr(self.trainer.ctx, 'optimizer', None) if hasattr(self.trainer, 'ctx') else None
                    self._track_extended_compute_metrics(
                        round, compute_seconds, self.trainer.ctx.model, optimizer
                    )
                    
                    if self.fah_enabled:
                        self.fah_last_training_time = compute_seconds
                        # FAH-QLoRA: Perform evaluation at two ranks and send stats
                        self._fah_evaluate_and_send_stats(content, sender, round, timestamp)              
                    
                    if self._cfg.federate.share_local_model and not \
                            self._cfg.federate.online_aggr:
                        model_para_all = copy.deepcopy(model_para_all)
                    train_log_res = self._monitor.format_eval_res(
                        results,
                        rnd=self.state,
                        role='Client #{}'.format(self.ID),
                        return_raw=True)
                    logger.info(train_log_res)
                    if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                        self._monitor.save_formatted_results(train_log_res,
                                                            save_file_name="")

                # Return the feedbacks to the server after local update
                if self._cfg.federate.use_ss:
                    assert not self.is_unseen_client, \
                        "Un-support using secret sharing for unseen clients." \
                        "i.e., you set cfg.federate.use_ss=True and " \
                        "cfg.federate.unseen_clients_rate in (0, 1)"
                    single_model_case = True
                    if isinstance(model_para_all, list):
                        assert isinstance(model_para_all[0], dict), \
                            "model_para should a list of " \
                            "multiple state_dict for multiple models"
                        single_model_case = False
                    else:
                        assert isinstance(model_para_all, dict), \
                            "model_para should a state_dict for single model case"
                        model_para_all = [model_para_all]
                    model_para_list_all = []
                    for model_para in model_para_all:
                        for key in model_para:
                            model_para[key] = model_para[key] * sample_size
                        model_para_list = self.ss_manager.secret_split(model_para)
                        model_para_list_all.append(model_para_list)
                    frame_idx = 0
                    for neighbor in self.comm_manager.neighbors:
                        if neighbor != self.server_id:
                            content_frame = model_para_list_all[0][frame_idx] if \
                                single_model_case else \
                                [model_para_list[frame_idx] for model_para_list
                                in model_para_list_all]
                            self.comm_manager.send(
                                Message(msg_type='ss_model_para',
                                        sender=self.ID,
                                        receiver=[neighbor],
                                        state=self.state,
                                        timestamp=self._gen_timestamp(
                                            init_timestamp=timestamp,
                                            instance_number=sample_size),
                                        content=content_frame))
                            frame_idx += 1
                    content_frame = model_para_list_all[0][frame_idx] if \
                        single_model_case else \
                        [model_para_list[frame_idx] for model_para_list in
                        model_para_list_all]
                    self.msg_buffer['train'][self.state] = [(sample_size,
                                                            content_frame)]
                else:
                    if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                            ['krum', 'normbounding', 'median', 'trimmedmean',
                            'bulyan']:
                        # Return the model delta when using asynchronous training
                        # protocol, because the staled updated might be discounted
                        # and cause that the sum of the aggregated weights might
                        # not be equal to 1
                        shared_model_para = self._calculate_model_delta(
                            init_model=content, updated_model=model_para_all)
                    else:
                        shared_model_para = model_para_all

                    # [HetLoRA] Perform rank self-pruning if enabled
                    # This may truncate LoRA weights and send rank update to server
                    if self.hetlora_enabled and self.hetlora_pruning_enabled:
                        shared_model_para = self._hetlora_prune_and_send_rank(
                            shared_model_para, sender, round, timestamp
                        )
                    # [AdaSparse-LoRA] Perform component-based pruning if enabled
                    # This slices LoRA weights and updates indices
                    adasparse_indices_for_upload = None
                    if self.adasparse_enabled:
                        shared_model_para, adasparse_indices_for_upload = \
                            self._adasparse_prune_and_prepare_upload(shared_model_para)

                    # [AdaSparse-LoRAv2] Stage 1 pruning + Stage 2 upload preparation
                    adasparse_v2_upload_content = None
                    if self.adasparse_v2_enabled:
                        # Stage 1: Structural pruning (updates survivor_indices)
                        new_survivor_indices = self._adasparse_v2_stage1_prune()
                        
                        # Import Stage 2 utility functions
                        from federatedscope.contrib.common.adasparse_lora_utils import (
                            compute_model_update_from_snapshot,
                            apply_residual_to_update,
                            compute_stage2_upload_scores,
                            compute_component_upload_cost,
                            greedy_select_by_score_cost_ratio,
                            slice_model_update_by_indices,
                            update_residual_buffers_after_upload,
                            prune_residual_buffers,
                            validate_upload_subset,
                            compute_residual_norm_summary,
                        )
                        
                        # Stage 2: Compute fresh local model updates for survivors
                        # Compare post-training LoRA tensors against pre-round snapshot
                        
                        # Defensive check: ensure pre-round snapshot exists
                        if self.adasparse_v2_pre_round_lora_snapshot is None:
                            raise RuntimeError(
                                f"[AdaSparse-LoRAv2] Client {self.ID}: Pre-round LoRA snapshot "
                                f"missing before Stage 2 upload computation. This snapshot should "
                                f"have been created at the start of the round after receiving "
                                f"the server payload."
                            )
                        
                        fresh_model_update = compute_model_update_from_snapshot(
                            current_state_dict=self.trainer.ctx.model.state_dict(),
                            snapshot_state_dict=self.adasparse_v2_pre_round_lora_snapshot,
                            survivor_indices=new_survivor_indices,
                            cfg=self._cfg
                        )
                        
                        # Remove residual entries for components pruned by Stage 1
                        pre_prune_indices = self.adasparse_v2_indices_before_stage1 or []
                        pruned_indices = set(pre_prune_indices) - set(new_survivor_indices)
                        if pruned_indices and self.adasparse_v2_residual_buffers:
                            self.adasparse_v2_residual_buffers = prune_residual_buffers(
                                self.adasparse_v2_residual_buffers,
                                pruned_indices
                            )
                        
                        # Form residual-corrected effective updates
                        if self.adasparse_v2_residual_enabled and self.adasparse_v2_residual_buffers:
                            effective_update = apply_residual_to_update(
                                delta_dict=fresh_model_update,
                                residual_buffers=self.adasparse_v2_residual_buffers,
                                survivor_indices=new_survivor_indices,
                                cfg=self._cfg
                            )
                        else:
                            effective_update = fresh_model_update
                        
                        # Get uplink budget (in bits) from cached bandwidth info
                        uplink_budget = self.adasparse_v2_uplink_budget_last or float('inf')
                        
                        # Stage 2: Budgeted upload selection
                        if self.adasparse_v2_stage2_enabled and uplink_budget < float('inf'):
                            # Compute upload scores from effective updates
                            upload_scores = compute_stage2_upload_scores(
                                effective_update_dict=effective_update,
                                survivor_indices=new_survivor_indices,
                            )
                            
                            # Compute per-component upload costs
                            upload_costs = compute_component_upload_cost(
                                effective_update_dict=effective_update,
                                survivor_indices=new_survivor_indices,
                                q_bits=self.adasparse_v2_q_up_bits,
                                cmeta_bits=self.adasparse_v2_cmeta_bits,
                            )
                            
                            # Greedy selection under uplink budget
                            upload_indices = greedy_select_by_score_cost_ratio(
                                scores=upload_scores,
                                costs=upload_costs,
                                budget=uplink_budget,
                                survivor_indices=new_survivor_indices,
                            )
                            
                            # Validate upload subset
                            if not validate_upload_subset(upload_indices, new_survivor_indices):
                                logger.warning(
                                    f"[AdaSparse-LoRAv2] Upload indices not subset of survivors, filtering"
                                )
                                upload_indices = [i for i in upload_indices if i in set(new_survivor_indices)]
                            
                            # Log selection statistics
                            used_budget = sum(upload_costs.get(idx, 0) for idx in upload_indices)
                            budget_ratio = used_budget / uplink_budget if uplink_budget > 0 else 0.0
                            
                            logger.info(
                                # Emit budget_ratio + used/avail bits in a format parseable
                                # equivalently to the v3 stage-2 line, so one parser covers both.
                                # Logging only -- no effect on training/RNG/selection.
                                f"[AdaSparse-LoRAv2] Client {self.ID} Stage 2 upload selection: "
                                f"survivors={len(new_survivor_indices)}, selected={len(upload_indices)}, "
                                f"budget_ratio={budget_ratio*100:.1f}%, "
                                f"budget={uplink_budget:.0f}bits, used={used_budget:.0f}bits"
                            )
                        else:
                            # No budget constraint or Stage 2 disabled - upload all survivors
                            upload_indices = list(new_survivor_indices)
                            if self.debug_mode:
                                logger.debug(
                                    f"[AdaSparse-LoRAv2] Client {self.ID}: Full upload (no budget constraint), "
                                    f"n_components={len(upload_indices)}"
                                )
                        
                        # Build sparse model_update_dict only for selected upload_indices
                        sparse_model_update = slice_model_update_by_indices(
                            model_update_dict=effective_update,
                            survivor_indices=new_survivor_indices,
                            selected_indices=upload_indices,
                        )
                        
                        # Update residual buffers by subtracting what was actually sent
                        self.adasparse_v2_residual_buffers = update_residual_buffers_after_upload(
                            residual_buffers=self.adasparse_v2_residual_buffers,
                            effective_update_dict=effective_update,
                            upload_indices=upload_indices,
                            survivor_indices=new_survivor_indices,
                            cfg=self._cfg
                        )

                        # Log residual state summary using the keys actually returned by
                        # compute_residual_norm_summary(...)
                        residual_summary = compute_residual_norm_summary(self.adasparse_v2_residual_buffers)
                        if self.debug_mode:
                            logger.debug(
                                f"[AdaSparse-LoRAv2] Client {self.ID}: residual state after upload: "
                                f"count={residual_summary.get('count', 0)}, "
                                f"total={residual_summary.get('total', 0.0):.6f}, "
                                f"avg={residual_summary.get('avg', 0.0):.6f}, "
                                f"max={residual_summary.get('max', 0.0):.6f}"
                            )
                        
                        # Store upload indices for next round
                        self.adasparse_v2_upload_indices_last = upload_indices
                        
                        # Build v2 upload content (dict-based per Milestone 5)
                        adasparse_v2_upload_content = {
                            'sample_size': sample_size,
                            'model_update_dict': sparse_model_update,
                            'upload_indices': upload_indices,
                            'survivor_indices': new_survivor_indices,
                            'measured_bandwidth': self._get_measured_bandwidth_for_feedback(),
                            # Synchronized task-head federation (absolute, separate from the
                            # LoRA delta path): upload the trainable classifier+pooler.
                            'head_params': self._extract_head_params_for_upload(),
                        }
                        
                        logger.info(
                            f"[AdaSparse-LoRAv2] Client {self.ID}: Upload prepared - "
                            f"survivor_count={len(new_survivor_indices)}, "
                            f"upload_count={len(upload_indices)}, "
                            f"sample_size={sample_size}"
                        )


                    # [AdaSparse-LoRAv3] Stage 1 pruning + Stage 2 upload preparation
                    adasparse_v3_upload_content = None
                    if self.adasparse_v3_enabled:
                        # Stage 1: structural pruning updates grouped survivor state
                        new_survivors_by_layer = self._adasparse_v3_stage1_prune()

                        # Stage 2: native v3 sparse upload preparation
                        _, sparse_model_update_v3, upload_indices_grouped = \
                            self._adasparse_v3_stage2_select_upload()

                        if sparse_model_update_v3 is None or upload_indices_grouped is None:
                            raise RuntimeError(
                                f"[AdaSparse-LoRAv3] Client {self.ID}: Failed to build native v3 "
                                f"upload payload after local training. The shared client upload "
                                f"path must send dict-based v3 content with grouped upload_indices "
                                f"and survivor_indices."
                            )

                        adasparse_v3_upload_content = {
                            'sample_size': sample_size,
                            'model_update_dict': sparse_model_update_v3,
                            'upload_indices': upload_indices_grouped,
                            'survivor_indices': new_survivors_by_layer,
                            'measured_bandwidth': self._get_measured_bandwidth_for_feedback(),
                            # Synchronized task-head federation (absolute, separate from the
                            # LoRA delta path): upload the trainable classifier+pooler.
                            'head_params': self._extract_head_params_for_upload(),
                        }

                        logger.info(
                            f"[AdaSparse-LoRAv3] Client {self.ID}: Upload prepared - "
                            f"survivor_components={len(self.adasparse_v3_survivor_components)}, "
                            f"upload_components={len(self.adasparse_v3_upload_components_last or [])}, "
                            f"sample_size={sample_size}"
                        )

                    # quantization
                    if self._cfg.quantization.method == 'uniform':
                        from federatedscope.core.compression import \
                            symmetric_uniform_quantization
                        nbits = self._cfg.quantization.nbits
                        
                        # [AdaSparse-LoRAv2] Quantize the sparse v2 upload payload, not shared_model_para
                        if self.adasparse_v2_enabled and adasparse_v2_upload_content is not None:
                            # Quantize only the model_update_dict tensor values, not metadata
                            quantized_update_dict = {}
                            for k, v in adasparse_v2_upload_content['model_update_dict'].items():
                                if isinstance(v, torch.Tensor):
                                    quantized_update_dict[k] = symmetric_uniform_quantization(v, nbits)
                                else:
                                    quantized_update_dict[k] = v
                            adasparse_v2_upload_content['model_update_dict'] = quantized_update_dict
                            if self.debug_mode:
                                logger.debug(
                                    f"[AdaSparse-LoRAv2] Client {self.ID}: Quantized sparse v2 upload "
                                    f"to {nbits} bits"
                                )
                        elif self.adasparse_v3_enabled and adasparse_v3_upload_content is not None:
                            # Quantize only the model_update_dict tensor values, not metadata
                            quantized_update_dict = {}
                            for k, v in adasparse_v3_upload_content['model_update_dict'].items():
                                if isinstance(v, torch.Tensor):
                                    quantized_update_dict[k] = symmetric_uniform_quantization(v, nbits)
                                else:
                                    quantized_update_dict[k] = v
                            adasparse_v3_upload_content['model_update_dict'] = quantized_update_dict
                            if self.debug_mode:
                                logger.debug(
                                    f"[AdaSparse-LoRAv3] Client {self.ID}: Quantized sparse v3 upload "
                                    f"to {nbits} bits"
                                )
                        elif isinstance(shared_model_para, list):
                            shared_model_para = [
                                symmetric_uniform_quantization(x, nbits)
                                for x in shared_model_para
                            ]
                        else:
                            shared_model_para = symmetric_uniform_quantization(
                                shared_model_para, nbits)

                    # [ExtMetrics] Track upload metrics before sending
                    # Track method-native sparse payloads when present
                    if self.adasparse_v2_enabled and adasparse_v2_upload_content is not None:
                        self._track_extended_upload_metrics(
                            round, adasparse_v2_upload_content['model_update_dict'], is_warmup=False
                        )
                    elif self.adasparse_v3_enabled and adasparse_v3_upload_content is not None:
                        self._track_extended_upload_metrics(
                            round, adasparse_v3_upload_content['model_update_dict'], is_warmup=False
                        )
                    else:
                        self._track_extended_upload_metrics(round, shared_model_para, is_warmup=False)

                    # Send method-native structured payloads when present
                    if self.adasparse_v2_enabled and adasparse_v2_upload_content is not None:
                        upload_content = adasparse_v2_upload_content
                    elif self.adasparse_v3_enabled and adasparse_v3_upload_content is not None:
                        upload_content = adasparse_v3_upload_content
                    # [AdaSparse-LoRA] Send 3-tuple format: (sample_size, model_para, indices)
                    elif self.adasparse_enabled and adasparse_indices_for_upload is not None:
                        upload_content = (sample_size, shared_model_para, adasparse_indices_for_upload)
                    else:
                        upload_content = (sample_size, shared_model_para)

                    # Wrap tuple uploads with measured bandwidth for server feedback
                    if isinstance(upload_content, tuple):
                        measured_bw = self._get_measured_bandwidth_for_feedback()
                        if measured_bw is not None:
                            upload_content = {
                                '_tuple_content': upload_content,
                                'measured_bandwidth': measured_bw,
                            }

                    self.comm_manager.send(
                        Message(msg_type='model_para',
                                sender=self.ID,
                                receiver=[sender],
                                state=self.state,
                                timestamp=self._gen_timestamp(
                                    init_timestamp=timestamp,
                                    instance_number=sample_size),
                                content=upload_content))

                    del upload_content, shared_model_para, model_para_all
                    del content
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    def callback_funcs_for_assign_id(self, message: Message):
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.

        Arguments:
            message: The received message
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    def callback_funcs_for_join_in_info(self, message: Message):
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    num_sample = self._cfg.train.local_update_steps * \
                                    self._cfg.dataloader.batch_size
                else:
                    num_sample = self._cfg.train.local_update_steps * \
                                    len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = \
                        self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                assert self.comm_bandwidth is not None and self.comp_speed \
                        is not None, "The requirement join_in_info " \
                                    "'client_resource' does not exist."
                join_in_info['client_resource'] = self.model_size / \
                    self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    def callback_funcs_for_address(self, message: Message):
        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology

        Arguments:
            message: The received message
        """
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            content = message.content
            
            # Handle wrapped content (contains model_para and hetero metadata)
            client_rank_config_from_msg = None
            bandwidth_info_from_msg = None
            adasparse_indices_from_msg = None
            # V2-specific fields
            survivor_indices_from_msg = None
            download_indices_from_msg = None
            is_partial_downlink_from_msg = False
            
            if isinstance(content, dict) and ('fah_ranks' in content or 'client_rank_config' in content or
                                              'bandwidth_info' in content or
                                              'adasparse_indices' in content or 'survivor_indices' in content or
                                              'download_indices' in content):
                client_rank_config_from_msg = content.get('client_rank_config', None)
                bandwidth_info_from_msg = content.get('bandwidth_info', None)
                adasparse_indices_from_msg = content.get('adasparse_indices', None)
                # V2-specific fields
                survivor_indices_from_msg = content.get('survivor_indices', None)
                download_indices_from_msg = content.get('download_indices', None)
                is_partial_downlink_from_msg = content.get('is_partial_downlink', False)
                content = content.get('model_para', content)
                
                # [AdaSparse-LoRAv2] Update three-state bookkeeping from server during eval
                if self.adasparse_v2_enabled and survivor_indices_from_msg is not None:
                    self._sync_adasparse_v2_state_from_message(
                        survivor_indices=survivor_indices_from_msg,
                        download_indices=download_indices_from_msg,
                        bandwidth_info=bandwidth_info_from_msg,
                        client_rank_config=client_rank_config_from_msg,
                    )

                # [AdaSparse-LoRAv3] Update grouped layer-aware state from server during eval
                if self.adasparse_v3_enabled and (
                    survivor_indices_from_msg is not None or
                    download_indices_from_msg is not None or
                    bandwidth_info_from_msg is not None or
                    client_rank_config_from_msg is not None
                ):
                    self._sync_adasparse_v3_state_from_message(
                        survivor_indices=survivor_indices_from_msg,
                        download_indices=download_indices_from_msg,
                        bandwidth_info=bandwidth_info_from_msg,
                        client_rank_config=client_rank_config_from_msg,
                    )

                if self.adasparse_enabled:
                    self._sync_adasparse_state_from_message(
                        client_rank_config=client_rank_config_from_msg,
                        adasparse_indices=adasparse_indices_from_msg,
                    )
                
                # [Shared] Sync bandwidth_info through shared helper
                self._sync_bandwidth_info_from_message(bandwidth_info_from_msg, self.state)
            
            # Handle heterogeneous LoRA loading if needed (same as model_para)
            has_distributed_format = any(
                '.' in key and key.split('.')[-1].isdigit() and
                ('lora_A' in key or 'lora_B' in key)
                for key in content.keys()
            )
            
            hetero_expected = self._expects_client_specific_hetero_payload()
            method_name = getattr(self._cfg.federate, 'method', '').lower()

            if hetero_expected:
                if method_name == 'hetlora' and not has_distributed_format:
                    raise RuntimeError(
                        f"[Client {self.ID}] Expected distributed heterogeneous LoRA payload "
                        f"during evaluation at round {self.state}, but received plain model_para instead."
                    )
                if method_name in ['adasparse_lora', 'adasparse-lora'] and adasparse_indices_from_msg is None:
                    raise RuntimeError(
                        f"[Client {self.ID}] Expected AdaSparse client-specific evaluation payload "
                        f"at round {self.state}, but received model_para without adasparse_indices."
                    )

            # [AdaSparse-LoRAv2] Handle partial downlink (Option A masked refresh) during evaluation
            # This must come BEFORE the normal hetero_lora path to bypass rank-resize
            is_v2_partial_downlink = (
                self.adasparse_v2_enabled and
                is_partial_downlink_from_msg and
                download_indices_from_msg is not None and
                survivor_indices_from_msg is not None
            )
            
            if is_v2_partial_downlink:
                # Option A partial downlink: apply sparse masked refresh to existing survivor-sized model
                # Do NOT resize model to payload subset rank - keep model at full survivor rank
                from federatedscope.contrib.common.adasparse_lora_utils import apply_sparse_update_to_model
                from federatedscope.contrib.common.heterolora_utils import load_weight_local
                
                # Extract LoRA tensors from distributed-format payload at download_indices rank
                download_rank = len(download_indices_from_msg)
                target_modules = getattr(
                    getattr(getattr(self._cfg, 'llm', None), 'adapter', None),
                    'target_modules', None
                ) or getattr(
                    getattr(getattr(self._cfg, 'glue', None), 'adapter', None),
                    'target_modules', []
                )
                download_rank_config = {m: download_rank for m in target_modules}
                
                # Load sparse payload at download_indices rank (NOT survivor rank)
                sparse_lora_dict = load_weight_local(
                    weighted_single_weights=content,
                    model=self.trainer.ctx.model,
                    client_rank_config=download_rank_config,
                    debug=False,
                )
                
                # Get current model state dict (at survivor rank)
                model_state_dict = self.trainer.ctx.model.state_dict()
                model_lora_only = {
                    k: v for k, v in model_state_dict.items()
                    if 'lora_A' in k or 'lora_B' in k
                }
                
                # Apply sparse masked refresh: overwrite only download_indices positions
                refreshed_lora = apply_sparse_update_to_model(
                    model_state_dict=model_lora_only,
                    sparse_update_dict=sparse_lora_dict,
                    download_indices=download_indices_from_msg,
                    survivor_indices=survivor_indices_from_msg,
                )
                
                # Combine refreshed LoRA with non-LoRA parameters
                non_lora_content = {
                    k: v for k, v in model_state_dict.items()
                    if 'lora_A' not in k and 'lora_B' not in k
                }
                content = {**non_lora_content, **refreshed_lora}
                
                logger.info(
                    f"[AdaSparse-LoRAv2] Client {self.ID} eval: Applied partial downlink "
                    f"(masked refresh) for {len(download_indices_from_msg)} of "
                    f"{len(survivor_indices_from_msg)} survivors"
                )

            
            content = self._apply_client_specific_heterolora_payload(
                content,
                client_rank_config_from_msg=client_rank_config_from_msg,
                context='eval',
            )

            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model)
            # [FAH] After loading parameters from the server, LoRA tensors may be fp32
            # (communication dtype) even when this client computes in fp16/bf16.
            # Recast trainable tensors back to the intended compute dtype to prevent
            # mixed-dtype attention matmuls.
            try:
                _cd = self._fah_resolve_compute_dtype()
                self._fah_cast_trainable_params(_cd, log_prefix=f"[FAH] Client {self.ID}:")
            except Exception as _e:
                logger.warning("[FAH] Client %s: failed to recast trainable params after update: %s", self.ID, _e)
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global"
        ]:
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval:
                self.trainer.finetune()
            for split in self._cfg.eval.split:
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        self._monitor.format_eval_res(eval_metrics,
                                                        rnd=self.state,
                                                        role='Client #{}'.format(
                                                            self.ID),
                                                        return_raw=True))

                metrics.update(**eval_metrics)

            formatted_eval_res = self._monitor.format_eval_res(
                metrics,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                forms=['raw'],
                return_raw=True)
            logger.info(formatted_eval_res)
            update_best_this_round = self._monitor.update_best_result(
                self.best_results,
                formatted_eval_res['Results_raw'],
                results_type=f"client #{self.ID}",
            )

            if update_best_this_round and self._cfg.federate.save_client_model:
                path = add_prefix_to_path(f'client_{self.ID}_',
                                            self._cfg.federate.save_to)
                if self.ds_rank == 0:
                    self.trainer.save_model(path, self.state)

            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key])

        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message):
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            content = message.content
            
            # Handle wrapped content (contains model_para and hetero metadata)
            client_rank_config_from_msg = None
            bandwidth_info_from_msg = None
            adasparse_indices_from_msg = None
            if isinstance(content, dict) and ('fah_ranks' in content or 'client_rank_config' in content or
                                              'bandwidth_info' in content or
                                              'adasparse_indices' in content):
                client_rank_config_from_msg = content.get('client_rank_config', None)
                bandwidth_info_from_msg = content.get('bandwidth_info', None)
                adasparse_indices_from_msg = content.get('adasparse_indices', None)
                content = content.get('model_para', content)

                if self.adasparse_enabled:
                    self._sync_adasparse_state_from_message(
                        client_rank_config=client_rank_config_from_msg,
                        adasparse_indices=adasparse_indices_from_msg,
                    )
                
                # [Shared] Sync bandwidth_info through shared helper
                self._sync_bandwidth_info_from_message(bandwidth_info_from_msg, self.state)

            # Handle heterogeneous LoRA loading if needed during finish.
            # If the finish payload is a plain max-rank LoRA state_dict rather than
            # distributed client-specific weights, preserve the local hetero LoRA
            # structure by filtering out LoRA tensors before update.
            has_distributed_format = any(
                '.' in key and key.split('.')[-1].isdigit() and
                ('lora_A' in key or 'lora_B' in key)
                for key in content.keys()
            )
            use_hetero_lora = (
                self._expects_client_specific_hetero_payload()
                or client_rank_config_from_msg is not None
                or fs_common.get_active_hetero_config_local(self._cfg) is not None
            )

            if use_hetero_lora:
                if has_distributed_format:
                    content = self._apply_client_specific_heterolora_payload(
                        content,
                        client_rank_config_from_msg=client_rank_config_from_msg,
                        context='finish',
                    )
                else:
                    logger.info(
                        f"Client {self.ID}: Filtering out max-rank LoRA weights "
                        f"from finish message to preserve local hetero configuration"
                    )
                    content = {
                        k: v for k, v in content.items()
                        if 'lora_A' not in k and 'lora_B' not in k
                    }

            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model)

            # [FAH] After loading parameters from the server, LoRA tensors may be fp32
            # (communication dtype) even when this client computes in fp16/bf16.
            # Recast trainable tensors back to the intended compute dtype to prevent
            # mixed-dtype attention matmuls.
            try:
                _cd = self._fah_resolve_compute_dtype()
                self._fah_cast_trainable_params(_cd, log_prefix=f"[FAH] Client {self.ID}:")
            except Exception as _e:
                logger.warning("[FAH] Client %s: failed to recast trainable params after update: %s", self.ID, _e)

        self._monitor.finish_fl()

    def callback_funcs_for_converged(self, message: Message):
        """
        The handling function for receiving the signal that the FL course \
        converged

        Arguments:
            message: The received message
        """
        self._monitor.global_converged()

    def _init_fah_qloRA(self):
        """Default FAH hook for non-FAH clients."""
        self.fah_enabled = False
        self.fah_current_rank = None
        self.fah_current_hat_rank = None
        self.fah_last_training_time = 0.0
        self.fah_validation_fraction = 0.0
        self.fah_validation_steps = 0

    def _fah_evaluate_and_send_stats(
        self,
        model_content: dict,
        server_id: int,
        round_idx: int,
        timestamp: float
    ):
        """Default no-op FAH stats hook for non-FAH clients."""
        return

    def _apply_heterolora_rank_config(self, client_rank_config: dict, debug: bool = False):
        """Default no-op heterogeneous rank-application hook."""
        return

    def _fah_evaluate_loss(self) -> float:
        """Default FAH evaluation hook for non-FAH clients."""
        return float('inf')

    def _fah_eval_val_loss(self) -> float:
        """Default validation-loss hook for non-FAH clients."""
        return float('inf')

    def _fah_evaluate_loss_at_rank(self, rank: int) -> float:
        """Default rank-probing hook for non-FAH clients."""
        return float('inf')

    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str