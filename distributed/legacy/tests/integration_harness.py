"""
Integration test harness for Gates 3-4.

This module provides utilities for running full FL message round-trips
through the device_agent relay boundary.

Components:
- IntegrationTestHarness: Manages server comm, device_agent stub, and client comm
- Utilities for creating test messages and verifying integrity

Usage:
    harness = IntegrationTestHarness(port_offset=6000)
    harness.start()
    
    # Send message from server to client
    harness.server_send(message)
    received = harness.client_receive()
    
    # Send message from client to server
    harness.client_send(message)
    received = harness.server_receive()
    
    harness.stop()
"""

import logging
import os
import threading
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    """Configuration for integration test harness."""
    port_offset: int = 6000
    device_name: str = "test_device"
    container_name: str = "test_container"
    client_id: int = 1
    server_host: str = "127.0.0.1"
    recv_timeout_ms: int = 5000
    send_timeout_ms: int = 2000


class IntegrationTestHarness:
    """
    Integration test harness for FL message round-trips.
    
    This harness sets up the full communication stack:
    - ZMQServerCommManager (binds FL payload ports)
    - DeviceAgentStub (relays between server and worker)
    - ZMQClientCommManager (connects to device_agent)
    
    Port assignments (with offset):
    - Server REP: 60001 + offset
    - Server PUB: 60002 + offset
    - Device Agent worker REP: 60011 + offset
    - Device Agent worker PUB: 60012 + offset
    """
    
    def __init__(self, config: Optional[HarnessConfig] = None):
        """
        Initialize the integration test harness.
        
        Args:
            config: Harness configuration (uses defaults if None)
        """
        self.config = config or HarnessConfig()
        
        self._server_comm = None
        self._device_agent = None
        self._client_comm = None
        self._started = False
        
        logger.info(f"IntegrationTestHarness created with port_offset={self.config.port_offset}")
    
    def start(self) -> None:
        """Start all components in correct order."""
        if self._started:
            return
        
        from distributed.comm import ZMQServerCommManager, ZMQClientCommManager, DeviceAgentStub
        
        # 1. Start server comm manager (binds FL payload ports)
        routing_table = {
            self.config.client_id: {
                'device_name': self.config.device_name,
                'container_name': self.config.container_name,
            }
        }
        
        self._server_comm = ZMQServerCommManager(
            host=self.config.server_host,
            port_offset=self.config.port_offset,
            recv_timeout_ms=self.config.recv_timeout_ms,
            send_timeout_ms=self.config.send_timeout_ms,
            client_routing_table=routing_table,
        )
        
        # Add client as neighbor
        self._server_comm.add_neighbors(
            self.config.client_id,
            routing_table[self.config.client_id]
        )
        
        # Initialize server (starts receive thread, binds sockets)
        self._server_comm._initialize()
        logger.info("Server comm manager started")
        
        # 2. Start device agent stub (connects to server, binds worker ports)
        self._device_agent = DeviceAgentStub(
            device_name=self.config.device_name,
            server_host=self.config.server_host,
            control_plane_host=self.config.server_host,
            local_host=self.config.server_host,
            port_offset=self.config.port_offset,
        )
        
        self._device_agent.register_container(
            self.config.container_name,
            self.config.client_id,
        )
        
        self._device_agent.start()
        logger.info("Device agent stub started")
        
        # Give device_agent time to connect and subscribe
        time.sleep(0.5)
        
        # 3. Start client comm manager (connects to device_agent)
        self._client_comm = ZMQClientCommManager(
            client_id=self.config.client_id,
            container_name=self.config.container_name,
            device_agent_host=self.config.server_host,
            device_agent_port_offset=self.config.port_offset,
            recv_timeout_ms=self.config.recv_timeout_ms,
            send_timeout_ms=self.config.send_timeout_ms,
        )
        
        # Add server as neighbor
        self._client_comm.add_neighbors(0, {'host': self.config.server_host})
        
        # Initialize client (starts receive thread, connects sockets)
        self._client_comm._initialize()
        logger.info("Client comm manager started")
        
        # Give everything time to settle
        time.sleep(0.3)
        
        self._started = True
        logger.info("Integration test harness fully started")
    
    def stop(self) -> None:
        """Stop all components in reverse order."""
        if not self._started:
            return
        
        if self._client_comm:
            self._client_comm.close()
            logger.info("Client comm manager stopped")
        
        if self._device_agent:
            self._device_agent.stop()
            logger.info("Device agent stub stopped")
        
        if self._server_comm:
            self._server_comm.close()
            logger.info("Server comm manager stopped")
        
        self._started = False
        logger.info("Integration test harness stopped")
    
    @property
    def server_comm(self):
        """Get the server comm manager."""
        return self._server_comm
    
    @property
    def client_comm(self):
        """Get the client comm manager."""
        return self._client_comm
    
    @property
    def device_agent(self):
        """Get the device agent stub."""
        return self._device_agent


def create_test_model_para_message(
    sender: int,
    receiver: List[int],
    state: int = 0,
    model_size: int = 100,
) -> 'Message':
    """
    Create a test model_para message with realistic content.
    
    Args:
        sender: Sender ID
        receiver: List of receiver IDs
        state: Training round state
        model_size: Size of model parameters (number of elements per tensor)
        
    Returns:
        FedLoRA Message with model_para content
    """
    from federatedscope.core.message import Message
    
    # Create realistic model state dict
    state_dict = {
        'layer1.weight': torch.randn(model_size, model_size),
        'layer1.bias': torch.randn(model_size),
        'layer2.weight': torch.randn(model_size, model_size // 2),
        'layer2.bias': torch.randn(model_size // 2),
    }
    
    return Message(
        msg_type='model_para',
        sender=sender,
        receiver=receiver,
        state=state,
        timestamp=time.time(),
        content=state_dict,
    )


def create_test_client_upload_message(
    client_id: int,
    server_id: int = 0,
    state: int = 0,
    sample_count: int = 100,
    model_size: int = 100,
) -> 'Message':
    """
    Create a test client upload message with (sample_count, state_dict).
    
    This matches the expected FedIT upload format.
    
    Args:
        client_id: Client sender ID
        server_id: Server receiver ID
        state: Training round state
        sample_count: Number of training samples
        model_size: Size of model parameters
        
    Returns:
        FedLoRA Message with (sample_count, state_dict) content
    """
    from federatedscope.core.message import Message
    
    state_dict = {
        'layer1.weight': torch.randn(model_size, model_size),
        'layer1.bias': torch.randn(model_size),
        'layer2.weight': torch.randn(model_size, model_size // 2),
        'layer2.bias': torch.randn(model_size // 2),
    }
    
    return Message(
        msg_type='model_para',
        sender=client_id,
        receiver=[server_id],
        state=state,
        timestamp=time.time(),
        content=(sample_count, state_dict),
    )


def verify_message_integrity(original: 'Message', received: 'Message') -> Dict[str, bool]:
    """
    Verify that a received message matches the original.
    
    Args:
        original: Original message
        received: Received message after transport
        
    Returns:
        Dict of check names to pass/fail booleans
    """
    results = {}
    
    # Basic fields
    results['msg_type_match'] = original.msg_type == received.msg_type
    results['sender_match'] = original.sender == received.sender
    results['receiver_match'] = original.receiver == received.receiver
    results['state_match'] = original.state == received.state
    
    # Content check
    if isinstance(original.content, dict) and isinstance(received.content, dict):
        results['content_keys_match'] = set(original.content.keys()) == set(received.content.keys())
        
        # For tensor content, check shapes and values
        all_tensors_match = True
        for key in original.content.keys():
            if isinstance(original.content[key], torch.Tensor):
                if not isinstance(received.content[key], torch.Tensor):
                    all_tensors_match = False
                    break
                if original.content[key].shape != received.content[key].shape:
                    all_tensors_match = False
                    break
                if not torch.allclose(original.content[key], received.content[key]):
                    all_tensors_match = False
                    break
        results['tensor_values_match'] = all_tensors_match
        
    elif isinstance(original.content, tuple) and isinstance(received.content, tuple):
        results['tuple_length_match'] = len(original.content) == len(received.content)
        
        if len(original.content) == 2:  # (sample_count, state_dict) format
            results['sample_count_match'] = original.content[0] == received.content[0]
            
            if isinstance(original.content[1], dict) and isinstance(received.content[1], dict):
                all_tensors_match = True
                for key in original.content[1].keys():
                    if isinstance(original.content[1][key], torch.Tensor):
                        if not torch.allclose(original.content[1][key], received.content[1][key]):
                            all_tensors_match = False
                            break
                results['state_dict_match'] = all_tensors_match
    else:
        results['content_type_match'] = type(original.content) == type(received.content)
    
    return results


def gate4_minimal_quadratic_config_path() -> Path:
    """Path to the checked-in minimal quadratic YAML for real Client tests."""
    return Path(__file__).resolve().parents[2] / "distributed/tests/configs/gate4_minimal_quadratic.yaml"


def gate4_partition_config_path() -> Path:
    """Preferred Gate 4 config path using the same GLUE-oriented config family as partition prep."""
    return Path(__file__).resolve().parents[2] / "distributed/configs/fedit_partition_prep.yaml"


def gate4_real_glue_config_path() -> Path:
    """Primary Gate 4 config path aligned with the real working FedLoRA GLUE setup."""
    return Path(__file__).resolve().parents[2] / "2_yamls/fedit/fedit-NO_quantized.yaml"


def gate4_generated_partition_path(client_id: int) -> Path:
    """Path to a previously prepared and validated partition artifact for Gate 4."""
    return Path(__file__).resolve().parents[2] / f"distributed/data/generated_partitions/client_{int(client_id)}.pkl"


def _best_available_gate4_paths(client_id: int, yaml_path: Optional[Path], partition_path: Optional[str]):
    """Choose the safest Gate 4 config/data paths available on disk.

    Priority order:
    1. Explicit yaml/partition arguments
    2. Generated validated partition + real working GLUE config
    3. Partition-prep config
    4. Minimal quadratic fallback
    """
    chosen_yaml = Path(yaml_path) if yaml_path is not None else None
    chosen_partition = Path(partition_path) if partition_path is not None else None

    if chosen_partition is None:
        candidate_partition = gate4_generated_partition_path(client_id)
        if candidate_partition.exists():
            chosen_partition = candidate_partition

    if chosen_yaml is None:
        real_glue_yaml = gate4_real_glue_config_path()
        if real_glue_yaml.exists():
            chosen_yaml = real_glue_yaml
        else:
            prep_yaml = gate4_partition_config_path()
            if prep_yaml.exists():
                chosen_yaml = prep_yaml
            else:
                chosen_yaml = gate4_minimal_quadratic_config_path()

    return chosen_yaml, chosen_partition


def _normalize_gate4_cfg(cfg, client_id: int):
    """Normalize Gate 4 config against the real current FedLoRA interfaces.

    This aligns the harness with:
    - real GLUE YAML conventions from 2_yamls/
    - current Client/Monitor expectations in federatedscope/
    - current worker build_client(...) helper signature in distributed/
    """
    import torch

    try:
        cfg.defrost()
    except Exception:
        pass

    # Gate 4 runs the existing FedLoRA Client in distributed mode with static ID.
    cfg.federate.mode = "distributed"
    cfg.federate.client_id = int(client_id)
    try:
        cfg.distribute.data_idx = int(client_id)
    except Exception:
        pass

    # Ensure monitor/trainer expectations are satisfied even if a reduced config is used.
    try:
        cfg.trainer.type = getattr(cfg.trainer, 'type', None) or 'gluetrainer'
    except Exception:
        pass
    try:
        cfg.criterion.type = getattr(cfg.criterion, 'type', None) or 'CrossEntropyLoss'
    except Exception:
        pass
    try:
        metrics = list(getattr(cfg.eval, 'metrics', []))
        if not metrics:
            cfg.eval.metrics = ['acc', 'f1', 'loss']
        elif 'acc' not in metrics:
            metrics.append('acc')
            cfg.eval.metrics = metrics
    except Exception:
        pass
    try:
        cfg.eval.report = list(getattr(cfg.eval, 'report', [])) or ['weighted_avg']
    except Exception:
        pass
    try:
        best_key = getattr(cfg.eval, 'best_res_update_round_wise_key', '')
        if not best_key or 'acc' not in str(best_key):
            cfg.eval.best_res_update_round_wise_key = 'val_acc'
    except Exception:
        pass

    # Keep GLUE/task defaults aligned with the real SST-2 path if absent.
    try:
        cfg.data.type = getattr(cfg.data, 'type', None) or 'sst2@glue'
    except Exception:
        pass
    try:
        cfg.model.task = getattr(cfg.model, 'task', None) or 'SequenceClassification'
    except Exception:
        pass
    try:
        if not hasattr(cfg.model, 'out_channels') or cfg.model.out_channels in [None, 0]:
            cfg.model.out_channels = 2
    except Exception:
        pass

    # Device placement must match the real current model_builder behavior.
    # federatedscope/glue/model/model_builder.py reads config.device directly.
    if torch.cuda.is_available():
        try:
            cfg.use_gpu = True
        except Exception:
            pass
        try:
            cfg.device = 0
        except Exception:
            pass
        try:
            cfg.gpu_id = 0
        except Exception:
            pass
    else:
        try:
            cfg.use_gpu = False
        except Exception:
            pass
        try:
            cfg.device = 'cpu'
        except Exception:
            pass
        try:
            cfg.gpu_id = -1
        except Exception:
            pass

    try:
        cfg.train.is_enable_half = False
    except Exception:
        pass
    try:
        cfg.freeze()
    except Exception:
        pass
    return cfg


def _build_model_for_gate4(cfg, data):
    """Build the real model using the normalized live FedLoRA config.

    No guessed wrapper kwargs are injected here. The goal is to exercise the
    actual current model-builder path the codebase uses for GLUE/FedIT.
    """
    from federatedscope.core.auxiliaries.model_builder import get_model
    return get_model(cfg, data)


def load_gate4_minimal_client_bundle(
    harness: IntegrationTestHarness,
    *,
    yaml_path: Optional[Path] = None,
    partition_path: Optional[str] = None,
    client_id: Optional[int] = None,
) -> Tuple[Any, Any, Any]:  # (cfg, data, model)
    """
    Build ``(cfg, data, model)`` for a real FedLoRA Client on the harness.

    Preferred path for Gate 4 is to reuse the already generated and validated
    partition artifact from the partition-prep prerequisite. This matches the
    Phase 1 worker design more closely than rebuilding data from a tiny fixture.
    """
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.data_builder import get_data

    cfg_client_id = int(client_id if client_id is not None else harness.config.client_id)
    cfg_path, resolved_partition = _best_available_gate4_paths(
        cfg_client_id, yaml_path, partition_path
    )

    cfg = global_cfg.clone()
    cfg.merge_from_file(str(cfg_path))
    cfg = _normalize_gate4_cfg(cfg, cfg_client_id)

    if resolved_partition is not None:
        from distributed.worker.main import load_partition
        data = load_partition(str(resolved_partition), cfg)
    else:
        data, _ = get_data(cfg.clone())

    model = _build_model_for_gate4(cfg, data)
    return cfg, data, model




def build_gate2_dry_run_client(
    *,
    client_id: int,
    container_name: str,
    device_agent_host: str = "localhost",
    device_agent_port_offset: int = 0,
    yaml_path: Optional[Path] = None,
    local_address: Optional[Dict[str, Any]] = None,
):
    """
    Build a real FedLoRA ``Client`` for Gate 2 dry-run validation without
    depending on the full ``get_data()``/``get_model()`` path.

    Gate 2 only needs to prove that the worker can construct an injected
    ``Client`` successfully and exit before ``join_in()`` / ``run()``.
    In this repo, the tiny quadratic fixture can still fall through code paths
    that expect data/model structures not required for dry-run validation.

    This helper therefore:
    - loads the real FedLoRA config
    - constructs a tiny torch model that is pickle-safe
    - temporarily patches the ``Client`` module's ``get_trainer`` symbol with a
      minimal trainer object sufficient for ``Client.__init__``
    - returns the fully constructed injected ``Client``

    It does *not* alter normal Gate 4 training helpers.
    """
    from types import SimpleNamespace
    import torch
    import federatedscope.core.workers.client as client_mod
    from federatedscope.core.configs.config import global_cfg
    from distributed.worker.main import build_client

    cfg_path = yaml_path or gate4_minimal_quadratic_config_path()
    cfg = global_cfg.clone()
    cfg.merge_from_file(str(cfg_path))
    cfg.federate.mode = "distributed"
    cfg.federate.client_id = int(client_id)

    # Normalize fixture fields used by some FedLoRA utilities even though the
    # dry-run path below does not depend on full dataset conversion.
    try:
        cfg.distribute.data_idx = int(getattr(cfg.distribute, 'data_idx', client_id))
    except Exception:
        pass
    cfg.freeze()

    class _Gate2DryRunModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.x = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x * self.x

    class _DummyTrainData:
        def __len__(self):
            return 1

    class _DummyDataBundle:
        def __init__(self):
            self.train_data = _DummyTrainData()

    class _DummyTrainer:
        def __init__(self, model, data):
            self.data = data
            self.ctx = SimpleNamespace(model=model, optimizer=None)

        def get_model_para(self):
            return self.ctx.model.state_dict()

    model = _Gate2DryRunModel()
    data = _DummyDataBundle()

    original_get_trainer = client_mod.get_trainer
    client_mod.get_trainer = lambda **kwargs: _DummyTrainer(kwargs.get('model'), kwargs.get('data'))
    try:
        client = build_client(
            client_id=int(client_id),
            container_name=container_name,
            device_agent_host=device_agent_host,
            device_agent_port_offset=device_agent_port_offset,
            cfg=cfg,
            data=data,
            model=model,
            local_address=local_address if local_address is not None else {"host": "gate2_dryrun_worker", "port": 0},
        )
    finally:
        client_mod.get_trainer = original_get_trainer

    return client
def build_injected_client_for_harness(
    harness: IntegrationTestHarness,
    *,
    yaml_path: Optional[Path] = None,
    partition_path: Optional[str] = None,
    client_id: Optional[int] = None,
    local_address: Optional[Dict[str, Any]] = None,
):
    """
    Instantiate the real FedLoRA ``Client`` with ``harness.client_comm`` injected.

    This must follow the *actual current* ``distributed.worker.main.build_client``
    signature in the codebase. Do not pass guessed kwargs such as ``device=...``.
    Device selection is handled by the worker helper itself.
    """
    from distributed.worker.main import build_client

    cfg, data, model = load_gate4_minimal_client_bundle(
        harness,
        yaml_path=yaml_path,
        partition_path=partition_path,
        client_id=client_id,
    )
    cid = int(client_id if client_id is not None else harness.config.client_id)
    return build_client(
        client_id=cid,
        container_name=harness.config.container_name,
        device_agent_host=harness.config.server_host,
        device_agent_port_offset=harness.config.port_offset,
        cfg=cfg,
        data=data,
        model=model,
        comm_manager=harness.client_comm,
        local_address=local_address if local_address is not None else {"host": "gate4_test_worker", "port": 0},
    )


def capture_model_para_after_training(
    harness: IntegrationTestHarness,
    client,
    *,
    training_round: int = 0,
):
    """
    End-to-end: server sends ``model_para`` (global ``state_dict``) over the
    relay; client receives it; ``callback_funcs_for_model_para`` runs local
    training and uploads; return the ``Message`` observed on the server comm.
    """
    from federatedscope.core.message import Message

    state_dict = client.model.state_dict()
    down_msg = Message(
        msg_type="model_para",
        sender=0,
        receiver=[client.ID],
        state=training_round,
        timestamp=time.time(),
        content=state_dict,
    )
    harness.server_comm.send(down_msg)
    received = harness.client_comm.receive()
    client.callback_funcs_for_model_para(received)
    return harness.server_comm.receive()


class _Gate5MinimalModel(torch.nn.Module):
    """Tiny model for Gate 5 server aggregation tests."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def build_injected_server_for_harness(
    harness: IntegrationTestHarness,
    *,
    yaml_path: Optional[Path] = None,
    client_num: int = 1,
):
    """
    Instantiate the real FedLoRA ``Server`` with ``harness.server_comm`` injected.

    Prerequisites: ``harness.start()`` has completed so ``server_comm`` sockets
    are live. Intended for Gate 5 (real aggregation + rebroadcast).

    Args:
        harness: Running IntegrationTestHarness
        yaml_path: Path to YAML config (defaults to gate4 minimal quadratic)
        client_num: Expected number of clients for FL course

    Returns:
        Configured FedLoRA Server instance with injected comm_manager
    """
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.workers.server import Server

    cfg_path = yaml_path or gate4_minimal_quadratic_config_path()
    cfg = global_cfg.clone()
    cfg.merge_from_file(str(cfg_path))

    try:
        cfg.defrost()
    except Exception:
        pass

    cfg.federate.mode = "standalone"
    cfg.federate.client_num = client_num
    cfg.federate.sample_client_num = client_num
    cfg.federate.total_round_num = 2

    try:
        cfg.federate.make_global_eval = False
    except Exception:
        pass

    try:
        cfg.distribute.data_idx = 0
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            cfg.use_gpu = True
            cfg.device = 0
        except Exception:
            pass
    else:
        try:
            cfg.use_gpu = False
            cfg.device = "cpu"
        except Exception:
            pass

    try:
        cfg.freeze()
    except Exception:
        pass

    model = _Gate5MinimalModel(dim=4)
    device = "cpu"

    server = Server(
        ID=0,
        state=0,
        config=cfg,
        data=None,
        model=model,
        client_num=client_num,
        total_round_num=cfg.federate.total_round_num,
        device=device,
        comm_manager=harness.server_comm,
    )

    logger.info(f"Built Server with injected ZMQServerCommManager, client_num={client_num}")
    return server, cfg, model


def run_simple_roundtrip_test(port_offset: int = 6000) -> bool:
    """
    Run a simple server->client->server round-trip test.
    
    Args:
        port_offset: Port offset for this test
        
    Returns:
        True if test passed, False otherwise
    """
    harness = IntegrationTestHarness(HarnessConfig(port_offset=port_offset))
    
    try:
        harness.start()
        
        # Server -> Client
        from federatedscope.core.message import Message
        
        server_msg = Message(
            msg_type='model_para',
            sender=0,
            receiver=[1],
            state=0,
            timestamp=0,
            content={'test': torch.tensor([1.0, 2.0, 3.0])},
        )
        
        harness.server_comm.send(server_msg)
        
        # Client receives
        received = harness.client_comm.receive()
        
        # Verify
        checks = verify_message_integrity(server_msg, received)
        
        if not all(checks.values()):
            logger.error(f"Server->Client integrity check failed: {checks}")
            return False
        
        logger.info("Server->Client round-trip passed")
        
        # Client -> Server
        client_msg = Message(
            msg_type='model_para',
            sender=1,
            receiver=[0],
            state=0,
            timestamp=0,
            content=(100, {'test': torch.tensor([4.0, 5.0, 6.0])}),
        )
        
        harness.client_comm.send(client_msg)
        
        # Server receives
        received = harness.server_comm.receive()
        
        # Verify
        checks = verify_message_integrity(client_msg, received)
        
        if not all(checks.values()):
            logger.error(f"Client->Server integrity check failed: {checks}")
            return False
        
        logger.info("Client->Server round-trip passed")
        
        return True
        
    except Exception as e:
        logger.error(f"Round-trip test failed with exception: {e}")
        return False
        
    finally:
        harness.stop()


@dataclass
class MultiClientHarnessConfig:
    """Configuration for multi-client integration test harness (Gate 6)."""
    port_offset: int = 6000
    client_ids: Tuple[int, ...] = (1, 2)
    device_name: str = "gate6_device"
    server_host: str = "127.0.0.1"
    recv_timeout_ms: int = 10000
    send_timeout_ms: int = 5000


class MultiClientTestHarness:
    """
    Multi-client integration test harness for Gate 6.

    Sets up:
    - ZMQServerCommManager with routing for N clients
    - Single DeviceAgentStub with N container registrations
    - N ZMQClientCommManager instances
    """

    def __init__(self, config: Optional[MultiClientHarnessConfig] = None):
        self.config = config or MultiClientHarnessConfig()
        self._server_comm = None
        self._device_agent = None
        self._client_comms: Dict[int, Any] = {}
        self._started = False

        logger.info(
            f"MultiClientTestHarness created with port_offset={self.config.port_offset}, "
            f"client_ids={self.config.client_ids}"
        )

    def start(self) -> None:
        """Start all components in correct order."""
        if self._started:
            return

        from distributed.comm import ZMQServerCommManager, ZMQClientCommManager, DeviceAgentStub

        routing_table = {}
        for cid in self.config.client_ids:
            routing_table[cid] = {
                "device_name": self.config.device_name,
                "container_name": f"fl_worker_{cid}",
            }

        self._server_comm = ZMQServerCommManager(
            host=self.config.server_host,
            port_offset=self.config.port_offset,
            recv_timeout_ms=self.config.recv_timeout_ms,
            send_timeout_ms=self.config.send_timeout_ms,
            client_routing_table=routing_table,
        )

        for cid, info in routing_table.items():
            self._server_comm.add_neighbors(cid, info)

        self._server_comm._initialize()
        logger.info("Server comm manager started (multi-client)")

        self._device_agent = DeviceAgentStub(
            device_name=self.config.device_name,
            server_host=self.config.server_host,
            control_plane_host=self.config.server_host,
            local_host=self.config.server_host,
            port_offset=self.config.port_offset,
        )

        for cid in self.config.client_ids:
            self._device_agent.register_container(f"fl_worker_{cid}", cid)

        self._device_agent.start()
        logger.info("Device agent stub started (multi-client)")
        time.sleep(0.5)

        for cid in self.config.client_ids:
            client_comm = ZMQClientCommManager(
                client_id=cid,
                container_name=f"fl_worker_{cid}",
                device_agent_host=self.config.server_host,
                device_agent_port_offset=self.config.port_offset,
                recv_timeout_ms=self.config.recv_timeout_ms,
                send_timeout_ms=self.config.send_timeout_ms,
            )
            client_comm.add_neighbors(0, {"host": self.config.server_host})
            client_comm._initialize()
            self._client_comms[cid] = client_comm
            logger.info(f"Client comm manager started for client_id={cid}")

        time.sleep(0.3)
        self._started = True
        logger.info("MultiClientTestHarness fully started")

    def stop(self) -> None:
        """Stop all components in reverse order."""
        if not self._started:
            return

        for cid, cc in self._client_comms.items():
            cc.close()
            logger.info(f"Client comm manager stopped for client_id={cid}")

        if self._device_agent:
            self._device_agent.stop()
            logger.info("Device agent stub stopped")

        if self._server_comm:
            self._server_comm.close()
            logger.info("Server comm manager stopped")

        self._started = False
        logger.info("MultiClientTestHarness stopped")

    @property
    def server_comm(self):
        return self._server_comm

    def client_comm(self, client_id: int):
        return self._client_comms.get(client_id)

    @property
    def client_comms(self) -> Dict[int, Any]:
        return self._client_comms


def build_injected_server_for_multi_client_harness(
    harness: MultiClientTestHarness,
    *,
    yaml_path: Optional[Path] = None,
    total_round_num: int = 5,
):
    """
    Instantiate the real FedLoRA Server for Gate 6 multi-client testing.

    Args:
        harness: Running MultiClientTestHarness
        yaml_path: Path to YAML config (defaults to gate4 minimal quadratic)
        total_round_num: Number of FL rounds

    Returns:
        (server, cfg, model) tuple
    """
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.workers.server import Server

    client_num = len(harness.config.client_ids)

    cfg_path = yaml_path or gate4_minimal_quadratic_config_path()
    cfg = global_cfg.clone()
    cfg.merge_from_file(str(cfg_path))

    try:
        cfg.defrost()
    except Exception:
        pass

    cfg.federate.mode = "standalone"
    cfg.federate.client_num = client_num
    cfg.federate.sample_client_num = client_num
    cfg.federate.total_round_num = total_round_num

    try:
        cfg.federate.make_global_eval = False
    except Exception:
        pass

    try:
        cfg.distribute.data_idx = 0
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            cfg.use_gpu = True
            cfg.device = 0
        except Exception:
            pass
    else:
        try:
            cfg.use_gpu = False
            cfg.device = "cpu"
        except Exception:
            pass

    try:
        cfg.freeze()
    except Exception:
        pass

    model = _Gate5MinimalModel(dim=4)
    device = "cpu"

    server = Server(
        ID=0,
        state=0,
        config=cfg,
        data=None,
        model=model,
        client_num=client_num,
        total_round_num=total_round_num,
        device=device,
        comm_manager=harness.server_comm,
    )

    logger.info(
        f"Built Server for Gate 6: client_num={client_num}, total_rounds={total_round_num}"
    )
    return server, cfg, model


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Running simple round-trip test...")
    success = run_simple_roundtrip_test(port_offset=7000)
    print(f"Test result: {'PASSED' if success else 'FAILED'}")
