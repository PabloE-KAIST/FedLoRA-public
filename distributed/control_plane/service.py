"""
Control Plane Service for distributed FedLoRA deployment.

Supports two wire modes:
- **pickle mode** (default, ``proto_mode=False``): Python dict control messages
  via pickle+zlib. Historical path, retained for legacy tests in
  ``distributed/legacy/tests/`` (see ``distributed/legacy/`` for the archived
  in-process stub).
- **proto mode** (``proto_mode=True``): Protobuf wire format matching the real
  C++ device_agent (AegisGov). Uses ``proto_bridge`` for encoding/decoding.

Proto-mode wire format (matches C++ device_agent):
  REQ/REP: ``b"{WIRE_MSG_TYPE} {protobuf_bytes}"``
  PUB:     ``b"{device_name}| {WIRE_MSG_TYPE} {protobuf_bytes}"``

Ports (with offset):
- REP socket (receive): 60101 + offset
- PUB socket (broadcast): 60102 + offset

Relay-mode (proto_mode + relay):
  The C++ device_agent uses the *same* controller REQ/PUB sockets for
  both control and FL payload relay.  In relay mode the ControlPlaneService
  acts as a unified gateway:
    - REP receives both DEV_AD (control) and forwarded relay envelopes
    - PUB publishes both CONT_START (control) and FWD_CONT (FL payloads)
  ZMQServerCommManager bridges through queues exposed by this service.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Queue
from typing import Dict, Optional, List, Any
from enum import Enum

from distributed.comm.serialization import (
    serialize_any,
    deserialize_any,
    pack_control_message,
    unpack_control_message,
)

logger = logging.getLogger(__name__)


class ControlMessageType(str, Enum):
    """
    Control plane message types.

    Python values use full strings for readability.  The C++ device_agent
    (AegisGov) uses abbreviated wire tokens — the bridge layer must translate:
      DEVICE_ADVERTISEMENT -> "DEV_AD"
      CONTAINER_START      -> "CONT_START"
      CONTAINER_STOP       -> "CONT_STOP"
      CONNECT_DEVICE       -> "CON_DEV"
    """
    DEVICE_ADVERTISEMENT = "DEVICE_ADVERTISEMENT"
    CONTAINER_START = "CONTAINER_START"
    CONTAINER_STOP = "CONTAINER_STOP"
    CONNECT_DEVICE = "CONNECT_DEVICE"
    SYSTEM_INFO = "SYSTEM_INFO"
    DEVICE_SHUTDOWN = "DEVICE_SHUTDOWN"
    ACK = "ACK"
    ERROR = "ERROR"


@dataclass
class DeviceInfo:
    """
    Device information for DEVICE_ADVERTISEMENT.
    
    This is the Python equivalent of the protobuf DeviceInfo message.
    Production deployment should use actual protobuf.
    """
    name: str
    device_type: str
    ip_address: str
    agent_port_offset: int = 0
    processors: int = 1
    memory: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DeviceInfo':
        return cls(**data)


@dataclass
class ContainerConfig:
    """
    Container configuration for CONTAINER_START.
    
    This is the Python equivalent of the protobuf ContainerConfig message.
    
    The worker targets the local device_agent for FL message relay,
    NOT the FL server directly. Fields reflect this architecture:
    - device_agent_host: Usually "localhost" from container's perspective
    - device_agent_port_offset: Port offset for device_agent (60011/60012 + offset)
    """
    name: str  # Container name (used for device_agent routing)
    executable: str = "python -m distributed.worker.main"
    docker_tag: str = ""
    json_config: str = ""
    device: int = 0
    control_port: int = 0
    
    # FedLoRA-specific fields for worker launch
    client_id: int = -1
    partition_path: str = ""
    config_path: str = ""
    
    # Device-agent routing (worker connects to local device_agent, not server)
    device_agent_host: str = "localhost"  # From container's view
    device_agent_port_offset: int = 0

    # Bandwidth measurement
    nic: str = ""
    shared_nic: bool = False

    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ContainerConfig':
        # Filter out unknown fields for compatibility
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def to_worker_args(self) -> List[str]:
        """
        Convert to command-line arguments for worker.

        Arguments target the device_agent relay, not the FL server.
        """
        args = [
            f"--client-id={self.client_id}",
            f"--container-name={self.name}",
            f"--device-agent-host={self.device_agent_host}",
            f"--device-agent-port-offset={self.device_agent_port_offset}",
        ]
        if self.partition_path:
            args.append(f"--partition-path={self.partition_path}")
        if self.config_path:
            args.append(f"--config={self.config_path}")
        if self.nic:
            args.append(f"--nic={self.nic}")
        if self.shared_nic:
            args.append("--shared-nic")
        return args


def container_config_from_manifest_client(
    client_entry: Dict[str, Any],
    *,
    config_path: str,
    device_agent_host: str = "localhost",
    device_agent_port_offset: int = 0,
    partition_root: Optional[str] = None,
    executable: str = "python -m distributed.worker.main",
) -> ContainerConfig:
    """
    Build a ``ContainerConfig`` from a single static-manifest client record.

    This is the **single source of truth** for mapping manifest fields to
    worker CLI arguments: ``client_id``, ``container_name``, ``partition_path``,
    ``config_path``, and device_agent connectivity.

    Args:
        client_entry: One element from ``manifest["clients"]`` (must include
            ``client_id`` and ``container_name``; optional ``partition_path``).
        config_path: Path passed to the worker as ``--config`` (canonical
            experiment YAML).
        device_agent_host / device_agent_port_offset: Relay boundary seen from
            the container (defaults match local stub / dev layout).
        partition_root: If set and ``partition_path`` is relative, join this
            root with the manifest path to produce the path given to the worker.
        executable: Worker entry command (default thin worker module).

    Returns:
        ``ContainerConfig`` ready for ``to_worker_args()`` / CONTAINER_START.
    """
    cid = int(client_entry["client_id"])
    name = str(client_entry["container_name"])
    raw_part = client_entry.get("partition_path") or ""
    partition_path = ""
    if raw_part:
        partition_path = (
            str(Path(partition_root) / raw_part) if partition_root else str(raw_part)
        )
    nic = client_entry.get("nic", "")
    # Crystal virtual devices share a NIC — NIC-level stats show combined traffic
    device_name = client_entry.get("device_name", "")
    shared_nic = device_name.startswith("crystal_")
    return ContainerConfig(
        name=name,
        executable=executable,
        client_id=cid,
        partition_path=partition_path,
        config_path=str(config_path),
        device_agent_host=device_agent_host,
        device_agent_port_offset=int(device_agent_port_offset),
        nic=nic,
        shared_nic=shared_nic,
    )


@dataclass
class ContainerSignal:
    """Container signal for CONTAINER_STOP."""
    name: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ContainerSignal':
        return cls(**data)


@dataclass
class RegisteredDevice:
    """A device that has registered with the control plane."""
    info: DeviceInfo
    registered_at: float
    last_seen: float
    containers: Dict[str, ContainerConfig] = field(default_factory=dict)
    is_healthy: bool = True


class ControlPlaneService:
    """
    Control Plane Service for distributed FedLoRA.

    Responsibilities:
    - Receive DEVICE_ADVERTISEMENT from device agents
    - Maintain device registry
    - Send CONTAINER_START and CONTAINER_STOP commands
    - Validate manifest matches
    - In relay mode: act as unified gateway for FL payload relay between
      device_agents and ZMQServerCommManager (see module docstring).
    """

    DEFAULT_REP_PORT = 60101
    DEFAULT_PUB_PORT = 60102

    def __init__(
        self,
        host: str = "0.0.0.0",
        port_offset: int = 0,
        manifest_path: Optional[str] = None,
        proto_mode: bool = False,
    ):
        self.host = host
        self.port_offset = port_offset
        self.rep_port = self.DEFAULT_REP_PORT + port_offset
        self.pub_port = self.DEFAULT_PUB_PORT + port_offset
        self.proto_mode = proto_mode

        # Device registry
        self.devices: Dict[str, RegisteredDevice] = {}
        self._devices_lock = threading.Lock()

        # Manifest
        self.manifest: Optional[dict] = None
        if manifest_path:
            self.load_manifest(manifest_path)

        # ZMQ sockets (initialized in start())
        self._context = None
        self._rep_socket = None
        self._pub_socket = None

        # Service state
        self._running = False
        self._receive_thread = None

        # Bandwidth shaping profile index (sent to device_agents via SystemInfo)
        # When per-device differentiation is enabled, _device_bw_map overrides
        # the global bandwidth_setting for each device.
        self.bandwidth_setting: int = 0
        self._device_bw_map: Dict[str, int] = {}
        if self.manifest:
            for c in self.manifest.get("clients", []):
                bw_id = c.get("bandwidth_setting_id")
                if bw_id is not None:
                    self._device_bw_map[c["device_name"]] = int(bw_id)

        # ── FL relay gateway ────────────────────────────────────────────
        # In relay mode, the REP socket receives both control messages
        # (DEV_AD) and forwarded FL relay envelopes (from device_agent's
        # ForwardToController).  Relay data is pushed here for
        # ZMQServerCommManager to consume.
        self._fl_relay_queue: Queue = Queue()
        # Lock for PUB socket access (shared between control and relay)
        self._pub_lock = threading.Lock()

        mode_str = "PROTO" if proto_mode else "PICKLE"
        logger.info(
            f"ControlPlaneService initialized ({mode_str} mode): "
            f"REP={self.host}:{self.rep_port}, PUB={self.host}:{self.pub_port}"
        )
    
    def load_manifest(self, manifest_path: str) -> None:
        """Load client manifest from JSON file."""
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        logger.info(f"Loaded manifest with {len(self.manifest.get('clients', []))} clients")
        # Rebuild per-device bandwidth map
        self._device_bw_map = {}
        for c in self.manifest.get("clients", []):
            bw_id = c.get("bandwidth_setting_id")
            if bw_id is not None:
                self._device_bw_map[c["device_name"]] = int(bw_id)
    
    def start(self) -> None:
        """Start the control plane service."""
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required. Install with: pip install pyzmq")
        
        if self._running:
            logger.warning("ControlPlaneService already running")
            return
        
        self._context = zmq.Context()
        
        # REP socket for receiving device advertisements
        self._rep_socket = self._context.socket(zmq.REP)
        self._rep_socket.bind(f"tcp://{self.host}:{self.rep_port}")
        self._rep_socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        # PUB socket for broadcasting commands to devices
        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.bind(f"tcp://{self.host}:{self.pub_port}")
        
        self._running = True
        
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name="ControlPlane-Receive"
        )
        self._receive_thread.start()
        
        logger.info(
            f"ControlPlaneService started: "
            f"REP=tcp://{self.host}:{self.rep_port}, "
            f"PUB=tcp://{self.host}:{self.pub_port}"
        )
    
    def stop(self) -> None:
        """Stop the control plane service."""
        self._running = False
        
        if self._receive_thread:
            self._receive_thread.join(timeout=2.0)
        
        if self._rep_socket:
            self._rep_socket.close()
        if self._pub_socket:
            self._pub_socket.close()
        if self._context:
            self._context.term()
        
        logger.info("ControlPlaneService stopped")
    
    def _receive_loop(self) -> None:
        """Main receive loop for device messages and FL relay data."""
        import zmq
        from distributed.control_plane.proto_bridge import is_relay_data

        while self._running:
            try:
                try:
                    raw_msg = self._rep_socket.recv()
                except zmq.Again:
                    continue

                # ── Relay data: forwarded by device_agent's ForwardToController
                # The device_agent strips the "FWD_CTRL " prefix and sends the
                # raw relay_envelope.  First byte < 0x20 → binary relay data.
                if self.proto_mode and is_relay_data(raw_msg):
                    self._fl_relay_queue.put(raw_msg)
                    self._rep_socket.send(b"ACK")
                    logger.debug(
                        "Relay envelope received (%d bytes), queued for FL plane",
                        len(raw_msg),
                    )
                    continue

                # ── Control message (DEV_AD, etc.)
                try:
                    if self.proto_mode:
                        msg_type, payload = self._parse_proto_req(raw_msg)
                    else:
                        msg_type, payload = unpack_control_message(raw_msg)
                except Exception as e:
                    logger.warning(f"Failed to parse control message: {e}")
                    self._send_error_reply(str(e))
                    continue

                response_type, response_payload = self._handle_message(msg_type, payload)
                self._send_reply(response_type, response_payload)

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                try:
                    self._send_error_reply(str(e))
                except Exception:
                    pass

    # ── Proto-mode REQ parsing / reply encoding ──────────────────────────

    def _parse_proto_req(self, raw: bytes):
        from distributed.control_plane.proto_bridge import (
            decode_req_frame,
            device_info_from_proto,
        )
        python_type, proto_payload = decode_req_frame(raw)
        if python_type == ControlMessageType.DEVICE_ADVERTISEMENT.value:
            payload = device_info_from_proto(proto_payload)
        else:
            payload = {"raw": proto_payload}
        return python_type, payload

    def _send_reply(self, msg_type: ControlMessageType, payload: dict) -> None:
        if self.proto_mode:
            self._send_proto_reply(msg_type, payload)
        else:
            reply = pack_control_message(msg_type.value, payload)
            self._rep_socket.send(reply)

    def _send_proto_reply(self, msg_type: ControlMessageType, payload: dict) -> None:
        from distributed.control_plane.proto_bridge import system_info_to_proto
        # The C++ device_agent parses the raw reply bytes directly as a
        # protobuf message (e.g. SystemInfo.ParseFromString(reply)).  It does
        # NOT expect a "MSG_TYPE <payload>" prefix on the REP reply — only PUB
        # frames carry that prefix.
        if msg_type == ControlMessageType.SYSTEM_INFO:
            self._rep_socket.send(system_info_to_proto(payload))
        else:
            self._rep_socket.send(b"")

    def _send_error_reply(self, error_msg: str) -> None:
        """Best-effort error reply that respects REQ/REP discipline."""
        if self.proto_mode:
            self._rep_socket.send(b"ERROR " + error_msg.encode("utf-8"))
        else:
            reply = pack_control_message(
                ControlMessageType.ERROR.value, {"error": error_msg}
            )
            self._rep_socket.send(reply)

    def _handle_message(self, msg_type: str, payload: dict) -> tuple:
        if msg_type == ControlMessageType.DEVICE_ADVERTISEMENT.value:
            return self._handle_device_advertisement(payload)
        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return ControlMessageType.ERROR, {"error": f"Unknown message type: {msg_type}"}
    
    def _handle_device_advertisement(self, payload: dict) -> tuple:
        """Handle DEVICE_ADVERTISEMENT from a device agent."""
        try:
            device_info = DeviceInfo.from_dict(payload)
        except Exception as e:
            logger.error(f"Failed to parse DeviceInfo: {e}")
            return ControlMessageType.ERROR, {"error": str(e)}
        
        # Validate against manifest if available
        manifest_valid = True
        if self.manifest:
            manifest_valid = self._validate_device_manifest(device_info)
        
        # Register device
        with self._devices_lock:
            now = time.time()
            if device_info.name in self.devices:
                self.devices[device_info.name].info = device_info
                self.devices[device_info.name].last_seen = now
                self.devices[device_info.name].is_healthy = manifest_valid
                logger.info(f"Device re-registered: {device_info.name}")
            else:
                self.devices[device_info.name] = RegisteredDevice(
                    info=device_info,
                    registered_at=now,
                    last_seen=now,
                    is_healthy=manifest_valid,
                )
                logger.info(
                    f"Device registered: {device_info.name} "
                    f"(type={device_info.device_type}, ip={device_info.ip_address})"
                )
        
        # Per-device bandwidth_setting from manifest, fall back to global
        device_bw = self._device_bw_map.get(
            device_info.name, self.bandwidth_setting
        )

        # Return system info response
        system_info = {
            "name": "fedlora_distributed",
            "experiment": "fedit_phase1",
            "ctrl_ip_address": self.host,
            "manifest_valid": manifest_valid,
            "bandwidth_setting": device_bw,
        }
        logger.info(
            "DEV_AD reply to %s: bandwidth_setting=%d",
            device_info.name, device_bw,
        )
        return ControlMessageType.SYSTEM_INFO, system_info
    
    def _validate_device_manifest(self, device_info: DeviceInfo) -> bool:
        """Validate that device matches manifest expectations."""
        if not self.manifest:
            return True
        
        clients = self.manifest.get('clients', [])
        for client in clients:
            if client.get('device_name') == device_info.name:
                logger.debug(f"Device {device_info.name} matches manifest entry")
                return True
        
        logger.warning(f"Device {device_info.name} not found in manifest")
        return False
    
    def send_container_start(
        self,
        device_name: str,
        container_config: ContainerConfig,
    ) -> bool:
        """Send CONTAINER_START to a device via PUB socket."""
        if not self._pub_socket:
            logger.error("Service not started")
            return False

        try:
            if self.proto_mode:
                frame = self._encode_pub_container_start(device_name, container_config)
            else:
                topic = f"{device_name}|"
                msg = pack_control_message(
                    ControlMessageType.CONTAINER_START.value,
                    container_config.to_dict(),
                )
                frame = topic.encode('utf-8') + msg

            with self._pub_lock:
                self._pub_socket.send(frame)
            logger.info(f"Sent CONTAINER_START to {device_name}: {container_config.name}")

            with self._devices_lock:
                if device_name in self.devices:
                    self.devices[device_name].containers[container_config.name] = container_config

            return True
        except Exception as e:
            logger.error(f"Failed to send CONTAINER_START: {e}")
            return False

    def _encode_pub_container_start(self, device_name: str, cc: ContainerConfig) -> bytes:
        from distributed.control_plane.proto_bridge import (
            container_config_to_proto,
            encode_pub_frame,
        )
        proto_bytes = container_config_to_proto(cc.to_dict())
        return encode_pub_frame(
            device_name,
            ControlMessageType.CONTAINER_START.value,
            proto_bytes,
        )

    def send_container_stop(self, device_name: str, container_name: str) -> bool:
        """Send CONTAINER_STOP to a device via PUB socket."""
        if not self._pub_socket:
            logger.error("Service not started")
            return False

        try:
            if self.proto_mode:
                from distributed.control_plane.proto_bridge import (
                    container_signal_to_proto,
                    encode_pub_frame,
                )
                proto_bytes = container_signal_to_proto(container_name)
                frame = encode_pub_frame(
                    device_name,
                    ControlMessageType.CONTAINER_STOP.value,
                    proto_bytes,
                )
            else:
                topic = f"{device_name}|"
                signal = ContainerSignal(name=container_name)
                msg = pack_control_message(
                    ControlMessageType.CONTAINER_STOP.value,
                    signal.to_dict(),
                )
                frame = topic.encode('utf-8') + msg

            with self._pub_lock:
                self._pub_socket.send(frame)
            logger.info(f"Sent CONTAINER_STOP to {device_name}: {container_name}")

            with self._devices_lock:
                if device_name in self.devices:
                    self.devices[device_name].containers.pop(container_name, None)

            return True
        except Exception as e:
            logger.error(f"Failed to send CONTAINER_STOP: {e}")
            return False
    
    # ── FL relay gateway methods ──────────────────────────────────────────

    def get_fl_relay_queue(self) -> Queue:
        """
        Return the queue into which the REP handler pushes relay envelopes
        forwarded by device_agents.  ZMQServerCommManager reads from this
        instead of binding its own REP socket in relay mode.
        """
        return self._fl_relay_queue

    def publish_fl_relay(
        self,
        device_name: str,
        container_name: str,
        relay_payload: bytes,
    ) -> None:
        """
        Publish an FL payload through this service's PUB socket so the
        device_agent can relay it to the target container.

        Wire format:
          b"{device_name}| FWD_CONT {PackagedMsg_proto}"

        Called by ZMQServerCommManager.send() in relay mode.
        """
        from distributed.control_plane.proto_bridge import encode_relay_pub_frame

        frame = encode_relay_pub_frame(device_name, container_name, relay_payload)
        with self._pub_lock:
            self._pub_socket.send(frame)
        logger.debug(
            "Published FL relay to %s/%s (%d bytes)",
            device_name, container_name, len(relay_payload),
        )

    def get_registered_devices(self) -> Dict[str, RegisteredDevice]:
        """Get a copy of the device registry."""
        with self._devices_lock:
            return dict(self.devices)
    
    def get_device(self, device_name: str) -> Optional[RegisteredDevice]:
        """Get a specific device by name."""
        with self._devices_lock:
            return self.devices.get(device_name)
    
    def is_device_registered(self, device_name: str) -> bool:
        """Check if a device is registered."""
        with self._devices_lock:
            return device_name in self.devices
    
    def wait_for_devices(
        self,
        device_names: List[str],
        timeout: float = 60.0,
        poll_interval: float = 1.0,
    ) -> bool:
        """Wait for specific devices to register."""
        start_time = time.time()
        device_set = set(device_names)
        
        while time.time() - start_time < timeout:
            with self._devices_lock:
                registered = set(self.devices.keys())
            
            if device_set.issubset(registered):
                logger.info(f"All {len(device_names)} expected devices registered")
                return True
            
            missing = device_set - registered
            logger.debug(f"Waiting for devices: {missing}")
            time.sleep(poll_interval)
        
        logger.warning(f"Timeout waiting for devices. Missing: {device_set - registered}")
        return False
    
    def build_client_routing_table(self) -> Dict[int, Dict[str, str]]:
        """
        Build a client routing table from manifest for FL server use.
        
        Returns:
            Dict mapping client_id -> {device_name, container_name}
        """
        if not self.manifest:
            return {}
        
        routing = {}
        for client in self.manifest.get('clients', []):
            client_id = client['client_id']
            routing[client_id] = {
                'device_name': client['device_name'],
                'container_name': client['container_name'],
            }
        return routing

    def build_container_config(
        self,
        client_entry: Dict[str, Any],
        *,
        config_path: str,
        device_agent_host: str = "localhost",
        device_agent_port_offset: int = 0,
        partition_root: Optional[str] = None,
    ) -> ContainerConfig:
        """
        Same as ``container_config_from_manifest_client`` but keeps launch
        configuration derived from the manifest currently loaded on this
        service instance (optional consistency check in callers).
        """
        if self.manifest:
            ids = {c.get("client_id") for c in self.manifest.get("clients", [])}
            if client_entry.get("client_id") not in ids:
                logger.warning(
                    "client_id %s not found in loaded manifest client list %s",
                    client_entry.get("client_id"),
                    ids,
                )
        return container_config_from_manifest_client(
            client_entry,
            config_path=config_path,
            device_agent_host=device_agent_host,
            device_agent_port_offset=device_agent_port_offset,
            partition_root=partition_root,
        )


def main():
    """Standalone control-plane entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='FedLoRA Control Plane Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port-offset', type=int, default=0, help='Port offset')
    parser.add_argument('--manifest', default=None, help='Path to client manifest')
    parser.add_argument('--proto', action='store_true',
                        help='Use protobuf wire format for real device_agent')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    service = ControlPlaneService(
        host=args.host,
        port_offset=args.port_offset,
        manifest_path=args.manifest,
        proto_mode=args.proto,
    )

    try:
        service.start()
        mode = "PROTO" if args.proto else "PICKLE"
        logger.info(f"Control Plane Service running ({mode} mode). Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        service.stop()


if __name__ == '__main__':
    main()
