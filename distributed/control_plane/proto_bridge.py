"""
Protobuf bridge between Python ControlPlaneService and C++ device_agent.

The C++ device_agent (AegisGov) speaks protobuf over ZMQ with abbreviated
message-type strings and a specific PUB/SUB topic format.  The Python
ControlPlaneService uses pickle+zlib with full message-type strings.

This bridge translates between the two:
  - Python-side ContainerConfig/DeviceInfo → protobuf bytes
  - Protobuf bytes from device_agent → Python-side dicts
  - MSG_TYPE string translation (CONTAINER_START ↔ CONT_START)
  - PUB topic format: "{name}| {MSG_TYPE} {protobuf_bytes}"
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

from distributed.proto import controlmessages_pb2 as pb

# Raw bytes is the default relay encoding.  The proto field is `bytes`
# and the C++ device_agent stores SerializeAsString() output (binary),
# so raw bytes flow through without issues.
#
# Legacy base64 wrapping (33% overhead) can be re-enabled for debugging
# by setting FEDLORA_RELAY_BASE64=1.  The client auto-detects both
# encodings, so only the server side needs the flag.
_FORCE_BASE64_ENV = "FEDLORA_RELAY_BASE64"


def relay_uses_raw_bytes() -> bool:
    return os.environ.get(_FORCE_BASE64_ENV, "0").lower() not in ("1", "true", "yes", "on")

logger = logging.getLogger(__name__)

# ── MSG_TYPE translation ────────────────────────────────────────────────

_PYTHON_TO_WIRE: Dict[str, str] = {
    "CONTAINER_START": "CONT_START",
    "DEVICE_ADVERTISEMENT": "DEV_AD",
    "CONTAINER_STOP": "CONT_STOP",
    "CONNECT_DEVICE": "CON_DEV",
    "TO_CONTROLLER": "FWD_CTRL",
    "TO_CONTAINER": "FWD_CONT",
}

_WIRE_TO_PYTHON: Dict[str, str] = {v: k for k, v in _PYTHON_TO_WIRE.items()}


def msg_type_to_wire(python_type: str) -> str:
    return _PYTHON_TO_WIRE.get(python_type, python_type)


def msg_type_from_wire(wire_type: str) -> str:
    return _WIRE_TO_PYTHON.get(wire_type, wire_type)


# ── Device type mapping ─────────────────────────────────────────────────

_DEVICE_TYPE_TO_ENUM: Dict[str, int] = {
    "agxorin": pb.OrinAGX,
    "orinagx": pb.OrinAGX,
    "agxavier": pb.AGXXavier,
}

_ENUM_TO_DEVICE_TYPE: Dict[int, str] = {
    pb.OrinAGX: "agxorin",
    pb.AGXXavier: "agxavier",
}


def device_type_to_enum(device_type_str: str) -> int:
    return _DEVICE_TYPE_TO_ENUM.get(device_type_str.lower(), 0)


def device_type_from_enum(enum_val: int) -> str:
    return _ENUM_TO_DEVICE_TYPE.get(enum_val, "unknown")


# ── ContainerConfig: Python → protobuf ──────────────────────────────────

def container_config_to_proto(
    python_config: dict,
) -> bytes:
    """
    Convert a Python ContainerConfig dict to protobuf ContainerConfig bytes.

    FedLoRA-specific fields (client_id, partition_path, config_path,
    device_agent_host, device_agent_port_offset) are packed into
    ``json_config`` since the protobuf schema has no matching fields.

    The ``executable`` field receives the full worker launch command.
    """
    pb_msg = pb.ContainerConfig()
    pb_msg.name = python_config.get("name", "")
    pb_msg.docker_tag = python_config.get("docker_tag", "")
    pb_msg.device = python_config.get("device", 0)
    pb_msg.control_port = python_config.get("control_port", 0)

    fedlora_args = {
        "client_id": python_config.get("client_id", -1),
        "container_name": python_config.get("name", ""),
        "partition_path": python_config.get("partition_path", ""),
        "config_path": python_config.get("config_path", ""),
        "device_agent_host": python_config.get("device_agent_host", "localhost"),
        "device_agent_port_offset": python_config.get("device_agent_port_offset", 0),
        "nic": python_config.get("nic", ""),
        "shared_nic": python_config.get("shared_nic", False),
    }
    extra = python_config.get("extra_args")
    if extra:
        fedlora_args["extra_args"] = extra

    pb_msg.json_config = json.dumps(fedlora_args)

    # AegisGov's runCompose() passes EXECUTABLE unquoted in its system()
    # call, so it MUST be a single token (no spaces).  We use a launcher
    # script that reads START_STRING (= json_config) for the actual args.
    pb_msg.executable = "/workspace/launch_worker.sh"

    return pb_msg.SerializeToString()


def _build_worker_cli_args(fedlora_args: dict) -> str:
    parts = []
    cid = fedlora_args.get("client_id", -1)
    if cid >= 0:
        parts.append(f"--client-id {cid}")
    name = fedlora_args.get("container_name", "")
    if name:
        parts.append(f"--container-name {name}")
    pp = fedlora_args.get("partition_path", "")
    if pp:
        parts.append(f"--partition-path {pp}")
    cp = fedlora_args.get("config_path", "")
    if cp:
        parts.append(f"--config {cp}")
    dah = fedlora_args.get("device_agent_host", "localhost")
    parts.append(f"--device-agent-host {dah}")
    dapo = fedlora_args.get("device_agent_port_offset", 0)
    parts.append(f"--device-agent-port-offset {dapo}")

    nic = fedlora_args.get("nic", "")
    if nic:
        parts.append(f"--nic {nic}")
    if fedlora_args.get("shared_nic", False):
        parts.append("--shared-nic")

    extra = fedlora_args.get("extra_args", {})
    if isinstance(extra, dict):
        fl_host = extra.get("fl_server_host", "")
        if fl_host:
            parts.append(f"--fl-server-host {fl_host}")
        fl_port_offset = extra.get("fl_server_port_offset", 0)
        if fl_port_offset:
            parts.append(f"--fl-server-port-offset {fl_port_offset}")
        if extra.get("verbose"):
            parts.append("--verbose")
        config_opts = extra.get("config_opts", [])
        if config_opts:
            parts.append("--config-opts")
            parts.append(json.dumps(config_opts))

    return " ".join(parts)


def container_config_from_proto(data: bytes) -> dict:
    """Parse protobuf ContainerConfig bytes to a Python dict."""
    pb_msg = pb.ContainerConfig()
    pb_msg.ParseFromString(data)
    result = {
        "name": pb_msg.name,
        "docker_tag": pb_msg.docker_tag,
        "json_config": pb_msg.json_config,
        "executable": pb_msg.executable,
        "device": pb_msg.device,
        "control_port": pb_msg.control_port,
        "model_type": pb_msg.model_type,
        "input_shape": list(pb_msg.input_shape),
    }
    if pb_msg.json_config:
        try:
            result.update(json.loads(pb_msg.json_config))
        except json.JSONDecodeError:
            pass
    return result


# ── DeviceInfo: protobuf → Python ───────────────────────────────────────

def device_info_from_proto(data: bytes) -> dict:
    """Parse protobuf DeviceInfo bytes to a Python dict."""
    pb_msg = pb.DeviceInfo()
    pb_msg.ParseFromString(data)
    return {
        "name": pb_msg.name,
        "device_type": device_type_from_enum(pb_msg.type),
        "processors": pb_msg.processors,
        "memory": list(pb_msg.memory),
        "ip_address": pb_msg.ip_address,
        "agent_port_offset": pb_msg.agent_port_offset,
    }


def device_info_to_proto(python_info: dict) -> bytes:
    """Convert a Python DeviceInfo dict to protobuf bytes."""
    pb_msg = pb.DeviceInfo()
    pb_msg.name = python_info.get("name", "")
    pb_msg.type = device_type_to_enum(python_info.get("device_type", ""))
    pb_msg.processors = python_info.get("processors", 1)
    for m in python_info.get("memory", []):
        pb_msg.memory.append(m)
    pb_msg.ip_address = python_info.get("ip_address", "")
    pb_msg.agent_port_offset = python_info.get("agent_port_offset", 0)
    return pb_msg.SerializeToString()


# ── SystemInfo: Python → protobuf ───────────────────────────────────────

def system_info_to_proto(python_info: dict) -> bytes:
    """Convert a Python SystemInfo dict to protobuf bytes for DEV_AD reply."""
    pb_msg = pb.SystemInfo()
    pb_msg.name = python_info.get("name", "")
    pb_msg.experiment = python_info.get("experiment", "")
    pb_msg.ctrl_ip_address = python_info.get("ctrl_ip_address", "")
    pb_msg.bandwidth_setting = python_info.get("bandwidth_setting", 0)
    return pb_msg.SerializeToString()


def system_info_from_proto(data: bytes) -> dict:
    pb_msg = pb.SystemInfo()
    pb_msg.ParseFromString(data)
    return {
        "name": pb_msg.name,
        "experiment": pb_msg.experiment,
        "ctrl_ip_address": pb_msg.ctrl_ip_address,
        "bandwidth_setting": pb_msg.bandwidth_setting,
    }


# ── ContainerSignal: Python → protobuf ──────────────────────────────────

def container_signal_to_proto(name: str, forced: bool = False) -> bytes:
    pb_msg = pb.ContainerSignal()
    pb_msg.name = name
    pb_msg.forced = forced
    return pb_msg.SerializeToString()


# ── PUB/SUB wire format ─────────────────────────────────────────────────

def encode_pub_frame(
    device_name: str,
    msg_type_python: str,
    proto_payload: bytes,
) -> bytes:
    """
    Encode a PUB frame in the format the C++ device_agent expects:
      b"{device_name}| {WIRE_MSG_TYPE} {proto_bytes}"

    The device_agent subscribes to "{name}|" and splits the remainder
    on the first two spaces to get msg_type and payload.
    """
    wire_type = msg_type_to_wire(msg_type_python)
    topic = f"{device_name}| ".encode("utf-8")
    type_bytes = wire_type.encode("utf-8")
    return topic + type_bytes + b" " + proto_payload


def decode_pub_frame(frame: bytes) -> Tuple[str, str, bytes]:
    """
    Decode a PUB frame from the C++ device_agent.

    Returns (device_name, python_msg_type, proto_payload).
    """
    pipe_idx = frame.index(b"|")
    device_name = frame[:pipe_idx].decode("utf-8")
    remainder = frame[pipe_idx + 2:]  # skip "| "
    space_idx = remainder.index(b" ")
    wire_type = remainder[:space_idx].decode("utf-8")
    proto_payload = remainder[space_idx + 1:]
    return device_name, msg_type_from_wire(wire_type), proto_payload


def encode_req_reply(
    msg_type_python: str,
    proto_payload: bytes,
) -> bytes:
    """
    Encode a REQ/REP frame:  b"{WIRE_MSG_TYPE} {proto_bytes}"
    """
    wire_type = msg_type_to_wire(msg_type_python)
    return wire_type.encode("utf-8") + b" " + proto_payload


def decode_req_frame(frame: bytes) -> Tuple[str, bytes]:
    """
    Decode a REQ frame from the device_agent:  b"{WIRE_MSG_TYPE} {proto_bytes}"

    Returns (python_msg_type, proto_payload).
    """
    space_idx = frame.index(b" ")
    wire_type = frame[:space_idx].decode("utf-8")
    proto_payload = frame[space_idx + 1:]
    return msg_type_from_wire(wire_type), proto_payload


# ── FL relay support (device_agent data plane) ─────────────────────────
#
# The C++ device_agent relays FL traffic between workers and the server
# using the same controller REQ/REP and PUB/SUB sockets used for control.
#
# Uplink:  Worker → device_agent REP (60011+offset) "FWD_CTRL <payload>"
#          → device_agent strips prefix, forwards <payload> via REQ to
#          ControlPlaneService REP (60101+offset).
#
# Downlink: ControlPlaneService PUB (60102+offset)
#           → "device_name| FWD_CONT <PackagedMsg_proto>"
#           → device_agent unpacks PackagedMsg, publishes
#           → "container_name| <inner_payload>" on device PUB (60012+offset).

# Wire type for relay data forwarded by device_agent's ForwardToController.
WIRE_FWD_CTRL = "FWD_CTRL"
WIRE_FWD_CONT = "FWD_CONT"


def is_relay_data(raw: bytes) -> bool:
    """
    Detect whether a frame received on the ControlPlaneService REP socket
    is FL relay data (forwarded by device_agent's ForwardToController)
    vs. a control message (like DEV_AD).

    Control messages start with an ASCII wire-type string (e.g. "DEV_AD ").
    Relay envelopes start with a binary byte (msg_type < 0x20).

    This works because ForwardToController strips the "FWD_CTRL " prefix
    and sends the raw relay_envelope, whose first byte is the binary
    MessageType (0x01 for TO_CONTROLLER).
    """
    if not raw:
        return False
    return raw[0] < 0x20


def packaged_msg_to_proto(target_name: str, payload: bytes) -> bytes:
    """
    Wrap an FL relay payload in a PackagedMsg protobuf for the FWD_CONT
    downlink path.  The device_agent's ForwardToContainer() parses this
    to route the payload to the named container.

    Both our proto and the AegisGov C++ proto use ``bytes`` for the
    payload field, so raw binary flows through without issues.

    Raw bytes is the default.  Legacy base64 wrapping (33% overhead)
    can be re-enabled for debugging via ``FEDLORA_RELAY_BASE64=1``.
    """
    msg = pb.PackagedMsg()
    msg.target_name = target_name
    if relay_uses_raw_bytes():
        msg.payload = payload
    else:
        import base64
        msg.payload = base64.b64encode(payload)
    return msg.SerializeToString()


def encode_relay_pub_frame(
    device_name: str,
    container_name: str,
    relay_payload: bytes,
) -> bytes:
    """
    Encode a PUB frame for FL payload relay to a specific container
    through the device_agent.

    Wire format:
      b"{device_name}| FWD_CONT {PackagedMsg_proto}"

    The device_agent subscribes to "{device_name}|", receives this,
    dispatches to ForwardToContainer which unpacks PackagedMsg and
    publishes "{container_name}| {inner_payload}" to the container.
    """
    pkg_bytes = packaged_msg_to_proto(container_name, relay_payload)
    return encode_pub_frame(device_name, "TO_CONTAINER", pkg_bytes)
