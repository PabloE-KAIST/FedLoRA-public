"""
Communication managers for distributed FedLoRA deployment.

Provides ZeroMQ-based comm managers that can be injected into
existing FedLoRA Server and Client classes.

ARCHITECTURE NOTE:
The comm managers operate through device_agent relay boundaries,
NOT direct server-worker connections. See zmq_client_comm.py and
zmq_server_comm.py for details.

Components:
- ZMQClientCommManager: Worker-side comm manager (targets device_agent)
- ZMQServerCommManager: Server-side comm manager (FL payload plane)
- Serialization utilities for message transport

Heavy submodules (ZMQ managers) are loaded lazily so that
``from distributed.comm.serialization import ...`` via this package does not
require torch / pyzmq / full federatedscope at import time.
"""

from importlib import import_module
from typing import Any

from distributed.comm.serialization import (
    serialize_message,
    deserialize_message,
    serialize_any,
    deserialize_any,
    pack_relay_envelope,
    unpack_relay_envelope,
    pack_control_message,
    unpack_control_message,
    MessageType,
)

__all__ = [
    'serialize_message',
    'deserialize_message',
    'serialize_any',
    'deserialize_any',
    'pack_relay_envelope',
    'unpack_relay_envelope',
    'pack_control_message',
    'unpack_control_message',
    'MessageType',
    'ZMQClientCommManager',
    'ZMQServerCommManager',
]

_LAZY_ATTRS = {
    'ZMQClientCommManager': ('distributed.comm.zmq_client_comm', 'ZMQClientCommManager'),
    'ZMQServerCommManager': ('distributed.comm.zmq_server_comm', 'ZMQServerCommManager'),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        mod_path, attr = _LAZY_ATTRS[name]
        mod = import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
