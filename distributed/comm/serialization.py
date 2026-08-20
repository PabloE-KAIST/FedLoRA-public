"""
Serialization utilities for FedLoRA Message transport.

This module provides:
1. FL Message serialization (pickle + zlib)
2. Relay envelope packing/unpacking for device-agent routing

Wire Formats:

FL Payload:
    pickle(Message) -> zlib.compress() -> bytes

Relay Envelope (for device-agent routing):
    <msg_type:1 byte><source_len:2 bytes><source:N bytes><payload:M bytes>

Message Types:
- TO_CONTROLLER (0x01): Worker -> Device Agent -> Server
- FL_PAYLOAD (0x02): Server -> Device Agent -> Worker
- CONTROL_PLANE (0x10): Control plane messages
"""

import io
import pickle
import zlib
import struct
import logging
from typing import Any, Tuple
from enum import IntEnum

import torch

logger = logging.getLogger(__name__)

# Compression level (1-9, higher = more compression, slower)
COMPRESSION_LEVEL = 6


class MessageType(IntEnum):
    """Message types for relay envelope routing."""
    TO_CONTROLLER = 0x01    # Worker -> Server (via device_agent)
    FL_PAYLOAD = 0x02       # Server -> Worker (via device_agent)
    CONTROL_PLANE = 0x10    # Control plane messages
    ACK = 0x20              # Acknowledgment
    ERROR = 0xFF            # Error


def serialize_message(message: 'Message') -> bytes:
    """
    Serialize a FedLoRA Message to bytes for transport.
    Uses torch.save so that torch.load with map_location can remap
    CUDA devices on the receiving end.

    Args:
        message: A FedLoRA Message object

    Returns:
        Compressed bytes representation of the message
    """
    buf = io.BytesIO()
    torch.save(message, buf, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(buf.getvalue(), level=COMPRESSION_LEVEL)
    return compressed


def deserialize_message(data: bytes) -> 'Message':
    """
    Deserialize bytes back to a FedLoRA Message.
    Uses torch.load with map_location to remap CUDA tensors to cuda:0,
    so workers with fewer GPUs can receive models serialized on
    higher-numbered GPUs (e.g. server on cuda:2, worker has only cuda:0).

    Args:
        data: Compressed bytes from transport

    Returns:
        The original FedLoRA Message object
    """
    decompressed = zlib.decompress(data)
    buf = io.BytesIO(decompressed)
    message = torch.load(buf, map_location='cpu', weights_only=False)
    return message


def serialize_any(obj: Any) -> bytes:
    """
    Serialize any Python object to bytes.
    Used for control plane messages.
    
    Args:
        obj: Any Python object
        
    Returns:
        Compressed bytes representation
    """
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(pickled, level=COMPRESSION_LEVEL)
    return compressed


def deserialize_any(data: bytes) -> Any:
    """
    Deserialize bytes back to a Python object.
    Used for control plane messages.
    
    Args:
        data: Compressed bytes
        
    Returns:
        The original Python object
    """
    decompressed = zlib.decompress(data)
    return pickle.loads(decompressed)


def pack_relay_envelope(
    msg_type: MessageType,
    source_container: str,
    payload: bytes,
) -> bytes:
    """
    Pack a message into a relay envelope for device-agent routing.
    
    Envelope format:
        <msg_type:1 byte><source_len:2 bytes><source:N bytes><payload:M bytes>
    
    Args:
        msg_type: Message type for routing
        source_container: Source container name
        payload: Serialized FL message or other payload
        
    Returns:
        Packed envelope bytes
    """
    source_bytes = source_container.encode('utf-8')
    source_len = len(source_bytes)
    
    # Pack: 1 byte msg_type + 2 bytes source_len + source + payload
    header = struct.pack('!BH', int(msg_type), source_len)
    return header + source_bytes + payload


def unpack_relay_envelope(envelope: bytes) -> Tuple[MessageType, bytes]:
    """
    Unpack a relay envelope to extract message type and payload.
    
    Args:
        envelope: Packed envelope bytes
        
    Returns:
        Tuple of (msg_type, payload)
    """
    # Unpack header
    header_size = 3  # 1 byte msg_type + 2 bytes source_len
    msg_type_int, source_len = struct.unpack('!BH', envelope[:header_size])
    
    # Extract source container name (for logging/debugging)
    source_start = header_size
    source_end = source_start + source_len
    # source_container = envelope[source_start:source_end].decode('utf-8')
    
    # Extract payload
    payload = envelope[source_end:]
    
    return MessageType(msg_type_int), payload


def pack_control_message(
    msg_type: str,
    payload: dict,
) -> bytes:
    """
    Pack a control plane message.
    
    Format matches device_agent expectation:
        "<MSG_TYPE> <serialized_payload>"
    
    Args:
        msg_type: Control message type string (e.g., "DEVICE_ADVERTISEMENT")
        payload: Payload dict to serialize
        
    Returns:
        Packed message bytes
    """
    payload_bytes = serialize_any(payload)
    type_bytes = msg_type.encode('utf-8')
    return type_bytes + b' ' + payload_bytes


def unpack_control_message(data: bytes) -> Tuple[str, dict]:
    """
    Unpack a control plane message.
    
    Args:
        data: Packed control message bytes
        
    Returns:
        Tuple of (msg_type, payload_dict)
    """
    # Find space separator
    space_idx = data.find(b' ')
    if space_idx < 0:
        raise ValueError("Invalid control message format: no space separator")
    
    msg_type = data[:space_idx].decode('utf-8')
    payload_bytes = data[space_idx + 1:]
    payload = deserialize_any(payload_bytes)
    
    return msg_type, payload
