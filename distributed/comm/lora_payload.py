"""
LoRA-adapter-only payload helpers for the distributed wire path.

Priority 2 of ``docs/ARCHITECTURE.md``: only the LoRA
adapter parameters (plus any PEFT ``modules_to_save`` wrapped heads such
as the classifier) need to cross the wire. Frozen backbone weights are
identical on every device and waste bandwidth.

This module is intentionally stateless and transport-agnostic so that it
can be reused from both the server and client comm managers, and from
offline tests.

Activation
----------
The filter is a *wire-level* shim applied inside the ZMQ comm managers
just before ``serialize_message``. It is opt-in via the environment
variable ``FEDLORA_LORA_ONLY_PAYLOAD=1``. When disabled (the default)
this module is a no-op.

Correctness note
----------------
The receive-side trainer (``GeneralTorchTrainer.update``) merges the
received partial ``state_dict`` into the local ``state_dict`` before
loading with ``strict=True``, so dropping non-LoRA keys on the wire is
semantically safe *provided* both ends key their adapter params the
same way (PEFT LoRA wrapping). If the server is not PEFT-wrapped, the
filter will drop every key and yield an empty payload — enable the
filter only when both ends share the adapter config.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


ENV_FLAG = "FEDLORA_LORA_ONLY_PAYLOAD"

# Substrings that identify a LoRA adapter parameter in a PEFT state_dict.
# ``.lora_`` catches ``lora_A.default.weight``, ``lora_B.default.weight``,
# ``lora_embedding_A``, ``lora_magnitude_vector`` etc.
# ``modules_to_save.`` catches PEFT-wrapped trainable heads (classifier,
# pooler, etc. for SeqCls).
_DEFAULT_MARKERS: Tuple[str, ...] = (".lora_", "modules_to_save.")

# Message types whose ``content`` carries a state_dict payload that should
# be filtered. Other message types (join_in, join_in_info, finish, ack,
# metrics, address, ...) pass through untouched.
_FILTERED_MSG_TYPES: frozenset = frozenset({
    "model_para",
    "evaluate",
    "ss_model_para",
})


def is_enabled() -> bool:
    """Whether the LoRA-only payload filter is active this process."""
    return os.environ.get(ENV_FLAG, "0").lower() in ("1", "true", "yes", "on")


def is_lora_key(key: str, markers: Iterable[str] = _DEFAULT_MARKERS) -> bool:
    return any(marker in key for marker in markers)


def filter_state_dict_lora(
    state_dict: Mapping[str, Any],
    markers: Iterable[str] = _DEFAULT_MARKERS,
) -> Dict[str, Any]:
    """Return a dict containing only keys that look like LoRA adapter params."""
    return {k: v for k, v in state_dict.items() if is_lora_key(k, markers)}


def _looks_like_state_dict(obj: Any) -> bool:
    """Heuristic: a state_dict is a dict with string keys and tensor-ish values.

    We check the first few entries to avoid paying O(n) for large dicts.
    ``Tensor`` is not imported here so we duck-type on ``shape`` or ``size``.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    sampled = 0
    for k, v in obj.items():
        if not isinstance(k, str):
            return False
        if not (hasattr(v, "shape") or hasattr(v, "size")):
            return False
        sampled += 1
        if sampled >= 3:
            break
    return True


def _filter_if_state_dict(content: Any, markers) -> Tuple[Any, bool]:
    """Try to filter ``content`` in-place. Return (new_content, did_filter)."""
    if _looks_like_state_dict(content):
        filtered = filter_state_dict_lora(content, markers)
        return filtered, True
    return content, False


def filter_message_content(content: Any, markers=_DEFAULT_MARKERS) -> Tuple[Any, bool]:
    """Rewrite ``content`` so any embedded state_dict is LoRA-only.

    Handles the three shapes produced by FedLoRA/FederatedScope:

    1. Raw state_dict  (server broadcast of ``model_para``)
    2. ``(sample_size, state_dict)`` tuple (client upload)
    3. Wrapped dict with a ``model_para`` key and method metadata
       (FAH-QLoRA / HetLoRA / AdaSparse-LoRA variants)

    Returns the (possibly new) content and a bool indicating whether
    any filtering actually happened.
    """
    # Case 1: raw state_dict
    new_content, did = _filter_if_state_dict(content, markers)
    if did:
        return new_content, True

    # Case 2: (sample_size, state_dict) from client upload
    if isinstance(content, tuple) and len(content) == 2:
        head, payload = content
        if isinstance(head, (int, float)):
            new_payload, did = _filter_if_state_dict(payload, markers)
            if did:
                return (head, new_payload), True
            # payload may itself be a wrapped dict
            if isinstance(payload, dict) and "model_para" in payload:
                new_wrapped, wrapped_did = filter_message_content(payload, markers)
                if wrapped_did:
                    return (head, new_wrapped), True

    # Case 3: wrapped metadata dict — rewrite the ``model_para`` key in place
    if isinstance(content, dict) and "model_para" in content:
        inner = content["model_para"]
        new_inner, did = _filter_if_state_dict(inner, markers)
        if did:
            new_content = dict(content)
            new_content["model_para"] = new_inner
            return new_content, True

    return content, False


def should_filter_message(msg_type: Optional[str]) -> bool:
    if not msg_type:
        return False
    return msg_type in _FILTERED_MSG_TYPES


def estimate_bytes(obj: Any) -> int:
    """Approximate in-memory footprint of a state_dict-like payload.

    Uses tensor ``element_size * numel`` where available; falls back to
    pickle for unknown objects. Diagnostic only — not exact wire size.
    """
    try:
        if _looks_like_state_dict(obj):
            total = 0
            for v in obj.values():
                try:
                    total += int(v.element_size()) * int(v.numel())
                except Exception:
                    pass
            return total
        if isinstance(obj, tuple) and len(obj) == 2:
            return estimate_bytes(obj[1])
        if isinstance(obj, dict) and "model_para" in obj:
            return estimate_bytes(obj["model_para"])
    except Exception:
        pass

    # Fallback: pickle length (costly, only for unknown payloads).
    try:
        import pickle
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return -1


def _relay_wire_factor() -> float:
    """Approximate wire-expansion factor above the pickle+zlib payload.

    Raw bytes is the default (factor ~1.0).  Legacy base64 wrapping
    (factor 4/3) is only active when ``FEDLORA_RELAY_BASE64=1``.
    """
    force_base64 = os.environ.get("FEDLORA_RELAY_BASE64", "0").lower() in (
        "1", "true", "yes", "on",
    )
    return 4.0 / 3.0 if force_base64 else 1.0


def apply_if_enabled(message, *, role: str):
    """Filter ``message.content`` in place when the env flag is set.

    ``role`` is ``"server"`` or ``"client"`` for log attribution.
    Returns the (possibly mutated) message.
    """
    if not is_enabled():
        return message

    if not should_filter_message(getattr(message, "msg_type", None)):
        return message

    before_bytes = estimate_bytes(message.content)
    new_content, did = filter_message_content(message.content)
    if not did:
        return message

    after_bytes = estimate_bytes(new_content)
    wire_factor = _relay_wire_factor()
    if isinstance(new_content, dict) and len(new_content) == 0:
        logger.warning(
            "[%s] LoRA filter produced an EMPTY payload for msg_type=%s. "
            "Is the %s model PEFT-wrapped with matching adapter config?",
            role, message.msg_type, role,
        )
    else:
        logger.info(
            "[%s] LoRA filter: %s content %d -> %d bytes "
            "(%.1f%% reduction; ~%d bytes on the wire at %.2fx expansion)",
            role, message.msg_type, before_bytes, after_bytes,
            (100.0 * (1 - after_bytes / before_bytes)) if before_bytes > 0 else 0.0,
            int(after_bytes * wire_factor), wire_factor,
        )
    message.content = new_content
    return message
