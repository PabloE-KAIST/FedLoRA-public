"""Shared payload helpers.

The goal here is not to redesign the message schema.
These functions only make the current wrapped-payload contract easier to read
and reuse from client.py and server.py.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Optional


PAYLOAD_KEYS = {
    'model_para',
    'client_rank_config',
    'fah_ranks',
    'bandwidth_info',
    'max_rank_lora_bytes',
    'adasparse_indices',
    'survivor_indices',
    'download_indices',
    'is_partial_downlink',
}



def is_wrapped_payload(content: Any) -> bool:
    """Return True when content looks like the current dict-based payload."""
    return isinstance(content, dict) and 'model_para' in content



def extract_model_para(content: Any, default: Any = None) -> Any:
    """Return model_para when the payload is wrapped, else return content."""
    if is_wrapped_payload(content):
        return content.get('model_para', default)
    return content if content is not None else default



def get_payload_field(content: Any, field_name: str, default: Any = None) -> Any:
    """Safely read a field from a wrapped payload."""
    if not is_wrapped_payload(content):
        return default
    return content.get(field_name, default)



def copy_non_model_fields(content: Any, *, deep: bool = True) -> Dict[str, Any]:
    """Copy all wrapped-payload fields except model_para."""
    if not is_wrapped_payload(content):
        return {}

    extra = {k: v for k, v in content.items() if k != 'model_para'}
    return deepcopy(extra) if deep else dict(extra)



def build_wrapped_model_payload(
    model_para: Any,
    extra_fields: Optional[Dict[str, Any]] = None,
    *,
    drop_none: bool = False,
) -> Dict[str, Any]:
    """Build a wrapped payload while preserving the current message contract."""
    payload = {'model_para': model_para}
    if extra_fields:
        for key, value in extra_fields.items():
            if drop_none and value is None:
                continue
            payload[key] = value
    return payload



def merge_wrapped_payload(
    base_payload: Any,
    updates: Optional[Dict[str, Any]] = None,
    *,
    preserve_model_para: bool = True,
) -> Dict[str, Any]:
    """Return a merged wrapped payload.

    This is useful when a server/client wants to add metadata fields to an
    existing wrapped payload without changing anything else.
    """
    model_para = extract_model_para(base_payload)
    extra = copy_non_model_fields(base_payload, deep=True)
    if updates:
        extra.update(updates)
    if preserve_model_para:
        return build_wrapped_model_payload(model_para, extra_fields=extra)
    payload = dict(extra)
    if model_para is not None:
        payload['model_para'] = model_para
    return payload



def select_known_payload_fields(content: Any, allowed_fields: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Return a shallow dict of recognized payload fields.

    By default this keeps all known metadata fields. It is handy when a call
    site wants to forward the current message contract but exclude any accidental
    local-only keys.
    """
    if not is_wrapped_payload(content):
        return {}

    keys = set(allowed_fields) if allowed_fields is not None else set(PAYLOAD_KEYS)
    return {k: v for k, v in content.items() if k in keys}
