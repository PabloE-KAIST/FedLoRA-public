"""Rank and index helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence



def resolve_client_key(config_local: Any, client_id: int) -> Optional[str]:
    """Resolve a client key from hetero config.

    Supports the patterns already present in your server logic:
    - 'Client_1' (1-indexed)
    - 'Client_0' (0-indexed)
    Also tolerates a plain integer/string key as a future convenience.
    """
    if not config_local:
        return None

    preferred_keys = [
        f'Client_{client_id}',
        f'Client_{client_id - 1}',
        client_id,
        str(client_id),
        client_id - 1,
        str(client_id - 1),
    ]
    for key in preferred_keys:
        if key in config_local:
            return key
    return None



def get_client_config(config_local: Any, client_id: int, default: Any = None) -> Any:
    """Return the config entry for a client from config_local."""
    key = resolve_client_key(config_local, client_id)
    if key is None:
        return default
    return config_local.get(key, default)



def infer_rank_from_client_rank_config(client_rank_config: Any, default: Optional[int] = None) -> Optional[int]:
    """Infer a logical rank from a module->rank mapping.

    Current code often uses a single uniform rank per target module. This helper
    mirrors that assumption while staying tolerant to malformed inputs.
    """
    if not client_rank_config:
        return default
    try:
        first_value = next(iter(client_rank_config.values()))
        return int(first_value)
    except Exception:
        return default



def logical_rank_from_indices(indices: Optional[Sequence[int]]) -> int:
    """Return the logical rank implied by an index list."""
    if indices is None:
        return 0
    return len(list(indices))



def indices_from_rank(rank: int) -> List[int]:
    """Return [0, 1, ..., rank-1] for a dense prefix rank."""
    rank = int(rank)
    if rank <= 0:
        return []
    return list(range(rank))



def normalize_indices(indices: Optional[Iterable[Any]], *, unique: bool = True, sort_values: bool = True) -> List[int]:
    """Convert any iterable of indices into a clean int list."""
    if indices is None:
        return []

    cleaned: List[int] = []
    seen = set()
    for value in indices:
        idx = int(value)
        if unique:
            if idx in seen:
                continue
            seen.add(idx)
        cleaned.append(idx)

    if sort_values:
        cleaned.sort()
    return cleaned



def validate_nonempty_indices(
    indices: Optional[Sequence[int]],
    *,
    client_id: Optional[int] = None,
    method_name: str = 'method',
    allow_empty: bool = False,
) -> List[int]:
    """Normalize indices and enforce the non-empty invariant when required."""
    normalized = normalize_indices(indices)
    if normalized or allow_empty:
        return normalized

    if client_id is None:
        raise RuntimeError(f'[{method_name}] Empty indices are not allowed.')
    raise RuntimeError(
        f'[{method_name}] Empty indices are not allowed for client {client_id}.')
