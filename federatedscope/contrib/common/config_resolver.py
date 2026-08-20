"""Shared config-resolution helpers.

These helpers are intentionally conservative.
They mirror the current behavior in client.py, server.py, and trainer_glue.py
without changing the message contract or method semantics.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional


DEFAULT_TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'down_proj', 'up_proj'
]


METHOD_ALIASES = {
    'adasparse-lora': 'adasparse_lora',
    'adasparse_lora': 'adasparse_lora',
    'adasparse-lorav2': 'adasparse_lorav2',
    'adasparse_lorav2': 'adasparse_lorav2',
    'adasparse-lorav3': 'adasparse_lorav3',
    'adasparse_lorav3': 'adasparse_lorav3',
    'fah-qlora': 'fah_qlora',
    'fah_qlora': 'fah_qlora',
    'hetlora': 'hetlora',
    'heterolora': 'heterolora',
    'hetero-lora': 'heterolora',
}


def normalize_method_name(method: Any) -> str:
    """Normalize federated method names so alias handling is centralized."""
    if method is None:
        return ''
    name = str(method).strip().lower()
    return METHOD_ALIASES.get(name, name)



def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default) if obj is not None else default



def is_vlm_task(cfg: Any) -> bool:
    data_type = _safe_getattr(_safe_getattr(cfg, 'data', None), 'type', '')
    if not isinstance(data_type, str):
        return False
    return '@vlm' in data_type.lower()


def is_glue_task(cfg: Any) -> bool:
    """Return True when the current run is a GLUE-style task.

    Mirrors the existing client/server checks but also tolerates a plain
    'glue' data type in case the pipeline uses a slightly different name.
    """
    data_type = _safe_getattr(_safe_getattr(cfg, 'data', None), 'type', '')
    if not isinstance(data_type, str):
        return False
    lowered = data_type.lower()
    return ('@glue' in lowered) or lowered.startswith('glue')



def get_adapter_root(cfg: Any, prefer_glue: Optional[bool] = None) -> Any:
    """Return the active adapter root, preferring the task-specific one.

    The return value is usually cfg.vlm.adapter, cfg.glue.adapter, or
    cfg.llm.adapter.  When none exists, returns None.
    """
    vlm_adapter = None
    if hasattr(cfg, 'vlm') and hasattr(cfg.vlm, 'adapter'):
        vlm_adapter = cfg.vlm.adapter

    glue_adapter = None
    if hasattr(cfg, 'glue') and hasattr(cfg.glue, 'adapter'):
        glue_adapter = cfg.glue.adapter

    llm_adapter = None
    if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter'):
        llm_adapter = cfg.llm.adapter

    if is_vlm_task(cfg):
        return vlm_adapter if vlm_adapter is not None else llm_adapter

    if prefer_glue is None:
        prefer_glue = is_glue_task(cfg)

    if prefer_glue:
        return glue_adapter if glue_adapter is not None else llm_adapter
    return llm_adapter if llm_adapter is not None else glue_adapter



def get_adapter_args_list(cfg: Any, prefer_glue: Optional[bool] = None) -> list:
    """Return adapter.args as a list, or an empty list when unavailable."""
    root = get_adapter_root(cfg, prefer_glue=prefer_glue)
    args = _safe_getattr(root, 'args', None)
    if args is None:
        return []
    try:
        return list(args)
    except TypeError:
        return []



def _get_named_feature_cfg(adapter_root: Any, feature_name: str, enabled_only: bool = True) -> Any:
    if adapter_root is None or not hasattr(adapter_root, feature_name):
        return None
    feature_cfg = getattr(adapter_root, feature_name)
    if enabled_only and not getattr(feature_cfg, 'enabled', False):
        return None
    return feature_cfg



def _get_method_scoped_cfg(
    cfg: Any,
    *,
    feature_name: str,
    allowed_methods: Optional[Iterable[str]] = None,
    enabled_only: bool = True,
    prefer_glue: Optional[bool] = None,
) -> Any:
    method = normalize_method_name(_safe_getattr(_safe_getattr(cfg, 'federate', None), 'method', ''))
    if allowed_methods is not None:
        normalized_allowed = {normalize_method_name(m) for m in allowed_methods}
        if method not in normalized_allowed:
            return None

    if prefer_glue is None:
        prefer_glue = is_glue_task(cfg)

    roots = []
    primary = get_adapter_root(cfg, prefer_glue=prefer_glue)
    secondary = get_adapter_root(cfg, prefer_glue=not prefer_glue)
    if primary is not None:
        roots.append(primary)
    if secondary is not None and secondary is not primary:
        roots.append(secondary)

    for root in roots:
        feature_cfg = _get_named_feature_cfg(root, feature_name, enabled_only=enabled_only)
        if feature_cfg is not None:
            return feature_cfg
    return None



def get_hetero_ranks_cfg(cfg: Any, enabled_only: bool = False) -> Any:
    """Return hetero_ranks config from GLUE or LLM adapter roots.

    We do not gate this on a single federated method because your current code
    uses hetero_ranks for HetLoRA, HeteroLoRA, and AdaSparse-compatible payload
    preparation when client-specific rank configs are available.
    """
    return _get_method_scoped_cfg(
        cfg,
        feature_name='hetero_ranks',
        allowed_methods=None,
        enabled_only=enabled_only,
    )



def get_hetlora_cfg(cfg: Any) -> Any:
    return _get_method_scoped_cfg(
        cfg,
        feature_name='hetlora',
        allowed_methods={'hetlora'},
        enabled_only=True,
    )



def get_adasparse_cfg(cfg: Any) -> Any:
    return _get_method_scoped_cfg(
        cfg,
        feature_name='adasparse_lora',
        allowed_methods={'adasparse_lora', 'adasparse-lora'},
        enabled_only=True,
    )



def get_adasparse_v2_cfg(cfg: Any) -> Any:
    return _get_method_scoped_cfg(
        cfg,
        feature_name='adasparse_lorav2',
        allowed_methods={'adasparse_lorav2', 'adasparse-lorav2'},
        enabled_only=True,
    )



def get_adasparse_v3_cfg(cfg: Any) -> Any:
    """Return AdaSparse-LoRAv3 config if method matches and is enabled."""
    return _get_method_scoped_cfg(
        cfg,
        feature_name='adasparse_lorav3',
        allowed_methods={'adasparse_lorav3', 'adasparse-lorav3'},
        enabled_only=True,
    )



def get_fah_cfg(cfg: Any) -> Any:
    """Return FAH config.

    Current code primarily uses cfg.llm.adapter.fah, but this helper also
    tolerates a future glue.adapter.fah location so later refactors do not have
    to duplicate the fallback logic.
    """
    return _get_method_scoped_cfg(
        cfg,
        feature_name='fah',
        allowed_methods={'fah_qlora', 'fah-qlora'},
        enabled_only=True,
    )



def get_active_hetero_config_local(cfg: Any) -> Any:
    """Resolve the active hetero_ranks.config_local.

    This preserves the current behavior: prefer the task-specific config,
    otherwise fall back to whichever one is defined.
    """
    vlm_config_local = None
    if hasattr(cfg, 'vlm') and hasattr(cfg.vlm, 'adapter') and \
            hasattr(cfg.vlm.adapter, 'hetero_ranks') and \
            hasattr(cfg.vlm.adapter.hetero_ranks, 'config_local'):
        vlm_config_local = cfg.vlm.adapter.hetero_ranks.config_local

    glue_config_local = None
    if hasattr(cfg, 'glue') and hasattr(cfg.glue, 'adapter') and \
            hasattr(cfg.glue.adapter, 'hetero_ranks') and \
            hasattr(cfg.glue.adapter.hetero_ranks, 'config_local'):
        glue_config_local = cfg.glue.adapter.hetero_ranks.config_local

    llm_config_local = None
    if hasattr(cfg, 'llm') and hasattr(cfg.llm, 'adapter') and \
            hasattr(cfg.llm.adapter, 'hetero_ranks') and \
            hasattr(cfg.llm.adapter.hetero_ranks, 'config_local'):
        llm_config_local = cfg.llm.adapter.hetero_ranks.config_local

    if is_vlm_task(cfg):
        return vlm_config_local or llm_config_local or glue_config_local
    if is_glue_task(cfg):
        return glue_config_local or llm_config_local or vlm_config_local
    return llm_config_local or glue_config_local or vlm_config_local



def get_effective_target_modules(cfg: Any, default: Optional[Iterable[str]] = None) -> list:
    """Resolve adapter target_modules from the preferred adapter root."""
    target_modules = list(default) if default is not None else list(DEFAULT_TARGET_MODULES)

    args_list = get_adapter_args_list(cfg)
    if args_list:
        first = args_list[0]
        if isinstance(first, dict):
            tm = first.get('target_modules', None)
            if tm:
                return list(tm)

    secondary_args_list = get_adapter_args_list(cfg, prefer_glue=not is_glue_task(cfg))
    if secondary_args_list:
        first = secondary_args_list[0]
        if isinstance(first, dict):
            tm = first.get('target_modules', None)
            if tm:
                return list(tm)

    return target_modules



def get_effective_max_rank(cfg: Any, default: int = 64) -> int:
    """Resolve max_rank from the preferred adapter root, then fallback root."""
    primary = get_adapter_root(cfg)
    secondary = get_adapter_root(cfg, prefer_glue=not is_glue_task(cfg))

    for root in [primary, secondary]:
        if root is not None and hasattr(root, 'max_rank'):
            return int(root.max_rank)
    return int(default)
