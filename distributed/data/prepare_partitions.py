#!/usr/bin/env python3
"""
Prepare per-client data partitions for distributed FedLoRA (v2.3).

Uses the existing ``get_data()`` path with ``federate.mode=standalone`` so
federation splits match standalone semantics (same translator + splitter).

Writes one artifact per manifest client:
  - ``{output_dir}/client_{id}.pkl`` — partition payload (see ``--mode``)
  - ``{output_dir}/client_{id}.metadata.json`` — dataset/splitter/seed/hash

Modes:
  - ``full`` (default): pickle the ``ClientData`` object for that client.
  - ``index``: when per-client ``train_data`` / ``val_data`` / ``test_data``
    expose ``indices`` (``torch.utils.data.Subset``), write an index dict;
    otherwise falls back to ``full`` and logs a warning (index mode is the
    recommended portable baseline when indices exist).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _config_fingerprint(cfg) -> str:
    """Short stable fingerprint of merged config for metadata."""
    try:
        blob = cfg.dump()
    except Exception:
        blob = repr(cfg)
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()[:16]


def _maybe_subset_indices(ds) -> Optional[list]:
    from torch.utils.data import Subset

    if isinstance(ds, Subset):
        return list(ds.indices)
    return None


def _build_index_payload(client_data) -> Optional[dict]:
    train_i = _maybe_subset_indices(client_data.train_data)
    val_i = _maybe_subset_indices(client_data.val_data)
    test_i = _maybe_subset_indices(client_data.test_data)
    if train_i is None and val_i is None and test_i is None:
        return None
    out: Dict[str, Any] = {"format": "index_v1"}
    if train_i is not None:
        out["train_indices"] = train_i
    if val_i is not None:
        out["val_indices"] = val_i
    if test_i is not None:
        out["test_indices"] = test_i
    return out


def prepare_partitions(
    config_path: str,
    manifest_path: str,
    output_dir: str,
    mode: str = "full",
    seed: Optional[int] = None,
    model_type_override: Optional[str] = None,
    data_type_override: Optional[str] = None,
    opts: Optional[list] = None,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.data.base_data import StandaloneDataDict

    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    clients = manifest.get("clients", [])
    if not clients:
        raise ValueError("Manifest has no clients[]")

    cfg = global_cfg.clone()
    cfg.merge_from_file(config_path)
    cfg.federate.mode = "standalone"
    if seed is not None:
        cfg.seed = seed
    if model_type_override is not None or data_type_override is not None:
        try:
            cfg.defrost()
        except Exception:
            pass
        if model_type_override is not None:
            cfg.model.type = model_type_override
        if data_type_override is not None:
            cfg.data.type = data_type_override
    if opts:
        try:
            cfg.defrost()
        except Exception:
            pass
        cfg.merge_from_list(list(opts))
    cfg.freeze()

    data, _modified = get_data(cfg.clone())
    if not isinstance(data, StandaloneDataDict) and not (
        isinstance(data, dict) and all(isinstance(k, int) for k in data.keys())
    ):
        raise TypeError(
            f"Expected StandaloneDataDict from get_data in standalone mode, got {type(data)}"
        )

    fp = _config_fingerprint(cfg)
    data_type = str(cfg.data.type)
    splitter = str(getattr(cfg.data, "splitter", ""))

    for entry in clients:
        cid = int(entry["client_id"])
        if cid not in data:
            raise KeyError(f"client_id {cid} not in federated data keys {sorted(data.keys())}")

        client_data = data[cid]
        metadata = {
            "client_id": cid,
            "dataset_type": data_type,
            "splitter": splitter,
            "seed_used": int(cfg.seed),
            "config_fingerprint_sha256_16": fp,
            "source_config": os.path.abspath(config_path),
            "manifest_client": entry,
            "artifact_mode_requested": mode,
        }

        payload: Any
        artifact_mode = mode
        if mode == "index":
            idx = _build_index_payload(client_data)
            if idx is None:
                logger.warning(
                    "Index mode requested but no Subset.indices found for client %s; "
                    "writing full ClientData pickle instead. Prefer index artifacts when "
                    "the underlying datasets are torch Subsets.",
                    cid,
                )
                payload = client_data
                artifact_mode = "full_fallback_from_index"
            else:
                payload = idx
        else:
            payload = client_data

        metadata["artifact_mode_effective"] = artifact_mode

        pkl_path = os.path.join(output_dir, f"client_{cid}.pkl")
        meta_path = os.path.join(output_dir, f"client_{cid}.metadata.json")

        with open(pkl_path, "wb") as out:
            pickle.dump(payload, out, protocol=pickle.HIGHEST_PROTOCOL)
        with open(meta_path, "w", encoding="utf-8") as out:
            json.dump(metadata, out, indent=2)
        logger.info("Wrote %s and %s", pkl_path, meta_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Prepare FedLoRA client partitions")
    parser.add_argument("--config", required=True, help="FedLoRA YAML (merged into global_cfg)")
    parser.add_argument("--manifest", required=True, help="client_manifest.json")
    parser.add_argument("--output-dir", required=True, help="Directory for client_*.pkl")
    parser.add_argument(
        "--mode",
        choices=("full", "index"),
        default="full",
        help="full=pickle ClientData; index=torch Subset indices when available",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed before freeze")
    parser.add_argument("--model-type", default=None, help="Override cfg.model.type (e.g. host-side path when config carries container-side path)")
    parser.add_argument("--data-type", default=None, help="Override cfg.data.type (e.g. 'qnli@glue')")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                        help="Extra 'KEY VALUE ...' cfg overrides, applied last (e.g. federate.client_num 9 data.splitter iid)")
    args = parser.parse_args()
    prepare_partitions(
        config_path=args.config,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        model_type_override=args.model_type,
        data_type_override=args.data_type,
        opts=args.opts,
    )


if __name__ == "__main__":
    main()
