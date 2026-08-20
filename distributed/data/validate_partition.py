#!/usr/bin/env python3
"""
Validate client partition artifacts (v2.3).

For each ``client_*.pkl`` under an output directory (or an explicit list):
  - Unpickle the artifact
  - Classify as full ``ClientData`` vs index-dict vs unknown
  - Report pass/fail per client

If full-object unpickling fails for some environments, the report states that
**index-based artifacts** should be used as the phase-1 portable baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _classify(obj: Any) -> str:
    try:
        from federatedscope.core.data.base_data import ClientData
    except Exception:
        ClientData = ()  # type: ignore

    if isinstance(obj, dict) and obj.get("format") == "index_v1":
        return "index_v1"
    if isinstance(obj, dict) and "train_indices" in obj:
        return "index_legacy"
    if ClientData and isinstance(obj, ClientData):
        return "full_clientdata"
    return f"unknown:{type(obj).__name__}"


def validate_directory(
    partition_dir: str,
    expect_metadata: bool = True,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate all ``client_*.pkl`` in a directory.

    Returns:
        (all_ok, rows) where each row is a result dict suitable for JSON export.
    """
    rows: List[Dict[str, Any]] = []
    all_ok = True
    base = Path(partition_dir)
    if not base.is_dir():
        raise NotADirectoryError(partition_dir)

    for pkl in sorted(base.glob("client_*.pkl")):
        row: Dict[str, Any] = {"file": str(pkl), "ok": False}
        meta_path = pkl.with_suffix(".metadata.json")
        if expect_metadata and not meta_path.exists():
            row["error"] = "missing_metadata_json"
            all_ok = False
            rows.append(row)
            continue
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    row["metadata"] = json.load(f)
            except Exception as e:
                row["metadata_error"] = str(e)

        try:
            with open(pkl, "rb") as f:
                obj = pickle.load(f)
            row["kind"] = _classify(obj)
            row["ok"] = row["kind"] in ("full_clientdata", "index_v1", "index_legacy")
            if row["kind"] == "unknown":
                row["error"] = "unrecognized_payload"
                all_ok = False
            if row["kind"] == "full_clientdata":
                row["note"] = (
                    "Full ClientData pickle validated for round-trip load only; "
                    "for cross-machine portability prefer index_v1 when available."
                )
        except Exception as e:
            row["error"] = f"unpickle_failed:{e}"
            row["recommendation"] = (
                "Use index-based partition artifacts (see prepare_partitions --mode index) "
                "when full-object pickles are not robust across environments."
            )
            all_ok = False
        rows.append(row)

    if not rows:
        all_ok = False
        rows.append({"file": str(base), "ok": False, "error": "no_client_*.pkl found"})

    return all_ok, rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Validate partition artifacts")
    parser.add_argument("--partition-dir", required=True, help="Directory with client_*.pkl")
    parser.add_argument(
        "--json-report",
        default="",
        help="Optional path to write structured JSON report for Gate 4/5 review",
    )
    parser.add_argument(
        "--no-require-metadata",
        action="store_true",
        help="Do not require sidecar .metadata.json files",
    )
    args = parser.parse_args()

    ok, rows = validate_directory(
        args.partition_dir,
        expect_metadata=not args.no_require_metadata,
    )
    print("Partition validation summary")
    print("------------------------------")
    for r in rows:
        status = "PASS" if r.get("ok") else "FAIL"
        print(f"[{status}] {r.get('file')}")
        if "kind" in r:
            print(f"        kind={r['kind']}")
        if "error" in r:
            print(f"        error={r['error']}")
        if "recommendation" in r:
            print(f"        recommendation={r['recommendation']}")
        if "note" in r:
            print(f"        note={r['note']}")
    print("------------------------------")
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")

    if args.json_report:
        with open(args.json_report, "w", encoding="utf-8") as f:
            json.dump({"ok": ok, "results": rows}, f, indent=2)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
