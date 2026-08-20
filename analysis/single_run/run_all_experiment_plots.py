#!/usr/bin/env python3
"""
Run plotting utilities across all experiment result folders.

This script recursively finds strategy_* experiment directories under a root,
skips any path containing golden_reference, and runs the available plotting
utilities on the matching result files.

It is designed for a structure such as:
    exp/
      fedit/
        strategy_xxx__timestamp/
      hetlora/
        strategy_xxx__timestamp/
      adasparse_lora/
        strategy_xxx__timestamp/
      adasparse_lorav2/
        strategy_xxx__timestamp/
      adasparse_lorav3/
        strategy_xxx__timestamp/
      fahqlora/
        strategy_xxx__timestamp/
      golden_reference/
        ...

By default it looks for the plotting utilities in the same directory as this
script, but explicit paths can be provided via CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class CommandResult:
    name: str
    command: List[str]
    returncode: Optional[int]
    skipped: bool
    reason: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


@dataclass
class ExperimentResult:
    experiment_dir: str
    method: Optional[str]
    commands: List[CommandResult]


METHOD_TO_RANK_SCRIPT = {
    "fahqlora": "rank_evolution_fahqlora.py",
    "fah_qlora": "rank_evolution_fahqlora.py",
    "adasparse_lorav2": "rank_evolution_adasparsev2.py",
    "adasparse_lorav3": "rank_evolution_adasparsev3.py",
    "adasparse_lora": "rank_evolution_adasparse.py",
    "hetlora": "rank_evolution_hetlora.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run plotting utilities for all strategy_* experiment folders."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="exp",
        help="Root folder to scan recursively. Default: exp",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for subprocess calls. Default: current interpreter",
    )
    parser.add_argument(
        "--analysis-dir",
        default=None,
        help=(
            "Directory containing the plotting scripts. "
            "Default: same directory as this runner script"
        ),
    )
    parser.add_argument(
        "--bandwidth-script",
        default=None,
        help="Override path to bandwidth_history.py",
    )
    parser.add_argument(
        "--loss-script",
        default=None,
        help="Override path to loss_evolution_plots.py",
    )
    parser.add_argument(
        "--system-metrics-script",
        default=None,
        help="Override path to system_metrics.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run commands even if the target output subfolder already exists",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going even if a command fails",
    )
    parser.add_argument(
        "--strategy-prefix",
        default="strategy_",
        help="Directory-name prefix used to identify experiment folders. Default: strategy_",
    )
    parser.add_argument(
        "--bandwidth-per-client",
        action="store_true",
        help="Also generate individual bandwidth plots by forwarding --per-client-plots to bandwidth_history.py",
    )
    parser.add_argument(
        "--summary-file",
        default="plot_runner_summary.json",
        help="Name of the JSON summary file written under the scan root. Default: plot_runner_summary.json",
    )
    return parser.parse_args()


def resolve_scripts(args: argparse.Namespace) -> Dict[str, Path]:
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(__file__).resolve().parent

    scripts = {
        "bandwidth": Path(args.bandwidth_script) if args.bandwidth_script else analysis_dir / "bandwidth_history.py",
        "loss": Path(args.loss_script) if args.loss_script else analysis_dir / "loss_evolution_plots.py",
        "system_metrics": Path(args.system_metrics_script) if args.system_metrics_script else analysis_dir / "system_metrics.py",
    }

    for method, filename in METHOD_TO_RANK_SCRIPT.items():
        scripts[f"rank::{method}"] = analysis_dir / filename

    return scripts


def infer_method_from_path(exp_dir: Path) -> Optional[str]:
    methods = ("adasparse_lorav3", "adasparse_lorav2", "adasparse_lora", "fahqlora", "fah_qlora", "hetlora", "fedit")
    parts = [p.lower() for p in exp_dir.parts]

    for key in methods:
        if key in parts:
            return key

    # Handle distributed_* prefixed dirs (e.g. "distributed_fah_qlora" → "fah_qlora")
    for part in parts:
        stripped = part.removeprefix("distributed_")
        if stripped != part:
            for key in methods:
                if key == stripped:
                    return key

    # Fallback: inspect immediate parents
    current = exp_dir.parent
    while current != current.parent:
        name = current.name.lower()
        if name in set(methods):
            return name
        stripped = name.removeprefix("distributed_")
        if stripped != name and stripped in set(methods):
            return stripped
        current = current.parent
    return None


def should_skip_dir(path: Path) -> bool:
    return any(part == "golden_reference" for part in path.parts)


def find_experiment_dirs(root: Path, strategy_prefix: str) -> List[Path]:
    exp_dirs: List[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)

        # Prune golden_reference from traversal
        dirnames[:] = [d for d in dirnames if d != "golden_reference"]

        if should_skip_dir(current):
            continue

        if strategy_prefix in current.name:
            exp_dirs.append(current)
    return sorted(exp_dirs)


def run_command(command: Sequence[str], dry_run: bool) -> CommandResult:
    if dry_run:
        return CommandResult(
            name=Path(command[1]).stem if len(command) > 1 else "command",
            command=list(command),
            returncode=0,
            skipped=False,
            reason="dry-run",
        )

    proc = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return CommandResult(
        name=Path(command[1]).stem if len(command) > 1 else "command",
        command=list(command),
        returncode=proc.returncode,
        skipped=False,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def plan_commands(
    exp_dir: Path,
    method: Optional[str],
    scripts: Dict[str, Path],
    python_exe: str,
    force: bool,
    bandwidth_per_client: bool,
) -> List[CommandResult]:
    commands: List[CommandResult] = []

    bandwidth_file = None
    for candidate in ("bandwidth_history.csv", "fahqlora_bandwidth_history.csv", "hetlora_bandwidth_history.csv"):
        p = exp_dir / candidate
        if p.exists():
            bandwidth_file = p
            break
    if bandwidth_file is not None:
        outdir = exp_dir / "bandwidth_history"
        if outdir.exists() and not force:
            commands.append(CommandResult("bandwidth_history", [], None, True, f"skip existing folder: {outdir}"))
        else:
            bandwidth_cmd = [
                python_exe, str(scripts["bandwidth"]), str(bandwidth_file), "--output-dir", str(outdir)
            ]
            if bandwidth_per_client:
                bandwidth_cmd.append("--per-client-plots")
            commands.append(CommandResult(
                "bandwidth_history",
                bandwidth_cmd,
                None,
                False,
            ))
    else:
        commands.append(CommandResult("bandwidth_history", [], None, True, "no bandwidth history file found"))

    exp_print = exp_dir / "exp_print.log"
    if exp_print.exists():
        outdir = exp_dir / "loss_evolution_plots"
        if outdir.exists() and not force:
            commands.append(CommandResult("loss_evolution_plots", [], None, True, f"skip existing folder: {outdir}"))
        else:
            commands.append(CommandResult(
                "loss_evolution_plots",
                [python_exe, str(scripts["loss"]), "--log_file", str(exp_print)],
                None,
                False,
            ))
    else:
        commands.append(CommandResult("loss_evolution_plots", [], None, True, "no exp_print.log found"))

    system_metrics = exp_dir / "system_metrics.log"
    if system_metrics.exists():
        outdir = exp_dir / "system_metrics"
        if outdir.exists() and not force:
            commands.append(CommandResult("system_metrics", [], None, True, f"skip existing folder: {outdir}"))
        else:
            commands.append(CommandResult(
                "system_metrics",
                [python_exe, str(scripts["system_metrics"]), str(system_metrics)],
                None,
                False,
            ))
    else:
        commands.append(CommandResult("system_metrics", [], None, True, "no system_metrics.log found"))

    rank_script = scripts.get(f"rank::{method}") if method else None
    if rank_script is None:
        commands.append(CommandResult("rank_evolution", [], None, True, f"no rank script for method={method}"))
    elif not exp_print.exists():
        commands.append(CommandResult("rank_evolution", [], None, True, "no exp_print.log found"))
    else:
        folder_name = f"rank_evolution_{method}"
        outdir = exp_dir / folder_name
        if outdir.exists() and not force:
            commands.append(CommandResult("rank_evolution", [], None, True, f"skip existing folder: {outdir}"))
        else:
            commands.append(CommandResult(
                "rank_evolution",
                [python_exe, str(rank_script), "--log_file", str(exp_print)],
                None,
                False,
            ))

    return commands


def execute_experiment(
    exp_dir: Path,
    method: Optional[str],
    commands: List[CommandResult],
    dry_run: bool,
    continue_on_error: bool,
) -> ExperimentResult:
    results: List[CommandResult] = []

    print(f"\n=== {exp_dir} ===")
    print(f"Method: {method}")

    for planned in commands:
        if planned.skipped:
            print(f"[SKIP] {planned.name}: {planned.reason}")
            results.append(planned)
            continue

        print(f"[RUN ] {planned.name}: {' '.join(planned.command)}")
        result = run_command(planned.command, dry_run=dry_run)
        result.name = planned.name
        results.append(result)

        if result.returncode != 0:
            print(f"[FAIL] {planned.name}: return code {result.returncode}")
            if result.stderr:
                print(result.stderr.strip())
            if not continue_on_error and not dry_run:
                break
        else:
            print(f"[ OK ] {planned.name}")
            if result.stdout and result.stdout.strip():
                print(result.stdout.strip())

    return ExperimentResult(str(exp_dir), method, results)


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: root does not exist: {root}")
        return 1

    scripts = resolve_scripts(args)

    missing = [str(path) for path in scripts.values() if not path.exists()]
    if missing:
        print("Error: the following plotting scripts were not found:")
        for p in missing:
            print(f"  - {p}")
        return 1

    exp_dirs = find_experiment_dirs(root, args.strategy_prefix)
    if not exp_dirs:
        print(f"No experiment directories found under {root} with prefix {args.strategy_prefix!r}")
        return 0

    print(f"Found {len(exp_dirs)} experiment directories under {root}")

    all_results: List[ExperimentResult] = []
    for exp_dir in exp_dirs:
        method = infer_method_from_path(exp_dir)
        planned = plan_commands(
            exp_dir, method, scripts, args.python, args.force, args.bandwidth_per_client
        )
        result = execute_experiment(
            exp_dir,
            method,
            planned,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
        all_results.append(result)

    summary_path = root / args.summary_file
    summary_payload = {
        "root": str(root),
        "dry_run": args.dry_run,
        "force": args.force,
        "continue_on_error": args.continue_on_error,
        "bandwidth_per_client": args.bandwidth_per_client,
        "results": [
            {
                "experiment_dir": r.experiment_dir,
                "method": r.method,
                "commands": [asdict(cmd) for cmd in r.commands],
            }
            for r in all_results
        ],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())