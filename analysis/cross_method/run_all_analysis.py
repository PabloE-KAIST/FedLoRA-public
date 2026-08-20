#!/usr/bin/env python3
"""Run all analysis plots for a task: cross-method + per-method.

Supports two output modes:
  - sensitivity_study: all experiments, with best-per-shade highlighted
  - selection: only selected experiments, connecting lines, no highlighting

Usage:
    python -m analysis.cross_method.run_all_analysis \
        --task mrpc --exp-dir exp_distributed \
        --output-dir 0_results/final/mrpc --mode both

    # thesis-v2 mode: exclude v3, rename v2
    python -m analysis.cross_method.run_all_analysis \
        --task mrpc --exp-dir exp_distributed \
        --output-dir 0_results/final/thesis-v2/mrpc --mode both \
        --exclude-methods adasparse_lorav3 \
        --rename-method adasparse_lorav2 'AdaS-LoRA'
"""

import argparse
import subprocess
import sys


METHODS = ["hetlora", "adasparse_lorav2", "adasparse_lorav3", "fahqlora", "fedit"]

DISPLAY_NAME_DEFAULTS = {
    "hetlora": "HetLoRA",
    "adasparse_lorav2": "AdaSparse-LoRA v2",
    "adasparse_lorav3": "AdaSparse-LoRA v3",
    "fahqlora": "FAH-QLoRA",
    "fedit": "FedIT",
}


def run_cmd(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def run_suite(task, exp_dir, output_dir, selection_only=False,
              methods=None, exclude_display=None, renames=None,
              cross_method_extra_args=None, per_method_args=None,
              per_method_common_args=None):
    ok = True
    label = "selection" if selection_only else "sensitivity_study"
    extra = ["--selection-only"] if selection_only else []

    cross_extra = list(extra)
    if selection_only:
        cross_extra.append("--no-connecting-lines")
    if exclude_display:
        cross_extra += ["--exclude-methods"] + exclude_display
    if renames:
        for old, new in renames.items():
            cross_extra += ["--rename-method", old, new]
    if cross_method_extra_args:
        cross_extra += cross_method_extra_args

    print(f"\n{'#'*60}")
    print(f"  Cross-method analysis for {task.upper()} [{label}]")
    print(f"{'#'*60}")
    if not run_cmd([
        sys.executable, "-m", "analysis.cross_method.fleet_cross_method_final",
        "--task", task,
        "--exp-dir", exp_dir,
        "--output-dir", output_dir,
    ] + cross_extra):
        print(f"WARNING: cross-method plots failed for {task} [{label}]", file=sys.stderr)
        ok = False

    for method in methods:
        display = renames.get(DISPLAY_NAME_DEFAULTS.get(method, method),
                              DISPLAY_NAME_DEFAULTS.get(method, method)) if renames else None
        per_method_extra = list(extra)
        if per_method_common_args:
            per_method_extra += per_method_common_args
        if display:
            per_method_extra += ["--display-name", display]
        if per_method_args and method in per_method_args:
            per_method_extra += per_method_args[method]

        method_output_label = display or DISPLAY_NAME_DEFAULTS.get(method, method)
        print(f"\n{'#'*60}")
        print(f"  Per-method analysis: {method_output_label} / {task.upper()} [{label}]")
        print(f"{'#'*60}")
        method_output = f"{output_dir}/{method}"
        if not run_cmd([
            sys.executable, "-m", "analysis.cross_method.fleet_per_method_plots",
            "--method", method,
            "--task", task,
            "--exp-dir", exp_dir,
            "--output-dir", method_output,
        ] + per_method_extra):
            print(f"WARNING: per-method plots failed for {method}/{task} [{label}]", file=sys.stderr)
            ok = False

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--exp-dir", default="exp_distributed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["sensitivity_study", "selection", "both"],
                        default="both")
    parser.add_argument("--exclude-methods", nargs="*", default=[],
                        help="Method keys to exclude (e.g. adasparse_lorav3)")
    parser.add_argument("--rename-method", nargs=2, action="append", default=[],
                        metavar=("METHOD_KEY", "NEW_DISPLAY_NAME"),
                        help="Rename a method's display name (e.g. adasparse_lorav2 'AdaS-LoRA')")
    parser.add_argument("--exclude-shade", nargs=3, action="append", default=[],
                        metavar=("METHOD", "KEY", "VALUE"),
                        help="Exclude records matching method/key/value from ALL modes")
    parser.add_argument("--exclude-shade-selection-only", nargs=3, action="append", default=[],
                        metavar=("METHOD", "KEY", "VALUE"),
                        help="Exclude records matching method/key/value from selection mode only")
    parser.add_argument("--override-selection", nargs=2, action="append", default=[],
                        metavar=("METHOD_KEY", "DIRNAME_PATTERN"),
                        help="Force-select a record by dirname substring")
    parser.add_argument("--add-selection", nargs=2, action="append", default=[],
                        metavar=("METHOD_KEY", "DIRNAME_PATTERN"),
                        help="Add a record to the selection without deselecting others")
    parser.add_argument("--break-comm-hours", type=float, nargs="+", default=None)
    parser.add_argument("--break-compute-wc-hours", type=float, nargs="+", default=None)
    parser.add_argument("--break-total-wc-hours", type=float, nargs="+", default=None)
    parser.add_argument("--y-window", type=float, default=0.20,
                        help="Fixed y-axis range size for accuracy plots (default: 0.20)")
    parser.add_argument("--y-pad", type=float, default=0.01,
                        help="Padding above max / below min within the window (default: 0.01)")
    parser.add_argument("--no-legend", action="store_true",
                        help="Hide legend from cross-method plots")
    parser.add_argument("--no-title", action="store_true",
                        help="Hide plot titles")
    parser.add_argument("--fedit-golden-only", action="store_true",
                        help="Pin FedIT to its golden run only (stable, table-consistent)")
    args = parser.parse_args()

    task = args.task.lower()

    methods = [m for m in METHODS if m not in args.exclude_methods]

    # Build rename map: display_name_old -> display_name_new
    renames = {}
    for method_key, new_name in args.rename_method:
        old_display = DISPLAY_NAME_DEFAULTS.get(method_key, method_key)
        renames[old_display] = new_name

    # Build list of display names to exclude (for cross-method script)
    exclude_display = [DISPLAY_NAME_DEFAULTS.get(m, m) for m in args.exclude_methods]

    # Build extra args for cross-method script
    cross_extra = []
    for method, key, val in args.exclude_shade:
        display = renames.get(DISPLAY_NAME_DEFAULTS.get(method, method),
                              DISPLAY_NAME_DEFAULTS.get(method, method))
        cross_extra += ["--exclude-shade", display, key, val]
    selection_only_excludes = []
    for method, key, val in args.exclude_shade_selection_only:
        display = renames.get(DISPLAY_NAME_DEFAULTS.get(method, method),
                              DISPLAY_NAME_DEFAULTS.get(method, method))
        selection_only_excludes += ["--exclude-shade", display, key, val]
    for method_key, pattern in args.override_selection:
        display = renames.get(DISPLAY_NAME_DEFAULTS.get(method_key, method_key),
                              DISPLAY_NAME_DEFAULTS.get(method_key, method_key))
        cross_extra += ["--override-selection", display, pattern]
    for method_key, pattern in args.add_selection:
        display = renames.get(DISPLAY_NAME_DEFAULTS.get(method_key, method_key),
                              DISPLAY_NAME_DEFAULTS.get(method_key, method_key))
        cross_extra += ["--add-selection", display, pattern]
    if args.break_comm_hours:
        cross_extra += ["--break-comm-hours"] + [str(v) for v in args.break_comm_hours]
    if args.break_compute_wc_hours:
        cross_extra += ["--break-compute-wc-hours"] + [str(v) for v in args.break_compute_wc_hours]
    if args.break_total_wc_hours:
        cross_extra += ["--break-total-wc-hours"] + [str(v) for v in args.break_total_wc_hours]
    cross_extra += ["--y-window", str(args.y_window), "--y-pad", str(args.y_pad)]
    if args.no_legend:
        cross_extra += ["--no-legend"]
    if args.no_title:
        cross_extra += ["--no-title"]
    if args.fedit_golden_only:
        cross_extra += ["--fedit-golden-only"]

    # Build per-method args (keyed by method_key, no display name needed)
    pm_args = {}  # method_key -> list of CLI args
    pm_selection_only = {}  # method_key -> list of CLI args (selection mode only)
    for method_key, key, val in args.exclude_shade:
        pm_args.setdefault(method_key, []).extend(["--exclude-shade", key, val])
    for method_key, key, val in args.exclude_shade_selection_only:
        pm_selection_only.setdefault(method_key, []).extend(["--exclude-shade", key, val])
    for method_key, pattern in args.override_selection:
        pm_args.setdefault(method_key, []).extend(["--override-selection", pattern])
    for method_key, pattern in args.add_selection:
        pm_args.setdefault(method_key, []).extend(["--add-selection", pattern])

    pm_common = ["--y-window", str(args.y_window), "--y-pad", str(args.y_pad)]
    if args.no_title:
        pm_common += ["--no-title"]
    if args.fedit_golden_only:
        pm_common += ["--fedit-golden-only"]

    ok = True

    if args.mode in ("sensitivity_study", "both"):
        out = f"{args.output_dir}/sensitivity_study"
        if not run_suite(task, args.exp_dir, out, selection_only=False,
                         methods=methods, exclude_display=exclude_display,
                         renames=renames, cross_method_extra_args=cross_extra,
                         per_method_args=pm_args,
                         per_method_common_args=pm_common):
            ok = False

    if args.mode in ("selection", "both"):
        pm_sel = {}
        for k in set(list(pm_args.keys()) + list(pm_selection_only.keys())):
            pm_sel[k] = pm_args.get(k, []) + pm_selection_only.get(k, [])
        out = f"{args.output_dir}/selection"
        if not run_suite(task, args.exp_dir, out, selection_only=True,
                         methods=methods, exclude_display=exclude_display,
                         renames=renames,
                         cross_method_extra_args=cross_extra + selection_only_excludes,
                         per_method_args=pm_sel,
                         per_method_common_args=pm_common):
            ok = False

    status = "COMPLETE" if ok else "COMPLETE (with warnings)"
    print(f"\n{'#'*60}")
    print(f"  Analysis {status} for {task.upper()}")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*60}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
