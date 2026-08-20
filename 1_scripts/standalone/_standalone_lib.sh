#!/usr/bin/env bash
# Shared helpers for standalone queue scripts.
# Sourced by sibling runners; not intended to be run directly.

STANDALONE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDLORA_ROOT="$(cd "${STANDALONE_DIR}/../.." && pwd)"

# Activate the fedlora conda environment if python is not already available.
if ! command -v python &>/dev/null; then
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    conda activate fedlora
fi

source "${FEDLORA_ROOT}/1_scripts/baseline_runs/glue/_glue_lib.sh"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

# Format regularizer weight in scientific notation matching
# _format_regularizer_scientific() in federatedscope/core/auxiliaries/logging.py
fmt_rw() {
    python3 -c "v=float('$1'); print(f'{v:.0e}'.replace('e-0','e-').replace('e+0','e+'))"
}

# Normalize float to Python's default repr (strip trailing zeros).
# 0.50 -> 0.5, 0.80 -> 0.8, 0.65 -> 0.65
fmt_float() {
    python3 -c "print(float('$1'))"
}

# Check if a completed experiment already exists.
# Usage: is_completed <outdir> <glob_pattern>
#   outdir:       e.g. exp_standalone/hetlora
#   glob_pattern: e.g. "sst2__strategy_custom__regularizer_5e-3__decay_0.5__*"
# Returns 0 if a matching directory contains 'Round': 'Final' in eval_results.log.
is_completed() {
    local outdir="$1" pattern="$2"
    for d in "${outdir}/"${pattern}; do
        if [ -d "$d" ] && grep -q "'Round': 'Final'" "$d/eval_results.log" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}
