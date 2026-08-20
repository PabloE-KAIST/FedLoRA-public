#!/usr/bin/env bash
# Full FL pipeline: cleanup → restart DAs → launch server (foreground) →
#                   collect logs + metrics → merge metrics → run analysis.
#
# Usage:
#   bash 1_scripts/distributed/orchestrators/full_fl_run.sh \
#       --config 2_yamls/fedit/fedit_distributed.yaml \
#       [--manifest distributed/configs/client_manifest.json] \
#       [--port-offset 0] [--no-tc] [--no-restart-das]
#
# For concurrent runs (e.g. small tasks on split fleet), use different
# --port-offset values and --no-restart-das to avoid cross-instance interference.
# PORT_OFFSET env var is also accepted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

# Parse our flags, forward the rest to launch_fl_server.sh
CONFIG_REL=""
MANIFEST_REL=""
PORT_OFFSET="${PORT_OFFSET:-0}"
NO_RESTART_DAS="${NO_RESTART_DAS:-false}"
NO_CLEANUP="${NO_CLEANUP:-false}"
NO_TC=false
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG_REL="$2"; shift 2 ;;
        --manifest)        MANIFEST_REL="$2"; shift 2 ;;
        --port-offset)     PORT_OFFSET="$2"; shift 2 ;;
        --no-restart-das)  NO_RESTART_DAS=true; shift ;;
        --no-cleanup)      NO_CLEANUP=true; shift ;;
        --no-tc)           NO_TC=true; shift ;;
        *)                 LAUNCH_ARGS+=("$1"); shift ;;
    esac
done

# Build launch args
SERVER_ARGS=()
[[ -n "$CONFIG_REL" ]]   && SERVER_ARGS+=(--config "$CONFIG_REL")
[[ -n "$MANIFEST_REL" ]] && SERVER_ARGS+=(--manifest "$MANIFEST_REL")
SERVER_ARGS+=(--port-offset "$PORT_OFFSET")
[[ "$NO_TC" == "true" ]] && SERVER_ARGS+=(--no-tc)
SERVER_ARGS+=(--foreground)
SERVER_ARGS+=("${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"}")

MANIFEST="${MANIFEST_PATH}"
[[ -n "$MANIFEST_REL" ]] && MANIFEST="${FEDLORA_ROOT}/${MANIFEST_REL}"

export LOGFILE="${LOGFILE:-/tmp/fl_distributed_p${PORT_OFFSET}.log}"

# ── Step 1: Kill old server (targeted by port-offset) ────────────────
log "=== Step 1: Kill old server (port-offset=${PORT_OFFSET}) ==="
pkill -f "distributed.server.main.*--port-offset ${PORT_OFFSET}[^0-9]" 2>/dev/null || true
pkill -f "distributed.server.main.*--port-offset ${PORT_OFFSET}$" 2>/dev/null || true
sleep 1

# ── Step 2: Cleanup old worker containers ─────────────────────────────
if [[ -n "${MANIFEST:-}" ]]; then
    log "=== Step 2: Manifest-scoped cleanup (concurrent-safe) ==="
    MANIFEST="$MANIFEST" bash "${SCRIPT_DIR}/../prep/cleanup_workers_for_manifest.sh"
elif [[ "$NO_CLEANUP" == "true" ]]; then
    log "=== Step 2: Cleanup workers — SKIPPED (--no-cleanup / NO_CLEANUP) ==="
else
    log "=== Step 2: Cleanup old worker containers ==="
    bash "${SCRIPT_DIR}/../prep/cleanup_workers.sh"
fi

# ── Step 3: Restart DeviceAgents ─────────────────────────────────────
if [[ "$NO_RESTART_DAS" == "true" ]]; then
    log "=== Step 3: Restart DAs — SKIPPED (--no-restart-das) ==="
elif [[ -n "${CONTROLLER_IP:-}" && -n "${MANIFEST:-}" ]]; then
    log "=== Step 3: Restart DAs for manifest → ${CONTROLLER_IP} ==="
    CONTROLLER_IP="$CONTROLLER_IP" MANIFEST="$MANIFEST" \
        bash "${SCRIPT_DIR}/../prep/restart_das_for_manifest.sh"
else
    log "=== Step 3: Restart all DeviceAgents ==="
    bash "${SCRIPT_DIR}/../prep/restart_all_das.sh"
fi

# ── Step 4: Launch FL server (foreground, blocks until FL finishes) ───
log "=== Step 4: Launch FL server (foreground) ==="
bash "${SCRIPT_DIR}/launch_fl_server.sh" "${SERVER_ARGS[@]}"

# ── Detect experiment directory ───────────────────────────────────────
EXP_DIR=""

# Primary: parse server log for the actual nested experiment directory
if [[ -f "$LOGFILE" ]]; then
    EXP_DIR=$(sed 's/\x1b\[[0-9;]*m//g' "$LOGFILE" | grep -oP 'outdir updated to: \K\S+' | tail -1 || true)
fi

# Fallback: newest subdirectory under the YAML outdir
if [[ -z "$EXP_DIR" || ! -d "${FEDLORA_ROOT}/${EXP_DIR}" ]]; then
    EXP_DIR=""
    if [[ -n "$CONFIG_REL" ]]; then
        YAML_OUTDIR=$(grep -oP '^outdir:\s*\K\S+' "${FEDLORA_ROOT}/${CONFIG_REL}" 2>/dev/null || true)
        if [[ -n "$YAML_OUTDIR" && -d "${FEDLORA_ROOT}/${YAML_OUTDIR}" ]]; then
            NEWEST=$(ls -td "${FEDLORA_ROOT}/${YAML_OUTDIR}"/*/ 2>/dev/null | head -1 || true)
            if [[ -n "$NEWEST" ]]; then
                EXP_DIR="${NEWEST#${FEDLORA_ROOT}/}"
                EXP_DIR="${EXP_DIR%/}"
            fi
        fi
    fi
fi

if [[ -z "$EXP_DIR" || ! -d "${FEDLORA_ROOT}/${EXP_DIR}" ]]; then
    log "ERROR: Could not determine experiment directory. Skipping post-run steps."
    exit 1
fi

log "Experiment directory: ${EXP_DIR}"

# ── Step 5: Collect worker logs and metrics ───────────────────────────
log "=== Step 5: Collect worker logs and metrics ==="
bash "${SCRIPT_DIR}/../log_tools/collect_logs.sh" "${FEDLORA_ROOT}/${EXP_DIR}" "$MANIFEST"

# ── Step 5b: Merge worker eval logs into exp_print.log ───────────────
log "=== Step 5b: Merge worker eval logs ==="
python3 "${SCRIPT_DIR}/../log_tools/merge_eval_logs.py" "${FEDLORA_ROOT}/${EXP_DIR}" \
    && log "Worker eval logs merged successfully." \
    || log "WARNING: Worker eval log merge failed (client loss data may be unavailable)."

# ── Step 6: Merge system metrics ─────────────────────────────────────
log "=== Step 6: Merge system metrics ==="
python3 "${SCRIPT_DIR}/../log_tools/merge_system_metrics.py" "${FEDLORA_ROOT}/${EXP_DIR}" \
    && log "System metrics merged successfully." \
    || log "WARNING: System metrics merge failed (worker metrics may be unavailable)."

# ── Step 7: Run analysis ─────────────────────────────────────────────
log "=== Step 7: Run analysis ==="
ABS_EXP_DIR="${FEDLORA_ROOT}/${EXP_DIR}"
ANALYSIS_DIR="${FEDLORA_ROOT}/analysis/single_run"

run_analysis() {
    local script=$1; shift
    local name
    name=$(basename "$script" .py)
    log "[RUN ] ${name}"
    if python3 "$script" "$@" 2>&1; then
        log "[ OK ] ${name}"
    else
        log "[FAIL] ${name}"
    fi
}

if [[ -f "${ABS_EXP_DIR}/bandwidth_history.csv" ]]; then
    run_analysis "${ANALYSIS_DIR}/bandwidth_history.py" \
        "${ABS_EXP_DIR}/bandwidth_history.csv" --output-dir "${ABS_EXP_DIR}/bandwidth_history"
fi
if [[ -f "${ABS_EXP_DIR}/exp_print.log" ]]; then
    run_analysis "${ANALYSIS_DIR}/loss_evolution_plots.py" \
        --log_file "${ABS_EXP_DIR}/exp_print.log"
fi
if [[ -f "${ABS_EXP_DIR}/system_metrics.log" ]]; then
    run_analysis "${ANALYSIS_DIR}/system_metrics.py" \
        "${ABS_EXP_DIR}/system_metrics.log"
fi

# Rank evolution: detect method from config path and run the appropriate script
RANK_METHOD=""
case "${CONFIG_REL}" in
    *fahqlora*|*fah_qlora*)       RANK_METHOD="fahqlora" ;;
    *hetlora*)                     RANK_METHOD="hetlora" ;;
    *adasparse_lora_v3*|*lorav3*)  RANK_METHOD="adasparsev3" ;;
    *adasparse_lora_v2*|*lorav2*)  RANK_METHOD="adasparsev2" ;;
    *adasparse_lora*|*adasparse*)  RANK_METHOD="adasparse" ;;
esac
RANK_LOG="${ABS_EXP_DIR}/unified_log.log"
[[ ! -f "$RANK_LOG" ]] && RANK_LOG="${ABS_EXP_DIR}/exp_print.log"
if [[ -n "$RANK_METHOD" && -f "$RANK_LOG" ]]; then
    RANK_SCRIPT="${ANALYSIS_DIR}/rank_evolution_${RANK_METHOD}.py"
    if [[ -f "$RANK_SCRIPT" ]]; then
        run_analysis "$RANK_SCRIPT" \
            --log_file "$RANK_LOG"
    else
        log "[SKIP] rank_evolution: no script for method=${RANK_METHOD}"
    fi
elif [[ -z "$RANK_METHOD" ]]; then
    log "[SKIP] rank_evolution: method not detected from config"
fi
log "Analysis complete."

log "=== Pipeline complete ==="
log "Results: ${EXP_DIR}"
