#!/usr/bin/env bash
# Standalone master queue: runs all standalone sweeps sequentially by method.
# Each method launches gpu-host-a (local) and gpu-host-b (remote) scripts in parallel.
# Order: HetLoRA → FAH-QLoRA → AdaSparse v2 → AdaSparse v3
#
# Usage (run on gpu-host-a):
#   bash 1_scripts/sweeps/standalone_master_queue.sh
set -euo pipefail

cd ${FEDLORA_ROOT}

LUGIA="pablo@${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_SSH="ssh -o StrictHostKeyChecking=no $LUGIA"
QUEUE_LOG="exp2/standalone_master_queue.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$QUEUE_LOG"
}

run_method() {
    local name=$1
    local bulbasaur_script=$2
    local lugia_script=$3

    log_status "=== ${name}: STARTING ==="

    local pids=()

    bash "$bulbasaur_script" 2>&1 | tee -a "$QUEUE_LOG" &
    pids+=($!)
    log_status "${name}: gpu-host-a launched (PID ${pids[-1]})"

    if [ -n "$lugia_script" ]; then
        $LUGIA_SSH "cd ${FEDLORA_ROOT} && bash $lugia_script" 2>&1 | tee -a "$QUEUE_LOG" &
        pids+=($!)
        log_status "${name}: gpu-host-b launched (PID ${pids[-1]})"
    fi

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done

    if [ $failed -eq 0 ]; then
        log_status "=== ${name}: COMPLETE ==="
    else
        log_status "=== ${name}: FINISHED WITH ERRORS ==="
    fi
}

log_status "=========================================="
log_status "Standalone master queue started"
log_status "=========================================="

# Phase 1: HetLoRA (12 experiments: 8 gpu-host-a + 4 gpu-host-b)
run_method "HetLoRA" \
    "1_scripts/sweeps/standalone_sweep_hetlora.sh" \
    "1_scripts/archive/gpu-host-b/standalone_sweep_hetlora_lugia.sh"

# Phase 2: FAH-QLoRA (9 experiments: 6 gpu-host-a + 3 gpu-host-b)
run_method "FAH-QLoRA" \
    "1_scripts/sweeps/standalone_sweep_fahqlora.sh" \
    "1_scripts/archive/gpu-host-b/standalone_sweep_fahqlora_lugia.sh"

# Phase 3: AdaSparse v2 (96 experiments: 64 gpu-host-a + 32 gpu-host-b)
run_method "AdaSparse v2" \
    "1_scripts/sweeps/standalone_sweep_adasparselorav2_fullgrid_2gpu.sh" \
    "1_scripts/archive/gpu-host-b/standalone_sweep_adasparselorav2_lugia.sh"

# Phase 4: AdaSparse v3 (96 experiments: 32 gpu-host-a + 64 gpu-host-b)
run_method "AdaSparse v3" \
    "1_scripts/sweeps/standalone_sweep_adasparselorav3_bulbasaur.sh" \
    "1_scripts/archive/gpu-host-b/standalone_sweep_adasparselorav3_lugia.sh"

log_status "=========================================="
log_status "Standalone master queue finished"
log_status "=========================================="
