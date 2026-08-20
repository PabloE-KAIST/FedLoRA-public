#!/usr/bin/env bash
# fleet_campaign_v2.sh — Second main experiment campaign.
#
# Runs 4 small tasks × 5 methods on dual sub-fleets (Bulbasaur + Lugia).
# Completes all methods for a task before moving to the next, enabling
# immediate cross-method analysis after each task.
#
# Task order: stsb → mrpc → rte → cola
# Per-task allocation:
#   Phase 1: Bulbasaur runs FedIT + FAH-QLoRA, Lugia runs HetLoRA
#   Phase 2: Bulbasaur runs v2, Lugia runs v3
#   Phase 3: Rsync Lugia → Bulbasaur, auto-analysis
#
# Parameter grids (set in queue scripts):
#   HetLoRA: rw={0.005,0.05,0.1} × decay={0.50,0.65,0.80} = 9/task
#   v2/v3:   rw={0.005,0.05,0.1} × gamma={0.50,0.65,0.80} × UL={230,460} = 18/task each
#   FAH-QLoRA: init_rank={32,64} × lambda={1,5,10} = 6/task
#   FedIT:   rank=64, 1/task
#   Total: 52/task, 208 total
#
# Resumable: writes completion markers to exp_distributed/.completed/
#
# Usage:
#   bash 1_scripts/distributed/orchestrators/fleet_campaign_v2.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"
source "${FEDLORA_ROOT}/1_scripts/baseline_runs/glue/_glue_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

TASKS=(mrpc rte stsb cola)

MANIFEST_A="distributed/configs/client_manifest_group_a.json"
MANIFEST_B="distributed/configs/client_manifest_group_b.json"
BULBASAUR_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_HOST="gpu-host-b"
LUGIA_FEDLORA="${FEDLORA_ROOT}"

COMPLETED_DIR="${FEDLORA_ROOT}/exp_distributed/.completed"
mkdir -p "$COMPLETED_DIR"

CAMPAIGN_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_campaign_v2.log"

export CUDA_VISIBLE_DEVICES=3

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$CAMPAIGN_LOG"
}

is_done() {
    [[ -f "${COMPLETED_DIR}/${1}__${2}" ]]
}

mark_done() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')" > "${COMPLETED_DIR}/${1}__${2}"
}

# ── Helpers ────────────────────────────────────────────────────────

run_on_bulbasaur() {
    local task=$1 method=$2
    local -a env_vars=(TASK="$task" MANIFEST="$MANIFEST_A" PORT_OFFSET=0
                       CONTROLLER_IP="$BULBASAUR_IP" NO_CLEANUP=true)
    env "${env_vars[@]}" bash "${QUEUE_DIR}/fleet_queue_${method}.sh" 2>&1
}

run_on_lugia() {
    local task=$1 method=$2
    ssh "$LUGIA_HOST" bash <<REMOTE_QUEUE
cd ${LUGIA_FEDLORA}
export PATH="\$HOME/miniconda3/envs/fedlora/bin:\$PATH"
export CUDA_VISIBLE_DEVICES=0
TASK="${task}" MANIFEST="${MANIFEST_B}" PORT_OFFSET=0 \
    CONTROLLER_IP="${LUGIA_IP}" NO_CLEANUP=true \
    bash 1_scripts/distributed/queues/fleet_queue_${method}.sh 2>&1
REMOTE_QUEUE
}

prep_both_groups() {
    local task=$1
    log_status "Prepping partitions for ${task} (both groups)..."
    TASK="$task" MANIFEST="$MANIFEST_A" bash "${PREP_DIR}/activate_partitions.sh"
    TASK="$task" MANIFEST="$MANIFEST_B" bash "${PREP_DIR}/activate_partitions.sh"

    log_status "Restarting DAs for both groups..."
    CONTROLLER_IP="$BULBASAUR_IP" MANIFEST="$MANIFEST_A" \
        bash "${PREP_DIR}/restart_das_for_manifest.sh" 2>/dev/null || true
    CONTROLLER_IP="$LUGIA_IP" MANIFEST="$MANIFEST_B" \
        bash "${PREP_DIR}/restart_das_for_manifest.sh" 2>/dev/null || true
    sleep 15
}

sync_from_lugia() {
    local method_dir=$1 task=$2
    log_status "Syncing ${method_dir}/${task}* from Lugia..."
    rsync -avz --ignore-existing \
        "${LUGIA_HOST}:${LUGIA_FEDLORA}/exp_distributed/${method_dir}/${task}__strategy_*" \
        "${FEDLORA_ROOT}/exp_distributed/${method_dir}/" 2>/dev/null || true
}

run_analysis() {
    local task=$1
    log_status "Running cross-method analysis for ${task}..."
    python3 -m analysis.cross_method.run_all_analysis \
        --task "$task" \
        --exp-dir "exp_distributed" \
        --output-dir "0_results/second_mainExperiment_artifacts/${task}" \
        2>&1 | tee -a "${CAMPAIGN_LOG%.log}_analysis_${task}.log" || true
}

# ── Dry-run mode ───────────────────────────────────────────────────

if $DRY_RUN; then
    echo "=== DRY RUN: Fleet Campaign v2 ==="
    echo "Tasks: ${TASKS[*]}"
    echo ""
    for task in "${TASKS[@]}"; do
        clients=$(glue_clients_for_task "$task")
        rounds=$(glue_total_rounds "$task")
        steps=$(glue_local_steps "$task")
        echo "--- ${task} (${clients} clients, ${rounds} rounds, ${steps} local steps) ---"
        echo "  Bulbasaur: FedIT (1) → FAH-QLoRA (6) → v2 (18) = 25 runs"
        echo "  Lugia:     HetLoRA (9) → v3 (18) = 27 runs"
        echo "  (no barrier between methods — each server runs independently)"
        echo "  Total: 52 experiments"
        echo ""

        for method in fedit_r64 fahqlora hetlora v2 v3; do
            if is_done "$method" "$task"; then
                echo "  [SKIP] ${method}__${task} (already completed)"
            else
                echo "  [TODO] ${method}__${task}"
            fi
        done
        echo ""
    done
    echo "Grand total: ${#TASKS[@]} tasks × 52 = $((${#TASKS[@]} * 52)) experiments"
    exit 0
fi

# ── Main campaign loop ─────────────────────────────────────────────

log_status "=== Fleet Campaign v2 START ==="
log_status "Tasks: ${TASKS[*]}"
log_status "Servers: Bulbasaur (${BULBASAUR_IP}, GPU 3) + Lugia (${LUGIA_IP}, GPU 0)"

for task in "${TASKS[@]}"; do
    log_status "========================================="
    log_status "TASK: ${task}"
    log_status "========================================="

    all_done=true
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        is_done "$method" "$task" || { all_done=false; break; }
    done
    if $all_done; then
        log_status "[${task}] All methods already completed, skipping."
        continue
    fi

    prep_both_groups "$task"

    # ── Two independent pipelines, single barrier at task end ───────
    # Bulbasaur: FedIT → FAH-QLoRA → v2 (sequential, no waiting for Lugia)
    # Lugia:     HetLoRA → v3       (sequential, no waiting for Bulbasaur)
    log_status "[${task}] Bulbasaur: FedIT → FAH-QLoRA → v2"
    log_status "[${task}] Lugia:     HetLoRA → v3"

    BULB_NEEDED=false
    for m in fedit_r64 fahqlora v2; do is_done "$m" "$task" || { BULB_NEEDED=true; break; }; done
    LUGIA_NEEDED=false
    for m in hetlora v3; do is_done "$m" "$task" || { LUGIA_NEEDED=true; break; }; done

    if $BULB_NEEDED; then
        (
            for method in fedit_r64 fahqlora v2; do
                if is_done "$method" "$task"; then
                    log_status "[${task}] Bulbasaur: ${method} already done, skipping."
                    continue
                fi
                log_status "[${task}] Bulbasaur: ${method} starting..."
                if run_on_bulbasaur "$task" "$method" \
                    | tee -a "${CAMPAIGN_LOG%.log}_${task}_${method}.log"; then
                    mark_done "$method" "$task"
                    log_status "[${task}] ${method}: DONE"
                else
                    log_status "[${task}] ${method}: FAILED"
                fi
                sleep 30
            done
        ) &
        BULB_PID=$!
    fi

    if $LUGIA_NEEDED; then
        (
            for method in hetlora v3; do
                if is_done "$method" "$task"; then
                    log_status "[${task}] Lugia: ${method} already done, skipping."
                    continue
                fi
                log_status "[${task}] Lugia: ${method} starting..."
                if run_on_lugia "$task" "$method" \
                    | tee -a "${CAMPAIGN_LOG%.log}_${task}_${method}.log"; then
                    mark_done "$method" "$task"
                    log_status "[${task}] ${method}: DONE"
                else
                    log_status "[${task}] ${method}: FAILED"
                fi
                sleep 30
            done
        ) &
        LUGIA_PID=$!
    fi

    $BULB_NEEDED && { wait $BULB_PID || true; }
    $LUGIA_NEEDED && { wait $LUGIA_PID || true; }

    # ── Sync + analysis ────────────────────────────────────────────
    is_done "hetlora" "$task" && sync_from_lugia "hetlora" "$task"
    is_done "v3" "$task" && sync_from_lugia "adasparse_lorav3" "$task"

    all_done=true
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        is_done "$method" "$task" || { all_done=false; break; }
    done

    if $all_done; then
        run_analysis "$task"
        log_status "[${task}] === TASK COMPLETE ==="
    else
        log_status "[${task}] WARNING: Not all methods succeeded, skipping analysis."
        for method in fedit_r64 fahqlora hetlora v2 v3; do
            is_done "$method" "$task" \
                && log_status "  [OK]   ${method}" \
                || log_status "  [FAIL] ${method}"
        done
    fi

    sleep 30
done

log_status "=== Fleet Campaign v2 COMPLETE ==="
