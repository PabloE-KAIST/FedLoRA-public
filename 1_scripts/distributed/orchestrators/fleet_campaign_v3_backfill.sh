#!/usr/bin/env bash
# fleet_campaign_v3_backfill.sh — Full-grid rerun for cola, stsb, sst2.
#
# Runs all 5 methods × full parameter grid for 3 tasks to produce
# complete v3 (layer-aware) coverage alongside v2 and baselines.
#
# Parameter grids (identical to fleet_campaign_v2.sh):
#   FedIT:     rank=64, 1 run/task
#   FAH-QLoRA: init_rank={32,64} × lambda={1,5,10} = 6/task
#   HetLoRA:   rw={0.005,0.05,0.1} × decay={0.50,0.65,0.80} = 9/task
#   v2:        rw={0.005,0.05,0.1} × gamma={0.50,0.65,0.80} × UL={230,460} = 18/task
#   v3:        same as v2 = 18/task
#   Total: 52/task, 156 total across 3 tasks
#
# Server allocation:
#   cola, stsb (6 clients): dual-server split
#     Bulbasaur: FedIT → FAH-QLoRA → v2
#     Lugia:     HetLoRA → v3
#   sst2 (12 clients): single-server (Bulbasaur, full manifest)
#     Sequential: FedIT → FAH-QLoRA → HetLoRA → v2 → v3
#
# Estimated wall-clock:
#   cola:  ~55h dual-server (~110h serial ÷ 2)
#   stsb:  ~46h dual-server (~91h serial ÷ 2)
#   sst2:  ~85h single-server (serial, 12 clients)
#   Total: ~186h ≈ 7.8 days continuous
#
# Experiment outputs: exp_distributed/{method}/{task}__strategy_...
# Analysis outputs: generated post-campaign via generate_thesis_v3.sh
#   → 0_results/final/thesis-v3-{Legend,noLegend}/{task}-trueAdaSv3/
#
# Resumable: writes completion markers to exp_distributed/.completed_v3/
#
# Usage:
#   bash 1_scripts/distributed/orchestrators/fleet_campaign_v3_backfill.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"
source "${FEDLORA_ROOT}/1_scripts/baseline_runs/glue/_glue_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# 6-client tasks first (dual-server), then 12-client (single-server)
DUAL_TASKS=(cola stsb)
SINGLE_TASKS=(sst2)

MANIFEST_A="distributed/configs/client_manifest_group_a.json"
MANIFEST_B="distributed/configs/client_manifest_group_b.json"
MANIFEST_FULL="distributed/configs/client_manifest.json"
BULBASAUR_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_HOST="gpu-host-b"
LUGIA_FEDLORA="${FEDLORA_ROOT}"

COMPLETED_DIR="${FEDLORA_ROOT}/exp_distributed/.completed_v3"
mkdir -p "$COMPLETED_DIR"

CAMPAIGN_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_campaign_v3_backfill.log"

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
    local task=$1 method=$2 manifest=$3
    local -a env_vars=(TASK="$task" MANIFEST="$manifest" PORT_OFFSET=0
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

prep_dual() {
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

prep_single() {
    local task=$1
    log_status "Prepping partitions for ${task} (full fleet)..."
    TASK="$task" MANIFEST="$MANIFEST_FULL" bash "${PREP_DIR}/activate_partitions.sh"

    log_status "Restarting DAs for full fleet..."
    CONTROLLER_IP="$BULBASAUR_IP" MANIFEST="$MANIFEST_FULL" \
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

run_method() {
    local task=$1 method=$2 where=$3 manifest=${4:-}
    if is_done "$method" "$task"; then
        log_status "[${task}] ${where}: ${method} already done, skipping."
        return 0
    fi
    log_status "[${task}] ${where}: ${method} starting..."
    local ok=false
    if [[ "$where" == "Bulbasaur" ]]; then
        run_on_bulbasaur "$task" "$method" "${manifest:-$MANIFEST_A}" \
            | tee -a "${CAMPAIGN_LOG%.log}_${task}_${method}.log" && ok=true
    else
        run_on_lugia "$task" "$method" \
            | tee -a "${CAMPAIGN_LOG%.log}_${task}_${method}.log" && ok=true
    fi
    if $ok; then
        mark_done "$method" "$task"
        log_status "[${task}] ${method}: DONE"
    else
        log_status "[${task}] ${method}: FAILED"
    fi
    sleep 30
}

# ── Dry-run mode ───────────────────────────────────────────────────

if $DRY_RUN; then
    echo "=== DRY RUN: Fleet Campaign v3 Backfill ==="
    echo ""
    total=0
    for task in "${DUAL_TASKS[@]}"; do
        clients=$(glue_clients_for_task "$task")
        rounds=$(glue_total_rounds "$task")
        echo "--- ${task} (${clients} clients, ${rounds} rounds) — DUAL-SERVER ---"
        echo "  Bulbasaur: FedIT (1) → FAH-QLoRA (6) → v2 (18) = 25 runs"
        echo "  Lugia:     HetLoRA (9) → v3 (18) = 27 runs"
        for method in fedit_r64 fahqlora hetlora v2 v3; do
            if is_done "$method" "$task"; then
                echo "  [SKIP] ${method}__${task}"
            else
                echo "  [TODO] ${method}__${task}"
            fi
        done
        echo ""
        total=$((total + 52))
    done
    for task in "${SINGLE_TASKS[@]}"; do
        clients=$(glue_clients_for_task "$task")
        rounds=$(glue_total_rounds "$task")
        echo "--- ${task} (${clients} clients, ${rounds} rounds) — SINGLE-SERVER ---"
        echo "  Bulbasaur: FedIT (1) → FAH-QLoRA (6) → HetLoRA (9) → v2 (18) → v3 (18) = 52 runs"
        for method in fedit_r64 fahqlora hetlora v2 v3; do
            if is_done "$method" "$task"; then
                echo "  [SKIP] ${method}__${task}"
            else
                echo "  [TODO] ${method}__${task}"
            fi
        done
        echo ""
        total=$((total + 52))
    done
    echo "Grand total: $total experiments"
    echo ""
    echo "Estimated wall-clock:"
    echo "  cola:  ~55h (dual-server)"
    echo "  stsb:  ~46h (dual-server)"
    echo "  sst2:  ~85h (single-server, 12 clients)"
    echo "  Total: ~186h ≈ 7.8 days continuous"
    exit 0
fi

# ── Main campaign loop ─────────────────────────────────────────────

log_status "=== Fleet Campaign v3 Backfill START ==="
log_status "Dual-server tasks: ${DUAL_TASKS[*]}"
log_status "Single-server tasks: ${SINGLE_TASKS[*]}"
log_status "Servers: Bulbasaur (${BULBASAUR_IP}, GPU 3) + Lugia (${LUGIA_IP}, GPU 0)"

# ── Phase 1: Dual-server tasks (cola, stsb) ───────────────────────

for task in "${DUAL_TASKS[@]}"; do
    log_status "========================================="
    log_status "TASK: ${task} (DUAL-SERVER)"
    log_status "========================================="

    all_done=true
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        is_done "$method" "$task" || { all_done=false; break; }
    done
    if $all_done; then
        log_status "[${task}] All methods already completed, skipping."
        continue
    fi

    prep_dual "$task"

    # Bulbasaur pipeline: FedIT → FAH-QLoRA → v2
    BULB_NEEDED=false
    for m in fedit_r64 fahqlora v2; do is_done "$m" "$task" || { BULB_NEEDED=true; break; }; done

    # Lugia pipeline: HetLoRA → v3
    LUGIA_NEEDED=false
    for m in hetlora v3; do is_done "$m" "$task" || { LUGIA_NEEDED=true; break; }; done

    if $BULB_NEEDED; then
        (
            for method in fedit_r64 fahqlora v2; do
                run_method "$task" "$method" "Bulbasaur" "$MANIFEST_A"
            done
        ) &
        BULB_PID=$!
    fi

    if $LUGIA_NEEDED; then
        (
            for method in hetlora v3; do
                run_method "$task" "$method" "Lugia"
            done
        ) &
        LUGIA_PID=$!
    fi

    $BULB_NEEDED && { wait $BULB_PID || true; }
    $LUGIA_NEEDED && { wait $LUGIA_PID || true; }

    # Sync Lugia results
    is_done "hetlora" "$task" && sync_from_lugia "hetlora" "$task"
    is_done "v3" "$task" && sync_from_lugia "adasparse_lorav3" "$task"

    all_done=true
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        is_done "$method" "$task" || { all_done=false; break; }
    done

    if $all_done; then
        log_status "[${task}] === TASK COMPLETE ==="
    else
        log_status "[${task}] WARNING: Not all methods succeeded."
        for method in fedit_r64 fahqlora hetlora v2 v3; do
            is_done "$method" "$task" \
                && log_status "  [OK]   ${method}" \
                || log_status "  [FAIL] ${method}"
        done
    fi

    sleep 30
done

# ── Phase 2: Single-server tasks (sst2) ───────────────────────────

for task in "${SINGLE_TASKS[@]}"; do
    log_status "========================================="
    log_status "TASK: ${task} (SINGLE-SERVER, 12 clients)"
    log_status "========================================="

    all_done=true
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        is_done "$method" "$task" || { all_done=false; break; }
    done
    if $all_done; then
        log_status "[${task}] All methods already completed, skipping."
        continue
    fi

    prep_single "$task"

    # All methods run sequentially on Bulbasaur with full manifest
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        run_method "$task" "$method" "Bulbasaur" "$MANIFEST_FULL"
    done

    all_done=true
    for method in fedit_r64 fahqlora hetlora v2 v3; do
        is_done "$method" "$task" || { all_done=false; break; }
    done

    if $all_done; then
        log_status "[${task}] === TASK COMPLETE ==="
    else
        log_status "[${task}] WARNING: Not all methods succeeded."
        for method in fedit_r64 fahqlora hetlora v2 v3; do
            is_done "$method" "$task" \
                && log_status "  [OK]   ${method}" \
                || log_status "  [FAIL] ${method}"
        done
    fi

    sleep 30
done

log_status "=== Fleet Campaign v3 Backfill COMPLETE ==="
log_status ""
log_status "Next: generate analysis plots with:"
log_status "  bash 1_scripts/analysis/generate_thesis_v3_trueAdaSv3.sh"
