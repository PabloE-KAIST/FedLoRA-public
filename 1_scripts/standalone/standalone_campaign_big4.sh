#!/usr/bin/env bash
# Standalone campaign: 4 big 12-client GLUE tasks.
# Split across Lugia (GPUs 0,1,2) and Bulbasaur (GPU 2).
#
# GPU allocation per task:
#   Lugia GPU 0:     v2/v3 pool worker 0/3     = 12 experiments
#   Lugia GPU 1:     v2/v3 pool worker 1/3     = 12 experiments
#   Lugia GPU 2:     2× concurrent small methods = 16 experiments (8 pairs)
#   Bulbasaur GPU 2: v2/v3 pool worker 2/3     = 12 experiments  (separate script)
#
# This script runs the Lugia side (40 experiments/task).
# Bulbasaur runs standalone_v2v3_bulbasaur.sh independently.
#
# Usage (run on Lugia):
#   bash 1_scripts/standalone/standalone_campaign_big4.sh
#   bash 1_scripts/standalone/standalone_campaign_big4.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_standalone_lib.sh"
cd "$FEDLORA_ROOT"

# ── Configuration ──────────────────────────────────────────────────────────────
TASKS=(sst2 qnli mnli qqp)

COMPLETED_DIR="exp_standalone/.completed"
LOG_DIR="exp_standalone/logs/standalone_campaign_big4"
CAMPAIGN_LOG="exp_standalone/standalone_campaign_big4.log"
QUEUE_DIR="${SCRIPT_DIR}/queues"

mkdir -p "$COMPLETED_DIR" "$LOG_DIR"

# ── Helpers ────────────────────────────────────────────────────────────────────
campaign_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$CAMPAIGN_LOG"; }

is_task_done() { [ -f "${COMPLETED_DIR}/TASK_DONE__${1}" ]; }
mark_task_done() { touch "${COMPLETED_DIR}/TASK_DONE__${1}"; }

# ── Dry-run mode ───────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=== DRY RUN: Standalone Big-4 Campaign (Lugia side) ==="
    echo ""
    echo "Tasks: ${TASKS[*]}"
    echo "Lugia GPU 0: v2/v3 pool worker 0/3 (12 exp/task)"
    echo "Lugia GPU 1: v2/v3 pool worker 1/3 (12 exp/task)"
    echo "Lugia GPU 2: 2× concurrent small methods (16 exp/task, 8 pairs)"
    echo "Bulbasaur GPU 2: v2/v3 pool worker 2/3 (12 exp/task, separate script)"
    echo ""
    for task in "${TASKS[@]}"; do
        clients=$(glue_clients_for_task "$task")
        rounds=$(glue_total_rounds "$task")
        metric=$(glue_eval_key "$task")
        echo "── Task: ${task} (${clients} clients, ${rounds} rounds, metric=${metric})"
        if is_task_done "$task"; then
            echo "   STATUS: ALREADY COMPLETE"
        else
            echo "   Lugia:     40 experiments (12 v2/v3 + 12 v2/v3 + 16 small)"
            echo "   Bulbasaur: 12 experiments (v2/v3 pool)"
            echo "   Total:     52 experiments"
        fi
        echo ""
    done
    exit 0
fi

# ── Campaign loop ──────────────────────────────────────────────────────────────
campaign_log "=========================================="
campaign_log "Standalone Big-4 Campaign started (Lugia side)"
campaign_log "Tasks: ${TASKS[*]}"
campaign_log "GPU 0: v2/v3 W0/3 | GPU 1: v2/v3 W1/3 | GPU 2: small serial"
campaign_log "=========================================="

for task in "${TASKS[@]}"; do
    if is_task_done "$task"; then
        campaign_log "SKIP task ${task} (already complete)"
        continue
    fi

    campaign_log "── Starting task: ${task} ──"

    # GPU 0: v2/v3 pool worker 0/3
    (
        GPU=0 TASK=$task WORKER_ID=0 TOTAL_WORKERS=3 \
            bash "${QUEUE_DIR}/standalone_queue_v2v3_pool.sh"
    ) > "${LOG_DIR}/gpu0_${task}.log" 2>&1 &
    PID0=$!

    # GPU 1: v2/v3 pool worker 1/3
    (
        GPU=1 TASK=$task WORKER_ID=1 TOTAL_WORKERS=3 \
            bash "${QUEUE_DIR}/standalone_queue_v2v3_pool.sh"
    ) > "${LOG_DIR}/gpu1_${task}.log" 2>&1 &
    PID1=$!

    # GPU 2: small methods serial (concurrent OOMs on 12-client tasks)
    (
        GPU=2 TASK=$task bash "${QUEUE_DIR}/standalone_queue_fedit.sh"
        GPU=2 TASK=$task bash "${QUEUE_DIR}/standalone_queue_fahqlora.sh"
        GPU=2 TASK=$task bash "${QUEUE_DIR}/standalone_queue_hetlora.sh"
    ) > "${LOG_DIR}/gpu2_${task}.log" 2>&1 &
    PID2=$!

    campaign_log "Launched: GPU 0 (PID ${PID0}), GPU 1 (PID ${PID1}), GPU 2 (PID ${PID2})"

    TASK_FAILED=0
    for pid in $PID0 $PID1 $PID2; do
        if ! wait "$pid"; then
            TASK_FAILED=1
        fi
    done

    if [ $TASK_FAILED -ne 0 ]; then
        campaign_log "WARNING: task ${task} had failures — check GPU logs in ${LOG_DIR}/"
    fi

    # Per-experiment analysis
    campaign_log "Running per-experiment analysis for ${task}..."
    for method_dir in exp_standalone/fedit exp_standalone/fahqlora exp_standalone/hetlora exp_standalone/adasparse_lorav2 exp_standalone/adasparse_lorav3; do
        if [ -d "$method_dir" ]; then
            python3 analysis/single_run/run_all_experiment_plots.py "$method_dir" \
                --analysis-dir analysis/single_run --force --continue-on-error \
                >> "${LOG_DIR}/analysis_${task}.log" 2>&1 || true
        fi
    done

    mark_task_done "$task"
    campaign_log "── Task ${task} DONE (Lugia side) ──"
done

campaign_log "=========================================="
campaign_log "Standalone Big-4 Campaign finished (Lugia side)"
campaign_log "Next: rsync Bulbasaur v2/v3 results, then run cross-method analysis"
campaign_log "=========================================="
