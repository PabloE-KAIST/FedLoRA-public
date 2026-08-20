#!/usr/bin/env bash
# Standalone v2/v3 pool runner on Bulbasaur GPU 2 (worker 2/3).
# Companion to standalone_campaign_big4.sh which runs the other workers on Lugia.
#
# Usage (run on Bulbasaur):
#   bash 1_scripts/standalone/standalone_v2v3_bulbasaur.sh
#   bash 1_scripts/standalone/standalone_v2v3_bulbasaur.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_standalone_lib.sh"
cd "$FEDLORA_ROOT"

TASKS=(sst2 qnli mnli qqp)
GPU=2
WORKER_ID=2
TOTAL_WORKERS=3
QUEUE_DIR="${SCRIPT_DIR}/queues"
LOG_DIR="exp_standalone/logs/standalone_v2v3_bulbasaur"
CAMPAIGN_LOG="exp_standalone/standalone_v2v3_bulbasaur.log"

mkdir -p "$LOG_DIR"

campaign_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$CAMPAIGN_LOG"; }

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=== DRY RUN: v2/v3 pool on Bulbasaur GPU ${GPU} (worker ${WORKER_ID}/${TOTAL_WORKERS}) ==="
    echo "Tasks: ${TASKS[*]}"
    echo "12 experiments per task × 4 tasks = 48 experiments"
    exit 0
fi

campaign_log "=========================================="
campaign_log "v2/v3 Bulbasaur runner started (GPU ${GPU}, worker ${WORKER_ID}/${TOTAL_WORKERS})"
campaign_log "Tasks: ${TASKS[*]}"
campaign_log "=========================================="

for task in "${TASKS[@]}"; do
    campaign_log "── Starting v2/v3 pool for ${task} ──"
    GPU=$GPU TASK=$task WORKER_ID=$WORKER_ID TOTAL_WORKERS=$TOTAL_WORKERS \
        bash "${QUEUE_DIR}/standalone_queue_v2v3_pool.sh" \
        > "${LOG_DIR}/gpu${GPU}_${task}.log" 2>&1
    campaign_log "── v2/v3 ${task} DONE ──"
done

campaign_log "=========================================="
campaign_log "v2/v3 Bulbasaur runner finished"
campaign_log "=========================================="
