#!/usr/bin/env bash
# Standalone v3 runner on Bulbasaur GPU 2.
# Companion to standalone_campaign_big4.sh which runs the other methods on Lugia.
#
# After both finish, rsync v3 results to Lugia and run cross-method analysis.
#
# Usage (run on Bulbasaur):
#   bash 1_scripts/standalone/standalone_v3_bulbasaur.sh
#   bash 1_scripts/standalone/standalone_v3_bulbasaur.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_standalone_lib.sh"
cd "$FEDLORA_ROOT"

TASKS=(sst2 qnli mnli qqp)
GPU=2
QUEUE_DIR="${SCRIPT_DIR}/queues"
LOG_DIR="exp_standalone/logs/standalone_v3_bulbasaur"
CAMPAIGN_LOG="exp_standalone/standalone_v3_bulbasaur.log"

mkdir -p "$LOG_DIR"

campaign_log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$CAMPAIGN_LOG"; }

if [[ "${1:-}" == "--dry-run" ]]; then
    echo "=== DRY RUN: v3 on Bulbasaur GPU ${GPU} ==="
    echo "Tasks: ${TASKS[*]}"
    echo "18 experiments per task × 4 tasks = 72 experiments"
    exit 0
fi

campaign_log "=========================================="
campaign_log "v3 Bulbasaur runner started (GPU ${GPU})"
campaign_log "Tasks: ${TASKS[*]}"
campaign_log "=========================================="

for task in "${TASKS[@]}"; do
    campaign_log "── Starting v3 for ${task} ──"
    GPU=$GPU TASK=$task bash "${QUEUE_DIR}/standalone_queue_v3.sh" \
        > "${LOG_DIR}/gpu${GPU}_${task}.log" 2>&1
    campaign_log "── v3 ${task} DONE ──"
done

campaign_log "=========================================="
campaign_log "v3 Bulbasaur runner finished"
campaign_log "=========================================="
