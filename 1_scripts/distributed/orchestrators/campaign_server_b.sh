#!/usr/bin/env bash
# Server B (Lugia) campaign: mrpc → rte
# Methods: v3 (mrpc) → HetLoRA (mrpc), then HetLoRA + v3 (rte)
#
# Usage (via tmux on Lugia):
#   tmux new-session -d -s campaign_b bash 1_scripts/distributed/orchestrators/campaign_server_b.sh
set -euo pipefail

export PATH="$HOME/miniconda3/envs/fedlora/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2
export LOGFILE="${FEDLORA_ROOT}/exp_distributed/fl_server_b.log"
export TMPDIR="${FEDLORA_WORKSPACE:-$HOME}/tmp"
mkdir -p "$TMPDIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

MANIFEST="distributed/configs/client_manifest_group_b.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
CAMPAIGN_LOG="${FEDLORA_ROOT}/exp_distributed/campaign_server_b.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Server-B] $*" | tee -a "$CAMPAIGN_LOG"
}

run_queue() {
    local task=$1 method=$2
    log_status ">>> ${task} / ${method} starting"
    if TASK="$task" MANIFEST="$MANIFEST" PORT_OFFSET=0 \
       CONTROLLER_IP="$CONTROLLER_IP" NO_CLEANUP=true \
       bash "${QUEUE_DIR}/fleet_queue_${method}.sh" 2>&1 \
       | tee -a "${CAMPAIGN_LOG%.log}_${task}_${method}.log"; then
        log_status "<<< ${task} / ${method} DONE"
    else
        log_status "<<< ${task} / ${method} FAILED (continuing)"
    fi
    sleep 30
}

activate_task() {
    local task=$1
    log_status "Activating partitions for ${task}..."
    TASK="$task" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
    sleep 5
}

log_status "=== Campaign Server B START ==="

# ── MRPC ──────────────────────────────────────────────────
log_status "===== TASK: mrpc ====="
activate_task mrpc
run_queue mrpc v3
run_queue mrpc hetlora

# ── RTE ───────────────────────────────────────────────────
log_status "===== TASK: rte ====="
activate_task rte
run_queue rte hetlora
run_queue rte v3

log_status "=== Campaign Server B COMPLETE ==="
