#!/usr/bin/env bash
# Server B (Lugia) — v3 RTE rw=0.1 subset (parallel with Server A doing rw=0.05)
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
LOG="${FEDLORA_ROOT}/exp_distributed/campaign_v3_rte_parallel_b.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Parallel-B] $*" | tee -a "$LOG"
}

log_status "=== Parallel v3 RTE (Server B: rw=0.1) START ==="

log_status "Activating RTE partitions for Group B..."
TASK="rte" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
sleep 5

log_status ">>> rte / v3 (rw=0.1) starting"
if TASK="rte" MANIFEST="$MANIFEST" PORT_OFFSET=0 \
   CONTROLLER_IP="$CONTROLLER_IP" NO_CLEANUP=true \
   RW_LIST="0.1" \
   bash "${QUEUE_DIR}/fleet_queue_v3.sh" 2>&1 \
   | tee -a "${LOG%.log}_stdout.log"; then
    log_status "<<< rte / v3 (rw=0.1) DONE"
else
    log_status "<<< rte / v3 (rw=0.1) FAILED"
fi

log_status "=== Parallel v3 RTE (Server B) COMPLETE ==="
