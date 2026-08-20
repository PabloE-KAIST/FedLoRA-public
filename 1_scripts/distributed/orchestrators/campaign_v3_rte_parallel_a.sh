#!/usr/bin/env bash
# Server A (Bulbasaur) — v3 RTE rw=0.05 subset (parallel with Server B doing rw=0.1)
# Skip-if-exists handles the 2 already completed rw=0.05 experiments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

export CUDA_VISIBLE_DEVICES=3

MANIFEST="distributed/configs/client_manifest_group_a.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LOG="${FEDLORA_ROOT}/exp_distributed/campaign_v3_rte_parallel_a.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Parallel-A] $*" | tee -a "$LOG"
}

log_status "=== Parallel v3 RTE (Server A: rw=0.05) START ==="

log_status "Activating RTE partitions for Group A..."
TASK="rte" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
sleep 5

log_status ">>> rte / v3 (rw=0.05) starting"
if TASK="rte" MANIFEST="$MANIFEST" PORT_OFFSET=0 \
   CONTROLLER_IP="$CONTROLLER_IP" NO_CLEANUP=true \
   RW_LIST="0.05" \
   bash "${QUEUE_DIR}/fleet_queue_v3.sh" 2>&1 \
   | tee -a "${LOG%.log}_stdout.log"; then
    log_status "<<< rte / v3 (rw=0.05) DONE"
else
    log_status "<<< rte / v3 (rw=0.05) FAILED"
fi

log_status "=== Parallel v3 RTE (Server A) COMPLETE ==="
