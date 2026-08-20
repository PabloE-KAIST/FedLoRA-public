#!/usr/bin/env bash
# Server A (Bulbasaur) v3 offload: runs a subset of v3 RTE experiments
# that were originally assigned to Server B, to balance wall-clock time.
#
# Split: Server A takes rw={0.05, 0.1} (12 experiments)
#        Server B takes rw={0.005}      (6 experiments)
#
# This script waits for campaign_server_a.sh to finish before starting,
# so it is safe to launch immediately in a separate tmux session.
#
# Usage:
#   tmux new-session -d -s offload_a bash 1_scripts/distributed/orchestrators/campaign_server_a_v3_offload.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

export CUDA_VISIBLE_DEVICES=3

MANIFEST="distributed/configs/client_manifest_group_a.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
OFFLOAD_LOG="${FEDLORA_ROOT}/exp_distributed/campaign_server_a_v3_offload.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Offload-A] $*" | tee -a "$OFFLOAD_LOG"
}

# ── Wait for campaign_a to release Group A clients ────────
# Check for the actual bash process, not the tmux wrapper
if pgrep -x bash -a 2>/dev/null | grep -q "campaign_server_a\.sh"; then
    log_status "Waiting for campaign_server_a.sh to finish..."
    while pgrep -x bash -a 2>/dev/null | grep -q "campaign_server_a\.sh"; do
        sleep 60
    done
    log_status "campaign_server_a.sh finished — starting v3 offload after 60s cooldown"
    sleep 60
else
    log_status "campaign_server_a.sh already finished — starting v3 offload after 30s cooldown"
    sleep 30
fi

# ── Activate RTE partitions (idempotent) ─────────────────
log_status "Activating RTE partitions for Group A..."
TASK="rte" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
sleep 5

# ── Run v3 with rw=0.05 and rw=0.1 only ─────────────────
log_status ">>> rte / v3 (offloaded: rw=0.05,0.1) starting"
if TASK="rte" MANIFEST="$MANIFEST" PORT_OFFSET=0 \
   CONTROLLER_IP="$CONTROLLER_IP" NO_CLEANUP=true \
   RW_LIST="0.05 0.1" \
   bash "${QUEUE_DIR}/fleet_queue_v3.sh" 2>&1 \
   | tee -a "${OFFLOAD_LOG%.log}_rte_v3.log"; then
    log_status "<<< rte / v3 (offloaded) DONE"
else
    log_status "<<< rte / v3 (offloaded) FAILED"
fi

log_status "=== Offload Server A COMPLETE ==="
