#!/usr/bin/env bash
# Injection: run v3 rw=0.1 experiments for stsb on Bulbasaur (Group A)
# to offload work from Lugia while Bulbasaur is idle.
#
# Lugia keeps: rw=0.005 (done) + rw=0.05 (in progress)
# Bulbasaur takes: rw=0.1 (6 experiments)
#
# Usage:
#   tmux new-session -d -s inject_stsb_v3 \
#       bash 1_scripts/distributed/orchestrators/inject_stsb_v3_bulbasaur.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

export CUDA_VISIBLE_DEVICES=3

TASK="stsb"
MANIFEST="distributed/configs/client_manifest_group_a.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
INJECT_LOG="${FEDLORA_ROOT}/exp_distributed/inject_stsb_v3_bulbasaur.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Inject-stsb-v3] $*" | tee -a "$INJECT_LOG"
}

log_status "=== stsb v3 injection on Bulbasaur START ==="
log_status "Taking rw=0.1 block (6 experiments)"

# Partitions should already be active from the campaign prep.
# Re-activate to be safe (idempotent).
log_status "Activating stsb partitions for Group A..."
TASK="$TASK" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
sleep 5

log_status ">>> stsb / v3 (rw=0.1) starting"
if TASK="$TASK" MANIFEST="$MANIFEST" PORT_OFFSET=0 \
   CONTROLLER_IP="$CONTROLLER_IP" NO_CLEANUP=true \
   RW_LIST="0.1" \
   bash "${QUEUE_DIR}/fleet_queue_v3.sh" 2>&1 \
   | tee -a "${INJECT_LOG%.log}_queue.log"; then
    log_status "<<< stsb / v3 (rw=0.1) DONE"
else
    log_status "<<< stsb / v3 (rw=0.1) FAILED (check queue log)"
fi

log_status "=== stsb v3 injection COMPLETE ==="
