#!/usr/bin/env bash
# Master fleet queue: runs all 8 GLUE tasks × all 5 methods.
#
# Phase 1 — Small tasks (6 clients): rte → mrpc → stsb → cola
#   Dual-server concurrent execution: Group A on Bulbasaur, Group B on Lugia.
#   Methods are paired and run simultaneously on the two sub-fleets.
#
# Phase 2 — Big tasks (12 clients): sst2 → qnli → qqp → mnli
#   Full 12-device fleet on Bulbasaur, one experiment at a time.
#
# Usage:
#   bash 1_scripts/distributed/orchestrators/fleet_master_queue.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"
source "${FEDLORA_ROOT}/1_scripts/baseline_runs/glue/_glue_lib.sh"

QUEUE_DIR="${SCRIPT_DIR}/../queues"
PREP_DIR="${SCRIPT_DIR}/../prep"

QUEUE_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_master_queue.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$QUEUE_LOG"
}

SMALL_TASKS=(rte mrpc stsb cola)
BIG_TASKS=(sst2 qnli qqp mnli)
ALL_METHODS=(fedit_r64 hetlora fahqlora v2 v3)

MANIFEST_A="distributed/configs/client_manifest_group_a.json"
MANIFEST_B="distributed/configs/client_manifest_group_b.json"
MANIFEST_FULL="distributed/configs/client_manifest.json"

BULBASAUR_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_HOST="gpu-host-b"
LUGIA_FEDLORA="${FEDLORA_ROOT}"

log_status "=== Fleet master queue started ==="
log_status "Small tasks: ${SMALL_TASKS[*]}"
log_status "Big tasks: ${BIG_TASKS[*]}"
log_status "Methods: ${ALL_METHODS[*]}"
log_status "Servers: Bulbasaur (${BULBASAUR_IP}) + Lugia (${LUGIA_IP})"

# ── Phase 1: Small tasks — dual-server concurrent sub-fleets ─────
log_status "=== PHASE 1: Small tasks (6-client, dual-server) ==="

bash "${PREP_DIR}/cleanup_workers.sh"

for task in "${SMALL_TASKS[@]}"; do
    log_status "--- Task: ${task} ---"

    # Activate partitions for both groups
    TASK="$task" MANIFEST="$MANIFEST_A" bash "${PREP_DIR}/activate_partitions.sh"
    TASK="$task" MANIFEST="$MANIFEST_B" bash "${PREP_DIR}/activate_partitions.sh"

    # Process methods in pairs: A on Bulbasaur, B on Lugia
    methods=("${ALL_METHODS[@]}")
    slot=0

    while [[ ${#methods[@]} -gt 0 ]]; do
        method_a="${methods[0]}"
        methods=("${methods[@]:1}")

        if [[ ${#methods[@]} -gt 0 ]]; then
            method_b="${methods[0]}"
            methods=("${methods[@]:1}")
            slot=$((slot + 1))

            log_status "[${task}] slot ${slot}: ${method_a} (Bulbasaur) + ${method_b} (Lugia)"

            # Group A on Bulbasaur (local) — restarts its own DAs via CONTROLLER_IP
            (
                TASK="$task" MANIFEST="$MANIFEST_A" PORT_OFFSET=0 \
                    CONTROLLER_IP="$BULBASAUR_IP" NO_CLEANUP=true \
                    bash "${QUEUE_DIR}/fleet_queue_${method_a}.sh" 2>&1 | \
                    tee -a "${QUEUE_LOG%.log}_${task}_${method_a}.log"
            ) &
            PID_A=$!

            # Group B on Lugia (remote) — restarts its own DAs via CONTROLLER_IP
            (
                ssh "$LUGIA_HOST" bash <<REMOTE_QUEUE
cd ${LUGIA_FEDLORA}
export PATH="\$HOME/miniconda3/envs/fedlora/bin:\$PATH"
TASK="${task}" MANIFEST="${MANIFEST_B}" PORT_OFFSET=0 \
    CONTROLLER_IP="${LUGIA_IP}" NO_CLEANUP=true \
    bash 1_scripts/distributed/queues/fleet_queue_${method_b}.sh 2>&1
REMOTE_QUEUE
            ) 2>&1 | tee -a "${QUEUE_LOG%.log}_${task}_${method_b}.log" &
            PID_B=$!

            wait $PID_A && log_status "[${task}] ${method_a} (Bulbasaur): DONE" \
                        || log_status "[${task}] ${method_a} (Bulbasaur): FAILED"
            wait $PID_B && log_status "[${task}] ${method_b} (Lugia): DONE" \
                        || log_status "[${task}] ${method_b} (Lugia): FAILED"
        else
            # Odd method out — run on Bulbasaur (group A) only
            slot=$((slot + 1))
            log_status "[${task}] slot ${slot}: ${method_a} (Bulbasaur only)"

            TASK="$task" MANIFEST="$MANIFEST_A" PORT_OFFSET=0 \
                CONTROLLER_IP="$BULBASAUR_IP" \
                bash "${QUEUE_DIR}/fleet_queue_${method_a}.sh" 2>&1 | \
                tee -a "${QUEUE_LOG%.log}_${task}_${method_a}.log" \
                && log_status "[${task}] ${method_a}: DONE" \
                || log_status "[${task}] ${method_a}: FAILED"
        fi

        sleep 15
    done

    log_status "--- Task ${task} complete ---"
done

log_status "=== PHASE 1 complete ==="

# ── Phase 2: Big tasks — full fleet on Bulbasaur ─────────────────
log_status "=== PHASE 2: Big tasks (12-client full fleet) ==="

log_status "Restarting all DAs → Bulbasaur for full fleet..."
CONTROLLER_IP="$BULBASAUR_IP" bash "${PREP_DIR}/restart_all_das.sh"
bash "${PREP_DIR}/cleanup_workers.sh"

for task in "${BIG_TASKS[@]}"; do
    log_status "--- Task: ${task} ---"

    TASK="$task" bash "${PREP_DIR}/activate_partitions.sh"

    for method in "${ALL_METHODS[@]}"; do
        log_status "[${task}] Running: ${method}"

        TASK="$task" MANIFEST="$MANIFEST_FULL" PORT_OFFSET=0 \
            bash "${QUEUE_DIR}/fleet_queue_${method}.sh" 2>&1 | \
            tee -a "${QUEUE_LOG%.log}_${task}_${method}.log" \
            && log_status "[${task}] ${method}: DONE" \
            || log_status "[${task}] ${method}: FAILED"
    done

    log_status "--- Task ${task} complete ---"
done

log_status "=== PHASE 2 complete ==="
log_status "=== Fleet master queue finished ==="
