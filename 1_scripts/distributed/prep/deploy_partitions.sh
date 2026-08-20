#!/usr/bin/env bash
# Deploy partitions for a given GLUE task to the fleet.
# Partitions persist in task-specific subdirs on each device — deploy once,
# reuse across methods (FedIT, HetLoRA, FAH-QLoRA, etc.).
#
# Each device receives only the partition(s) assigned to it by the manifest.
#
# Usage:
#   TASK=qnli  bash 1_scripts/distributed/prep/deploy_partitions.sh          # all devices
#   TASK=qnli  bash 1_scripts/distributed/prep/deploy_partitions.sh agxorin1  # single device
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"
require_cmd jq
require_cmd rsync

TASK="${TASK:?TASK env var required (e.g. sst2, qnli, qqp, mnli, cola, stsb, mrpc, rte)}"
SRC_DIR="${FEDLORA_ROOT}/distributed/data/generated_partitions/${TASK}"
[[ -d "$SRC_DIR" ]] || die "Partition dir not found: ${SRC_DIR}. Run prepare_partitions.py --data-type '${TASK}@glue' first."

# Override manifest path if MANIFEST env var is set (for sub-fleet deploys)
if [[ -n "${MANIFEST:-}" ]]; then
    if [[ "$MANIFEST" == /* ]]; then
        MANIFEST_PATH="$MANIFEST"
    else
        MANIFEST_PATH="${FEDLORA_ROOT}/${MANIFEST}"
    fi
    log "Using override manifest: ${MANIFEST_PATH}"
fi

SINGLE_DEVICE="${1:-}"

JETSON_RUNTIME="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
CRYSTAL_RUNTIME="/ssd0/pablo/fedlora_runtime"
JETSONS=(agxorin1 agxorin2 agxorin3 agxorin4 agxavier1 agxavier2 orinnx1 orinnx2 orinnx3)

deploy_to_host() {
    local ssh_host=$1 user=$2 runtime=$3 jq_filter=$4
    local cids
    cids=$(jq -r ".clients[] | select(${jq_filter}) | .client_id" "$MANIFEST_PATH")
    if [[ -z "$cids" ]]; then
        log "  ${ssh_host}: no matching clients, skipping."
        return 0
    fi
    for cid in $cids; do
        local pkl="${SRC_DIR}/client_${cid}.pkl"
        local meta="${SRC_DIR}/client_${cid}.metadata.json"
        [[ -f "$pkl" ]] || die "Missing: ${pkl}"
        log "  ${ssh_host}: deploying client_${cid}.pkl → partitions/${TASK}/ ($(du -h "$pkl" | cut -f1))"
        ssh "${user}@${ssh_host}" "mkdir -p ${runtime}/partitions/${TASK}"
        rsync -avz "$pkl" "${user}@${ssh_host}:${runtime}/partitions/${TASK}/"
        [[ -f "$meta" ]] && rsync -avz "$meta" "${user}@${ssh_host}:${runtime}/partitions/${TASK}/"
    done
}

declare -A CRYSTAL_GPU=( [crystal_gpu5]=5 [crystal_gpu4]=4 [crystal_gpu3]=3 [crystal_gpu2]=2 )

deploy_to_crystal_per_gpu() {
    local jq_filter=$1
    local cids devs
    devs=$(jq -r ".clients[] | select(${jq_filter}) | .device_name" "$MANIFEST_PATH")
    for dev in $devs; do
        local gpu=${CRYSTAL_GPU[$dev]}
        local cid=$(jq -r ".clients[] | select(.device_name==\"${dev}\") | .client_id" "$MANIFEST_PATH")
        local pkl="${SRC_DIR}/client_${cid}.pkl"
        local meta="${SRC_DIR}/client_${cid}.metadata.json"
        [[ -f "$pkl" ]] || die "Missing: ${pkl}"
        # Crystal containers bind-mount the SHARED partitions/ dir (not partitions_gpu*/),
        # so deploy the fresh partition there. The per-GPU dirs were mounted by nothing,
        # leaving containers on a stale symlink (root cause of the STSB->RTE nll_loss assert).
        # Crystal's workers use distinct client_ids (4,5) so the shared dir has no collision.
        local pdir="${CRYSTAL_RUNTIME}/partitions"
        log "  x86-worker (gpu${gpu}): deploying client_${cid}.pkl → partitions/${TASK}/ ($(du -h "$pkl" | cut -f1))"
        ssh "${FEDLORA_X86_USER:-ubuntu}@${FEDLORA_X86_HOST:?set FEDLORA_X86_HOST}" "mkdir -p ${pdir}/${TASK}"
        rsync -avz "$pkl" "${FEDLORA_X86_USER:-ubuntu}@${FEDLORA_X86_HOST:?set FEDLORA_X86_HOST}:${pdir}/${TASK}/"
        [[ -f "$meta" ]] && rsync -avz "$meta" "${FEDLORA_X86_USER:-ubuntu}@${FEDLORA_X86_HOST:?set FEDLORA_X86_HOST}:${pdir}/${TASK}/"
    done
}

if [[ -n "$SINGLE_DEVICE" ]]; then
    if [[ "$SINGLE_DEVICE" == "x86-worker" ]]; then
        deploy_to_crystal_per_gpu '.device_class=="x86"'
    else
        deploy_to_host "$SINGLE_DEVICE" ${FEDLORA_JETSON_USER:-ubuntu} "$JETSON_RUNTIME" ".device_name==\"${SINGLE_DEVICE}\""
    fi
else
    for dev in "${JETSONS[@]}"; do
        deploy_to_host "$dev" ${FEDLORA_JETSON_USER:-ubuntu} "$JETSON_RUNTIME" ".device_name==\"${dev}\""
    done
    deploy_to_crystal_per_gpu '.device_class=="x86"'
    log "All devices: ${TASK} partitions deployed."
fi
