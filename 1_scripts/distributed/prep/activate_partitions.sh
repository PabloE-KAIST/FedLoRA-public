#!/usr/bin/env bash
# Activate a deployed task's partitions via symlinks. No data transfer.
#
# Creates symlinks so that partitions/client_N.pkl → partitions/{task}/client_N.pkl
# on every device (and on Bulbasaur). This is what the Docker volume mount and
# manifest partition_path fields expect.
#
# Usage:
#   TASK=qnli bash 1_scripts/distributed/prep/activate_partitions.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"
require_cmd jq

TASK="${TASK:?TASK env var required (e.g. sst2, qnli, qqp, mnli, cola, stsb, mrpc, rte)}"

# Override manifest path if MANIFEST env var is set (for sub-fleet activations)
if [[ -n "${MANIFEST:-}" ]]; then
    if [[ "$MANIFEST" == /* ]]; then
        MANIFEST_PATH="$MANIFEST"
    else
        MANIFEST_PATH="${FEDLORA_ROOT}/${MANIFEST}"
    fi
    log "Using override manifest: ${MANIFEST_PATH}"
fi

JETSON_RUNTIME="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
CRYSTAL_RUNTIME="/ssd0/pablo/fedlora_runtime"
JETSONS=(agxorin1 agxorin2 agxorin3 agxorin4 agxavier1 agxavier2 orinnx1 orinnx2 orinnx3)

activate_on_host() {
    local ssh_host=$1 user=$2 runtime=$3 jq_filter=$4
    local cids
    cids=$(jq -r ".clients[] | select(${jq_filter}) | .client_id" "$MANIFEST_PATH")
    if [[ -z "$cids" ]]; then
        return 0
    fi
    local cmds=""
    for cid in $cids; do
        cmds+="ln -sf ${TASK}/client_${cid}.pkl ${runtime}/partitions/client_${cid}.pkl; "
        cmds+="ln -sf ${TASK}/client_${cid}.metadata.json ${runtime}/partitions/client_${cid}.metadata.json; "
    done
    ssh "${user}@${ssh_host}" "$cmds"
    log "  ${ssh_host}: activated ${TASK}"
}

# Activate on Bulbasaur (for verify_manifest.sh compatibility)
LOCAL_GEN="${FEDLORA_ROOT}/distributed/data/generated_partitions"
if [[ -d "${LOCAL_GEN}/${TASK}" ]]; then
    for f in "${LOCAL_GEN}/${TASK}"/client_*.pkl "${LOCAL_GEN}/${TASK}"/client_*.metadata.json; do
        [[ -f "$f" ]] || continue
        ln -sf "${TASK}/$(basename "$f")" "${LOCAL_GEN}/$(basename "$f")"
    done
    log "Bulbasaur: activated ${TASK}"
else
    log "WARNING: ${LOCAL_GEN}/${TASK} not found on Bulbasaur; skipping local symlinks."
fi

# Activate on fleet
for dev in "${JETSONS[@]}"; do
    activate_on_host "$dev" ${FEDLORA_JETSON_USER:-ubuntu} "$JETSON_RUNTIME" ".device_name==\"${dev}\""
done
# Crystal: per-GPU partition directories (each virtual DA has its own)
declare -A CRYSTAL_GPU=( [crystal_gpu5]=5 [crystal_gpu4]=4 [crystal_gpu3]=3 [crystal_gpu2]=2 )
crystal_devs=$(jq -r '.clients[] | select(.device_class=="x86") | .device_name' "$MANIFEST_PATH")
if [[ -n "$crystal_devs" ]]; then
    local_cmds=""
    for dev in $crystal_devs; do
        gpu=${CRYSTAL_GPU[$dev]}
        cid=$(jq -r ".clients[] | select(.device_name==\"${dev}\") | .client_id" "$MANIFEST_PATH")
        # Symlink in the SHARED partitions/ dir the container actually mounts (see deploy_partitions.sh)
        pdir="${CRYSTAL_RUNTIME}/partitions"
        local_cmds+="ln -sf ${TASK}/client_${cid}.pkl ${pdir}/client_${cid}.pkl; "
        local_cmds+="ln -sf ${TASK}/client_${cid}.metadata.json ${pdir}/client_${cid}.metadata.json; "
    done
    ssh "${FEDLORA_X86_USER:-ubuntu}@${FEDLORA_X86_HOST:?set FEDLORA_X86_HOST}" "$local_cmds"
    log "  x86-worker: activated ${TASK} (per-GPU dirs)"
fi
log "All devices: ${TASK} partitions activated."
