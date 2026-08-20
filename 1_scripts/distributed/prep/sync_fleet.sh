#!/usr/bin/env bash
# Sync updated source overlays to all fleet devices.
#
# Deploys: distributed/, federatedscope/, 2_yamls/, launch_worker.sh,
#          and the compose template to each device's fedlora_runtime/.
#
# Usage:
#   bash 1_scripts/distributed/prep/sync_fleet.sh            # all devices
#   bash 1_scripts/distributed/prep/sync_fleet.sh agxorin1   # single device
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"

SINGLE_DEVICE="${1:-}"

JETSON_RUNTIME="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
CRYSTAL_RUNTIME="/ssd0/pablo/fedlora_runtime"
COMPOSE_SRC="${FEDLORA_ROOT}/distributed/docker/docker-compose.aegisgov.jetson.yml"
COMPOSE_X86_SRC="${FEDLORA_ROOT}/distributed/docker/docker-compose.server.fedlora.yml"
COMPOSE_DST_JETSON="pablo/AegisGov-master/dockerfiles/docker-compose.jetson.yml"

sync_device() {
    local name="$1"
    local user="$2"
    local runtime="$3"
    local is_crystal="$4"

    log "Syncing ${name} (${runtime})..."

    rsync -avz --delete \
        --exclude='data/' \
        --exclude='__pycache__/' \
        "${FEDLORA_ROOT}/distributed/" \
        "${user}@${name}:${runtime}/distributed/"

    rsync -avz --delete \
        --exclude='__pycache__/' \
        "${FEDLORA_ROOT}/federatedscope/" \
        "${user}@${name}:${runtime}/federatedscope/"

    rsync -avz --delete \
        --exclude='__pycache__/' \
        "${FEDLORA_ROOT}/2_yamls/" \
        "${user}@${name}:${runtime}/2_yamls/"

    scp -q "${FEDLORA_ROOT}/distributed/docker/launch_worker.sh" \
        "${user}@${name}:${runtime}/launch_worker.sh"

    if [[ "$is_crystal" == "yes" ]]; then
        scp -q "$COMPOSE_X86_SRC" \
            "${user}@${name}:/ssd0/pablo/AegisGov-master/dockerfiles/docker-compose.server.yml"
    else
        scp -q "$COMPOSE_SRC" \
            "${user}@${name}:/home/${user}/${COMPOSE_DST_JETSON}"
    fi

    # Deploy bandwidth shaping JSONs if they exist
    local bw_json_dir="${FEDLORA_ROOT}/1_scripts/distributed/infra/generated_bandwidths"
    if [[ -f "${bw_json_dir}/bandwidth_limits_ul.json" ]]; then
        local bw_dst
        if [[ "$is_crystal" == "yes" ]]; then
            bw_dst="/ssd0/pablo/AegisGov-master/jsons/bandwidths"
        else
            bw_dst="/home/${user}/pablo/AegisGov-master/jsons/bandwidths"
        fi
        scp -q "${bw_json_dir}/bandwidth_limits_ul.json" \
            "${user}@${name}:${bw_dst}/bandwidth_limits1.json"
        log "  ${name}: deployed bandwidth_limits_ul.json as bandwidth_limits1.json"
    fi

    log "  ${name} done."
}

# Jetson devices (all use ${FEDLORA_JETSON_USER:-ubuntu} user)
JETSONS=(agxorin1 agxorin2 agxorin3 agxorin4 agxavier1 agxavier2 orinnx1 orinnx2 orinnx3)

# Crystal x86 virtual workers
CRYSTALS=(x86-worker)

if [[ -n "$SINGLE_DEVICE" ]]; then
    if [[ "$SINGLE_DEVICE" == "x86-worker" ]]; then
        sync_device x86-worker pablo "$CRYSTAL_RUNTIME" yes
    else
        sync_device "$SINGLE_DEVICE" ${FEDLORA_JETSON_USER:-ubuntu} "$JETSON_RUNTIME" no
    fi
else
    for dev in "${JETSONS[@]}"; do
        sync_device "$dev" ${FEDLORA_JETSON_USER:-ubuntu} "$JETSON_RUNTIME" no
    done
    for dev in "${CRYSTALS[@]}"; do
        sync_device "$dev" pablo "$CRYSTAL_RUNTIME" yes
    done
    log "All devices synced."
fi
