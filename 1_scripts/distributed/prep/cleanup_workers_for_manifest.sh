#!/usr/bin/env bash
# Stop and remove fl_worker containers only for devices in the given manifest.
# Safe to run during concurrent dual-server execution.
#
# Usage:
#   MANIFEST=distributed/configs/client_manifest_group_a.json \
#       bash 1_scripts/distributed/prep/cleanup_workers_for_manifest.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"
require_cmd jq

if [[ -n "${MANIFEST:-}" ]]; then
    if [[ "$MANIFEST" == /* ]]; then
        MANIFEST_PATH="$MANIFEST"
    else
        MANIFEST_PATH="${FEDLORA_ROOT}/${MANIFEST}"
    fi
fi

DEVICES=($(jq -r '.clients[].device_name' "$MANIFEST_PATH"))
log "Cleaning worker containers for ${#DEVICES[@]} manifest devices..."

CRYSTAL_DEVICES=()
JETSON_DEVICES=()
for dev in "${DEVICES[@]}"; do
    if [[ "$dev" == crystal_* ]]; then
        CRYSTAL_DEVICES+=("$dev")
    else
        JETSON_DEVICES+=("$dev")
    fi
done

for host in "${JETSON_DEVICES[@]}"; do
    (
        ssh -o ConnectTimeout=5 "$host" bash -s -- "$host" <<'REMOTE'
CONTAINER="fl_worker_${1}"
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true
REMOTE
        echo "$host: cleaned"
    ) &
done

if [[ ${#CRYSTAL_DEVICES[@]} -gt 0 ]]; then
    NAMES=""
    for dev in "${CRYSTAL_DEVICES[@]}"; do
        NAMES+="fl_worker_${dev} "
    done
    ssh -o ConnectTimeout=5 x86-worker bash -s -- $NAMES <<'REMOTE'
for c in "$@"; do
    docker stop "$c" 2>/dev/null || true
    docker rm "$c" 2>/dev/null || true
done
REMOTE
    echo "x86-worker (${CRYSTAL_DEVICES[*]}): cleaned"
fi

wait
log "Manifest-scoped cleanup done."
