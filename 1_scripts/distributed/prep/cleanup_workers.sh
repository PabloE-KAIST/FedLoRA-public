#!/usr/bin/env bash
# Stop and remove all fl_worker containers across the fleet.
# Usage: bash 1_scripts/distributed/prep/cleanup_workers.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"

ALL_HOSTS=( agxorin1 agxorin2 agxorin3 agxorin4 agxavier1 agxavier2 orinnx1 orinnx2 orinnx3 )
CRYSTAL_HOST="${CRYSTAL_HOST:-x86-worker}"

cleanup_host() {
    local host=$1
    local logs_base="${2:-${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}/logs}"
    ssh -o ConnectTimeout=5 "$host" bash -s -- "$logs_base" <<'REMOTE'
docker stop $(docker ps -q --filter "name=fl_worker") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=fl_worker") 2>/dev/null || true
# Worker containers create files as root; rm as non-root user silently fails.
# Use a disposable container to delete as root, with non-root fallback.
docker run --rm -v "${1}:/cleanup" alpine:latest sh -c 'rm -rf /cleanup/client_*' 2>/dev/null || \
    rm -rf "${1}"/client_*/ 2>/dev/null || true
REMOTE
    echo "$host: cleaned"
}

log "Cleaning worker containers on Jetsons..."
for host in "${ALL_HOSTS[@]}"; do
    cleanup_host "$host" &
done
wait

log "Cleaning worker containers on x86-worker..."
cleanup_host "$CRYSTAL_HOST" "/ssd0/pablo/fedlora_runtime/logs"

log "All worker containers removed."
