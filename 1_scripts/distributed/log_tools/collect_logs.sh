#!/usr/bin/env bash
# Collect worker container logs from all fleet devices into the experiment
# directory, then merge with the server log into a unified timeline.
#
# Usage:
#   bash 1_scripts/distributed/log_tools/collect_logs.sh <exp_dir> [manifest]
#
# The merged output reproduces a standalone-like single-file view where
# server and all client events are interleaved chronologically.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"

EXP_DIR="${1:?Usage: collect_logs.sh <exp_dir> [manifest]}"
MANIFEST="${2:-${MANIFEST_PATH}}"
CRYSTAL_HOST="${CRYSTAL_HOST:-x86-worker}"

[[ -d "$EXP_DIR" ]] || die "Experiment directory does not exist: $EXP_DIR"
[[ -f "$MANIFEST" ]] || die "Manifest not found: $MANIFEST"

WORKER_LOG_DIR="${EXP_DIR}/worker_logs"
mkdir -p "$WORKER_LOG_DIR"

num_clients=$(jq '.clients | length' "$MANIFEST")
log "Collecting logs from ${num_clients} workers into ${WORKER_LOG_DIR}/"

collect_one() {
    local cid=$1 device_name=$2 container_name=$3

    # crystal_* devices are virtual containers on the x86-worker host
    if [[ "$device_name" == crystal_* ]]; then
        ssh_host="$CRYSTAL_HOST"
    else
        ssh_host="$device_name"
    fi

    local outfile="${WORKER_LOG_DIR}/client_${cid}_${device_name}.log"
    if ssh -o ConnectTimeout=5 "$ssh_host" \
         "docker logs ${container_name} 2>&1" > "$outfile" 2>/dev/null; then
        local lines
        lines=$(wc -l < "$outfile")
        echo "${device_name} (client ${cid}): ${lines} lines"
    else
        echo "${device_name} (client ${cid}): FAILED (container may already be removed)"
        rm -f "$outfile"
    fi
}

for i in $(seq 0 $((num_clients - 1))); do
    cid=$(jq -r ".clients[$i].client_id" "$MANIFEST")
    dname=$(jq -r ".clients[$i].device_name" "$MANIFEST")
    cname=$(jq -r ".clients[$i].container_name" "$MANIFEST")
    collect_one "$cid" "$dname" "$cname" &
done
wait
log "Worker log collection complete."

# ── Collect system_metrics.log from worker containers ──────────────────
log "Collecting system_metrics.log from worker containers..."
METRICS_DIR="${EXP_DIR}/worker_metrics"
mkdir -p "$METRICS_DIR"

collect_metrics_one() {
    local cid=$1 device_name=$2 container_name=$3

    if [[ "$device_name" == crystal_* ]]; then
        ssh_host="$CRYSTAL_HOST"
        local runtime_base="/ssd0/pablo/fedlora_runtime"
    else
        ssh_host="$device_name"
        local runtime_base="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
    fi
    local runtime_logs="${runtime_base}/logs/client_${cid}"

    local outfile="${METRICS_DIR}/system_metrics_client_${cid}.log"
    if ssh -o ConnectTimeout=5 "$ssh_host" \
         "cat ${runtime_logs}/system_metrics.log 2>/dev/null" > "$outfile" 2>/dev/null; then
        if [[ -s "$outfile" ]]; then
            echo "${device_name} (client ${cid}): metrics collected"
        else
            echo "${device_name} (client ${cid}): no system_metrics.log in runtime logs"
            rm -f "$outfile"
        fi
    else
        echo "${device_name} (client ${cid}): FAILED to collect metrics"
        rm -f "$outfile"
    fi
}

for i in $(seq 0 $((num_clients - 1))); do
    cid=$(jq -r ".clients[$i].client_id" "$MANIFEST")
    dname=$(jq -r ".clients[$i].device_name" "$MANIFEST")
    cname=$(jq -r ".clients[$i].container_name" "$MANIFEST")
    collect_metrics_one "$cid" "$dname" "$cname" &
done
wait
log "System metrics collection complete."

# Merge into unified timeline
MERGE_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/merge_logs.py"
SERVER_LOG="${EXP_DIR}/exp_print.log"

if [[ -f "$MERGE_SCRIPT" && -f "$SERVER_LOG" ]]; then
    log "Merging server + worker logs into unified_log.log..."
    python3 "$MERGE_SCRIPT" "$SERVER_LOG" "$WORKER_LOG_DIR" \
        > "${EXP_DIR}/unified_log.log" 2>/dev/null \
        && log "Unified log: $(wc -l < "${EXP_DIR}/unified_log.log") lines" \
        || log "WARNING: merge failed, individual logs still available in worker_logs/"
else
    [[ -f "$SERVER_LOG" ]] || log "WARNING: server log not found at ${SERVER_LOG}"
    [[ -f "$MERGE_SCRIPT" ]] || log "WARNING: merge script not found at ${MERGE_SCRIPT}"
fi

log "Done. Logs in ${WORKER_LOG_DIR}/"
