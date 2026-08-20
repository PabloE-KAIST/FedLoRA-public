#!/usr/bin/env bash
# Launch the FL server in proto/relay mode for the 12-client fleet.
# Usage:
#   bash 1_scripts/distributed/orchestrators/launch_fl_server.sh \
#       --config 2_yamls/hetlora/hetlora_distributed.yaml \
#       --manifest distributed/configs/client_manifest.json
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"

# Defaults (overridable via flags)
CONFIG="${FEDLORA_ROOT}/${DIST_CONFIG_REL}"
MANIFEST="${MANIFEST_PATH}"
WORKER_CONFIG=""
LOGFILE="${LOGFILE:-/tmp/fl_distributed_p${PORT_OFFSET:-0}.log}"
BANDWIDTH_SETTING="${BANDWIDTH_SETTING:-1}"
BANDWIDTH_JSON="${BANDWIDTH_JSON:-${FEDLORA_ROOT}/1_scripts/distributed/infra/generated_bandwidths/bandwidth_limits_dl.json}"
SERVER_NIC="${SERVER_NIC:-enp66s0f0}"
PORT_OFFSET="${PORT_OFFSET:-0}"
NO_TC=false
FOREGROUND=false
EXTRA_OPTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)            CONFIG="${FEDLORA_ROOT}/$2"; shift 2 ;;
        --manifest)          MANIFEST="${FEDLORA_ROOT}/$2"; shift 2 ;;
        --worker-config)     WORKER_CONFIG="$2"; shift 2 ;;
        --logfile)           LOGFILE="$2"; shift 2 ;;
        --bandwidth-setting) BANDWIDTH_SETTING="$2"; shift 2 ;;
        --bandwidth-json)    BANDWIDTH_JSON="$2"; shift 2 ;;
        --server-nic)        SERVER_NIC="$2"; shift 2 ;;
        --port-offset)       PORT_OFFSET="$2"; shift 2 ;;
        --no-tc)             NO_TC=true; shift ;;
        --foreground)        FOREGROUND=true; shift ;;
        --)                  shift; EXTRA_OPTS+=("$@"); break ;;
        *)                   EXTRA_OPTS+=("$1"); shift ;;
    esac
done

# Kill any existing server with the same port-offset (avoids killing concurrent instances)
pkill -f "distributed.server.main.*--port-offset ${PORT_OFFSET}[^0-9]" 2>/dev/null || true
pkill -f "distributed.server.main.*--port-offset ${PORT_OFFSET}$" 2>/dev/null || true
sleep 1

log "Launching FL server: config=${CONFIG}, manifest=${MANIFEST}"
log "Log: ${LOGFILE}"

cd "$FEDLORA_ROOT"

CMD=(python -m distributed.server.main
    --config "$CONFIG"
    --manifest "$MANIFEST"
    --proto
    --port-offset "$PORT_OFFSET"
    --device-port-offset 100
    --device-wait-timeout 180
    --verbose)

# Default worker config to same yaml as server config (container-relative path)
if [[ -z "$WORKER_CONFIG" ]]; then
    # Strip FEDLORA_ROOT prefix to get the relative path workers use
    WORKER_CONFIG="${CONFIG#${FEDLORA_ROOT}/}"
fi
CMD+=(--worker-config-path "$WORKER_CONFIG")
log "Worker config: ${WORKER_CONFIG}"

if [[ "$NO_TC" == "true" ]]; then
    CMD+=(--bandwidth-setting 0)
    log "TC disabled (--no-tc)"
else
    CMD+=(--bandwidth-setting "$BANDWIDTH_SETTING")
    CMD+=(--bandwidth-json "$BANDWIDTH_JSON")
    CMD+=(--server-nic "$SERVER_NIC")
    log "TC enabled: setting=${BANDWIDTH_SETTING}, json=${BANDWIDTH_JSON}, nic=${SERVER_NIC}"
fi

if [[ ${#EXTRA_OPTS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_OPTS[@]}")
fi

if [[ "$FOREGROUND" == "true" ]]; then
    log "Running server in foreground..."
    "${CMD[@]}" 2>&1 | tee "$LOGFILE"
else
    "${CMD[@]}" > "$LOGFILE" 2>&1 &
    SERVER_PID=$!
    log "Server PID: ${SERVER_PID}"
    echo "$SERVER_PID"
fi
