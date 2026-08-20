#!/usr/bin/env bash
# Verify that the client manifest is internally consistent AND that all
# referenced artifacts exist on the local (server/build) filesystem.
# Read-only. Does not touch remote systems.
#
# Exit codes:
#   0 - manifest and artifacts OK
#   1 - any check failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_lib.sh
source "${SCRIPT_DIR}/../_lib.sh"

require_cmd jq

log "manifest: ${MANIFEST_PATH}"
NUM_CLIENTS="$(manifest_num_clients)"
log "num_clients=${NUM_CLIENTS}"

[[ "${NUM_CLIENTS}" -ge 2 ]] || die "manifest must define >= 2 clients for Milestone 1"

fail=0
for idx in $(seq 0 $((NUM_CLIENTS - 1))); do
    CID="$(manifest_field "${idx}" client_id)"
    DN="$(manifest_field "${idx}" device_name)"
    DC="$(manifest_field "${idx}" device_class)"
    CN="$(manifest_field "${idx}" container_name)"
    IMG="$(manifest_field "${idx}" image)"
    PP="$(manifest_field "${idx}" partition_path)"

    log "client[${idx}]: id=${CID} device=${DN} class=${DC} container=${CN} image=${IMG}"
    log "  partition_path=${PP}"

    # Image tag must be fedlora-worker:<device_class>
    if [[ "${IMG}" != "fedlora-worker:${DC}" ]]; then
        log "  [FAIL] image '${IMG}' does not match fedlora-worker:${DC}"
        fail=1
    fi

    # Container name convention: fl_worker_<device_name>
    if [[ "${CN}" != "fl_worker_${DN}" ]]; then
        log "  [WARN] container_name '${CN}' does not follow fl_worker_<device_name> convention"
    fi

    # Partition artifact must exist locally.
    ABS_PP="${FEDLORA_ROOT}/${PP}"
    if [[ ! -f "${ABS_PP}" ]]; then
        log "  [FAIL] partition artifact missing: ${ABS_PP}"
        fail=1
    else
        SIZE="$(stat -c%s "${ABS_PP}")"
        log "  partition present (${SIZE} bytes)"
    fi

    # Metadata sidecar (expected next to the .pkl) — soft check.
    META="${ABS_PP%.pkl}.metadata.json"
    if [[ ! -f "${META}" ]]; then
        log "  [WARN] metadata sidecar missing: ${META}"
    fi
done

# Config file must exist.
if [[ ! -f "${FEDLORA_ROOT}/${DIST_CONFIG_REL}" ]]; then
    log "[FAIL] distributed config missing: ${FEDLORA_ROOT}/${DIST_CONFIG_REL}"
    fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
    die "manifest verification FAILED"
fi

log "manifest verification OK"
