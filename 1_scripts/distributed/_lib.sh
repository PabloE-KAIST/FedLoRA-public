#!/usr/bin/env bash
# Shared helpers for distributed FL scripts.
# Sourced by sibling scripts. Not intended to be run directly.

set -euo pipefail

# Resolve repo root from this file's location (1_scripts/distributed/_lib.sh).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FEDLORA_ROOT="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

export MANIFEST_PATH="${FEDLORA_ROOT}/distributed/configs/client_manifest.json"
export DIST_CONFIG_REL="2_yamls/fedit/fedit_distributed.yaml"
export PARTITION_DIR_REL="distributed/data/generated_partitions"
export MILESTONE_LOG_DIR_REL="milestone1_logs"

log()  { printf '[fl] %s\n' "$*" >&2; }
die()  { printf '[fl][ERROR] %s\n' "$*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

# manifest_clients -> JSON array at .clients
manifest_clients() {
    require_cmd jq
    [[ -f "${MANIFEST_PATH}" ]] || die "manifest not found: ${MANIFEST_PATH}"
    jq -c '.clients[]' "${MANIFEST_PATH}"
}

# manifest_field <client_index> <jq-path-without-leading-dot>
# Example: manifest_field 0 client_id
manifest_field() {
    require_cmd jq
    local idx="$1"; local path="$2"
    jq -r ".clients[${idx}].${path}" "${MANIFEST_PATH}"
}

manifest_num_clients() {
    require_cmd jq
    jq '.clients | length' "${MANIFEST_PATH}"
}

# Collect the set of distinct device_class values from the manifest.
manifest_device_classes() {
    require_cmd jq
    jq -r '[.clients[].device_class] | unique | .[]' "${MANIFEST_PATH}"
}
