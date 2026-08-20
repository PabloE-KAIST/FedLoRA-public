#!/usr/bin/env bash
# Render the docker-run command that a given manifest client entry would use
# for the Milestone 1 manual dry-run init (Step 8 of the runbook).
#
# This script is LOCAL and does TWO things:
#   1) Prints the exact docker run command (--dry-run-init) for the client.
#   2) If --execute is passed, runs it locally using the LOCAL image.
#      Only meaningful on a host where the per-class image was built AND
#      the mounted runtime paths exist locally (typically the build host
#      for a smoke test, NOT the target device).
#
# On the real devices, use the printed command verbatim (after copying the
# image and runtime files). The production path is control-plane driven —
# this script exists for pre-flight proofs, not production launch.
#
# Usage:
#   distributed/scripts/launch_worker_dry_run.sh <client_id> [--execute]
#
# Example:
#   distributed/scripts/launch_worker_dry_run.sh 1            # print only
#   distributed/scripts/launch_worker_dry_run.sh 1 --execute  # run locally

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

require_cmd jq

[[ $# -ge 1 ]] || die "usage: $0 <client_id> [--execute]"
WANT_CID="$1"
EXECUTE=0
if [[ "${2:-}" == "--execute" ]]; then
    EXECUTE=1
fi

# Locate matching client in manifest.
idx="$(jq -r --argjson cid "${WANT_CID}" \
    '.clients | map(.client_id) | index($cid)' "${MANIFEST_PATH}")"
[[ "${idx}" != "null" ]] || die "client_id=${WANT_CID} not in manifest"

CID="$(manifest_field "${idx}" client_id)"
DN="$(manifest_field "${idx}" device_name)"
CN="$(manifest_field "${idx}" container_name)"
IMG="$(manifest_field "${idx}" image)"
PP="$(manifest_field "${idx}" partition_path)"

MODEL_DIR="${MODEL_DIR:-${FEDLORA_MODEL_ROOT:-$HOME/models}}"
PARTITION_HOST_DIR="${FEDLORA_ROOT}/${PARTITION_DIR_REL}"
CONFIG_HOST_DIR="${FEDLORA_ROOT}/distributed/configs"
LOGS_HOST_DIR="${FEDLORA_ROOT}/${MILESTONE_LOG_DIR_REL}"

mkdir -p "${LOGS_HOST_DIR}"

# Partition path inside the container (mount maps the host partition dir onto
# the same relative path under /workspace/FedLoRA).
CONTAINER_PP="${PP}"

cmd=(
    docker run --rm --network host --runtime=nvidia
    -e PYTHONPATH=/workspace/FedLoRA
    -e HF_HOME=/workspace/.cache/huggingface
    -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface
    -e TOKENIZERS_PARALLELISM=false
    -v "${PARTITION_HOST_DIR}:/workspace/FedLoRA/distributed/data/generated_partitions"
    -v "${CONFIG_HOST_DIR}:/workspace/FedLoRA/distributed/configs"
    -v "${MODEL_DIR}:/workspace/models"
    -v "${LOGS_HOST_DIR}:/workspace/FedLoRA/milestone1_logs"
    --name "${CN}_manual_check"
    "${IMG}"
    python3 -m distributed.worker.main
    --client-id "${CID}"
    --container-name "${CN}"
    --partition-path "${CONTAINER_PP}"
    --config "${DIST_CONFIG_REL}"
    --device-agent-host 127.0.0.1
    --device-agent-port-offset 0
    --dry-run-init
)

log "client_id=${CID} device=${DN} image=${IMG}"
log "rendered command:"
# Print as a safely quoted line the user can copy.
printf '  %q' "${cmd[@]}"; printf '\n'

if [[ "${EXECUTE}" -eq 1 ]]; then
    require_cmd docker
    log "EXECUTING locally (image must exist on this host)"
    exec "${cmd[@]}"
fi
