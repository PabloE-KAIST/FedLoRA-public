#!/usr/bin/env bash
# Print — but do not execute — the exact SSH/SCP/docker commands required to
# seed each real device with the worker image and mounted runtime files,
# and to launch the Milestone 1 manual dry-run.
#
# This script NEVER touches remote hosts. It only renders command templates
# to stdout, parameterized from the manifest and the MILESTONE_DEVICE_*
# environment variables. Copy-paste the output after review.
#
# Required env vars (one set per manifest client):
#   MILESTONE_DEVICE_<client_id>_HOST   e.g. <device-ip>
#   MILESTONE_DEVICE_<client_id>_USER   e.g. ${FEDLORA_JETSON_USER:-ubuntu}
#   MILESTONE_DEVICE_<client_id>_REPO   absolute path to FedLoRA on the device,
#                                       e.g. /home/${FEDLORA_JETSON_USER:-ubuntu}/FedLoRA
#   MILESTONE_DEVICE_<client_id>_MODELS absolute path to model dir on the device,
#                                       e.g. /home/${FEDLORA_JETSON_USER:-ubuntu}/models
#
# Example:
#   export MILESTONE_DEVICE_1_HOST=<device-ip>
#   export MILESTONE_DEVICE_1_USER=${FEDLORA_JETSON_USER:-ubuntu}
#   export MILESTONE_DEVICE_1_REPO=/home/${FEDLORA_JETSON_USER:-ubuntu}/FedLoRA
#   export MILESTONE_DEVICE_1_MODELS=/home/${FEDLORA_JETSON_USER:-ubuntu}/models
#   export MILESTONE_DEVICE_2_HOST=<device-ip>
#   export MILESTONE_DEVICE_2_USER=${FEDLORA_JETSON_USER:-ubuntu}
#   export MILESTONE_DEVICE_2_REPO=/home/${FEDLORA_JETSON_USER:-ubuntu}/FedLoRA
#   export MILESTONE_DEVICE_2_MODELS=/home/${FEDLORA_JETSON_USER:-ubuntu}/models
#   distributed/scripts/print_remote_runtime_commands.sh
#
# Any missing env var is flagged clearly in the output with a TODO marker so
# the operator cannot accidentally run an undefined command.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

require_cmd jq

IMAGES_DIR_HOST="${FEDLORA_ROOT}/${MILESTONE_LOG_DIR_REL}/images"

get_or_todo() {
    local var="$1"
    local val="${!var:-}"
    if [[ -z "${val}" ]]; then
        echo "<TODO:${var}>"
    else
        echo "${val}"
    fi
}

printf '# === Milestone 1 remote-prep commands (REVIEW BEFORE RUNNING) ===\n'
printf '# Source: %s\n' "${MANIFEST_PATH}"
printf '# Rendered at: %s\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

NUM_CLIENTS="$(manifest_num_clients)"
for idx in $(seq 0 $((NUM_CLIENTS - 1))); do
    CID="$(manifest_field "${idx}" client_id)"
    DN="$(manifest_field "${idx}" device_name)"
    DC="$(manifest_field "${idx}" device_class)"
    CN="$(manifest_field "${idx}" container_name)"
    IMG="$(manifest_field "${idx}" image)"
    PP="$(manifest_field "${idx}" partition_path)"
    META="${PP%.pkl}.metadata.json"

    HOST_VAR="MILESTONE_DEVICE_${CID}_HOST"
    USER_VAR="MILESTONE_DEVICE_${CID}_USER"
    REPO_VAR="MILESTONE_DEVICE_${CID}_REPO"
    MODELS_VAR="MILESTONE_DEVICE_${CID}_MODELS"

    DHOST="$(get_or_todo "${HOST_VAR}")"
    DUSER="$(get_or_todo "${USER_VAR}")"
    DREPO="$(get_or_todo "${REPO_VAR}")"
    DMODELS="$(get_or_todo "${MODELS_VAR}")"

    TAR_BASENAME="fedlora-worker_${DC}.tar"
    LOCAL_TAR="${IMAGES_DIR_HOST}/${TAR_BASENAME}"

    printf '# --- client_id=%s  device=%s  class=%s  image=%s ---\n' \
        "${CID}" "${DN}" "${DC}" "${IMG}"

    printf '# Step A: ensure remote directory skeleton exists\n'
    printf 'ssh %s@%s "mkdir -p %s/distributed/data/generated_partitions %s/distributed/configs %s/milestone1_logs %s"\n\n' \
        "${DUSER}" "${DHOST}" "${DREPO}" "${DREPO}" "${DREPO}" "${DMODELS}"

    printf '# Step B: copy partition + metadata + distributed config for this client\n'
    printf 'scp %s/%s %s@%s:%s/%s\n' \
        "${FEDLORA_ROOT}" "${PP}" "${DUSER}" "${DHOST}" "${DREPO}" "${PP}"
    printf 'scp %s/%s %s@%s:%s/%s\n' \
        "${FEDLORA_ROOT}" "${META}" "${DUSER}" "${DHOST}" "${DREPO}" "${META}"
    printf 'scp %s/%s %s@%s:%s/%s\n\n' \
        "${FEDLORA_ROOT}" "${DIST_CONFIG_REL}" "${DUSER}" "${DHOST}" "${DREPO}" "${DIST_CONFIG_REL}"

    printf '# Step C: copy the per-class worker image tarball and load it on the device\n'
    printf 'scp %s %s@%s:/tmp/%s\n' \
        "${LOCAL_TAR}" "${DUSER}" "${DHOST}" "${TAR_BASENAME}"
    printf 'ssh %s@%s "docker load -i /tmp/%s"\n\n' \
        "${DUSER}" "${DHOST}" "${TAR_BASENAME}"

    printf '# Step D: manual dry-run launch on the device (no device_agent, no FL round)\n'
    printf '# --runtime=nvidia is required on agxavier1 (default runtime = runc there).\n'
    printf '# It is a no-op on agxorin1 where default runtime is already nvidia.\n'
    printf 'ssh %s@%s \\\n' "${DUSER}" "${DHOST}"
    printf '  "docker run --rm --network host --runtime=nvidia \\\n'
    printf '     -e PYTHONPATH=/workspace/FedLoRA \\\n'
    printf '     -e HF_HOME=/workspace/.cache/huggingface \\\n'
    printf '     -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface \\\n'
    printf '     -e TOKENIZERS_PARALLELISM=false \\\n'
    printf '     -v %s/distributed/data/generated_partitions:/workspace/FedLoRA/distributed/data/generated_partitions \\\n' "${DREPO}"
    printf '     -v %s/distributed/configs:/workspace/FedLoRA/distributed/configs \\\n' "${DREPO}"
    printf '     -v %s:/workspace/models \\\n' "${DMODELS}"
    printf '     -v %s/milestone1_logs:/workspace/FedLoRA/milestone1_logs \\\n' "${DREPO}"
    printf '     --name %s_manual_check \\\n' "${CN}"
    printf '     %s \\\n' "${IMG}"
    printf '     python3 -m distributed.worker.main \\\n'
    printf '       --client-id %s \\\n' "${CID}"
    printf '       --container-name %s \\\n' "${CN}"
    printf '       --partition-path %s \\\n' "${PP}"
    printf '       --config %s \\\n' "${DIST_CONFIG_REL}"
    printf '       --device-agent-host 127.0.0.1 \\\n'
    printf '       --device-agent-port-offset 0 \\\n'
    printf '       --dry-run-init"\n\n'
done

printf '# === End of rendered commands ===\n'
printf '# REMINDER: Do NOT start device_agent, ControlPlaneService, or\n'
printf '#           production worker containers from this script.\n'
printf '#           Those steps are covered by the runbook and require\n'
printf '#           explicit operator action after dry-run init succeeds.\n'
