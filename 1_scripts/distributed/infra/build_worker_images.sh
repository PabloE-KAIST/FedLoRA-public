#!/usr/bin/env bash
# Build one fedlora-worker image per device_class referenced in the manifest.
# Runs locally on the build host. Does not push, does not SSH.
#
# Usage:
#   1_scripts/distributed/infra/build_worker_images.sh [device_class ...]
#   - no args: build every distinct device_class in the manifest
#   - args:    build only the listed classes (must appear in the manifest)
#
# Per-class base image can be overridden via env vars:
#   BASE_IMAGE_AGXORIN=nvcr.io/nvidia/l4t-pytorch:r35.4.1-pth2.1-py3
#   BASE_IMAGE_AGXAVIER=nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
# If unset, the Dockerfile's ARG default is used. A wrong tag will fail
# silently at torch CUDA init on the real device — set these explicitly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_lib.sh
source "${SCRIPT_DIR}/../_lib.sh"

require_cmd docker
require_cmd jq

cd "${FEDLORA_ROOT}"

if [[ $# -gt 0 ]]; then
    classes=("$@")
else
    mapfile -t classes < <(manifest_device_classes)
fi

[[ "${#classes[@]}" -gt 0 ]] || die "no device classes to build"

# Record the build decisions for the evidence bundle.
IMAGE_TAG_LOG="${FEDLORA_ROOT}/${MILESTONE_LOG_DIR_REL}/image_tags.txt"
mkdir -p "$(dirname "${IMAGE_TAG_LOG}")"
{
    echo "# Built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host=$(hostname) user=${USER:-unknown}"
} >> "${IMAGE_TAG_LOG}"

for dc in "${classes[@]}"; do
    dockerfile="distributed/docker/Dockerfile.worker.${dc}"
    if [[ ! -f "${dockerfile}" ]]; then
        die "no Dockerfile for device_class='${dc}' (expected ${dockerfile})"
    fi

    tag="fedlora-worker:${dc}"
    base_var="BASE_IMAGE_${dc^^}"
    base_val="${!base_var:-}"

    build_args=()
    if [[ -n "${base_val}" ]]; then
        build_args+=(--build-arg "BASE_IMAGE=${base_val}")
        log "build ${tag} with BASE_IMAGE=${base_val}"
    else
        log "build ${tag} with Dockerfile default BASE_IMAGE (override via ${base_var})"
    fi

    docker build -f "${dockerfile}" -t "${tag}" "${build_args[@]}" .

    echo "${tag}	${base_val:-<dockerfile-default>}" >> "${IMAGE_TAG_LOG}"
done

log "image_tags.txt updated: ${IMAGE_TAG_LOG}"
log "done"
