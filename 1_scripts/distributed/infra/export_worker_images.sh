#!/usr/bin/env bash
# Export built fedlora-worker:<class> images to tar archives for transfer to
# real devices. Local-only (docker save). No SSH, no remote load.
#
# Usage:
#   1_scripts/distributed/infra/export_worker_images.sh [device_class ...]
#   - no args: export every distinct device_class in the manifest
#
# Output:
#   ${FEDLORA_ROOT}/milestone1_logs/images/fedlora-worker_<class>.tar
#
# The tarball path is what you will scp to each device and `docker load` on
# that device. This script does NOT transfer or load; see
# print_remote_runtime_commands.sh for the command template.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../_lib.sh
source "${SCRIPT_DIR}/../_lib.sh"

require_cmd docker
require_cmd jq

OUT_DIR="${FEDLORA_ROOT}/${MILESTONE_LOG_DIR_REL}/images"
mkdir -p "${OUT_DIR}"

if [[ $# -gt 0 ]]; then
    classes=("$@")
else
    mapfile -t classes < <(manifest_device_classes)
fi

for dc in "${classes[@]}"; do
    tag="fedlora-worker:${dc}"
    out="${OUT_DIR}/fedlora-worker_${dc}.tar"

    if ! docker image inspect "${tag}" >/dev/null 2>&1; then
        die "image not found locally: ${tag} (run build_worker_images.sh first)"
    fi

    log "save ${tag} -> ${out}"
    docker save "${tag}" -o "${out}"
    log "  size=$(stat -c%s "${out}") bytes"
done

log "done. Transfer with: scp <tar> <user>@<device>:/tmp/ && ssh <user>@<device> 'docker load -i /tmp/<tar>'"
