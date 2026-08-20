#!/usr/bin/env bash
# Deploy per-device bandwidth JSON profiles to all fleet devices.
#
# Each device_agent loads bandwidth_limits{0-20}.json from
# ~/pablo/AegisGov-master/jsons/bandwidths/. This script copies the
# generated per-device profiles to every device so each DA has access
# to all 12 profiles. The server then sends each device its specific
# bandwidth_setting_id to select the right profile.
#
# For x86-worker virtual DAs, the files go to the per-DA AegisGov dir.
#
# Usage:
#   bash 1_scripts/distributed/infra/deploy_bandwidth_jsons.sh [--source-dir DIR]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

SOURCE_DIR="${1:-${FEDLORA_ROOT}/1_scripts/distributed/infra/generated_bandwidths}"
REMOTE_BW_DIR="~/pablo/AegisGov-master/jsons/bandwidths"

# Physical Jetson devices
JETSONS=(agxorin1 agxorin2 agxorin3 agxorin4 agxavier1 agxavier2 orinnx1 orinnx2 orinnx3)

# Crystal virtual DA paths (multiple DAs share the same host)
CRYSTAL_HOST="x86-worker"
CRYSTAL_DA_DIRS=(
    "/ssd0/pablo/AegisGov-master/jsons/bandwidths"
    "/ssd0/pablo/AegisGov-master-2/jsons/bandwidths"
    "/ssd0/pablo/AegisGov-master-3/jsons/bandwidths"
)

# Count how many per-device profiles exist
PROFILE_COUNT=$(find "$SOURCE_DIR" -maxdepth 1 -name 'bandwidth_limits[0-9]*.json' | wc -l)
if [[ "$PROFILE_COUNT" -eq 0 ]]; then
    die "No per-device bandwidth profiles found in $SOURCE_DIR. Run generate_bandwidth_json.py --per-device first."
fi
log "Found $PROFILE_COUNT per-device bandwidth profiles in $SOURCE_DIR"

# Deploy to physical Jetsons
for host in "${JETSONS[@]}"; do
    log "Deploying to $host..."
    ssh -o ConnectTimeout=5 "$host" "mkdir -p $REMOTE_BW_DIR" 2>/dev/null || {
        log "WARNING: Cannot reach $host, skipping"
        continue
    }
    scp -o ConnectTimeout=5 -q "$SOURCE_DIR"/bandwidth_limits[0-9]*.json \
        "$host:$REMOTE_BW_DIR/" 2>/dev/null && \
        log "  $host: OK ($PROFILE_COUNT profiles)" || \
        log "  WARNING: scp to $host failed"
done

# Deploy to x86-worker virtual DA directories
for da_dir in "${CRYSTAL_DA_DIRS[@]}"; do
    log "Deploying to $CRYSTAL_HOST:$da_dir..."
    ssh -o ConnectTimeout=5 "$CRYSTAL_HOST" "mkdir -p $da_dir" 2>/dev/null || {
        log "WARNING: Cannot create $da_dir on $CRYSTAL_HOST"
        continue
    }
    scp -o ConnectTimeout=5 -q "$SOURCE_DIR"/bandwidth_limits[0-9]*.json \
        "$CRYSTAL_HOST:$da_dir/" 2>/dev/null && \
        log "  $CRYSTAL_HOST:$da_dir: OK" || \
        log "  WARNING: scp to $CRYSTAL_HOST:$da_dir failed"
done

log "Deployment complete. Restart DAs for profiles to take effect."
