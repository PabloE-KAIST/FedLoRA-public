#!/usr/bin/env bash
# Restart DeviceAgents for devices in a given manifest, pointing them
# at a specific controller IP. Used for dual-server concurrent sub-fleets.
#
# Usage:
#   CONTROLLER_IP=${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP} MANIFEST=distributed/configs/client_manifest_group_b.json \
#       bash 1_scripts/distributed/prep/restart_das_for_manifest.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"
require_cmd jq

CONTROLLER_IP="${CONTROLLER_IP:?CONTROLLER_IP env var required}"

if [[ -n "${MANIFEST:-}" ]]; then
    if [[ "$MANIFEST" == /* ]]; then
        MANIFEST_PATH="$MANIFEST"
    else
        MANIFEST_PATH="${FEDLORA_ROOT}/${MANIFEST}"
    fi
fi

declare -A DEVICE_TYPE=(
    [agxorin1]=orinagx [agxorin2]=orinagx [agxorin3]=orinagx [agxorin4]=orinagx
    [agxavier1]=agxavier [agxavier2]=agxavier
    [orinnx1]=orinnx [orinnx2]=orinnx [orinnx3]=orinnx
)
declare -A DEVICE_NIC=(
    [agxorin1]=eno1 [agxorin2]=eth0 [agxorin3]=eno1 [agxorin4]=eno1
    [agxavier1]=eth0 [agxavier2]=eth0
    [orinnx1]=enP8p1s0 [orinnx2]=eno1 [orinnx3]=enP8p1s0
)

# Crystal virtual DA config: name → gpu_id:dev_port_offset
declare -A CRYSTAL_DA=(
    [crystal_gpu5]=5:0
    [crystal_gpu4]=4:100
    [crystal_gpu3]=3:400
    [crystal_gpu2]=2:300
)

DEVICES=($(jq -r '.clients[].device_name' "$MANIFEST_PATH"))
log "Restarting DAs for ${#DEVICES[@]} devices → controller ${CONTROLLER_IP}"

restart_jetson_da() {
    local host=$1
    local dtype=${DEVICE_TYPE[$host]}
    local nic=${DEVICE_NIC[$host]:-eth0}
    ssh "$host" bash <<REMOTE
ps aux | grep "[D]eviceAgent" | grep -- "--name ${host}" | awk '{print \$2}' | xargs -r kill 2>/dev/null
sleep 2
ps aux | grep "[D]eviceAgent" | grep -- "--name ${host}" | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep 1
cd ~/pablo/AegisGov-master/build_host
setsid ./DeviceAgent --name ${host} --device_type ${dtype} \
    --controller_url ${CONTROLLER_IP} \
    --dev_system_port_offset 100 --dev_port_offset 0 \
    --dev_networkInterface ${nic} \
    --dev_verbose 1 </dev/null > /tmp/da_${host}.log 2>&1 &
disown
sleep 2
count=\$(ps aux | grep "[D]eviceAgent" | grep -- "--name ${host}" | wc -l)
echo "${host}: \${count} DA(s) → ${CONTROLLER_IP}"
REMOTE
}

restart_crystal_da() {
    local name=$1
    local config=${CRYSTAL_DA[$name]}
    local gpu=${config%%:*}
    local offset=${config##*:}
    ssh x86-worker bash <<REMOTE
export LD_LIBRARY_PATH=/ssd0/pablo/da_libs
ps aux | grep "[D]eviceAgent" | grep -- "--name ${name}" | awk '{print \$2}' | xargs -r kill 2>/dev/null
sleep 2
ps aux | grep "[D]eviceAgent" | grep -- "--name ${name}" | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep 1
cd /ssd0/pablo/AegisGov-master/build_host
setsid ./DeviceAgent --name ${name} --device_type virtual \
    --controller_url ${CONTROLLER_IP} \
    --dev_gpuID ${gpu} --dev_system_port_offset 100 --dev_port_offset ${offset} \
    --dev_networkInterface enp129s0f0 --dev_verbose 1 \
    </dev/null > /tmp/da_${name}.log 2>&1 &
disown
sleep 2
count=\$(ps aux | grep "[D]eviceAgent" | grep -- "--name ${name}" | wc -l)
echo "${name}: \${count} DA(s) → ${CONTROLLER_IP}"
REMOTE
}

for dev in "${DEVICES[@]}"; do
    if [[ -n "${CRYSTAL_DA[$dev]:-}" ]]; then
        restart_crystal_da "$dev" &
    elif [[ -n "${DEVICE_TYPE[$dev]:-}" ]]; then
        restart_jetson_da "$dev" &
    else
        log "WARNING: unknown device ${dev}, skipping"
    fi
done
wait

log "DAs restarted for manifest devices → ${CONTROLLER_IP}"
