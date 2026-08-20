#!/usr/bin/env bash
# Restart all DeviceAgents across the fleet for a fresh FL run.
# Usage: bash 1_scripts/distributed/prep/restart_all_das.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../_lib.sh"

CONTROLLER_IP="${CONTROLLER_IP:-${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}}"

# Device lists — edit these when the fleet changes.
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
JETSON_HOSTS=( agxorin1 agxorin2 agxorin3 agxorin4 agxavier1 agxavier2 orinnx1 orinnx2 orinnx3 )
CRYSTAL_HOST="${CRYSTAL_HOST:-x86-worker}"
CRYSTAL_LAUNCH_SCRIPT="${CRYSTAL_LAUNCH_SCRIPT:-/ssd0/pablo/launch_virtual_das.sh}"

restart_jetson_da() {
    local host=$1
    local dtype=${DEVICE_TYPE[$host]}
    local nic=${DEVICE_NIC[$host]:-eth0}
    ssh "$host" bash <<REMOTE
ps aux | grep "[D]eviceAgent" | grep "${CONTROLLER_IP}" | awk '{print \$2}' | xargs -r kill 2>/dev/null
sleep 2
ps aux | grep "[D]eviceAgent" | grep "${CONTROLLER_IP}" | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep 1
cd ~/pablo/AegisGov-master/build_host
setsid ./DeviceAgent --name ${host} --device_type ${dtype} \
    --controller_url ${CONTROLLER_IP} \
    --dev_system_port_offset 100 --dev_port_offset 0 \
    --dev_networkInterface ${nic} \
    --dev_verbose 1 </dev/null > /tmp/da_${host}.log 2>&1 &
disown
sleep 2
count=\$(ps aux | grep "[D]eviceAgent" | grep "${CONTROLLER_IP}" | wc -l)
echo "${host}: \${count} DA(s)"
REMOTE
}

log "Restarting Jetson DAs..."
for host in "${JETSON_HOSTS[@]}"; do
    restart_jetson_da "$host" &
done
wait
log "Jetson DAs restarted."

log "Restarting x86-worker virtual DAs..."
ssh "$CRYSTAL_HOST" "bash ${CRYSTAL_LAUNCH_SCRIPT}" 2>&1
CRYSTAL_COUNT=$(ssh "$CRYSTAL_HOST" "ps aux | grep '[D]eviceAgent' | grep '${CONTROLLER_IP}' | wc -l" 2>&1)
log "Crystal: ${CRYSTAL_COUNT} DA(s) running"

log "All DAs restarted."
