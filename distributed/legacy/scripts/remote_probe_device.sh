#!/usr/bin/env bash
# On-device verification probe for Milestone 1.
#
# READ-ONLY. This script does NOT build, pull, load, or run any image.
# It does NOT start device_agent or ControlPlaneService. It only inspects
# the state of the device so we can pick base images and decide the build
# path.
#
# Run this on each real device (agxorin1, agxavier1). Capture stdout+stderr
# and send it back to the operator on the server.
#
# Usage on the device:
#   bash remote_probe_device.sh 2>&1 | tee ~/milestone1_probe_$(hostname).log
#
# Or via one-shot here-doc from the server once SSH is available:
#   ssh ${FEDLORA_JETSON_USER:-ubuntu}@<device> 'bash -s' < distributed/scripts/remote_probe_device.sh \
#       > milestone1_logs/probe_<device>.log 2>&1

set -u  # do NOT -e: we want the probe to continue past missing commands.

section() { printf '\n===== %s =====\n' "$*"; }
run()     { printf '\n$ %s\n' "$*"; eval "$*" 2>&1 || printf '  [exit=%s]\n' "$?"; }

section "identity"
run 'date -u +%Y-%m-%dT%H:%M:%SZ'
run 'hostname'
run 'id -un'
run 'uname -a'
run 'uname -m'

section "os release"
run 'cat /etc/os-release 2>/dev/null | head -20'
run 'lsb_release -a 2>/dev/null || true'

section "L4T / JetPack"
# Expected file on Jetson devices.
run 'cat /etc/nv_tegra_release 2>/dev/null'
run 'ls -l /etc/nv_tegra_release 2>/dev/null'
run 'dpkg -l 2>/dev/null | grep -iE "nvidia-l4t|jetpack|cuda-toolkit" | head -40'
# tegrastats presence is a soft indicator the device is a Jetson.
run 'command -v tegrastats'

section "GPU / CUDA (best effort)"
run 'command -v nvidia-smi && nvidia-smi || echo "nvidia-smi not available (expected on Tegra)"'
run 'ls /usr/local/cuda 2>/dev/null | head'
run 'ls /usr/local 2>/dev/null | grep -i cuda || true'
run 'find /usr/local -maxdepth 2 -name "version.txt" -path "*cuda*" -exec cat {} + 2>/dev/null'

section "Docker runtime"
run 'command -v docker && docker --version'
run 'docker info --format "ServerVersion={{.ServerVersion}} Runtimes={{.Runtimes}} DefaultRuntime={{.DefaultRuntime}}" 2>&1 | head -5'
run 'docker info 2>/dev/null | grep -iE "runtime|cgroup|architecture|operating system|kernel" | head -20'
# Is nvidia container runtime wired in?
run 'ls /etc/docker/daemon.json 2>/dev/null && cat /etc/docker/daemon.json'
run 'dpkg -l 2>/dev/null | grep -iE "nvidia-container|nvidia-docker" | head'

section "disk / mounts"
run 'df -h / /home 2>/dev/null'
run 'df -h /ssd2 2>/dev/null || true'
run 'mount | grep -E " /ssd2| /home" 2>/dev/null | head'

section "network / listening ports of interest"
# Device-agent ports (60011/60012) and FL-payload ports (60001/60002).
run 'ss -ltnp 2>/dev/null | egrep "60001|60002|60011|60012|60101|60102" || echo "no matches"'

section "home layout"
run 'echo HOME=$HOME'
run 'ls -la ~ | head -40'
run 'ls -la ~/pablo 2>/dev/null | head -40 || echo "no ~/pablo"'

section "FedLoRA target paths on device"
# None of these should exist yet — absence is expected for Milestone 1 prep.
for p in ${FEDLORA_ROOT} \
         ${FEDLORA_ROOT}/distributed \
         ${FEDLORA_ROOT}/distributed/data/generated_partitions \
         ${FEDLORA_ROOT}/distributed/configs \
         ${FEDLORA_ROOT}/milestone1_logs \
         ${FEDLORA_MODEL_ROOT:-$HOME/models} \
         google/bert_uncased_L-2_H-128_A-2 ; do
    run "ls -ld '$p' 2>&1 | head -1"
done

section "AegisGov-master (device_agent side)"
# Exists on agxorin1 per operator note; unknown on agxavier1 (absence OK).
run 'ls -ld ~/pablo/AegisGov-master 2>&1 | head -1'
run 'ls -la ~/pablo/AegisGov-master 2>/dev/null | head -40 || true'
run 'find ~/pablo/AegisGov-master -maxdepth 3 -type d 2>/dev/null | head -60'
run 'ls ~/pablo/AegisGov-master/device_agent 2>/dev/null | head -40'
# Surface files likely to carry CONTAINER_START / DEVICE_ADVERTISEMENT wiring.
run 'grep -RIl --exclude-dir=.git --exclude-dir=build --exclude-dir=cmake-build* -E "CONTAINER_START|DEVICE_ADVERTISEMENT|ContainerConfig|DeviceInfo" ~/pablo/AegisGov-master 2>/dev/null | head -40'
run 'grep -RIl --exclude-dir=.git --include="*.proto" -E "." ~/pablo/AegisGov-master 2>/dev/null | head -40'
# Any start/launch scripts?
run 'find ~/pablo/AegisGov-master -maxdepth 4 -type f \( -name "*.sh" -o -name "*.md" -o -name "README*" -o -name "CMakeLists.txt" \) 2>/dev/null | head -40'

section "python (host, not container)"
run 'command -v python3 && python3 --version'
run 'command -v python  && python --version'

section "done"
run 'date -u +%Y-%m-%dT%H:%M:%SZ'
echo "PROBE_COMPLETE"
