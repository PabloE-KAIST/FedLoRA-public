#!/usr/bin/env bash
# Protocol probe — READ ONLY.
#
# Run on agxorin1 (the device that has ~/pablo/AegisGov-master). This
# extracts exactly what we need to write the real ControlPlaneService ->
# device_agent bridge:
#   1) controlmessages.proto  — protobuf wire format for DEVICE_ADVERTISEMENT
#      and CONTAINER_START (authoritative).
#   2) device_agent.h / .cpp  — how the device_agent reads its own name,
#      how it subscribes for control commands, what it does with the
#      CONTAINER_START payload, whether it expects mount specs.
#   3) jsons/  — static device-side config: device name, ports, image tags.
#   4) scripts + README — how the operator normally starts device_agent.
#
# This script does NOT build, run, or start any agent.
#
# Usage on agxorin1:
#   bash remote_probe_aegisgov.sh 2>&1 | tee ~/milestone1_aegisgov_probe.log
#
# The output is self-contained; paste it back to the server operator.

set -u

ROOT="${AEGIS_ROOT:-$HOME/pablo/AegisGov-master}"

section() { printf '\n===== %s =====\n' "$*"; }
dump()    { if [[ -f "$1" ]]; then printf '\n----- FILE: %s -----\n' "$1"; cat "$1"; else printf '\n[missing] %s\n' "$1"; fi; }

section "env"
printf 'host=%s user=%s date=%s\n' "$(hostname)" "$(id -un)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'ROOT=%s\n' "$ROOT"
ls -ld "$ROOT" 2>&1

section "top-level README"
if [[ -f "$ROOT/README.md" ]]; then
    wc -l "$ROOT/README.md"
    # Dump only the first ~200 lines so the probe stays small.
    sed -n '1,200p' "$ROOT/README.md"
fi

section "controlmessages.proto (authoritative wire format)"
dump "$ROOT/libs/utils/protobufprotocols/controlmessages.proto"

section "dataexchange.proto (payload plane, not needed for Milestone 1 but useful)"
dump "$ROOT/libs/utils/protobufprotocols/dataexchange.proto"

section "device_agent header"
dump "$ROOT/libs/device_agent/device_agent.h"

section "device_agent.cpp — full source (look for CONTAINER_START handler)"
dump "$ROOT/libs/device_agent/device_agent.cpp"

section "device_agent CMakeLists"
dump "$ROOT/libs/device_agent/CMakeLists.txt"

section "misc.h — shared definitions (DeviceInfo, ContainerConfig)"
dump "$ROOT/libs/misc/misc.h"

section "misc.cpp — first 400 lines (shared helpers)"
if [[ -f "$ROOT/libs/misc/misc.cpp" ]]; then
    wc -l "$ROOT/libs/misc/misc.cpp"
    sed -n '1,400p' "$ROOT/libs/misc/misc.cpp"
fi

section "controller.cpp — CONTAINER_START sender (first 400 lines + every CONTAINER_START site)"
if [[ -f "$ROOT/libs/controller/controller.cpp" ]]; then
    wc -l "$ROOT/libs/controller/controller.cpp"
    sed -n '1,400p' "$ROOT/libs/controller/controller.cpp"
    printf '\n--- grep CONTAINER_START / CONTAINER_STOP / DEVICE_ADVERTISEMENT ---\n'
    grep -n -E "CONTAINER_START|CONTAINER_STOP|DEVICE_ADVERTISEMENT|sendContainerStart|SendContainerStart" "$ROOT/libs/controller/controller.cpp" || true
fi

section "jsons/ layout (device/controller static config)"
find "$ROOT/jsons" -maxdepth 3 -type f | sort

section "jsons/*.json — top-level configs (likely device name, ports, image tags)"
for f in "$ROOT"/jsons/*.json; do
    [[ -f "$f" ]] || continue
    printf '\n----- JSON: %s -----\n' "$f"
    sed -n '1,200p' "$f"
done

section "device_agent start scripts (how operator normally launches it)"
find "$ROOT" -maxdepth 4 -type f \( -name "*.sh" -o -name "systemd*" \) 2>/dev/null | sort | head -40
for f in "$ROOT"/scripts/*.sh; do
    [[ -f "$f" ]] || continue
    printf '\n----- SCRIPT: %s -----\n' "$f"
    sed -n '1,120p' "$f"
done

section "recent device_agent logs (if any) — tail only"
for d in "$ROOT"/logs/*/ ; do
    [[ -d "$d" ]] || continue
    printf '\n----- LOG DIR: %s -----\n' "$d"
    ls -la "$d" | head -20
    for lf in "$d"/*.log "$d"/*.txt; do
        [[ -f "$lf" ]] || continue
        printf '\n----- tail -n 60 %s -----\n' "$lf"
        tail -n 60 "$lf"
    done
done

section "topic-prefix / subscribe / pub-sub hints in device_agent"
grep -RIn --include="*.cpp" --include="*.h" -E 'setsockopt.*ZMQ_SUBSCRIBE|zmq_setsockopt.*SUBSCRIBE|SUB\s*[|,]|subscribe\b|topic\s*=' "$ROOT/libs/device_agent" "$ROOT/libs/misc" "$ROOT/libs/communicator" 2>/dev/null | head -80

section "mount-spec hints in CONTAINER_START path (grep for docker run / mount)"
grep -RIn --include="*.cpp" --include="*.h" -E 'docker\s+run|--mount|-v\s|bind_mount|binds|mounts' "$ROOT/libs" 2>/dev/null | head -80

section "done"
printf 'AEGISGOV_PROBE_COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
