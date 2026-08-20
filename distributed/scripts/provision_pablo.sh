#!/usr/bin/env bash
# provision_pablo.sh — set up /home/${FEDLORA_JETSON_USER:-ubuntu}/pablo FedLoRA environment on a Jetson.
#
# Usage: provision_pablo.sh <device_arch>   (orin | xavier)
#
# Expects these files to already be in /home/${FEDLORA_JETSON_USER:-ubuntu}/:
#   AegisGov-master.zip   — FedLoRA-patched AegisGov (bytes payload proto)
#   deberta-large.zip     — DeBERTa-large model weights
#
# The distributed/ overlay is rsynced from this script's assumptions;
# edit OVERLAY_SRC if the gpu-host-a path changes.
#
# Run as ${FEDLORA_JETSON_USER:-ubuntu} on the target device.  Logs to ~/provision_pablo.log.

set -euo pipefail
ARCH="${1:-orin}"
LOG="$HOME/provision_pablo.log"
exec > >(tee -a "$LOG") 2>&1

PABLO="$HOME/pablo"
AEGIS="$PABLO/AegisGov-master"
RUNTIME="$PABLO/fedlora_runtime"

echo "=== provision_pablo.sh start: $(date) arch=$ARCH ==="

# ------------------------------------------------------------------
# 1. Extract AegisGov-master.zip → /home/${FEDLORA_JETSON_USER:-ubuntu}/pablo/AegisGov-master
# ------------------------------------------------------------------
if [ ! -d "$AEGIS/libs" ]; then
    echo "--- Extracting AegisGov-master ---"
    mkdir -p "$PABLO"
    unzip -q "$HOME/AegisGov-master.zip" -d "$PABLO"
fi

# Patch proto: ensure bytes payload (not string)
PROTO="$AEGIS/libs/utils/protobufprotocols/controlmessages.proto"
if grep -q "string payload = 2;" "$PROTO" 2>/dev/null; then
    echo "--- Patching proto: string -> bytes payload ---"
    sed -i 's/  string payload = 2;/  bytes payload = 2;/' "$PROTO"
fi
grep "payload = 2" "$PROTO"

# ------------------------------------------------------------------
# 2. Ensure OpenCV and cuDNN are installed (needed by AegisGov cmake)
# ------------------------------------------------------------------
if ! dpkg -l libopencv-dev 2>/dev/null | grep -q "^ii"; then
    echo "--- Installing OpenCV ---"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends libopencv libopencv-dev
fi

# Ensure CUDA ubuntu2204 apt repo is present (needed for cusparselt, cuDNN 9, etc.)
CUDA_APT_LIST="/etc/apt/sources.list.d/cuda-ubuntu2204-arm64.list"
if [ ! -f "$CUDA_APT_LIST" ]; then
    echo "--- Adding CUDA ubuntu2204 apt source ---"
    sudo apt-get install -y --no-install-recommends wget ca-certificates 2>/dev/null || true
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb 2>/dev/null && \
        sudo dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb && sudo apt-get update -qq || true
fi

if ! ldconfig -p | grep -q libcusparseLt; then
    echo "--- Installing cusparseLt ---"
    sudo apt-get install -y --no-install-recommends libcusparselt0 libcusparselt-dev 2>/dev/null || true
    sudo ldconfig
fi

if ! ldconfig -p | grep -q libcudnn; then
    echo "--- Installing cuDNN ---"
    sudo apt-get update -qq
    if [ "$ARCH" = "xavier" ]; then
        sudo apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev 2>/dev/null || \
        sudo apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 2>/dev/null || true
    else
        sudo apt-get install -y --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 2>/dev/null || \
        sudo apt-get install -y --no-install-recommends libcudnn8 libcudnn8-dev 2>/dev/null || true
    fi
    sudo ldconfig
fi

# ------------------------------------------------------------------
# 3. Build DeviceAgent in build_host/
# ------------------------------------------------------------------
if [ ! -f "$AEGIS/build_host/DeviceAgent" ]; then
    echo "--- Building DeviceAgent (arch=$ARCH) ---"
    rm -rf "$AEGIS/build_host"
    mkdir -p "$AEGIS/build_host"
    cd "$AEGIS/build_host"
    # LibTorch location differs by JetPack generation
    if [ "$ARCH" = "xavier" ]; then
        TORCH_PREFIX="/home/${FEDLORA_JETSON_USER:-ubuntu}/.local/lib/python3.8/site-packages/torch"
    else
        TORCH_PREFIX="/home/${FEDLORA_JETSON_USER:-ubuntu}/.local/lib/python3.10/site-packages/torch"
    fi
    CMAKE_PREFIX_PATH="$TORCH_PREFIX"
    if [ -d /grpc ]; then
        CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;/grpc"
    fi
    CUDA_ROOT=""
    for d in /usr/local/cuda /usr/local/cuda-12.6 /usr/local/cuda-11.4; do
        if [ -f "$d/bin/nvcc" ]; then CUDA_ROOT="$d"; break; fi
    done
    cmake -DSYSTEM_NAME=FCPO -DON_HOST=ON -DDEVICE_ARCH="$ARCH" \
          -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
          ${CUDA_ROOT:+-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"} ..
    make -j2 DeviceAgent
fi
echo "DeviceAgent: $(ls -lh $AEGIS/build_host/DeviceAgent)"

# ------------------------------------------------------------------
# 4. Set up fedlora_runtime directory tree
# ------------------------------------------------------------------
echo "--- Setting up fedlora_runtime ---"
mkdir -p "$RUNTIME/models" \
         "$RUNTIME/partitions" \
         "$RUNTIME/logs" \
         "$RUNTIME/configs" \
         "$RUNTIME/data/generated_partitions"

# ------------------------------------------------------------------
# 5. Extract deberta-large model weights
# ------------------------------------------------------------------
if [ ! -d "$RUNTIME/models/deberta-large" ]; then
    if [ -f "$HOME/deberta-large.zip" ]; then
        echo "--- Extracting deberta-large model ---"
        unzip -q "$HOME/deberta-large.zip" -d "$RUNTIME/models/"
    else
        echo "WARNING: deberta-large.zip not found at $HOME — skipping model extraction"
    fi
fi

# ------------------------------------------------------------------
# 6. Deploy compose template into AegisGov dockerfiles/
# ------------------------------------------------------------------
COMPOSE_SRC="$RUNTIME/distributed/docker/docker-compose.aegisgov.jetson.yml"
COMPOSE_DST="$AEGIS/dockerfiles/docker-compose.jetson.yml"
if [ -f "$COMPOSE_SRC" ]; then
    cp "$COMPOSE_SRC" "$COMPOSE_DST"
    echo "Compose deployed: $COMPOSE_DST"
fi

# Symlink at fedlora_runtime root (legacy path some scripts still reference)
if [ ! -e "$RUNTIME/docker-compose.jetson.yml" ]; then
    ln -sf "$COMPOSE_DST" "$RUNTIME/docker-compose.jetson.yml"
fi

# ------------------------------------------------------------------
# 7. Copy launch_worker.sh
# ------------------------------------------------------------------
LAUNCH_SRC="$RUNTIME/distributed/docker/launch_worker.sh"
if [ -f "$LAUNCH_SRC" ]; then
    cp "$LAUNCH_SRC" "$RUNTIME/launch_worker.sh"
    chmod +x "$RUNTIME/launch_worker.sh"
fi

echo "=== provision_pablo.sh done: $(date) ==="
ls -la "$RUNTIME/"
