#!/usr/bin/env bash
# provision_fcpo.sh — install build dependencies and build FCPO DeviceAgent
# on a freshly reflashed Jetson device.
#
# Usage: provision_fcpo.sh <device_arch>   (orin | xavier)
#
# Run as ${FEDLORA_JETSON_USER:-ubuntu} on the target device.  Logs to ~/provision_fcpo.log.
# Safe to re-run: each step checks if already done.

set -euo pipefail
ARCH="${1:-orin}"
LOG="$HOME/provision_fcpo.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== provision_fcpo.sh start: $(date) arch=$ARCH ==="

# ------------------------------------------------------------------
# 1. APT dependencies
# ------------------------------------------------------------------
echo "--- APT deps ---"
# CUDA package names differ between JetPack generations
if [ "$ARCH" = "xavier" ]; then
    CUDA_PKGS="cuda-cupti-11-4 cuda-nvcc-11-4 cuda-nvtx-11-4"
else
    CUDA_PKGS="cuda-cupti-12-6 cuda-nvcc-12-6 cuda-nvtx-12-6"
fi
sudo apt-get update --fix-missing -qq
sudo apt-get install -y --no-install-recommends \
    autoconf autotools-dev build-essential clang cmake $CUDA_PKGS g++ gdb git \
    libbpf-dev libavcodec-dev libavformat-dev libboost-all-dev libbz2-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgtk2.0-dev \
    libicu-dev libjpeg-dev libopenblas-dev libpng-dev libpqxx-dev \
    libssl-dev libswscale-dev libtbb-dev libtbb2 libtiff-dev libtool \
    libv4l-dev pkg-config postgresql-client python3-dev python3-pip \
    linux-tools-generic unzip wget

# Ensure CUDA ubuntu2204 apt repo is present (needed for cusparselt, etc.)
CUDA_APT_LIST="/etc/apt/sources.list.d/cuda-ubuntu2204-arm64.list"
if [ ! -f "$CUDA_APT_LIST" ]; then
    echo "--- Adding CUDA ubuntu2204 apt source ---"
    wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb 2>/dev/null && \
        sudo dpkg -i /tmp/cuda-keyring.deb && rm -f /tmp/cuda-keyring.deb && sudo apt-get update -qq || true
fi
if ! ldconfig -p | grep -q libcusparseLt; then
    echo "--- Installing cusparseLt ---"
    sudo apt-get install -y --no-install-recommends libcusparselt0 libcusparselt-dev 2>/dev/null || true
    sudo ldconfig
fi

# Ensure bpftool is in a standard PATH location (needed by guardrail cmake)
if ! command -v bpftool &>/dev/null; then
    BPFTOOL_BIN=$(find /usr/sbin /usr/lib/linux-tools-* -name bpftool -not -type d 2>/dev/null | head -1)
    if [ -n "$BPFTOOL_BIN" ]; then
        sudo ln -sf "$BPFTOOL_BIN" /usr/local/bin/bpftool
    fi
fi

# ------------------------------------------------------------------
# 2. CMake 3.25.2 (if not already installed)
# ------------------------------------------------------------------
if ! cmake --version 2>/dev/null | grep -q "3.25.2"; then
    echo "--- CMake 3.25.2 ---"
    wget -q https://cmake.org/files/v3.25/cmake-3.25.2-linux-aarch64.sh -O /tmp/cmake-install.sh
    chmod +x /tmp/cmake-install.sh
    sudo mkdir -p /opt/cmake-3.25.2
    sudo /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.25.2
    sudo ln -sf /opt/cmake-3.25.2/bin/* /usr/local/bin/
    rm -f /tmp/cmake-install.sh
fi
cmake --version | head -1

# ------------------------------------------------------------------
# 3. gRPC v1.62.0 (if not already installed)
# ------------------------------------------------------------------
if [ ! -f /usr/local/lib/libprotobuf.a ] && [ ! -d /grpc ]; then
    echo "--- gRPC v1.62.0 ---"
    sudo rm -rf /tmp/grpc-src
    git clone --recurse-submodules -b v1.62.0 --depth 1 --shallow-submodules \
        https://github.com/grpc/grpc /tmp/grpc-src
    cd /tmp/grpc-src

    for dep in "third_party/abseil-cpp -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE" \
               "third_party/cares/cares" \
               "third_party/re2 -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE" \
               "third_party/zlib"; do
        subdir=$(echo "$dep" | awk '{print $1}')
        extra=$(echo "$dep" | cut -s -d' ' -f2-)
        mkdir -p "${subdir}/cmake/build"
        cd "${subdir}/cmake/build"
        cmake -DCMAKE_BUILD_TYPE=Release $extra ../..
        sudo make -j"$(nproc)" install
        cd /tmp/grpc-src
    done

    mkdir -p third_party/protobuf/cmake/build
    cd third_party/protobuf/cmake/build
    cmake -Dprotobuf_BUILD_SHARED_LIBS=ON -Dprotobuf_BUILD_TESTS=OFF \
          -DCMAKE_BUILD_TYPE=Release -Dprotobuf_ABSL_PROVIDER=package ../..
    sudo make -j4 install
    cd /tmp/grpc-src

    git submodule update --init
    mkdir -p cmake/build && cd cmake/build
    sudo mkdir -p /grpc
    cmake -DgRPC_INSTALL=ON -DCMAKE_BUILD_TYPE=Release \
          -DgRPC_ABSL_PROVIDER=package -DgRPC_CARES_PROVIDER=package \
          -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_RE2_PROVIDER=package \
          -DgRPC_SSL_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package \
          -DBUILD_DEPS=ON -DCMAKE_INSTALL_PREFIX=/grpc ../..
    sudo make -j"$(nproc)" install
    sudo ldconfig
    cd /
    sudo rm -rf /tmp/grpc-src
fi
echo "gRPC: $(ls /grpc/bin/grpc_cpp_plugin 2>/dev/null || echo 'using system install')"

# ------------------------------------------------------------------
# 4. OpenCV — install JetPack-provided dpkg (no source build)
# ------------------------------------------------------------------
if ! dpkg -l libopencv-dev 2>/dev/null | grep -q "^ii"; then
    echo "--- Installing OpenCV via apt ---"
    sudo apt-get install -y --no-install-recommends libopencv libopencv-dev
fi
echo "OpenCV: $(pkg-config --modversion opencv4 2>/dev/null || echo 'not found via pkg-config')"

# ------------------------------------------------------------------
# 5. cppzmq (libzmq + header)
# ------------------------------------------------------------------
if [ ! -f /usr/local/include/zmq.hpp ]; then
    echo "--- cppzmq ---"
    git clone --depth 1 https://github.com/zeromq/libzmq.git /tmp/libzmq
    cd /tmp/libzmq && mkdir build && cd build
    cmake .. && sudo make -j"$(nproc)" install
    cd / && rm -rf /tmp/libzmq

    git clone --depth 1 https://github.com/zeromq/cppzmq.git /tmp/cppzmq
    cd /tmp/cppzmq && mkdir build && cd build
    cmake -DCPPZMQ_BUILD_TESTS=OFF .. && sudo make -j"$(nproc)" install
    cd / && rm -rf /tmp/cppzmq
    sudo ldconfig
fi

# ------------------------------------------------------------------
# 6. spdlog v1.5.0 (build from source for JP6; JP5 uses apt spdlog-dev)
# ------------------------------------------------------------------
if ! dpkg -l libspdlog-dev 2>/dev/null | grep -q "^ii" && \
   [ ! -f /usr/local/lib/libspdlog.a ]; then
    echo "--- spdlog v1.5.0 ---"
    git clone --branch v1.5.0 --depth 1 https://github.com/gabime/spdlog.git /tmp/spdlog
    cd /tmp/spdlog && mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          -DSPDLOG_BUILD_TESTS=OFF ..
    sudo make -j"$(nproc)" install
    sudo ldconfig
    cd / && rm -rf /tmp/spdlog
fi

# ------------------------------------------------------------------
# 7. Extract zip and build FCPO DeviceAgent
# ------------------------------------------------------------------
FCPO_DIR="$HOME/FCPO"
ZIP_PATH="$HOME/AegisGov-Governer.zip"

if [ ! -f "$ZIP_PATH" ]; then
    echo "ERROR: $ZIP_PATH not found. Copy the zip to $HOME first."
    exit 1
fi

if [ ! -d "$FCPO_DIR/libs" ]; then
    echo "--- Extracting AegisGov-Governer to $FCPO_DIR ---"
    mkdir -p "$FCPO_DIR"
    unzip -q "$ZIP_PATH" -d /tmp/aegis_extract
    mv /tmp/aegis_extract/AegisGov-Governer/* "$FCPO_DIR/"
    rm -rf /tmp/aegis_extract
fi

# Patch guardrail cmake: ensure guardrail library waits for BPF skeleton
GUARDRAIL_CMAKE="$FCPO_DIR/libs/device_agent/guardrail/CMakeLists.txt"
if [ -f "$GUARDRAIL_CMAKE" ] && ! grep -q "add_dependencies(guardrail bpf_skeleton)" "$GUARDRAIL_CMAKE"; then
    sed -i '/add_dependencies(actuator bpf_skeleton)/a add_dependencies(guardrail bpf_skeleton)' "$GUARDRAIL_CMAKE"
fi

if [ ! -f "$FCPO_DIR/build_host/DeviceAgent" ]; then
    echo "--- Building FCPO DeviceAgent (arch=$ARCH) ---"
    rm -rf "$FCPO_DIR/build_host"
    mkdir -p "$FCPO_DIR/build_host"
    cd "$FCPO_DIR/build_host"
    if [ "$ARCH" = "xavier" ]; then
        TORCH_PATH="/home/${FEDLORA_JETSON_USER:-ubuntu}/.local/lib/python3.8/site-packages/torch"
    else
        TORCH_PATH="/home/${FEDLORA_JETSON_USER:-ubuntu}/.local/lib/python3.10/site-packages/torch"
    fi
    CMAKE_PREFIX_PATH=""
    if [ -d /grpc ]; then
        CMAKE_PREFIX_PATH="/grpc"
    fi
    if [ -d "$TORCH_PATH" ]; then
        CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+${CMAKE_PREFIX_PATH};}${TORCH_PATH}"
    fi
    CUDA_ROOT=""
    for d in /usr/local/cuda /usr/local/cuda-12.6 /usr/local/cuda-11.4; do
        if [ -f "$d/bin/nvcc" ]; then CUDA_ROOT="$d"; break; fi
    done
    cmake -DSYSTEM_NAME=FCPO -DON_HOST=ON -DDEVICE_ARCH="$ARCH" \
          -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
          ${CUDA_ROOT:+-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT"} ..
    make -j4 DeviceAgent
fi

echo "=== provision_fcpo.sh done: $(date) ==="
echo "DeviceAgent: $FCPO_DIR/build_host/DeviceAgent"
ls -lh "$FCPO_DIR/build_host/DeviceAgent"
