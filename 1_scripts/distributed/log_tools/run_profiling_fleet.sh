#!/usr/bin/env bash
# Deploy and run memory profiling across 4 device classes.
# Usage: bash 1_scripts/distributed/log_tools/run_profiling_fleet.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

PROFILE_SCRIPT="${FEDLORA_ROOT}/1_scripts/distributed/log_tools/profile_memory.py"
RESULTS_DIR="${FEDLORA_ROOT}/exp/memory_profiling"
mkdir -p "$RESULTS_DIR"

# One representative device per class.
declare -A DEVICE_HOST=(
    [agxorin]="agxorin1"
    [agxavier]="agxavier1"
    [x86]="x86-worker"
    [orinnx]="orinnx1"
)
declare -A DEVICE_IMAGE=(
    [agxorin]="fedlora-worker:agxorin"
    [agxavier]="fedlora-worker:agxavier"
    [x86]="fedlora-worker:x86"
    [orinnx]="fedlora-worker:agxorin"
)
declare -A DEVICE_RANKS=(
    [agxorin]="8 64 100 150 200 250"
    [agxavier]="8 64 100 150 200"
    [x86]="8 32 64 100 120"
    [orinnx]="8 32 48 64 80 100"
)
declare -A DEVICE_GPU_ID=(
    [agxorin]=""
    [agxavier]=""
    [x86]="5"
    [orinnx]=""
)
# Per-class precision: 32 = fp32, 16 = auto (bf16 on sm_80+, fp32 otherwise).
declare -A DEVICE_NBITS=(
    [agxorin]="32"
    [agxavier]="16"
    [x86]="16"
    [orinnx]="16"
)
# Gradient checkpointing: disable on Jetsons (PyTorch NVML assert on Tegra),
# keep enabled on x86 (GTX 1080 Ti needs it to fit in 11 GB).
declare -A DEVICE_DISABLE_GC=(
    [agxorin]="1"
    [agxavier]="1"
    [x86]="0"
    [orinnx]="1"
)
# Host-side runtime root (differs per device class).
declare -A DEVICE_RUNTIME_ROOT=(
    [agxorin]="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
    [agxavier]="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
    [x86]="/ssd0/pablo/fedlora_runtime"
    [orinnx]="${FEDLORA_JETSON_RUNTIME:-/home/${FEDLORA_JETSON_USER:-ubuntu}/fedlora_runtime}"
)

CONTAINER_CONFIG="/workspace/FedLoRA/distributed/configs/fedit_distributed.yaml"
CONTAINER_SCRIPT="/workspace/FedLoRA/profile_memory.py"
OUTPUT_PATH="/workspace/FedLoRA/logs/profile_result.json"

deploy_and_run() {
    local class="$1"
    local host="${DEVICE_HOST[$class]}"
    local image="${DEVICE_IMAGE[$class]}"
    local ranks="${DEVICE_RANKS[$class]}"
    local gpu_id="${DEVICE_GPU_ID[$class]}"
    local nbits="${DEVICE_NBITS[$class]}"
    local disable_gc="${DEVICE_DISABLE_GC[$class]}"
    local runtime_root="${DEVICE_RUNTIME_ROOT[$class]}"

    log "=== Profiling class=$class on host=$host (nbits=$nbits, gc_disabled=$disable_gc) ==="

    # Deploy profiling script
    log "Deploying profiling script to $host..."
    scp -o ConnectTimeout=10 "$PROFILE_SCRIPT" \
        "${host}:${runtime_root}/profile_memory.py"

    # Build docker run command.
    # Mirrors the compose volumes so the environment matches real training.
    local docker_cmd="docker run --rm"

    if [[ "$class" == "x86" ]]; then
        docker_cmd+=" --gpus device=${gpu_id}"
        docker_cmd+=" -e CUDA_VISIBLE_DEVICES=0"
    else
        docker_cmd+=" --runtime nvidia"
    fi

    docker_cmd+=" --network host"
    docker_cmd+=" -e PYTHONPATH=/workspace/FedLoRA"
    docker_cmd+=" -e HF_HOME=/workspace/.cache/huggingface"
    docker_cmd+=" -e TRANSFORMERS_CACHE=/workspace/.cache/huggingface"
    docker_cmd+=" -e TOKENIZERS_PARALLELISM=false"
    docker_cmd+=" -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
    docker_cmd+=" -e FEDLORA_DISABLE_GRADIENT_CHECKPOINTING=${disable_gc}"
    docker_cmd+=" -v ${runtime_root}/partitions:/workspace/FedLoRA/distributed/data/generated_partitions"
    docker_cmd+=" -v ${runtime_root}/configs:/workspace/FedLoRA/distributed/configs"
    docker_cmd+=" -v ${runtime_root}/models:/workspace/models"
    docker_cmd+=" -v ${runtime_root}/logs:/workspace/FedLoRA/logs"
    docker_cmd+=" -v ${runtime_root}/distributed:/workspace/FedLoRA/distributed"
    docker_cmd+=" -v ${runtime_root}/federatedscope:/workspace/FedLoRA/federatedscope"
    docker_cmd+=" -v ${runtime_root}/profile_memory.py:${CONTAINER_SCRIPT}:ro"
    docker_cmd+=" ${image}"
    docker_cmd+=" python3 ${CONTAINER_SCRIPT}"
    docker_cmd+="   --config ${CONTAINER_CONFIG}"
    docker_cmd+="   --ranks ${ranks}"
    docker_cmd+="   --steps 30"
    docker_cmd+="   --nbits ${nbits}"
    docker_cmd+="   --output ${OUTPUT_PATH}"

    log "Running profiling on $host (ranks: $ranks)..."
    log "  docker cmd: $docker_cmd"

    ssh -o ConnectTimeout=10 "$host" "$docker_cmd"

    # Retrieve results
    local result_file="${RESULTS_DIR}/profile_${class}.json"
    scp -o ConnectTimeout=10 \
        "${host}:${runtime_root}/logs/profile_result.json" \
        "$result_file" 2>/dev/null || \
    log "WARNING: Could not retrieve results for $class"

    if [[ -f "$result_file" ]]; then
        log "Results saved to $result_file"
    fi
}

# Run sequentially to avoid resource contention.
for class in agxorin agxavier x86 orinnx; do
    deploy_and_run "$class" 2>&1 | tee "${RESULTS_DIR}/profile_${class}.log"
    echo
done

log "=== All profiling complete ==="
log "Results in: ${RESULTS_DIR}/"
