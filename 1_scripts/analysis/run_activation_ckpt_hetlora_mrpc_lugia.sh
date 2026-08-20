#!/usr/bin/env bash
# HetLoRA MRPC on Lugia (Group B fleet) — fallback for Bulbasaur OOM.
# Runs FROM Bulbasaur, SSHes to Lugia for the server.
# Uses Group B devices with MRPC partitions activated.
set -euo pipefail

cd ${FEDLORA_ROOT}
source 1_scripts/distributed/_lib.sh
source 1_scripts/baseline_runs/glue/_glue_lib.sh

PREP_DIR="1_scripts/distributed/prep"

MANIFEST="distributed/configs/client_manifest_group_b.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
LUGIA_HOST="gpu-host-b"
LUGIA_FEDLORA="${FEDLORA_ROOT}"

CKPT_BASE="ckpt/activation_analysis"
LOG="${FEDLORA_ROOT}/exp_distributed/activation_ckpt_hetlora_mrpc_lugia.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [HetLoRA-MRPC-Lugia] $*" | tee -a "$LOG"
}

TASK="mrpc"
METHOD="hetlora"
CONFIG="2_yamls/hetlora/hetlora_distributed.yaml"
CKPT_DIR="${CKPT_BASE}/${TASK}"
CKPT_PATH="${CKPT_DIR}/${METHOD}.ckpt"

# Check if checkpoint already exists on Lugia
if ssh "$LUGIA_HOST" "test -f ${LUGIA_FEDLORA}/${CKPT_DIR}/final_${METHOD}.ckpt" 2>/dev/null; then
    log_status "SKIP: ${TASK}/${METHOD} — checkpoint already exists on Lugia"
    exit 0
fi
# Also check locally (Bulbasaur)
if [[ -f "${CKPT_DIR}/final_${METHOD}.ckpt" ]]; then
    log_status "SKIP: ${TASK}/${METHOD} — checkpoint already exists locally"
    exit 0
fi

log_status "=== HetLoRA MRPC on Lugia (Group B fleet) START ==="

# Activate MRPC partitions on Group B devices
log_status "Activating MRPC partitions for Group B"
TASK="$TASK" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
sleep 5

# Cleanup old workers
MANIFEST="$MANIFEST" bash "${PREP_DIR}/cleanup_workers_for_manifest.sh" 2>/dev/null || true

# Restart DAs pointing to Lugia
log_status "Restarting DAs for Group B → ${CONTROLLER_IP}"
CONTROLLER_IP="$CONTROLLER_IP" MANIFEST="$MANIFEST" \
    bash "${PREP_DIR}/restart_das_for_manifest.sh" 2>/dev/null || true
sleep 15

CLIENTS=$(glue_clients_for_task "$TASK")
ROUNDS=$(glue_total_rounds "$TASK")
STEPS=$(glue_local_steps "$TASK")
EVAL_KEY=$(glue_eval_key "$TASK")
OUT_CH=$(glue_out_channels "$TASK")

log_status ">>> ${TASK}/${METHOD} starting (server on Lugia GPU 1)"
ssh "$LUGIA_HOST" bash <<REMOTE_CMD
source "\$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate fedlora
cd ${LUGIA_FEDLORA}
export CUDA_VISIBLE_DEVICES=1

mkdir -p "${CKPT_DIR}"

python -m distributed.server.main \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --proto \
    --port-offset 0 \
    --device-port-offset 100 \
    --device-wait-timeout 180 \
    --verbose \
    --worker-config-path "$CONFIG" \
    --bandwidth-setting 1 \
    --bandwidth-json "${LUGIA_FEDLORA}/1_scripts/distributed/infra/generated_bandwidths/bandwidth_limits_dl.json" \
    --server-nic enp66s0f0 \
    device 0 \
    data.type "${TASK}@glue" \
    federate.client_num "$CLIENTS" \
    federate.sample_client_num "$CLIENTS" \
    federate.total_round_num "$ROUNDS" \
    train.local_update_steps "$STEPS" \
    eval.best_res_update_round_wise_key "$EVAL_KEY" \
    model.out_channels "$OUT_CH" \
    federate.save_to "$CKPT_PATH" \
    glue.adapter.hetlora.pruning.regularizer_weight 0.05 \
    glue.adapter.hetlora.pruning.decay 0.50
REMOTE_CMD

if [ $? -eq 0 ]; then
    log_status "<<< ${TASK}/${METHOD} DONE"
else
    log_status "<<< ${TASK}/${METHOD} FAILED"
fi

log_status "=== HetLoRA MRPC on Lugia COMPLETE ==="
