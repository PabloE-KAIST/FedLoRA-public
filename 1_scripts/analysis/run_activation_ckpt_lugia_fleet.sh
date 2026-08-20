#!/usr/bin/env bash
# Distributed checkpoint runs on Lugia fleet (Group B): STS-B task.
# This script runs ON Lugia via SSH.
set -euo pipefail

cd ${FEDLORA_ROOT}
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate fedlora
source 1_scripts/distributed/_lib.sh
source 1_scripts/baseline_runs/glue/_glue_lib.sh

ORCH_DIR="1_scripts/distributed/orchestrators"
PREP_DIR="1_scripts/distributed/prep"

MANIFEST="distributed/configs/client_manifest_group_b.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
export CUDA_VISIBLE_DEVICES=0

CKPT_BASE="ckpt/activation_analysis"
LOG="${FEDLORA_ROOT}/exp_distributed/activation_ckpt_lugia_fleet.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [Lugia-Fleet] $*" | tee -a "$LOG"
}

run_dist() {
    local task=$1 method=$2 config=$3
    shift 3
    local extra_args=("$@")

    local ckpt_dir="${CKPT_BASE}/${task}"
    local ckpt_path="${ckpt_dir}/${method}.ckpt"
    mkdir -p "$ckpt_dir"

    if [[ -f "${ckpt_dir}/final_${method}.ckpt" ]]; then
        log_status "SKIP: ${task}/${method} — checkpoint already exists"
        return 0
    fi

    local clients=$(glue_clients_for_task "$task")
    local rounds=$(glue_total_rounds "$task")
    local steps=$(glue_local_steps "$task")
    local eval_key=$(glue_eval_key "$task")
    local out_ch=$(glue_out_channels "$task")

    local ARGS=(--config "$config" --manifest "$MANIFEST" --port-offset 0)
    ARGS+=(-- \
        device 0 \
        data.type "${task}@glue" \
        federate.client_num "$clients" \
        federate.sample_client_num "$clients" \
        federate.total_round_num "$rounds" \
        train.local_update_steps "$steps" \
        eval.best_res_update_round_wise_key "$eval_key" \
        model.out_channels "$out_ch" \
        federate.save_to "$ckpt_path" \
        "${extra_args[@]}")

    log_status ">>> ${task}/${method} starting"
    if bash "${ORCH_DIR}/full_fl_run.sh" "${ARGS[@]}" 2>&1 \
        | tee -a "${LOG%.log}_${task}_${method}.log"; then
        log_status "<<< ${task}/${method} DONE"
    else
        log_status "<<< ${task}/${method} FAILED"
    fi
    sleep 30
}

log_status "=== Lugia Fleet: STS-B Checkpoint Runs START ==="

TASK="stsb" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
sleep 5

# HetLoRA: decay=0.5, rw=0.05
run_dist stsb hetlora \
    2_yamls/hetlora/hetlora_distributed.yaml \
    glue.adapter.hetlora.pruning.regularizer_weight 0.05 \
    glue.adapter.hetlora.pruning.decay 0.50

# FAH-QLoRA: init_rank=64, lambda=10
run_dist stsb fahqlora \
    2_yamls/fahqlora/fah_qlora_distributed.yaml \
    glue.adapter.fah.init_rank 64 \
    glue.adapter.fah.lambda_inc 10 \
    glue.adapter.fah.lambda_dec 10

# AdaS-LoRA v2: gamma=0.5, UL=230, DL=63, rw=0.05
run_dist stsb adasparse_lorav2 \
    2_yamls/adasparse_lora_v2/adasparse_lorav2_distributed.yaml \
    glue.adapter.adasparse_lorav2.stage1.regularizer_weight 0.05 \
    glue.adapter.adasparse_lorav2.stage1.gamma 0.50 \
    glue.adapter.adasparse_lorav2.stage2.uplink_budget_window_s 230 \
    glue.adapter.adasparse_lorav2.stage2.downlink_budget_window_s 63

# AdaS-LoRA v3: gamma=0.5, UL=230, DL=63, rw=0.005
run_dist stsb adasparse_lorav3 \
    2_yamls/adasparse_lora_v3/adasparse_lorav3_distributed.yaml \
    glue.adapter.adasparse_lorav3.stage1.regularizer_weight 0.005 \
    glue.adapter.adasparse_lorav3.stage1.gamma 0.50 \
    glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s 230 \
    glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s 63

log_status "=== Lugia Fleet: STS-B COMPLETE ==="
