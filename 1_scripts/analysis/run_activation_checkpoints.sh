#!/usr/bin/env bash
# Run FL experiments with checkpoint saving for LoRA activation comparison.
#
# FedIT: standalone mode, rank=200 (the unconstrained optimal baseline).
# All other methods: distributed mode on the real fleet.
#
# Tasks: MRPC (6 clients), STS-B (6 clients).
#
# Usage:
#   bash 1_scripts/analysis/run_activation_checkpoints.sh [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEDLORA_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$FEDLORA_ROOT"

source 1_scripts/distributed/_lib.sh
source 1_scripts/baseline_runs/glue/_glue_lib.sh

ORCH_DIR="1_scripts/distributed/orchestrators"
PREP_DIR="1_scripts/distributed/prep"

MANIFEST="distributed/configs/client_manifest_group_a.json"
CONTROLLER_IP="${FEDLORA_CONTROLLER_IP:?set FEDLORA_CONTROLLER_IP}"
export CUDA_VISIBLE_DEVICES=3

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

CKPT_BASE="ckpt/activation_analysis"
LOG="${FEDLORA_ROOT}/exp_distributed/activation_checkpoints.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ActivationCkpt] $*" | tee -a "$LOG"
}

# ── Run a distributed experiment via full_fl_run.sh ───────
run_distributed() {
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

    if $DRY_RUN; then
        echo "[DRY-DIST] ${task}/${method}: ${config}"
        echo "           save_to: ${ckpt_path}"
        echo "           extra: ${extra_args[*]}"
        return 0
    fi

    log_status ">>> ${task}/${method} (distributed) starting"
    if bash "${ORCH_DIR}/full_fl_run.sh" "${ARGS[@]}" 2>&1 \
        | tee -a "${LOG%.log}_${task}_${method}.log"; then
        log_status "<<< ${task}/${method} DONE"
    else
        log_status "<<< ${task}/${method} FAILED"
    fi
    sleep 30
}

# ── Run FedIT standalone (rank=200) ───────────────────────
run_fedit_standalone() {
    local task=$1

    local ckpt_dir="${CKPT_BASE}/${task}"
    local ckpt_path="${ckpt_dir}/fedit.ckpt"
    mkdir -p "$ckpt_dir"

    if [[ -f "${ckpt_dir}/final_fedit.ckpt" ]]; then
        log_status "SKIP: ${task}/fedit — checkpoint already exists"
        return 0
    fi

    local clients=$(glue_clients_for_task "$task")
    local rounds=$(glue_total_rounds "$task")
    local steps=$(glue_local_steps "$task")
    local eval_key=$(glue_eval_key "$task")
    local out_ch=$(glue_out_channels "$task")

    if $DRY_RUN; then
        echo "[DRY-STANDALONE] ${task}/fedit: rank=200, clients=${clients}"
        echo "                 save_to: ${ckpt_path}"
        return 0
    fi

    log_status ">>> ${task}/fedit (standalone, rank=200) starting"
    if python federatedscope/main.py \
        --cfg 2_yamls/fedit/fedit-NO_quantized.yaml \
        device 0 \
        data.type "${task}@glue" \
        federate.client_num "$clients" \
        federate.total_round_num "$rounds" \
        train.local_update_steps "$steps" \
        eval.best_res_update_round_wise_key "$eval_key" \
        model.out_channels "$out_ch" \
        federate.save_to "$ckpt_path" \
        glue.adapter.max_rank 200 2>&1 \
        | tee -a "${LOG%.log}_${task}_fedit.log"; then
        log_status "<<< ${task}/fedit DONE"
    else
        log_status "<<< ${task}/fedit FAILED"
    fi
    sleep 10
}

# ── Task: MRPC ────────────────────────────────────────────
run_mrpc() {
    local task="mrpc"
    log_status "=== Task: ${task} ==="

    # FedIT standalone first (no fleet needed)
    run_fedit_standalone "$task"

    # Activate partitions for distributed methods
    TASK="$task" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
    sleep 5

    # HetLoRA: decay=0.5, rw=0.05
    run_distributed "$task" hetlora \
        2_yamls/hetlora/hetlora_distributed.yaml \
        glue.adapter.hetlora.pruning.regularizer_weight 0.05 \
        glue.adapter.hetlora.pruning.decay 0.50

    # FAH-QLoRA: init_rank=64, lambda_inc=lambda_dec=5
    run_distributed "$task" fahqlora \
        2_yamls/fahqlora/fah_qlora_distributed.yaml \
        glue.adapter.fah.init_rank 64 \
        glue.adapter.fah.lambda_inc 5 \
        glue.adapter.fah.lambda_dec 5

    # AdaS-LoRA v2: gamma=0.5, UL=230, DL=63, rw=0.1
    run_distributed "$task" adasparse_lorav2 \
        2_yamls/adasparse_lora_v2/adasparse_lorav2_distributed.yaml \
        glue.adapter.adasparse_lorav2.stage1.regularizer_weight 0.1 \
        glue.adapter.adasparse_lorav2.stage1.gamma 0.50 \
        glue.adapter.adasparse_lorav2.stage2.uplink_budget_window_s 230 \
        glue.adapter.adasparse_lorav2.stage2.downlink_budget_window_s 63

    # AdaS-LoRA v3: gamma=0.5, UL=230, DL=63, rw=0.005
    run_distributed "$task" adasparse_lorav3 \
        2_yamls/adasparse_lora_v3/adasparse_lorav3_distributed.yaml \
        glue.adapter.adasparse_lorav3.stage1.regularizer_weight 0.005 \
        glue.adapter.adasparse_lorav3.stage1.gamma 0.50 \
        glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s 230 \
        glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s 63
}

# ── Task: STS-B ───────────────────────────────────────────
run_stsb() {
    local task="stsb"
    log_status "=== Task: ${task} ==="

    # FedIT standalone first (no fleet needed)
    run_fedit_standalone "$task"

    # Activate partitions for distributed methods
    TASK="$task" MANIFEST="$MANIFEST" bash "${PREP_DIR}/activate_partitions.sh"
    sleep 5

    # HetLoRA: decay=0.5, rw=0.05
    run_distributed "$task" hetlora \
        2_yamls/hetlora/hetlora_distributed.yaml \
        glue.adapter.hetlora.pruning.regularizer_weight 0.05 \
        glue.adapter.hetlora.pruning.decay 0.50

    # FAH-QLoRA: init_rank=64, lambda_inc=lambda_dec=10
    run_distributed "$task" fahqlora \
        2_yamls/fahqlora/fah_qlora_distributed.yaml \
        glue.adapter.fah.init_rank 64 \
        glue.adapter.fah.lambda_inc 10 \
        glue.adapter.fah.lambda_dec 10

    # AdaS-LoRA v2: gamma=0.5, UL=230, DL=63, rw=0.05
    run_distributed "$task" adasparse_lorav2 \
        2_yamls/adasparse_lora_v2/adasparse_lorav2_distributed.yaml \
        glue.adapter.adasparse_lorav2.stage1.regularizer_weight 0.05 \
        glue.adapter.adasparse_lorav2.stage1.gamma 0.50 \
        glue.adapter.adasparse_lorav2.stage2.uplink_budget_window_s 230 \
        glue.adapter.adasparse_lorav2.stage2.downlink_budget_window_s 63

    # AdaS-LoRA v3: gamma=0.5, UL=230, DL=63, rw=0.005
    run_distributed "$task" adasparse_lorav3 \
        2_yamls/adasparse_lora_v3/adasparse_lorav3_distributed.yaml \
        glue.adapter.adasparse_lorav3.stage1.regularizer_weight 0.005 \
        glue.adapter.adasparse_lorav3.stage1.gamma 0.50 \
        glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s 230 \
        glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s 63
}

# ── Main ──────────────────────────────────────────────────

log_status "=== Activation Checkpoint Generation START ==="

if $DRY_RUN; then
    echo "=== DRY RUN ==="
    run_mrpc
    echo ""
    run_stsb
    echo ""
    echo "Total: 10 experiments (2 standalone + 8 distributed)"
    exit 0
fi

run_mrpc
run_stsb

log_status "=== Activation Checkpoint Generation COMPLETE ==="
log_status "Checkpoints in: ${CKPT_BASE}/{mrpc,stsb}/"
log_status "Next: python -m analysis.activation_comparison.run_activation_analysis"
