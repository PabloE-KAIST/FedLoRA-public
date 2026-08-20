#!/usr/bin/env bash
# FedIT standalone (rank=200) checkpoint runs for MRPC and STS-B.
# Intended to run on Lugia GPU 1.
set -euo pipefail

cd ${FEDLORA_ROOT}
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate fedlora
source 1_scripts/baseline_runs/glue/_glue_lib.sh

export CUDA_VISIBLE_DEVICES=1
CKPT_BASE="ckpt/activation_analysis"
LOG="exp_distributed/activation_ckpt_fedit_standalone.log"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [FedIT-Standalone] $*" | tee -a "$LOG"
}

run_fedit() {
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

    # STS-B needs pearson metric; add it for all tasks (harmless for others)
    log_status ">>> ${task}/fedit (standalone, rank=200) starting"
    if python federatedscope/main.py \
        --cfg 2_yamls/fedit/fedit-NO_quantized.yaml \
        device 0 \
        data.type "${task}@glue" \
        federate.client_num "$clients" \
        federate.total_round_num "$rounds" \
        train.local_update_steps "$steps" \
        eval.best_res_update_round_wise_key "$eval_key" \
        eval.metrics "['acc','f1','loss','pearson','mcc']" \
        model.out_channels "$out_ch" \
        federate.save_to "$ckpt_path" \
        glue.adapter.max_rank 200 2>&1 \
        | tee -a "${LOG%.log}_${task}.log"; then
        log_status "<<< ${task}/fedit DONE"
    else
        log_status "<<< ${task}/fedit FAILED"
    fi
    sleep 5
}

log_status "=== FedIT Standalone Checkpoint Runs START ==="
run_fedit mrpc
run_fedit stsb
log_status "=== FedIT Standalone Checkpoint Runs COMPLETE ==="
