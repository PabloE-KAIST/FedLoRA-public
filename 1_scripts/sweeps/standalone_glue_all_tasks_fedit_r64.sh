#!/usr/bin/env bash
# Standalone FedIT rank=64 on all 8 GLUE tasks across Bulbasaur GPUs 0 and 3.
# Task-aware client_num, eval metric, and local_update_steps (resolved via
# _glue_lib.sh). MRPC/RTE use local_update_steps=15 to avoid overfitting
# on their tiny per-client partitions; all others use 30.
#
# Schedule (2 GPUs, balanced by approximate wallclock):
#   GPU 0: mrpc (~3 min) → rte (~2 min) → sst2 (~60 min) → qnli (~90 min)  ≈ 2.5 h
#   GPU 3: cola (~8 min) → stsb (~6 min) → qqp (~5-6 h)  → mnli (~5-6 h)   ≈ 10 h
#
# Usage:
#   bash 1_scripts/sweeps/standalone_glue_all_tasks_fedit_r64.sh
set -euo pipefail

cd ${FEDLORA_ROOT}

source "1_scripts/baseline_runs/glue/_glue_lib.sh"

OUTDIR="exp_standalone/fedit"
MAX_RANK=64
CONFIG="2_yamls/fedit/fedit-NO_quantized.yaml"
SYSTEM_METRICS_MODE="extended"
EVAL_METRICS="['acc','f1','loss','pearson','spearman','mcc']"

run_task_on_gpu() {
    local gpu=$1
    local task=$2
    local clients
    clients=$(glue_clients_for_task "$task") || return 1
    local eval_key
    eval_key=$(glue_eval_key "$task")
    local local_steps
    local_steps=$(glue_local_steps "$task")
    echo "[GPU ${gpu}] task=${task} clients=${clients} steps=${local_steps} eval_key=${eval_key} — started $(date)"
    CUDA_VISIBLE_DEVICES=$gpu python federatedscope/main.py --cfg "$CONFIG" \
        device 0 \
        debug True \
        monitor.system_metrics_mode "$SYSTEM_METRICS_MODE" \
        federate.client_num "$clients" \
        data.type "${task}@glue" \
        glue.adapter.max_rank "$MAX_RANK" \
        train.local_update_steps "$local_steps" \
        early_stop.patience 0 \
        eval.metrics "$EVAL_METRICS" \
        eval.best_res_update_round_wise_key "$eval_key"
    echo "[GPU ${gpu}] task=${task} — finished $(date)"
}

# GPU 0: re-runs (MRPC, RTE with steps=15) then big tasks
( run_task_on_gpu 0 mrpc && run_task_on_gpu 0 rte && run_task_on_gpu 0 sst2 && run_task_on_gpu 0 qnli ) &
PID0=$!

# GPU 3: small tasks (steps=15 for cola/stsb already 30) then big tasks
( run_task_on_gpu 3 cola && run_task_on_gpu 3 stsb && run_task_on_gpu 3 qqp && run_task_on_gpu 3 mnli ) &
PID3=$!

wait $PID0 $PID3
echo "All FedIT rank=64 runs complete at $(date)"

echo "Running analysis plots over ${OUTDIR}..."
python3 analysis/single_run/run_all_experiment_plots.py "$OUTDIR" \
    --analysis-dir analysis/single_run --continue-on-error \
    && echo "Analysis complete." \
    || echo "WARNING: Some analysis plots may have failed."
