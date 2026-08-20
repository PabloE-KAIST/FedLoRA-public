#!/usr/bin/env bash
# Standalone FedIT rank=64 on the four small GLUE tasks across gpu-host-a
# GPUs 0 and 3 (GPUs 1, 2 currently in use). Per the campaign plan, these
# tasks run at 6 clients each (resolved via _glue_lib.sh in the inner
# baseline script).
#
# Schedule (balanced across 2 GPUs by approximate wallclock):
#   GPU 0: cola → rte       (~7-8 + 2-3 = ~9-11 min)
#   GPU 3: stsb → mrpc      (~5-6 + 3-4 = ~8-10 min)
#
# Total ~9-11 min wallclock. Followed by analysis dispatch over exp/fedit.
#
# Usage:
#   bash 1_scripts/sweeps/standalone_glue_small_tasks_fedit_r64.sh
set -euo pipefail

cd ${FEDLORA_ROOT}

source "1_scripts/baseline_runs/glue/_glue_lib.sh"

OUTDIR="exp_standalone/fedit"
MAX_RANK=64
CONFIG="2_yamls/fedit/fedit-NO_quantized.yaml"
SYSTEM_METRICS_MODE="extended"
EVAL_METRICS="['acc','f1','loss','pearson','spearman','mcc']"

glue_eval_key() {
    case "$1" in
        stsb) echo "val_pearson" ;;
        cola) echo "val_mcc" ;;
        mrpc|qqp) echo "val_f1" ;;
        *)    echo "val_acc" ;;
    esac
}

run_task_on_gpu() {
    local gpu=$1
    local task=$2
    local clients
    clients=$(glue_clients_for_task "$task") || return 1
    local eval_key
    eval_key=$(glue_eval_key "$task")
    echo "[GPU ${gpu}] task=${task} clients=${clients} max_rank=${MAX_RANK} eval_key=${eval_key} — started $(date)"
    CUDA_VISIBLE_DEVICES=$gpu python federatedscope/main.py --cfg "$CONFIG" \
        device 0 \
        debug True \
        monitor.system_metrics_mode "$SYSTEM_METRICS_MODE" \
        federate.client_num "$clients" \
        data.type "${task}@glue" \
        glue.adapter.max_rank "$MAX_RANK" \
        early_stop.patience 0 \
        eval.metrics "$EVAL_METRICS" \
        eval.best_res_update_round_wise_key "$eval_key"
    echo "[GPU ${gpu}] task=${task} — finished $(date)"
}

# Two GPUs (0 and 3) in parallel; each runs two tasks sequentially.
( run_task_on_gpu 0 cola && run_task_on_gpu 0 rte ) &
PID0=$!
( run_task_on_gpu 3 stsb && run_task_on_gpu 3 mrpc ) &
PID3=$!

wait $PID0 $PID3
echo "All small-task FedIT rank=64 runs complete at $(date)"

echo "Running analysis plots over ${OUTDIR}..."
python3 analysis/single_run/run_all_experiment_plots.py "$OUTDIR" \
    --analysis-dir analysis/single_run --continue-on-error \
    && echo "Analysis complete." \
    || echo "WARNING: Some analysis plots may have failed."
