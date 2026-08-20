#!/usr/bin/env bash
# AdaSparse v3 FULL GRID sweep on gpu-host-a GPUs 0,2.
# Runs rw=0.01 (32 experiments). rw=0.1,0.3 run on gpu-host-b.
# Skips experiments that already have a Final result in their output dir.
set -euo pipefail

cd ${FEDLORA_ROOT}

CONFIG="2_yamls/adasparse_lora_v3/adasparse-lorav3-NO_quantized.yaml"
OUTDIR="exp_standalone/adasparse_lorav3"
CLIENTS=12
MAX_RANK=200
STRATEGY="custom"

UL_WINDOWS=(50 100 150 190)
DL_WINDOWS=(51 38)

mkdir -p "$OUTDIR"

run_gpu() {
    local gpu=$1
    shift
    local gammas=("$@")
    local rw=0.01

    for gamma in "${gammas[@]}"; do
        for ul in "${UL_WINDOWS[@]}"; do
            for dl in "${DL_WINDOWS[@]}"; do
                echo "[GPU ${gpu}] rw=${rw} gamma=${gamma} ul=${ul} dl=${dl} — started $(date)"
                CUDA_VISIBLE_DEVICES=$gpu python federatedscope/main.py --cfg "$CONFIG" \
                    device 0 \
                    outdir "$OUTDIR" \
                    federate.client_num "$CLIENTS" \
                    glue.adapter.max_rank "$MAX_RANK" \
                    glue.adapter.hetero_strategy "$STRATEGY" \
                    glue.adapter.adasparse_lorav3.rank_max "$MAX_RANK" \
                    glue.adapter.adasparse_lorav3.stage1.regularizer_weight "$rw" \
                    glue.adapter.adasparse_lorav3.stage1.gamma "$gamma" \
                    glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s "$ul" \
                    glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s "$dl"
                echo "[GPU ${gpu}] rw=${rw} gamma=${gamma} ul=${ul} dl=${dl} — finished $(date)"
            done
        done
    done
}

# GPU 0: rw=0.01, gamma=0.50,0.60
run_gpu 0 0.50 0.60 &
PID0=$!

# GPU 2: rw=0.01, gamma=0.80,0.95
run_gpu 2 0.80 0.95 &
PID2=$!

wait $PID0 $PID2
echo "AdaSparse v3 (gpu-host-a, rw=0.01) complete at $(date)"
