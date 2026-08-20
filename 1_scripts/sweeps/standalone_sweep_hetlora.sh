#!/usr/bin/env bash
# Standalone HetLoRA sweep on gpu-host-a GPUs 0,2: rw=0.01,0.1 (8 experiments).
# Complements standalone_sweep_hetlora_lugia.sh on gpu-host-b (rw=0.3).
set -euo pipefail

cd ${FEDLORA_ROOT}

CONFIG="2_yamls/hetlora/hetlora-NO_quantized.yaml"
OUTDIR="exp_standalone/hetlora"
CLIENTS=12
MAX_RANK=200
STRATEGY="custom"

SPARSITY_RATIOS=(0.50 0.60 0.80 0.95)

run_gpu() {
    local gpu=$1
    local rw=$2
    shift 2
    local ratios=("$@")

    for sr in "${ratios[@]}"; do
        echo "[GPU ${gpu}] rw=${rw} sr=${sr} — started $(date)"
        CUDA_VISIBLE_DEVICES=$gpu python federatedscope/main.py --cfg "$CONFIG" \
            device 0 \
            outdir "$OUTDIR" \
            federate.client_num "$CLIENTS" \
            glue.adapter.max_rank "$MAX_RANK" \
            glue.adapter.hetero_strategy "$STRATEGY" \
            glue.adapter.hetlora.rank_max "$MAX_RANK" \
            glue.adapter.hetlora.pruning.regularizer_weight "$rw" \
            glue.adapter.hetlora.pruning.decay "$sr"
        echo "[GPU ${gpu}] rw=${rw} sr=${sr} — finished $(date)"
    done
}

# GPU 0: rw=0.01 (4 experiments)
run_gpu 0 0.01 "${SPARSITY_RATIOS[@]}" &
PID0=$!

# GPU 2: rw=0.1 (4 experiments)
run_gpu 2 0.1 "${SPARSITY_RATIOS[@]}" &
PID2=$!

wait $PID0 $PID2
echo "All HetLoRA standalone sweeps complete at $(date)"

echo "Running analysis plots..."
python3 analysis/single_run/run_all_experiment_plots.py "$OUTDIR" \
    --analysis-dir analysis/single_run --force --continue-on-error \
    && echo "Analysis complete." \
    || echo "WARNING: Some analysis plots may have failed."
