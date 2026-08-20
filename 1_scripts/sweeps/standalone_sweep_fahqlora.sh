#!/usr/bin/env bash
# Standalone FAH-QLoRA sweep on gpu-host-a GPUs 0,2: init_rank=16,32 (6 experiments).
# Complements standalone_sweep_fahqlora_lugia.sh on gpu-host-b (init_rank=64).
# Skips experiments that already have a Final result in their output dir.
set -euo pipefail

cd ${FEDLORA_ROOT}

CONFIG="2_yamls/fahqlora/fah_qlora-NO_quantized.yaml"
OUTDIR="exp_standalone/fahqlora"
CLIENTS=12
MAX_RANK=200
STRATEGY="custom"

INIT_RANKS=(16 32 64)
LAMBDAS=(1 5 10)

is_completed() {
    local ir=$1 lam=$2
    for d in "${OUTDIR}/"*"__strategy_custom__initr_${ir}__lambda_${lam}__"*; do
        if [ -d "$d" ] && grep -q "'Round': 'Final'" "$d/eval_results.log" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

run_combos() {
    local gpu=$1
    shift
    local combos=("$@")

    for combo in "${combos[@]}"; do
        IFS='|' read -r ir lam <<< "$combo"
        if is_completed "$ir" "$lam"; then
            echo "[GPU ${gpu}] SKIP (done): initr=${ir} lambda=${lam}"
            continue
        fi
        echo "[GPU ${gpu}] initr=${ir} lambda=${lam} — started $(date)"
        CUDA_VISIBLE_DEVICES=$gpu python federatedscope/main.py --cfg "$CONFIG" \
            device 0 \
            outdir "$OUTDIR" \
            federate.client_num "$CLIENTS" \
            glue.adapter.max_rank "$MAX_RANK" \
            glue.adapter.hetero_strategy "$STRATEGY" \
            glue.adapter.fah.init_rank "$ir" \
            glue.adapter.fah.lambda_inc "$lam" \
            glue.adapter.fah.lambda_dec "$lam"
        echo "[GPU ${gpu}] initr=${ir} lambda=${lam} — finished $(date)"
    done
}

# GPU 0: init_rank=16 (all lambdas) — 3 experiments
GPU0_COMBOS=("16|1" "16|5" "16|10")

# GPU 2: init_rank=32 (all lambdas) — 3 experiments
GPU2_COMBOS=("32|1" "32|5" "32|10")

mkdir -p "$OUTDIR"

run_combos 0 "${GPU0_COMBOS[@]}" &
PID0=$!
run_combos 2 "${GPU2_COMBOS[@]}" &
PID2=$!

wait $PID0 $PID2

echo "All FAH-QLoRA standalone sweeps complete at $(date)"

echo "Running analysis plots..."
python3 analysis/single_run/run_all_experiment_plots.py "$OUTDIR" \
    --analysis-dir analysis/single_run --force --continue-on-error \
    && echo "Analysis complete." \
    || echo "WARNING: Some analysis plots may have failed."
