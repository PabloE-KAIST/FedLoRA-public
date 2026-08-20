#!/usr/bin/env bash
# AdaSparse v2 FULL GRID sweep on gpu-host-a GPUs 0,2.
# Runs rw=0.01 and rw=0.1 (64 experiments). rw=0.3 runs on gpu-host-b.
# Skips experiments that already have a Final result in their output dir.
set -euo pipefail

cd ${FEDLORA_ROOT}

CONFIG="2_yamls/adasparse_lora_v2/adasparse-lorav2-NO_quantized.yaml"
OUTDIR="exp_standalone/adasparse_lorav2"
CLIENTS=12
MAX_RANK=200
STRATEGY="custom"

UL_WINDOWS=(50 100 150 190)
DL_WINDOWS=(51 38)

fmt_rw() {
    python3 -c "v=float('$1'); print(f'{v:.0e}'.replace('e-0','e-').replace('e+0','e+'))"
}

is_completed() {
    local rw_tag=$1 gamma=$2 ul=$3 dl=$4
    # Normalize gamma: strip trailing zeros (0.50 -> 0.5) to match dir naming
    local gamma_norm
    gamma_norm=$(python3 -c "print(float('$gamma'))")
    for d in "${OUTDIR}/"*"__strategy_custom__regularizer_${rw_tag}__gamma_${gamma_norm}__ul_${ul}__dl_${dl}__"*; do
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
    local count=0
    local total=${#combos[@]}

    for combo in "${combos[@]}"; do
        IFS='|' read -r rw gamma <<< "$combo"
        local rw_tag
        rw_tag=$(fmt_rw "$rw")
        for ul in "${UL_WINDOWS[@]}"; do
            for dl in "${DL_WINDOWS[@]}"; do
                count=$((count + 1))
                if is_completed "$rw_tag" "$gamma" "$ul" "$dl"; then
                    echo "[GPU ${gpu}] SKIP (done): rw=${rw} gamma=${gamma} ul=${ul} dl=${dl}"
                    continue
                fi
                echo "[GPU ${gpu}] (${count}) rw=${rw} gamma=${gamma} ul=${ul} dl=${dl} — started $(date)"
                CUDA_VISIBLE_DEVICES=$gpu python federatedscope/main.py --cfg "$CONFIG" \
                    device 0 \
                    outdir "$OUTDIR" \
                    federate.client_num "$CLIENTS" \
                    glue.adapter.max_rank "$MAX_RANK" \
                    glue.adapter.hetero_strategy "$STRATEGY" \
                    glue.adapter.adasparse_lorav2.rank_max "$MAX_RANK" \
                    glue.adapter.adasparse_lorav2.stage1.regularizer_weight "$rw" \
                    glue.adapter.adasparse_lorav2.stage1.gamma "$gamma" \
                    glue.adapter.adasparse_lorav2.stage2.uplink_budget_window_s "$ul" \
                    glue.adapter.adasparse_lorav2.stage2.downlink_budget_window_s "$dl"
                echo "[GPU ${gpu}] rw=${rw} gamma=${gamma} ul=${ul} dl=${dl} — finished $(date)"
            done
        done
    done
}

# GPU 0: rw=0.01 (all gammas)
GPU0_COMBOS=(
    "0.01|0.50" "0.01|0.60" "0.01|0.80" "0.01|0.95"
)

# GPU 2: rw=0.1 (all gammas)
GPU2_COMBOS=(
    "0.1|0.50" "0.1|0.60" "0.1|0.80" "0.1|0.95"
)

run_combos 0 "${GPU0_COMBOS[@]}" &
PID0=$!
run_combos 2 "${GPU2_COMBOS[@]}" &
PID1=$!

wait $PID0 $PID1
echo "All AdaSparse v2 full-grid (2-GPU continuation) complete at $(date)"

echo "Running analysis plots..."
python3 analysis/single_run/run_all_experiment_plots.py "$OUTDIR" \
    --analysis-dir analysis/single_run --force --continue-on-error \
    && echo "Analysis complete." \
    || echo "WARNING: Some analysis plots may have failed."
