#!/usr/bin/env bash
set -euo pipefail
cd ${FEDLORA_ROOT}

COMMON_RENAMES=(
    --rename-method adasparse_lorav2 'AdaS-LoRA-C'
    --rename-method adasparse_lorav3 'AdaS-LoRA-L'
)

run_variant() {
    local task=$1
    local exp_dir=$2
    local out_dir=$3
    shift 3
    local extra_args=("$@")

    echo ""
    echo "########################################"
    echo "  ${task} → ${out_dir}"
    echo "########################################"

    python -m analysis.cross_method.run_all_analysis \
        --task "$task" \
        --exp-dir "$exp_dir" \
        --output-dir "$out_dir" \
        --mode both \
        --fedit-golden-only \
        "${COMMON_RENAMES[@]}" \
        "${extra_args[@]}"
}

NL="0_results/final/thesis-v3-noLegend"
LG="0_results/final/thesis-v3-Legend"

# --- CoLA ---
COLA_ARGS=(--exclude-shade hetlora decay 0.95 --override-selection fahqlora initr_64__lambda_1)
run_variant cola exp_distributed "$NL/cola" --no-legend "${COLA_ARGS[@]}"
run_variant cola exp_distributed "$LG/cola" --no-title  "${COLA_ARGS[@]}"

# --- RTE (golden-only HetLoRA) ---
# Selections finalized 2026-06-23 after RTE rerun review:
#   HetLoRA decay=0.65 -> rw=5e-3 (was rw=5e-2); decay=0.5/0.8 kept.
#   v2 gamma=0.8 ul=230 override kept; v3/FAH selections kept.
RTE_ARGS=(--exclude-shade hetlora decay 0.95 --override-selection adasparse_lorav2 regularizer_5e-2__gamma_0.8__ul_230 --override-selection hetlora regularizer_5e-3__decay_0.65)
run_variant rte exp_distributed/golden "$NL/rte" --no-legend "${RTE_ARGS[@]}"
run_variant rte exp_distributed/golden "$LG/rte" --no-title  "${RTE_ARGS[@]}"

# --- RTE trueHetlora (full synced HetLoRA grid) ---
run_variant rte exp_distributed "$NL/rte-trueHetlora" --no-legend "${RTE_ARGS[@]}"
run_variant rte exp_distributed "$LG/rte-trueHetlora" --no-title  "${RTE_ARGS[@]}"

# --- MRPC ---
MRPC_ARGS=(--override-selection adasparse_lorav2 regularizer_5e-2__gamma_0.65__ul_230 --override-selection adasparse_lorav2 regularizer_5e-3__gamma_0.8__ul_230)
run_variant mrpc exp_distributed "$NL/mrpc" --no-legend "${MRPC_ARGS[@]}"
run_variant mrpc exp_distributed "$LG/mrpc" --no-title  "${MRPC_ARGS[@]}"

# --- STS-B ---
STSB_ARGS=(
    --override-selection hetlora regularizer_5e-2__decay_0.5
    --override-selection hetlora regularizer_5e-3__decay_0.65
    --override-selection adasparse_lorav2 regularizer_5e-2__gamma_0.5__ul_230
    --override-selection adasparse_lorav2 regularizer_5e-3__gamma_0.65__ul_230
    --override-selection adasparse_lorav2 regularizer_5e-3__gamma_0.8__ul_230
    --override-selection fahqlora initr_32__lambda_5
)
run_variant stsb exp_distributed "$NL/stsb" --no-legend "${STSB_ARGS[@]}"
run_variant stsb exp_distributed "$LG/stsb" --no-title  "${STSB_ARGS[@]}"

# --- SST-2 (full rerun: v2/v3 now use the standard grid; legacy archive auto-excluded) ---
# HetLoRA decay=0.8 -> rw=5e-2 (finalized 2026-06-30); all other natural picks kept.
SST2_ARGS=(
    --override-selection hetlora regularizer_5e-2__decay_0.8
)
run_variant sst2 exp_distributed "$NL/sst2" --no-legend "${SST2_ARGS[@]}"
run_variant sst2 exp_distributed "$LG/sst2" --no-title  "${SST2_ARGS[@]}"

echo ""
echo "========================================"
echo "  DONE"
echo "  noLegend: $NL"
echo "  Legend:   $LG"
echo "========================================"
