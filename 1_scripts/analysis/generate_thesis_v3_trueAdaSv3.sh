#!/usr/bin/env bash
# Generate thesis-v3 analysis plots for cola, sst2, stsb using the
# full-grid rerun data (all 5 methods with complete v3 coverage).
#
# Output dirs:
#   0_results/final/thesis-v3-noLegend/{task}-trueAdaSv3/
#   0_results/final/thesis-v3-Legend/{task}-trueAdaSv3/
#
# Mirrors the structure of rte-trueHetlora: full experiment grid
# with complete AdaS-LoRA v3 (layer-aware) data included.
#
# Run AFTER fleet_campaign_v3_backfill.sh completes.
# Requires: Lugia results synced to Bulbasaur.
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

# ── CoLA ──────────────────────────────────────────────────────────
# Selections finalized 2026-06-23 after CoLA rerun review:
#   HetLoRA  decay=0.5 -> rw=1e-1, decay=0.65 -> rw=1e-1, decay=0.8 keep (rw=5e-3)
#   v2 (AdaS-C) gamma=0.65 -> rw=1e-1 ul=230; gamma=0.5/0.8 keep
#   v3 (AdaS-L) gamma=0.5 -> rw=1e-1 ul=230; gamma=0.65/0.8 keep
COLA_ARGS=(
    --exclude-shade hetlora decay 0.95
    --override-selection fahqlora initr_64__lambda_1
    --override-selection hetlora regularizer_1e-1__decay_0.5
    --override-selection hetlora regularizer_1e-1__decay_0.65
    --override-selection adasparse_lorav2 regularizer_1e-1__gamma_0.65__ul_230
    --override-selection adasparse_lorav3 regularizer_1e-1__gamma_0.5__ul_230
)
run_variant cola exp_distributed "$NL/cola-trueAdaSv3" --no-legend "${COLA_ARGS[@]}"
run_variant cola exp_distributed "$LG/cola-trueAdaSv3" --no-title  "${COLA_ARGS[@]}"

# ── STS-B ─────────────────────────────────────────────────────────
# Selections finalized 2026-06-23: HetLoRA decay=0.5 -> rw=1e-1
# (was rw=5e-2); decay=0.65/0.8 and all v2/v3/FAH selections kept.
STSB_ARGS=(
    --override-selection hetlora regularizer_1e-1__decay_0.5
    --override-selection hetlora regularizer_5e-3__decay_0.65
    --override-selection adasparse_lorav2 regularizer_5e-2__gamma_0.5__ul_230
    --override-selection adasparse_lorav2 regularizer_5e-3__gamma_0.65__ul_230
    --override-selection adasparse_lorav2 regularizer_5e-3__gamma_0.8__ul_230
    --override-selection fahqlora initr_32__lambda_5
)
run_variant stsb exp_distributed "$NL/stsb-trueAdaSv3" --no-legend "${STSB_ARGS[@]}"
run_variant stsb exp_distributed "$LG/stsb-trueAdaSv3" --no-title  "${STSB_ARGS[@]}"

# ── SST-2 ─────────────────────────────────────────────────────────
# SST-2 rerun (v2 06-24→27, v3 06-27→29) now uses the SAME standard grid as the
# other tasks: rw {5e-3,5e-2,1e-1} x gamma {0.5,0.65,0.8} x ul {230,460}. The old
# legacy-archive overrides (ul_690/ul_870, decay_0.95) no longer apply and the
# legacy archive points are auto-excluded by the non-golden filter.
# Selections finalized 2026-06-30 after SST-2 rerun review: HetLoRA decay=0.8 -> rw=5e-2
# (was natural rw=5e-3); all other natural best-per-shade selections kept.
SST2_ARGS=(
    --override-selection hetlora regularizer_5e-2__decay_0.8
)
run_variant sst2 exp_distributed "$NL/sst2-trueAdaSv3" --no-legend "${SST2_ARGS[@]}"
run_variant sst2 exp_distributed "$LG/sst2-trueAdaSv3" --no-title  "${SST2_ARGS[@]}"

echo ""
echo "========================================"
echo "  DONE — trueAdaSv3 analysis"
echo "  noLegend: $NL/{cola,stsb,sst2}-trueAdaSv3/"
echo "  Legend:   $LG/{cola,stsb,sst2}-trueAdaSv3/"
echo "========================================"
