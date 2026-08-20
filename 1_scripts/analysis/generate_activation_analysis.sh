#!/usr/bin/env bash
set -euo pipefail
cd ${FEDLORA_ROOT}

CKPT_BASE="ckpt/activation_analysis"
OUT_BASE="0_results/activation_analysis"

COMMON_RENAMES=(
    --rename-method adasparse_lorav2 'AdaS-LoRA-C'
    --rename-method adasparse_lorav3 'AdaS-LoRA-L'
)

run_task() {
    local task=$1
    shift
    local extra=("$@")

    echo ""
    echo "########################################"
    echo "  ${task}"
    echo "########################################"

    python -m analysis.activation_comparison.run_activation_analysis \
        --task "$task" \
        --ckpt-dir "${CKPT_BASE}/${task}" \
        --output-dir "${OUT_BASE}/${task}" \
        "${COMMON_RENAMES[@]}" \
        "${extra[@]}"

    python -m analysis.activation_comparison.run_activation_analysis \
        --task "$task" \
        --ckpt-dir "${CKPT_BASE}/${task}" \
        --output-dir "${OUT_BASE}/${task}-noLegend" \
        --no-legend \
        "${COMMON_RENAMES[@]}" \
        "${extra[@]}"

    python -m analysis.activation_comparison.run_activation_analysis \
        --task "$task" \
        --ckpt-dir "${CKPT_BASE}/${task}" \
        --output-dir "${OUT_BASE}/${task}-noTitle" \
        --no-title \
        "${COMMON_RENAMES[@]}" \
        "${extra[@]}"
}

TASKS=()
if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    for d in "${CKPT_BASE}"/*/; do
        t=$(basename "$d")
        TASKS+=("$t")
    done
fi

if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "No tasks found in ${CKPT_BASE}. Run checkpoint experiments first."
    exit 1
fi

for task in "${TASKS[@]}"; do
    run_task "$task"
done

echo ""
echo "========================================"
echo "  DONE — outputs in ${OUT_BASE}/"
echo "========================================"
