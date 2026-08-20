#!/bin/bash

cd ${FEDLORA_ROOT}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_glue_lib.sh"
task="${TASK:-sst2}"
clients=$(glue_clients_for_task "$task") || exit 1

configs=(
    "2_yamls/adasparse_lora/adasparse-lora-NO_quantized.yaml"
)

device=1
debug=("True")
system_metrics_mode=("extended")
max_rank=200 #to match with the maximum rank as defined in paper types --> federatedscope/llm/utils/client_config_generator.py
rank_strategy=(
    custom
    #"homo"
    #"random"
    #"heavy_tail" 
    #"heavy_tail_strong"
    #"normal"
    )
# Sweep regularizer_weight values (using decimal notation)
regularizer_weight=(0.1) # 0.0003 0.001 0.003 0.01 0.03 0.1)
sparsity_ratio=(0.99)

for w in "${regularizer_weight[@]}"; do
    echo "=============================================="
    echo "Running: config=$configs | task=$task | debug=$debug | rank_strategy=$rank_strategy | max_rank=$max_rank | client_num=$clients"
    echo "Started at: $(date)"
    echo "=============================================="

    python federatedscope/main.py --cfg "$configs" \
        device "$device" \
        debug "$debug" \
        monitor.system_metrics_mode "$system_metrics_mode" \
        federate.client_num "$clients" \
        data.type "${task}@glue" \
        glue.adapter.max_rank "$max_rank" \
        glue.adapter.hetero_strategy "$rank_strategy" \
        glue.adapter.adasparse_lora.rank_max "$max_rank" \
        glue.adapter.adasparse_lora.pruning.regularizer_weight "$w" \
        glue.adapter.adasparse_lora.pruning.gamma "$sparsity_ratio"
    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"