#!/bin/bash

cd ${FEDLORA_ROOT}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_glue_lib.sh"
task="${TASK:-sst2}"
clients=$(glue_clients_for_task "$task") || exit 1

configs=(
    "2_yamls/hetlora/hetlora-NO_quantized.yaml"
)

device=2
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
regularizer_weight=(0.1) #(0.0001 0.0003 0.001 0.003)
sparsity_ratio=(0.65) #0.95, 0.80, 0.65 0.55

for strategy in "${rank_strategy[@]}"; do
    echo "=============================================="
    echo "Running: config=$configs | task=$task | debug=$debug | rank_strategy=$strategy | max_rank=$max_rank | client_num=$clients"
    echo "Started at: $(date)"
    echo "=============================================="

    python federatedscope/main.py --cfg "$configs" \
        device "$device" \
        debug "$debug" \
        monitor.system_metrics_mode "$system_metrics_mode" \
        federate.client_num "$clients" \
        data.type "${task}@glue" \
        glue.adapter.max_rank "$max_rank" \
        glue.adapter.hetero_strategy "$strategy" \
        glue.adapter.hetlora.rank_max "$max_rank" \
        glue.adapter.hetlora.pruning.regularizer_weight "$regularizer_weight" \
        glue.adapter.hetlora.pruning.decay "$sparsity_ratio"
    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"