#!/bin/bash

cd ${FEDLORA_ROOT}

configs=(
    "2_yamls/adasparse_lora/adasparse_lora_llm.yaml"
)

device=1
debug=("True")
system_metrics_mode=("extended")
clients=12
max_rank=64
rank_strategy=(
    custom
)
regularizer_weight=(0.1)
sparsity_ratio=(0.99)

for w in "${regularizer_weight[@]}"; do
    echo "=============================================="
    echo "Running: config=$configs | debug=$debug | rank_strategy=$rank_strategy | max_rank=$max_rank | client_num=$clients"
    echo "Started at: $(date)"
    echo "=============================================="

    python federatedscope/main.py --cfg "$configs" \
        device "$device" \
        debug "$debug" \
        monitor.system_metrics_mode "$system_metrics_mode" \
        federate.client_num "$clients" \
        llm.adapter.max_rank "$max_rank" \
        llm.adapter.hetero_strategy "$rank_strategy" \
        llm.adapter.adasparse_lora.rank_max "$max_rank" \
        llm.adapter.adasparse_lora.pruning.regularizer_weight "$w" \
        llm.adapter.adasparse_lora.pruning.gamma "$sparsity_ratio"
    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"
