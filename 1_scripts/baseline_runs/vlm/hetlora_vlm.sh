#!/bin/bash

cd ${FEDLORA_ROOT}

configs=(
    "2_yamls/hetlora/hetlora_vlm.yaml"
)

device=0
debug=("True")
system_metrics_mode=("extended")
clients=12
max_rank=200
rank_strategy=(
    custom
)
regularizer_weight=(0.1)
sparsity_ratio=(0.65)

for strategy in "${rank_strategy[@]}"; do
    echo "=============================================="
    echo "Running: config=$configs | debug=$debug | rank_strategy=$strategy | max_rank=$max_rank | client_num=$clients"
    echo "Started at: $(date)"
    echo "=============================================="

    python federatedscope/main.py --cfg "$configs" \
        device "$device" \
        debug "$debug" \
        monitor.system_metrics_mode "$system_metrics_mode" \
        federate.client_num "$clients" \
        vlm.adapter.max_rank "$max_rank" \
        vlm.adapter.hetero_strategy "$strategy" \
        vlm.adapter.hetlora.rank_max "$max_rank" \
        vlm.adapter.hetlora.pruning.regularizer_weight "$regularizer_weight" \
        vlm.adapter.hetlora.pruning.decay "$sparsity_ratio"
    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"
