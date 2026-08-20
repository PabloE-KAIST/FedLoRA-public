#!/bin/bash

cd ${FEDLORA_ROOT}

configs=(
    "2_yamls/adasparse_lora_v3/adasparse_lorav3_llm.yaml"
)

device=0
debug=("True")
system_metrics_mode=("extended")
clients=12
max_rank=64
rank_strategy=(
    custom
)
regularizer_weight=(0.01)
sparsity_ratio=(0.99)

uplink_window_s=(200)
downlink_window_s=(30000)

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
        llm.adapter.adasparse_lorav3.rank_max "$max_rank" \
        llm.adapter.adasparse_lorav3.stage1.regularizer_weight "$w" \
        llm.adapter.adasparse_lorav3.stage1.gamma "$sparsity_ratio" \
        llm.adapter.adasparse_lorav3.stage2.uplink_budget_window_s "$uplink_window_s" \
        llm.adapter.adasparse_lorav3.stage2.downlink_budget_window_s "$downlink_window_s" \
        llm.adapter.adasparse_lorav3.stage2.residual_enabled True \
        llm.adapter.adasparse_lorav3.stage1_global_competition False \
        llm.adapter.adasparse_lorav3.stage2_global_competition False
    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"
