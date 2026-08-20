#!/bin/bash

cd ${FEDLORA_ROOT}

configs=(
    "2_yamls/fahqlora/fah_qlora_vlm.yaml"
)

device=0
debug=("True")
system_metrics_mode=("extended")
clients=12
rank=64
max_rank=200
rank_strategy="custom"
lambda=1
network_trace_path="data/4Gnetwork_trace/"
bandwidth_mode=(
    "dynamic"
)

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
        vlm.adapter.fah.r_max "$max_rank" \
        vlm.adapter.fah.init_rank "$rank" \
        vlm.adapter.fah.lambda_inc "$lambda" \
        vlm.adapter.fah.lambda_dec "$lambda" \
        vlm.adapter.fah.network_trace_path "$network_trace_path" \
        vlm.adapter.fah.bandwidth_mode "$bandwidth_mode"

    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"
