#!/bin/bash

cd ${FEDLORA_ROOT}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_glue_lib.sh"
task="${TASK:-sst2}"
clients=$(glue_clients_for_task "$task") || exit 1

configs=(
    "2_yamls/fahqlora/fah_qlora-NO_quantized.yaml"
)

device=2
debug=("True")
system_metrics_mode=("extended")
rank=64 #warmup round rank init
max_rank=200 #to match with the maximum rank as defined in paper types --> federatedscope/llm/utils/client_config_generator.py
rank_strategy="custom" #After patch, now FAH first round must be homo (without user interaction). Rank caps are now mapped to the rank strategy.
lambda=1 #1, 5, 10 for example
network_trace_path="data/4Gnetwork_trace/" #"data/4Gnetwork_trace/pedestrian_extended/A_2017.12.18_04.44.30.csv" #"data/4Gnetwork_trace/static_extended/A_2017.12.18_12.20.34.csv"
bandwidth_mode=(
    "dynamic"
)

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
        glue.adapter.fah.r_max "$max_rank" \
        glue.adapter.fah.init_rank "$rank" \
        glue.adapter.fah.lambda_inc "$lambda" \
        glue.adapter.fah.lambda_dec "$lambda" \
        glue.adapter.fah.network_trace_path "$network_trace_path" \
        glue.adapter.fah.bandwidth_mode "$bandwidth_mode"

    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"