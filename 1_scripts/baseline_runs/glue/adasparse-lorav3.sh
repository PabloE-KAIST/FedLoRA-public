#!/bin/bash

cd ${FEDLORA_ROOT}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_glue_lib.sh"
task="${TASK:-sst2}"
clients=$(glue_clients_for_task "$task") || exit 1

configs=(
    "2_yamls/adasparse_lora_v3/adasparse-lorav3-NO_quantized.yaml"
)

device=0
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
regularizer_weight=(0.01) # 0.0003 0.001 0.003 0.01 0.03 0.1)
sparsity_ratio=(0.99)

bandwidth_mode=(
    "dynamic"
)
uplink_window_s=(200)      # 2000, 500, 100, 50
downlink_window_s=(30000)    # 30000 1000 600 250 | 200, 100, 50, 20
network_trace_path="" #"data/4Gnetwork_trace/"       # "data/4Gnetwork_trace_selection/ammend_ADAS_sampling.csv", "data/4Gnetwork_trace/static_extended/A_2017.12.18_12.20.34.csv"
pedestrian_extended=50.0
static_extended=50.0

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
        glue.adapter.adasparse_lorav3.rank_max "$max_rank" \
        glue.adapter.adasparse_lorav3.stage1.regularizer_weight "$w" \
        glue.adapter.adasparse_lorav3.stage1.gamma "$sparsity_ratio" \
        glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s "$uplink_window_s" \
        glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s "$downlink_window_s" \
        glue.adapter.adasparse_lorav3.stage2.residual_enabled True \
        glue.adapter.adasparse_lorav3.stage1_global_competition False \
        glue.adapter.adasparse_lorav3.stage2_global_competition False
    echo ""
    echo "Finished at: $(date) with $clients clients"
    echo ""
done

echo "All experiments completed!"