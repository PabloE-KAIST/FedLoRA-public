#!/bin/bash

cd ${FEDLORA_ROOT}

configs=(
    "2_yamls/fedit/fedit_vlm.yaml"
)

device=1
debug=("True")
system_metrics_mode=("extended")
clients=12
max_rank=200

echo "=============================================="
echo "Running: config=$configs | max_rank=$max_rank | client_num=$clients"
echo "Started at: $(date)"
echo "=============================================="

python federatedscope/main.py --cfg "$configs" \
    device "$device" \
    debug "$debug" \
    monitor.system_metrics_mode "$system_metrics_mode" \
    federate.client_num "$clients" \
    vlm.adapter.max_rank "$max_rank"

echo ""
echo "Finished at: $(date) with $clients clients"
echo ""

echo "All experiments completed!"
