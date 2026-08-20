#!/bin/bash

cd ${FEDLORA_ROOT}

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_glue_lib.sh"
task="${TASK:-sst2}"
clients=$(glue_clients_for_task "$task") || exit 1

configs=(
    "2_yamls/fedit/fedit-NO_quantized.yaml"
)

device=1
debug=("True")
system_metrics_mode=("extended")
max_rank=200 #Test theoretical maximum budget if all clients are homogeneous, max_rank=200 for example

echo "=============================================="
echo "Running: config=$configs | task=$task | max_rank=$max_rank | client_num=$clients"
echo "Started at: $(date)"
echo "=============================================="

python federatedscope/main.py --cfg "$configs" \
    device "$device" \
    debug "$debug" \
    monitor.system_metrics_mode "$system_metrics_mode" \
    federate.client_num "$clients" \
    data.type "${task}@glue" \
    glue.adapter.max_rank "$max_rank"

echo ""
echo "Finished at: $(date) with $clients clients"
echo ""


echo "All experiments completed!"