#!/usr/bin/env bash
# Standalone queue: FedIT r=64 — 1 experiment per task.
#
# Env vars:
#   TASK   GLUE task (required)
#   GPU    CUDA device index (required)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_standalone_lib.sh"
cd "$FEDLORA_ROOT"

: "${TASK:?TASK is required}"
: "${GPU:?GPU is required}"

CLIENTS=$(glue_clients_for_task "$TASK") || exit 1
TOTAL_ROUNDS=$(glue_total_rounds "$TASK")
LOCAL_STEPS=$(glue_local_steps "$TASK")
EVAL_KEY=$(glue_eval_key "$TASK")
OUT_CHANNELS=$(glue_out_channels "$TASK")

CONFIG="2_yamls/fedit/fedit-NO_quantized.yaml"
OUTDIR="exp_standalone/fedit"
QUEUE_LOG="${FEDLORA_ROOT}/exp_standalone/standalone_queue_fedit.log"
mkdir -p "$OUTDIR" "$(dirname "$QUEUE_LOG")"

PATTERN="${TASK}__strategy_homo__*"

if is_completed "$OUTDIR" "$PATTERN"; then
    log "[GPU ${GPU}] SKIP (done): FedIT r=64 ${TASK}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') SKIP fedit_r64__${TASK}" >> "$QUEUE_LOG"
else
    log "[GPU ${GPU}] Starting: FedIT r=64 ${TASK}"
    if CUDA_VISIBLE_DEVICES=$GPU python federatedscope/main.py --cfg "$CONFIG" \
        device 0 \
        outdir "$OUTDIR" \
        debug True \
        monitor.system_metrics_mode extended \
        federate.client_num "$CLIENTS" \
        federate.sample_client_num "$CLIENTS" \
        federate.total_round_num "$TOTAL_ROUNDS" \
        train.local_update_steps "$LOCAL_STEPS" \
        eval.best_res_update_round_wise_key "$EVAL_KEY" \
        model.out_channels "$OUT_CHANNELS" \
        data.type "${TASK}@glue" \
        glue.adapter.max_rank 64 \
        glue.adapter.hetero_strategy homo; then
        log "[GPU ${GPU}] PASS: FedIT r=64 ${TASK}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') PASS fedit_r64__${TASK}" >> "$QUEUE_LOG"
    else
        log "[GPU ${GPU}] FAIL: FedIT r=64 ${TASK}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL fedit_r64__${TASK}" >> "$QUEUE_LOG"
    fi
fi

log "[GPU ${GPU}] FedIT queue complete for ${TASK}"
