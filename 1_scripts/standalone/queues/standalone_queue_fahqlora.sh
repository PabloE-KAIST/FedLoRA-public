#!/usr/bin/env bash
# Standalone queue: FAH-QLoRA — 2 init_rank × 3 lambda = 6 experiments per task.
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

CONFIG="2_yamls/fahqlora/fah_qlora-NO_quantized.yaml"
OUTDIR="exp_standalone/fahqlora"
QUEUE_LOG="${FEDLORA_ROOT}/exp_standalone/standalone_queue_fahqlora.log"
mkdir -p "$OUTDIR" "$(dirname "$QUEUE_LOG")"

INIT_RANKS=(32 64)
LAMBDAS=(1 5 10)

PASS=0; FAIL=0

for ir in "${INIT_RANKS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
        TAG="${TASK}__initr_${ir}__lambda_${lam}"
        PATTERN="${TASK}__strategy_custom__initr_${ir}__lambda_${lam}__*"

        if is_completed "$OUTDIR" "$PATTERN"; then
            log "[GPU ${GPU}] SKIP (done): ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') SKIP ${TAG}" >> "$QUEUE_LOG"
            continue
        fi

        log "[GPU ${GPU}] Starting: ${TAG}"
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
            glue.adapter.max_rank 200 \
            glue.adapter.hetero_strategy custom \
            glue.adapter.fah.r_max 200 \
            glue.adapter.fah.init_rank "$ir" \
            glue.adapter.fah.lambda_inc "$lam" \
            glue.adapter.fah.lambda_dec "$lam"; then
            log "[GPU ${GPU}] PASS: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$QUEUE_LOG"
            PASS=$((PASS + 1))
        else
            log "[GPU ${GPU}] FAIL: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$QUEUE_LOG"
            FAIL=$((FAIL + 1))
        fi
    done
done

log "[GPU ${GPU}] FAH-QLoRA queue complete for ${TASK}: ${PASS} passed, ${FAIL} failed"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY fahqlora__${TASK}: ${PASS} passed, ${FAIL} failed" >> "$QUEUE_LOG"
