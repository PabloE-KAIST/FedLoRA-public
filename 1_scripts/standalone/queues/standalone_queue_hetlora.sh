#!/usr/bin/env bash
# Standalone queue: HetLoRA — 3 rw × 3 decay = 9 experiments per task.
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

CONFIG="2_yamls/hetlora/hetlora-NO_quantized.yaml"
OUTDIR="exp_standalone/hetlora"
QUEUE_LOG="${FEDLORA_ROOT}/exp_standalone/standalone_queue_hetlora.log"
mkdir -p "$OUTDIR" "$(dirname "$QUEUE_LOG")"

REGULARIZER_WEIGHTS=(0.005 0.05 0.1)
SPARSITY_RATIOS=(0.50 0.65 0.80)

PASS=0; FAIL=0

for rw in "${REGULARIZER_WEIGHTS[@]}"; do
    for sr in "${SPARSITY_RATIOS[@]}"; do
        rw_tag=$(fmt_rw "$rw")
        sr_norm=$(fmt_float "$sr")
        TAG="${TASK}__rw_${rw}__sr_${sr}"
        PATTERN="${TASK}__strategy_custom__regularizer_${rw_tag}__decay_${sr_norm}__*"

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
            glue.adapter.hetlora.rank_max 200 \
            glue.adapter.hetlora.pruning.regularizer_weight "$rw" \
            glue.adapter.hetlora.pruning.decay "$sr"; then
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

log "[GPU ${GPU}] HetLoRA queue complete for ${TASK}: ${PASS} passed, ${FAIL} failed"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY hetlora__${TASK}: ${PASS} passed, ${FAIL} failed" >> "$QUEUE_LOG"
