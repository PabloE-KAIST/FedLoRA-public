#!/usr/bin/env bash
# Standalone queue: AdaSparse-LoRA v3 — 3 rw × 3 gamma × 2 UL × 1 DL = 18 experiments per task.
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

CONFIG="2_yamls/adasparse_lora_v3/adasparse-lorav3-NO_quantized.yaml"
OUTDIR="exp_standalone/adasparse_lorav3"
QUEUE_LOG="${FEDLORA_ROOT}/exp_standalone/standalone_queue_v3.log"
mkdir -p "$OUTDIR" "$(dirname "$QUEUE_LOG")"

REGULARIZER_WEIGHTS=(0.005 0.05 0.1)
GAMMAS=(0.50 0.65 0.80)
UL_WINDOWS=(230 460)
DL_WINDOWS=(63)

PASS=0; FAIL=0

for rw in "${REGULARIZER_WEIGHTS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for ul in "${UL_WINDOWS[@]}"; do
            for dl in "${DL_WINDOWS[@]}"; do
                rw_tag=$(fmt_rw "$rw")
                gamma_norm=$(fmt_float "$gamma")
                TAG="${TASK}__rw_${rw}__g_${gamma}__ul_${ul}__dl_${dl}"
                PATTERN="${TASK}__strategy_custom__regularizer_${rw_tag}__gamma_${gamma_norm}__ul_${ul}__dl_${dl}__*"

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
                    federate.communication.source trace \
                    federate.communication.network_trace_path "data/4Gnetwork_trace/" \
                    federate.communication.network_trace_distribution.pedestrian_extended 25.0 \
                    federate.communication.network_trace_distribution.static_extended 75.0 \
                    glue.adapter.max_rank 200 \
                    glue.adapter.hetero_strategy custom \
                    glue.adapter.adasparse_lorav3.rank_max 200 \
                    glue.adapter.adasparse_lorav3.stage1.regularizer_weight "$rw" \
                    glue.adapter.adasparse_lorav3.stage1.gamma "$gamma" \
                    glue.adapter.adasparse_lorav3.stage2.uplink_budget_window_s "$ul" \
                    glue.adapter.adasparse_lorav3.stage2.downlink_budget_window_s "$dl" \
                    glue.adapter.adasparse_lorav3.stage2.residual_enabled True \
                    glue.adapter.adasparse_lorav3.stage1_global_competition False \
                    glue.adapter.adasparse_lorav3.stage2_global_competition False; then
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
    done
done

log "[GPU ${GPU}] v3 queue complete for ${TASK}: ${PASS} passed, ${FAIL} failed"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY v3__${TASK}: ${PASS} passed, ${FAIL} failed" >> "$QUEUE_LOG"
