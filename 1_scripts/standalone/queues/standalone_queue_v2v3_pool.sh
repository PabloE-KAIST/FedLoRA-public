#!/usr/bin/env bash
# Combined v2+v3 pool queue with worker-based round-robin splitting.
# Generates 36 experiments (18 v2 + 18 v3, interleaved) and runs only
# those assigned to this worker.
#
# Env vars:
#   TASK            GLUE task (required)
#   GPU             CUDA device index (required)
#   WORKER_ID       0-based worker index (required)
#   TOTAL_WORKERS   Total number of pool workers (required)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_standalone_lib.sh"
cd "$FEDLORA_ROOT"

: "${TASK:?TASK is required}"
: "${GPU:?GPU is required}"
: "${WORKER_ID:?WORKER_ID is required}"
: "${TOTAL_WORKERS:?TOTAL_WORKERS is required}"

CLIENTS=$(glue_clients_for_task "$TASK") || exit 1
TOTAL_ROUNDS=$(glue_total_rounds "$TASK")
LOCAL_STEPS=$(glue_local_steps "$TASK")
EVAL_KEY=$(glue_eval_key "$TASK")
OUT_CHANNELS=$(glue_out_channels "$TASK")

QUEUE_LOG="${FEDLORA_ROOT}/exp_standalone/standalone_queue_v2v3_pool.log"
mkdir -p exp_standalone/adasparse_lorav2 exp_standalone/adasparse_lorav3

REGULARIZER_WEIGHTS=(0.005 0.05 0.1)
GAMMAS=(0.50 0.65 0.80)
UL_WINDOWS=(230 460)
DL=63

# Build interleaved experiment list: v2, v3, v2, v3, ...
EXPERIMENTS=()
for rw in "${REGULARIZER_WEIGHTS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for ul in "${UL_WINDOWS[@]}"; do
            EXPERIMENTS+=("v2|${rw}|${gamma}|${ul}|${DL}")
            EXPERIMENTS+=("v3|${rw}|${gamma}|${ul}|${DL}")
        done
    done
done

COMMON_ARGS=(
    device 0
    debug True
    monitor.system_metrics_mode extended
    federate.client_num "$CLIENTS"
    federate.sample_client_num "$CLIENTS"
    federate.total_round_num "$TOTAL_ROUNDS"
    train.local_update_steps "$LOCAL_STEPS"
    eval.best_res_update_round_wise_key "$EVAL_KEY"
    model.out_channels "$OUT_CHANNELS"
    data.type "${TASK}@glue"
    federate.communication.source trace
    federate.communication.network_trace_path "data/4Gnetwork_trace/"
    federate.communication.network_trace_distribution.pedestrian_extended 25.0
    federate.communication.network_trace_distribution.static_extended 75.0
    glue.adapter.max_rank 200
    glue.adapter.hetero_strategy custom
)

PASS=0; FAIL=0; SKIP=0

for i in "${!EXPERIMENTS[@]}"; do
    if (( i % TOTAL_WORKERS != WORKER_ID )); then
        continue
    fi

    IFS='|' read -r method rw gamma ul dl <<< "${EXPERIMENTS[$i]}"
    rw_tag=$(fmt_rw "$rw")
    gamma_norm=$(fmt_float "$gamma")
    TAG="${TASK}__${method}__rw_${rw}__g_${gamma}__ul_${ul}__dl_${dl}"

    if [[ "$method" == "v2" ]]; then
        CONFIG="2_yamls/adasparse_lora_v2/adasparse-lorav2-NO_quantized.yaml"
        OUTDIR="exp_standalone/adasparse_lorav2"
        NS="adasparse_lorav2"
        EXTRA_ARGS=(glue.adapter.adasparse_lorav2.stage2.residual_enabled True)
    else
        CONFIG="2_yamls/adasparse_lora_v3/adasparse-lorav3-NO_quantized.yaml"
        OUTDIR="exp_standalone/adasparse_lorav3"
        NS="adasparse_lorav3"
        EXTRA_ARGS=(
            glue.adapter.adasparse_lorav3.stage2.residual_enabled True
            glue.adapter.adasparse_lorav3.stage1_global_competition False
            glue.adapter.adasparse_lorav3.stage2_global_competition False
        )
    fi

    PATTERN="${TASK}__strategy_custom__regularizer_${rw_tag}__gamma_${gamma_norm}__ul_${ul}__dl_${dl}__*"
    if is_completed "$OUTDIR" "$PATTERN"; then
        log "[W${WORKER_ID} GPU ${GPU}] SKIP (done): ${TAG}"
        SKIP=$((SKIP + 1))
        continue
    fi

    log "[W${WORKER_ID} GPU ${GPU}] Starting: ${TAG}"
    if CUDA_VISIBLE_DEVICES=$GPU python federatedscope/main.py --cfg "$CONFIG" \
        "${COMMON_ARGS[@]}" \
        outdir "$OUTDIR" \
        glue.adapter.${NS}.rank_max 200 \
        glue.adapter.${NS}.stage1.regularizer_weight "$rw" \
        glue.adapter.${NS}.stage1.gamma "$gamma" \
        glue.adapter.${NS}.stage2.uplink_budget_window_s "$ul" \
        glue.adapter.${NS}.stage2.downlink_budget_window_s "$dl" \
        "${EXTRA_ARGS[@]}"; then
        log "[W${WORKER_ID} GPU ${GPU}] PASS: ${TAG}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$QUEUE_LOG"
        PASS=$((PASS + 1))
    else
        log "[W${WORKER_ID} GPU ${GPU}] FAIL: ${TAG}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$QUEUE_LOG"
        FAIL=$((FAIL + 1))
    fi
done

log "[W${WORKER_ID} GPU ${GPU}] v2v3 pool complete for ${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY W${WORKER_ID}__${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped" >> "$QUEUE_LOG"
