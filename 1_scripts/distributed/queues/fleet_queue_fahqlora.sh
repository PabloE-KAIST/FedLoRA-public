#!/usr/bin/env bash
# Fleet: FAH-QLoRA — init_rank={32,64} × lambda={1,5,10} (6 experiments per task).
#
# Accepts env vars:
#   TASK          GLUE task (default: sst2). All 8 tasks supported.
#   MANIFEST      Manifest path override (for sub-fleet groups)
#   PORT_OFFSET   Server port offset (for concurrent runs, default: 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"
source "${FEDLORA_ROOT}/1_scripts/baseline_runs/glue/_glue_lib.sh"

TASK="${TASK:-sst2}"
case "$TASK" in
    sst2|mnli|qqp|qnli|cola|stsb|mrpc|rte) ;;
    *) die "Unsupported TASK='${TASK}'. Supported: sst2 mnli qqp qnli cola stsb mrpc rte." ;;
esac

CLIENTS=$(glue_clients_for_task "$TASK")
TOTAL_ROUNDS=$(glue_total_rounds "$TASK")
LOCAL_STEPS=$(glue_local_steps "$TASK")
EVAL_KEY=$(glue_eval_key "$TASK")
OUT_CHANNELS=$(glue_out_channels "$TASK")

CONFIG="2_yamls/fahqlora/fah_qlora_distributed.yaml"
QUEUE_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_queue_fahqlora.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

INIT_RANKS=(32 64)
LAMBDAS=(1 5 10)

EXP_OUTDIR="${FEDLORA_ROOT}/exp_distributed/fahqlora"

experiment_completed() {
    for d in "${EXP_OUTDIR}/${TASK}__strategy_"*"__initr_${1}__lambda_${2}__"*; do
        [[ -d "$d" ]] || continue
        grep -q "Training is finished" "$d/exp_print.log" 2>/dev/null && return 0
    done
    return 1
}

PASS=0
FAIL=0
SKIP=0
FAILED_TAGS=()

for ir in "${INIT_RANKS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
        TAG="${TASK}__initr_${ir}__lambda_${lam}"

        if experiment_completed "$ir" "$lam"; then
            log "SKIP (completed): ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') SKIP ${TAG}" >> "$QUEUE_LOG"
            SKIP=$((SKIP + 1))
            continue
        fi

        log "======== Starting: ${TAG} ========"

        ARGS=(--config "$CONFIG")
        [[ -n "${MANIFEST:-}" ]] && ARGS+=(--manifest "$MANIFEST")
        [[ -n "${PORT_OFFSET:-}" ]] && ARGS+=(--port-offset "$PORT_OFFSET")
        ARGS+=(-- \
            device 0 \
            data.type "${TASK}@glue" \
            federate.client_num "$CLIENTS" \
            federate.sample_client_num "$CLIENTS" \
            federate.total_round_num "$TOTAL_ROUNDS" \
            train.local_update_steps "$LOCAL_STEPS" \
            eval.best_res_update_round_wise_key "$EVAL_KEY" \
            model.out_channels "$OUT_CHANNELS" \
            glue.adapter.fah.init_rank "$ir" \
            glue.adapter.fah.lambda_inc "$lam" \
            glue.adapter.fah.lambda_dec "$lam")

        if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${ARGS[@]}"; then
            log "PASS: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$QUEUE_LOG"
            PASS=$((PASS + 1))
        else
            log "FAIL: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$QUEUE_LOG"
            FAIL=$((FAIL + 1))
            FAILED_TAGS+=("${ir}|${lam}")
        fi

        log "Cooldown 30s..."
        sleep 30
    done
done

log "======== FAH-QLoRA queue complete (task=${TASK}): ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ========"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY ${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped" >> "$QUEUE_LOG"
