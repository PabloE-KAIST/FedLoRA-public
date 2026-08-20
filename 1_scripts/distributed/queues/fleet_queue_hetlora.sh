#!/usr/bin/env bash
# Fleet: HetLoRA — 3 rw × 3 decay = 9 experiments per task.
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

CONFIG="2_yamls/hetlora/hetlora_distributed.yaml"
QUEUE_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_queue_hetlora.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

REGULARIZER_WEIGHTS=(${RW_LIST:-0.005 0.05 0.1})
SPARSITY_RATIOS=(${DECAY_LIST:-0.50 0.65 0.80})

EXP_OUTDIR="${FEDLORA_ROOT}/exp_distributed/hetlora"

rw_to_sci() {
    case "$1" in
        0.005) echo "5e-3" ;; 0.05) echo "5e-2" ;; 0.1) echo "1e-1" ;;
        *) echo "$1" ;;
    esac
}

norm() { printf '%g' "$1"; }

experiment_completed() {
    local rw_sci decay_n
    rw_sci=$(rw_to_sci "$1")
    decay_n=$(norm "$2")
    for d in "${EXP_OUTDIR}/${TASK}__strategy_"*"__regularizer_${rw_sci}__decay_${decay_n}__"*; do
        [[ -d "$d" ]] || continue
        grep -q "Training is finished" "$d/exp_print.log" 2>/dev/null && return 0
    done
    return 1
}

PASS=0
FAIL=0
SKIP=0
FAILED_TAGS=()

for rw in "${REGULARIZER_WEIGHTS[@]}"; do
    for sr in "${SPARSITY_RATIOS[@]}"; do
        TAG="${TASK}__rw_${rw}__sr_${sr}"

        if experiment_completed "$rw" "$sr"; then
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
            glue.adapter.hetlora.pruning.regularizer_weight "$rw" \
            glue.adapter.hetlora.pruning.decay "$sr")

        if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${ARGS[@]}"; then
            log "PASS: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$QUEUE_LOG"
            PASS=$((PASS + 1))
        else
            log "FAIL: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$QUEUE_LOG"
            FAIL=$((FAIL + 1))
            FAILED_TAGS+=("${rw}|${sr}")
        fi

        log "Cooldown 30s..."
        sleep 30
    done
done

log "======== HetLoRA queue complete (task=${TASK}): ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ========"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY ${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped" >> "$QUEUE_LOG"
