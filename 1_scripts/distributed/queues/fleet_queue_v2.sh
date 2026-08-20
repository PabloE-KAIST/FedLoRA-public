#!/usr/bin/env bash
# Fleet: AdaSparse v2 — 3 rw × 3 gamma × 2 UL = 18 experiments per task.
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

CONFIG="2_yamls/adasparse_lora_v2/adasparse_lorav2_distributed.yaml"
QUEUE_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_queue_v2.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

REGULARIZER_WEIGHTS=(${RW_LIST:-0.005 0.05 0.1})
GAMMAS=(${GAMMA_LIST:-0.50 0.65 0.80})

UL_WINDOWS=(230 460)
DL_WINDOWS=(63)

EXP_OUTDIR="${FEDLORA_ROOT}/exp_distributed/adasparse_lorav2"

rw_to_sci() {
    case "$1" in
        0.005) echo "5e-3" ;; 0.05) echo "5e-2" ;; 0.1) echo "1e-1" ;;
        *) echo "$1" ;;
    esac
}

norm() { printf '%g' "$1"; }

experiment_completed() {
    local rw_sci gamma_n
    rw_sci=$(rw_to_sci "$1")
    gamma_n=$(norm "$2")
    for d in "${EXP_OUTDIR}/${TASK}__strategy_"*"__regularizer_${rw_sci}__gamma_${gamma_n}__ul_${3}__dl_${4}__"*; do
        [[ -d "$d" ]] || continue
        grep -q "Training is finished" "$d/exp_print.log" 2>/dev/null && return 0
    done
    return 1
}

PASS=0
FAIL=0
SKIP=0

for rw in "${REGULARIZER_WEIGHTS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
    for ul in "${UL_WINDOWS[@]}"; do
        for dl in "${DL_WINDOWS[@]}"; do
            TAG="${TASK}__rw_${rw}__g_${gamma}__ul_${ul}__dl_${dl}"

            if experiment_completed "$rw" "$gamma" "$ul" "$dl"; then
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
                glue.adapter.adasparse_lorav2.stage1.regularizer_weight "$rw" \
                glue.adapter.adasparse_lorav2.stage1.gamma "$gamma" \
                glue.adapter.adasparse_lorav2.stage2.uplink_budget_window_s "$ul" \
                glue.adapter.adasparse_lorav2.stage2.downlink_budget_window_s "$dl")

            if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${ARGS[@]}"; then
                log "PASS: ${TAG}"
                echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$QUEUE_LOG"
                PASS=$((PASS + 1))
            else
                log "FAIL: ${TAG}"
                echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$QUEUE_LOG"
                FAIL=$((FAIL + 1))
            fi

            log "Cooldown 30s..."
            sleep 30
        done
    done
    done
done

log "======== v2 queue complete (task=${TASK}): ${PASS} passed, ${FAIL} failed, ${SKIP} skipped ========"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY ${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped" >> "$QUEUE_LOG"
