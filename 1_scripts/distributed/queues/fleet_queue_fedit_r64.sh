#!/usr/bin/env bash
# Fleet: FedIT rank-64 — homogeneous baseline at the maximum rank supported
# by the least capable device in the fleet (Jetson Orin Nano = rank 64).
#
# Accepts env vars:
#   TASK          GLUE task (default: sst2). All 8 tasks supported.
#   MANIFEST      Manifest path override (for sub-fleet groups)
#   PORT_OFFSET   Server port offset (for concurrent runs, default: 0)
#
# Usage:
#   TASK=sst2 bash 1_scripts/distributed/queues/fleet_queue_fedit_r64.sh
#   TASK=cola MANIFEST=distributed/configs/client_manifest_group_a.json \
#       PORT_OFFSET=0 bash 1_scripts/distributed/queues/fleet_queue_fedit_r64.sh
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

CONFIG="2_yamls/fedit/fedit_distributed.yaml"
QUEUE_LOG="${FEDLORA_ROOT}/exp_distributed/fleet_queue_fedit_r64.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

TAG="fedit_r64_${TASK}"
log "======== Starting: ${TAG} ========"

ARGS=(--config "$CONFIG")
[[ -n "${MANIFEST:-}" ]] && ARGS+=(--manifest "$MANIFEST")
[[ -n "${PORT_OFFSET:-}" ]] && ARGS+=(--port-offset "$PORT_OFFSET")
ARGS+=(-- \
    glue.adapter.max_rank 64 \
    data.type "${TASK}@glue" \
    federate.client_num "$CLIENTS" \
    federate.sample_client_num "$CLIENTS" \
    federate.total_round_num "$TOTAL_ROUNDS" \
    train.local_update_steps "$LOCAL_STEPS" \
    eval.best_res_update_round_wise_key "$EVAL_KEY" \
    model.out_channels "$OUT_CHANNELS" \
    outdir "exp_distributed/fedit")

if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${ARGS[@]}"; then
    log "PASS: ${TAG}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$QUEUE_LOG"
else
    log "FAIL: ${TAG}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$QUEUE_LOG"
fi

log "======== FedIT r64 queue complete (task=${TASK}) ========"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY: ${TAG} done" >> "$QUEUE_LOG"
