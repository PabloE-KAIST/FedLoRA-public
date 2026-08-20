#!/usr/bin/env bash
# Resume fleet queue after manual intervention.
# Order: HetLoRA retry (decay=0.50) → FAH-QLoRA remaining (initr_32,64) → v2 → v3
#
# Usage:
#   nohup bash 1_scripts/distributed/intervention/fleet_resume_after_intervention.sh \
#       2>&1 | tee -a exp2/fleet_master_queue.log &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_lib.sh"

QUEUE_LOG="${FEDLORA_ROOT}/exp2/fleet_master_queue.log"
mkdir -p "$(dirname "$QUEUE_LOG")"

log_status() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "$QUEUE_LOG"
}

log_status "=== Fleet RESUME queue started (after intervention) ==="

# ── Phase A: HetLoRA retry (decay=0.50) ──────────────────────────────
log_status "Phase A: HetLoRA retry (decay=0.50)"
HETLORA_CONFIG="2_yamls/hetlora/hetlora_distributed.yaml"
HETLORA_ARGS=(--config "$HETLORA_CONFIG")
HETLORA_ARGS+=(-- \
    glue.adapter.hetlora.pruning.regularizer_weight 0.01 \
    glue.adapter.hetlora.pruning.decay 0.50)
if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${HETLORA_ARGS[@]}"; then
    log_status "Phase A PASS: HetLoRA decay=0.50"
    echo "$(date '+%Y-%m-%d %H:%M:%S') RETRY_PASS rw_0.01__sr_0.50" >> "${FEDLORA_ROOT}/exp2/fleet_queue_hetlora.log"
else
    log_status "Phase A FAIL: HetLoRA decay=0.50"
    echo "$(date '+%Y-%m-%d %H:%M:%S') RETRY_FAIL rw_0.01__sr_0.50" >> "${FEDLORA_ROOT}/exp2/fleet_queue_hetlora.log"
fi
sleep 30

# ── Phase B: FAH-QLoRA remaining (initr_32, initr_64) ───────────────
log_status "Phase B: FAH-QLoRA remaining (initr_32, initr_64)"

FAH_CONFIG="2_yamls/fahqlora/fah_qlora_distributed.yaml"
FAH_QUEUE_LOG="${FEDLORA_ROOT}/exp2/fleet_queue_fahqlora.log"

INIT_RANKS=(32 64)
LAMBDAS=(1 5 10)

FAH_PASS=0
FAH_FAIL=0
FAH_FAILED_TAGS=()

for ir in "${INIT_RANKS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
        TAG="initr_${ir}__lambda_${lam}"
        log "======== Starting: ${TAG} ========"

        ARGS=(--config "$FAH_CONFIG")
        ARGS+=(-- \
            glue.adapter.fah.init_rank "$ir" \
            glue.adapter.fah.lambda_inc "$lam" \
            glue.adapter.fah.lambda_dec "$lam")

        if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${ARGS[@]}"; then
            log "PASS: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${TAG}" >> "$FAH_QUEUE_LOG"
            FAH_PASS=$((FAH_PASS + 1))
        else
            log "FAIL: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${TAG}" >> "$FAH_QUEUE_LOG"
            FAH_FAIL=$((FAH_FAIL + 1))
            FAH_FAILED_TAGS+=("${ir}|${lam}")
        fi

        log "Cooldown 30s..."
        sleep 30
    done
done

if [[ ${#FAH_FAILED_TAGS[@]} -gt 0 ]]; then
    log "======== Retrying ${#FAH_FAILED_TAGS[@]} failed FAH-QLoRA experiment(s) ========"
    for combo in "${FAH_FAILED_TAGS[@]}"; do
        IFS='|' read -r ir lam <<< "$combo"
        TAG="initr_${ir}__lambda_${lam}"
        log "======== RETRY: ${TAG} ========"

        ARGS=(--config "$FAH_CONFIG")
        ARGS+=(-- \
            glue.adapter.fah.init_rank "$ir" \
            glue.adapter.fah.lambda_inc "$lam" \
            glue.adapter.fah.lambda_dec "$lam")

        if bash "${SCRIPT_DIR}/../orchestrators/full_fl_run.sh" "${ARGS[@]}"; then
            log "RETRY PASS: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') RETRY_PASS ${TAG}" >> "$FAH_QUEUE_LOG"
            FAH_PASS=$((FAH_PASS + 1))
            FAH_FAIL=$((FAH_FAIL - 1))
        else
            log "RETRY FAIL: ${TAG}"
            echo "$(date '+%Y-%m-%d %H:%M:%S') RETRY_FAIL ${TAG}" >> "$FAH_QUEUE_LOG"
        fi

        log "Cooldown 30s..."
        sleep 30
    done
fi

log_status "Phase B COMPLETE: FAH-QLoRA remaining — ${FAH_PASS} passed, ${FAH_FAIL} failed"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY_RESUMED: ${FAH_PASS} passed, ${FAH_FAIL} failed" >> "$FAH_QUEUE_LOG"

# ── Phase C: AdaSparse v2 ───────────────────────────────────────────
log_status "Phase C: AdaSparse v2"
if bash "${SCRIPT_DIR}/../queues/fleet_queue_v2.sh" 2>&1 | tee -a "$QUEUE_LOG"; then
    log_status "Phase C COMPLETE: AdaSparse v2"
else
    log_status "Phase C FAILED: AdaSparse v2"
fi

# ── Phase D: AdaSparse v3 ───────────────────────────────────────────
log_status "Phase D: AdaSparse v3"
if bash "${SCRIPT_DIR}/../queues/fleet_queue_v3.sh" 2>&1 | tee -a "$QUEUE_LOG"; then
    log_status "Phase D COMPLETE: AdaSparse v3"
else
    log_status "Phase D FAILED: AdaSparse v3"
fi

log_status "=== Fleet RESUME queue finished ==="
