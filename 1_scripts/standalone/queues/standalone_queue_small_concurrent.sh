#!/usr/bin/env bash
# Concurrent FedIT + FAH-QLoRA + HetLoRA queue — runs 2 experiments at a time
# on the same GPU. Total: 1 + 6 + 9 = 16 experiments in 8 pairs.
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

QUEUE_LOG="${FEDLORA_ROOT}/exp_standalone/standalone_queue_small_concurrent.log"
mkdir -p exp_standalone/fedit exp_standalone/fahqlora exp_standalone/hetlora

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
)

PASS=0; FAIL=0; SKIP=0

# Run a single experiment. Args: method tag config outdir method_args...
run_one() {
    local method=$1 tag=$2 config=$3 outdir=$4 pattern=$5
    shift 5

    if is_completed "$outdir" "$pattern"; then
        log "[GPU ${GPU}] SKIP (done): ${tag}"
        return 2
    fi

    log "[GPU ${GPU}] Starting: ${tag}"
    if CUDA_VISIBLE_DEVICES=$GPU python federatedscope/main.py --cfg "$config" \
        "${COMMON_ARGS[@]}" outdir "$outdir" "$@"; then
        log "[GPU ${GPU}] PASS: ${tag}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') PASS ${tag}" >> "$QUEUE_LOG"
        return 0
    else
        log "[GPU ${GPU}] FAIL: ${tag}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL ${tag}" >> "$QUEUE_LOG"
        return 1
    fi
}

# Build flat experiment list as function calls stored in arrays.
# Each entry: "method|tag|config|outdir|pattern|extra_args..."
EXPERIMENTS=()

# FedIT (1)
EXPERIMENTS+=("fedit|fedit_r64__${TASK}|2_yamls/fedit/fedit-NO_quantized.yaml|exp_standalone/fedit|${TASK}__strategy_homo__*|glue.adapter.max_rank|64|glue.adapter.hetero_strategy|homo")

# FAH-QLoRA (6)
for ir in 32 64; do
    for lam in 1 5 10; do
        EXPERIMENTS+=("fahqlora|${TASK}__initr_${ir}__lambda_${lam}|2_yamls/fahqlora/fah_qlora-NO_quantized.yaml|exp_standalone/fahqlora|${TASK}__strategy_custom__initr_${ir}__lambda_${lam}__*|glue.adapter.max_rank|200|glue.adapter.hetero_strategy|custom|glue.adapter.fah.r_max|200|glue.adapter.fah.init_rank|${ir}|glue.adapter.fah.lambda_inc|${lam}|glue.adapter.fah.lambda_dec|${lam}")
    done
done

# HetLoRA (9)
for rw in 0.005 0.05 0.1; do
    for sr in 0.50 0.65 0.80; do
        rw_tag=$(fmt_rw "$rw")
        sr_norm=$(fmt_float "$sr")
        EXPERIMENTS+=("hetlora|${TASK}__rw_${rw}__sr_${sr}|2_yamls/hetlora/hetlora-NO_quantized.yaml|exp_standalone/hetlora|${TASK}__strategy_custom__regularizer_${rw_tag}__decay_${sr_norm}__*|glue.adapter.max_rank|200|glue.adapter.hetero_strategy|custom|glue.adapter.hetlora.rank_max|200|glue.adapter.hetlora.pruning.regularizer_weight|${rw}|glue.adapter.hetlora.pruning.decay|${sr}")
    done
done

# Dispatch: parse experiment spec and run
dispatch() {
    local spec=$1
    IFS='|' read -r method tag config outdir pattern args_str <<< "$spec"
    # Split remaining args back into array
    local -a extra_args=()
    local rest="${spec#*|*|*|*|*|}"
    IFS='|' read -ra extra_args <<< "$rest"
    run_one "$method" "$tag" "$config" "$outdir" "$pattern" "${extra_args[@]}"
}

# Run experiments in pairs
i=0
while (( i < ${#EXPERIMENTS[@]} )); do
    if (( i + 1 < ${#EXPERIMENTS[@]} )); then
        dispatch "${EXPERIMENTS[$i]}" &
        PID1=$!
        dispatch "${EXPERIMENTS[$((i+1))]}" &
        PID2=$!

        R1=0; wait $PID1 || R1=$?
        R2=0; wait $PID2 || R2=$?

        for r in $R1 $R2; do
            case $r in
                0) PASS=$((PASS + 1)) ;;
                2) SKIP=$((SKIP + 1)) ;;
                *) FAIL=$((FAIL + 1)) ;;
            esac
        done
        i=$((i + 2))
    else
        dispatch "${EXPERIMENTS[$i]}"
        r=$?
        case $r in
            0) PASS=$((PASS + 1)) ;;
            2) SKIP=$((SKIP + 1)) ;;
            *) FAIL=$((FAIL + 1)) ;;
        esac
        i=$((i + 1))
    fi
done

log "[GPU ${GPU}] Small-methods concurrent complete for ${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped"
echo "$(date '+%Y-%m-%d %H:%M:%S') SUMMARY small__${TASK}: ${PASS} passed, ${FAIL} failed, ${SKIP} skipped" >> "$QUEUE_LOG"
