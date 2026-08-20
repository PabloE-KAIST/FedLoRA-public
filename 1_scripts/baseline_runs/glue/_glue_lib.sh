#!/usr/bin/env bash
# Shared helpers for GLUE baseline scripts.
# Sourced by sibling baseline runners; not intended to be run directly.

# Map a GLUE task to the appropriate federate.client_num for IID partitioning.
#
# Rationale: with 12 clients and IID split, small-train tasks end up with too
# few per-client samples for meaningful LoRA updates per round. We drop
# client_num to 6 for the small tasks so each client retains ~400+ samples.
#
#   sst2  67,349  → 12 clients (~5,612 per client)
#   mnli  392,702 → 12 clients
#   qqp   363,846 → 12 clients
#   qnli  104,743 → 12 clients
#   cola    8,551 →  6 clients (~1,425 per client; closer to the small cluster)
#   stsb    5,749 →  6 clients (~958 per client)
#   mrpc    3,668 →  6 clients (~611 per client)
#   rte     2,490 →  6 clients (~415 per client)
#
# WNLI is intentionally disabled in the codebase (635 train samples, too small).
# AX is diagnostic-only (no train split); not added.
glue_clients_for_task() {
    case "$1" in
        rte|mrpc|stsb|cola)        echo 6 ;;
        sst2|mnli|qqp|qnli)        echo 12 ;;
        wnli)
            echo "[glue_lib] ERROR: WNLI is intentionally disabled (too few train samples)." >&2
            return 1
            ;;
        *)
            echo "[glue_lib] ERROR: unknown GLUE task '$1'. Supported: sst2 mnli qqp qnli cola stsb mrpc rte" >&2
            return 1
            ;;
    esac
}

# Map a GLUE task to its canonical eval metric key for early stopping / best
# model selection. Per Wang et al. (2018) / gluebenchmark.com:
#   CoLA → MCC, STS-B → Pearson, MRPC/QQP → F1, rest → Accuracy
glue_eval_key() {
    case "$1" in
        stsb)     echo "val_pearson" ;;
        cola)     echo "val_mcc" ;;
        mrpc|qqp) echo "val_f1" ;;
        *)        echo "val_acc" ;;
    esac
}

# Per-task total_round_num. Large-sample tasks (QQP, MNLI) need more rounds
# to accumulate enough effective epochs at 30 steps/round.
glue_total_rounds() {
    case "$1" in
        qqp|mnli) echo 30 ;;
        *)        echo 20 ;;
    esac
}

# Per-task local_update_steps to normalise effective epochs across tasks.
# Default 30 works for tasks with ≥1,000 samples/client. MRPC and RTE have
# tiny partitions (611 and 415 samples/client) causing 1.5–2.3 epochs per
# round → overfitting. Reducing to 15 brings them to 0.8–1.2 epochs/round.
glue_local_steps() {
    case "$1" in
        mrpc|rte) echo 15 ;;
        *)        echo 30 ;;
    esac
}

# STS-B is regression (1 output); all other GLUE tasks are classification (2+).
# CoLA/SST-2/RTE/MRPC/QNLI = 2, MNLI = 3, QQP = 2, STS-B = 1.
glue_out_channels() {
    case "$1" in
        stsb) echo 1 ;;
        mnli) echo 3 ;;
        *)    echo 2 ;;
    esac
}

# eval.metrics list (YACS list literal) that includes the best-key metric.
# STS-B needs 'pearson'; CoLA needs 'mcc'; MRPC/QQP use 'f1' + 'acc'; rest use 'acc' + 'f1'.
glue_eval_metrics() {
    case "$1" in
        stsb) echo "['pearson','loss']" ;;
        cola) echo "['mcc','acc','loss']" ;;
        *)    echo "['acc','f1','loss']" ;;
    esac
}

# Criterion type: classification vs regression.
glue_criterion() {
    case "$1" in
        stsb) echo "MSELoss" ;;
        *)    echo "CrossEntropyLoss" ;;
    esac
}

# Regularizer weight for pruning-based methods (HetLoRA, v2, v3).
# rw=0.1 prunes too aggressively on small-data tasks (<1k samples/client),
# causing 8-17% accuracy drops vs FedIT. rw=0.01 matches rw=0.1 on large
# tasks (SST-2 golden: identical v2/v3 accuracy at both values) while
# preserving capacity on small tasks.
# STS-B kept at 0.1 for campaign consistency (already mid-run at 0.1).
glue_regularizer_weight() {
    case "$1" in
        rte|mrpc|cola)            echo "0.01" ;;
        stsb|sst2|qnli|qqp|mnli) echo "0.1" ;;
        *)
            echo "[glue_lib] ERROR: unknown task '$1'" >&2
            return 1
            ;;
    esac
}
