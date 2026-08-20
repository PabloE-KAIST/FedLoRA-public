# HetLoRA (Heterogeneous LoRA) Configuration Files

This directory contains YAML configuration files for **HetLoRA (Heterogeneous LoRA)** experiments. The implementation provides two baselines: a **naive** one (zero-padding and truncation only) and the **complete** HetLoRA baseline from the paper (rank self-pruning on clients + sparsity-weighted aggregation on the server).

## Overview

HetLoRA addresses heterogeneity in federated LoRA training in two ways:

1. **Naive HetLoRA** (`federate.method: heterolora`): Same pipeline as used by FAH-QLoRA for weight handling—zero-padding and truncation; aggregation is by **sample size** (standard FedAvg-style). No rank adaptation; ranks can be set via `hetero_ranks.config_local` or left homogeneous.

2. **Complete HetLoRA** (`federate.method: hetlora`): Implements the paper’s full baseline:
   - **Client-side:** Tail-rank regularizer during training; after training, compare the tail norm product of the updated LoRA with the initially received LoRA. If the updated one is **smaller**, prune. Pruning intensity is determined by a **decay** parameter (paper’s γ).
   - **Server-side:** Can aggregate by **sparsity-weighted** weights (Frobenius norm of each client’s effective LoRA update) or by sample size.

The framework reuses the same broadcast format (client-specific LoRA slices, `client_rank_config`) as the naive baseline and FAH-QLoRA. Complete HetLoRA adds the prune decision, rank update message (`hetlora_rank`), and sparsity-weighted aggregation.

## Core Implementation Components

### 1. Rank Self-Pruning (Complete HetLoRA Only)

**Prune rule:** After local training, compare tail norm product of the **updated** LoRA with the **initial** (received) LoRA. If the updated one is smaller → prune. The condition is strictly `score_after < score_before`.

**Decay parameter (paper’s γ):**
- Defines the **tail fraction** (which ranks count as “tail” for the norm product).
- Defines **pruning intensity**: when pruning, `new_rank = max(rank_min, floor(current_rank * decay))`.

**Flow:**
1. After loading server weights: record `tail_score_before`.
2. During training: add tail regularizer to the loss (optional, configurable).
3. After training: compute `tail_score_after`. If `score_after < score_before`, prune to `new_rank`, truncate LoRA tensors, and send `hetlora_rank` to the server.
4. Server updates `hetero_ranks.config_local` from client-reported ranks and uses it in the next broadcast.

**Implementation:** Client logic in `federatedscope/core/workers/client.py` (`_hetlora_record_tail_score_before`, `_hetlora_prune_and_send_rank`); tail score/penalty and truncation in `federatedscope/llm/utils/heterolora_utils.py`.

### 2. Tail-Rank Regularizer and Score

**Tail penalty (training):** For each LoRA pair (A, B), tail = rows/columns beyond `floor(r * decay)`. Penalty term: sum over pairs of `||B[:, tail_start:]||_F * ||A[tail_start:, :]||_F`. This is added to the loss with weight `pruning.regularizer_weight`.

**Tail score (prune decision):** Same scalar (sum of tail norm products) used to decide whether to prune. Recorded once after loading weights, then again after training.

**Implementation:** `tail_penalty()`, `tail_score()`, `iter_lora_pairs()` in `federatedscope/llm/utils/heterolora_utils.py`. Trainers (GLUE/LLM) add the regularizer in `_hook_on_batch_forward_regularizer` when HetLoRA complete is enabled.

### 3. Sparsity-Weighted Aggregation (can be exchanged by sample size)

When `aggregation.mode == 'sparsity_weighted'`, the server weights each client by the Frobenius norm of its effective LoRA update:

- For each client \(k\): \(s_k = \|B_k A_k\|_F\) (computed efficiently via trace trick).
- Weights: \(p_k = s_k / (\sum_j s_j + \epsilon)\).
- Aggregation: same zero-padding/truncation to `max_rank` as in HeteroLoRA, but using \(p_k\) instead of sample-size weights.

**Implementation:** `federatedscope/core/aggregators/hetlora_aggregator.py` (`HetLoRAAggregator`).

### 4. Server-Side Rank Updates

Clients send `msg_type='hetlora_rank'` with `content={'rank': r_new}`. The server stores per-client ranks and updates `hetero_ranks.config_local` via the same path used for FAH-QLoRA (`_update_fah_hetero_config`), so the next broadcast sends client-specific LoRA slices at the updated ranks.

**Implementation:** `callback_funcs_for_hetlora_rank` in `federatedscope/core/workers/server.py`.

## Comparison with Other Baselines

| Feature | FedIT | FAH-QLoRA | HetLoRA (naive) | HetLoRA (complete) |
|---------|-------|-----------|------------------|---------------------|
| **Federate method** | `FedAvg` | `heterolora` | `heterolora` | `hetlora` |
| **Rank selection** | Fixed | Dynamic (FAH) | Static/fixed | Client self-pruning |
| **Aggregation** | Sample-size | Sample-size (HeteroLoRA) | Sample-size (HeteroLoRA) | Sparsity-weighted or sample-size |
| **Network-aware** | No | Yes | No | No |

## Configuration Files

### Using the Complete HetLoRA Baseline

**Federate method:** `hetlora` (not `heterolora`). Enable the HetLoRA block under `glue.adapter` (or `llm.adapter`):

```yaml
federate:
  method: hetlora
  client_num: 12
  total_round_num: 20

glue.adapter:
  use: True
  max_rank: 100
  hetero_ranks: {}   # Filled by server from client hetlora_rank messages

  hetlora:
    enabled: True
    rank_min: 4
    rank_max: 100
    init_rank: 64

    pruning:
      enabled: True
      decay: 0.99              # Paper's γ: tail fraction & pruning intensity (new_rank = floor(r*decay))
      regularizer_weight: 0.01

    aggregation:
      mode: sparsity_weighted  # or sample_size
      epsilon: 1e-8
```

**Important:** For complete HetLoRA, do **not** set `glue.adapter.fah.enabled: True` (FAH expects the  'heterolora' aggregator and will override any aggregator to it).

### Using the Naive HetLoRA Baseline

Use `federate.method: heterolora` and configure `hetero_ranks.config_local`. Do **not** set `glue.adapter.hetlora.enabled: True` if you only want zero-padding + truncation with sample-size aggregation.

## Key Configuration Parameters

### HetLoRA Complete (`method: hetlora`)

```yaml
glue.adapter.hetlora:
  enabled: True

  # Rank bounds
  rank_min: 4
  rank_max: 100
  init_rank: 64

  # Pruning (client-side)
  pruning:
    enabled: True
    decay: 0.99              # Single parameter (paper γ): tail fraction and pruning intensity
    regularizer_weight: 0.01 # Weight of tail regularizer in loss

  # Aggregation (server-side)
  aggregation:
    mode: sparsity_weighted  # 'sparsity_weighted' | 'sample_size'
    epsilon: 1e-8
```

- **decay:** Same as paper’s γ. Tail is defined as ranks beyond `floor(r * decay)`; when pruning, `new_rank = max(rank_min, floor(r * decay))`. Typical range ~0.9–0.99.
- **regularizer_weight:** Scaling of the tail penalty in the training loss.
- **aggregation.mode:** `sparsity_weighted` uses Frobenius norms of effective LoRA updates; `sample_size` matches the naive/FAH-style sample-size weighting.

### Dual Configuration (GLUE and LLM)

As with FAH-QLoRA and FedIT, both `glue.adapter` and `llm.adapter` should be kept in sync when using GLUE tasks (e.g. same `hetlora` and `max_rank` settings).

## Running Experiments

### Complete HetLoRA

```bash
python federatedscope/main.py --cfg 2_yamls/quantized_hetlora/hetlora-quantized.yaml
```

Ensure the chosen YAML has `federate.method: hetlora` and `glue.adapter.hetlora.enabled: True` for the full baseline.

### Overrides

```bash
python federatedscope/main.py --cfg 2_yamls/quantized_hetlora/hetlora-quantized.yaml \
    device 0 \
    federate.client_num 12 \
    glue.adapter.hetlora.pruning.decay 0.95 \
    glue.adapter.hetlora.aggregation.mode sparsity_weighted
```