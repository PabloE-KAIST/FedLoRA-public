# FedIT Configuration Files

This directory contains YAML configuration files for **FedIT (Federated Instruction Tuning)** experiments. FedIT is a baseline approach for federated learning with LoRA that uses standard FedAvg aggregation with fixed LoRA ranks, while still supporting heterogeneous base model quantization across clients.

## Overview

FedIT represents a simpler alternative to adaptive rank selection methods like FAH-QLoRA:
- **Fixed LoRA ranks:** All clients use the same predetermined rank throughout training
- **Standard FedAvg aggregation:** Simple weighted averaging, no special handling for heterogeneous ranks
- **Heterogeneous quantization:** Different clients can still use different quantization levels (4-bit, 8-bit, 16-bit, 32-bit)
- **No network-aware optimization:** Does not consider bandwidth constraints in rank selection

This baseline is useful for:
1. Comparing the benefits of adaptive rank selection (FAH-QLoRA) vs fixed ranks
2. Evaluating the impact of quantization heterogeneity alone
3. Establishing performance baselines with simpler algorithms

## Key Differences across baselines

| Feature | FedIT | FAH-QLoRA | HetLoRA (complete) |
|---------|-------|-----------|---------------------|
| **Federate method** | `FedAvg` | `heterolora` | `hetlora` |
| **Rank selection** | Fixed (via `max_rank`) | Adaptive (FAH scheduler) | Client self-pruning (tail regularizer + prune rule) |
| **FAH enabled** | `False` | `True` | `False` |
| **Aggregation** | Sample-size weighted | HeteroLoRA (sample-size) | Sparsity-weighted or sample-size |
| **Network-aware** | No | Yes | No |
| **Complexity** | Low | High | Medium |

## Core Implementation Characteristics

### 1. Fixed Rank LoRA

Unlike FAH-QLoRA which dynamically adapts ranks, FedIT uses a fixed rank specified by `max_rank`:

```yaml
glue.adapter:
  max_rank: 100        # Fixed rank for all clients, all rounds
  fah:
    enabled: False     # No adaptive rank selection
```

**Implications:**
- Simpler to configure and understand
- No profiling or warmup required
- May be suboptimal for heterogeneous resources
- Communication cost proportional to rank (cannot adapt to bandwidth)

### 2. Standard FedAvg Aggregation

FedIT uses the standard FedAvg algorithm:

```yaml
federate:
  method: FedAvg      # Standard federated averaging
```

**How it works:**
1. Server sends global model to selected clients
2. Clients train locally for `local_update_steps`
3. Clients send updated LoRA weights back to server
4. Server aggregates via weighted average: `w_global = Σ(n_k / N) * w_k`
5. Repeat for `total_round_num` rounds

**Note:** Even though quantization is heterogeneous, LoRA adapters themselves are in `fp32` (or `lora_dtype`), so aggregation is straightforward.

### 3. Heterogeneous Quantization Support

FedIT still supports heterogeneous base model quantization:

```yaml
glue.adapter.base_quant:
  enabled: True
  distribution:
    '16': 0.5          # 50% of clients use 16-bit quantized base
    '32': 0.5          # 50% of clients use 32-bit (full precision)
  lora_dtype: fp32     # LoRA adapters use fp32
```

**Why this works:**
- Base model is frozen (only used for forward pass)
- LoRA adapters are always in `lora_dtype` regardless of base quantization
- Gradients computed in appropriate compute dtype (fp16/bf16/fp32)
- Only LoRA weights communicated (not base model)

## Configuration Files

### `fedit-quantized.yaml`

**Purpose:** Standard FedIT with 16-bit and 32-bit quantization

**Key Settings:**
```yaml
federate:
  method: FedAvg           # Standard averaging (NOT heterolora)
  client_num: 12
  total_round_num: 20

glue.adapter:
  base_quant:
    enabled: True
    distribution:
      '16': 0.5            # 50% clients at 16-bit
      '32': 0.5            # 50% clients at 32-bit
    lora_dtype: fp32
  
  max_rank: 100            # Fixed rank (no adaptation)
  
  fah:
    enabled: False         # Disable adaptive rank selection
```

**Use Case:** Baseline for comparing against FAH-QLoRA

**Note:** The `r` parameter in `args` is can be left empty so that it's overridden by `max_rank`.

## Key Configuration Parameters

### Federate Settings

```yaml
federate:
  mode: standalone              # Simulation mode
  method: FedAvg                # MUST be "FedAvg" for FedIT
  client_num: 12                # Number of clients
  total_round_num: 20           # Training rounds
  online_aggr: False            # Use batch aggregation
  ignore_weight: False          # Use weighted averaging
  share_local_model: False      # Keep models separate
```

**Critical:** `method: FedAvg` is what distinguishes FedIT from FAH-QLoRA.

### Fixed Rank Configuration

```yaml
glue.adapter:
  max_rank: 100                 # Fixed LoRA rank for all clients
  
  args:
    - adapter_package: peft
      adapter_method: lora
      r:                        # Leave empty, overridden by max_rank
      lora_alpha: 16
      lora_dropout: 0.05
```

**Parameter Selection:**

- **max_rank:** Choose based on model size, task complexity, and memory constraints
  - Small models (BERT-base): 8-32
  - Medium models (DeBERTa-large): 32-64
  - Large models (RoBERTa-large): 64-128
  
- **lora_alpha:** Typically `2 * r` (e.g., if rank=8, use alpha=16)
- **lora_dropout:** 0.05-0.1 for regularization

### Base Quantization

```yaml
glue.adapter.base_quant:
  enabled: True                 # Enable heterogeneous quantization
  distribution:                 # Must sum to ~1.0
    '4': 0.0                   # Fraction of 4-bit clients
    '8': 0.0                   # Fraction of 8-bit clients
    '16': 0.5                  # Fraction of 16-bit clients
    '32': 0.5                  # Fraction of 32-bit clients
  lora_dtype: fp32             # LoRA adapter dtype
```

**Quantization Strategy:**

For FedIT experiments, common distributions include:

1. **Moderate heterogeneity** (default):
   ```yaml
   distribution: { '16': 0.5, '32': 0.5 }
   ```

2. **High heterogeneity**:
   ```yaml
   distribution: { '4': 0.25, '8': 0.25, '16': 0.25, '32': 0.25 }
   ```

3. **Homogeneous** (control):
   ```yaml
   distribution: { '16': 1.0 }
   ```

### FAH Settings (Disabled)

```yaml
glue.adapter.fah:
  enabled: False               # MUST be False for FedIT
```

**Important:** Even though FAH is disabled, the section may still be present in the config for compatibility. The `enabled: False` flag ensures adaptive rank selection is not used.

### LoRA Adapter Settings

```yaml
glue.adapter:
  use: True                     # Enable LoRA adapters
  mv_to_cpu: False              # Keep adapters on GPU
  
  args:
    - adapter_package: peft     # Use PEFT library
      adapter_method: lora      # LoRA method
      target_modules:           # Modules to apply LoRA
        - 'attention.self.in_proj'
        - 'attention.output.dense'
        - 'intermediate.dense'
        - 'output.dense'
      r:                        # Empty, use max_rank instead
      lora_alpha: 16            # LoRA scaling factor
      lora_dropout: 0.05        # Dropout rate
  
  max_rank: 100                 # Fixed rank
  hetero_strategy: homo         # Heterogeneity strategy
```

**Target Modules for Different Models:**

- **DeBERTa-large** (as in config):
  ```yaml
  target_modules: ['attention.self.in_proj', 'attention.output.dense',
                   'intermediate.dense', 'output.dense']
  ```

- **BERT/RoBERTa**:
  ```yaml
  target_modules: ['query', 'key', 'value', 'dense']
  ```

- **GPT-2/GPT-Neo**:
  ```yaml
  target_modules: ['c_attn', 'c_proj', 'c_fc']
  ```

## Data Configuration

```yaml
data:
  root: data/
  type: 'sst2@glue'             # Dataset (sst2@glue, qnli@glue, etc.)
  splits: [0.90, 0.10]          # Train/val split
  splitter: 'iid'               # Data distribution (iid or non-iid)

dataloader:
  batch_size: 32                # Batch size per client
```

**Supported Datasets:**
- `sst2@glue`: Sentiment analysis (binary)
- `qnli@glue`: Question NLI
- `mnli@glue`: Multi-genre NLI
- `qqp@glue`: Question pair similarity
- Other GLUE tasks

## Training Configuration

```yaml
train:
  local_update_steps: 30        # Steps per round per client
  batch_or_epoch: batch         # Update by batch count
  
  optimizer:
    type: AdamW                 # Optimizer
    lr: 0.0005                  # Learning rate
    weight_decay: 0.01          # Weight decay
  
  scheduler:
    type: CosineAnnealingLR     # LR scheduler
    T_max: 20                   # Total FL rounds
  
  is_enable_half: False         # Use fp32 training (better for DeBERTa)
```

**Training Hyperparameters:**

- **Learning rate:** 
  - Higher ranks may need lower LR (e.g., 1e-4 for rank 64+)
  - Lower ranks can use higher LR (e.g., 5e-4 for rank 8-32)
  
- **Local steps:**
  - More steps = better local convergence but slower rounds
  - Fewer steps = faster but may need more rounds
  - Typical range: 10-50 steps

## Evaluation Configuration

```yaml
eval:
  freq: 5                       # Evaluate every N rounds
  metrics: ['acc', 'f1', 'loss']
  split: ['val']
  report: ['weighted_avg']
  best_res_update_round_wise_key: val_acc
  count_flops: False
```

## Important Configuration Notes

### 1. Dual Configuration Requirement

Both `glue.adapter` and `llm.adapter` sections must match:

```yaml
glue.adapter:
  max_rank: 100
  base_quant:
    enabled: True
    distribution: { '16': 0.5, '32': 0.5 }

llm.adapter:
  max_rank: 100              # MUST MATCH
  base_quant:
    enabled: True            # MUST MATCH
    distribution: { '16': 0.5, '32': 0.5 }  # MUST MATCH
```

### 2. Federate Method

```yaml
federate:
  method: FedAvg  # CRITICAL: Must be "FedAvg", not "heterolora"
```

Using `heterolora` would enable the HeteroLoRA aggregator and potentially cause confusion about which baseline is being evaluated.

### 3. Empty r Parameter

In the YAML, you'll see:
```yaml
args:
  - r:               # Empty
```

This is intentional. The `r` parameter is overridden by `max_rank`, so leaving it empty avoids confusion.

## Debugging and Monitoring

### Debug Mode

```yaml
debug:
  heterolora: True   # Can still be True for general debugging
```

Even though FAH is disabled, this flag enables useful debug logging for heterogeneous quantization.

### System Metrics

```yaml
monitor:
  system_metrics_mode: "fah_extended"
```

Tracks:
- Per-client losses
- Training time per round
- Memory usage
- (Note: Rank tracking not applicable since ranks are fixed)

## Running Experiments

### Basic Experiment

```bash
cd ${FEDLORA_ROOT}
python federatedscope/main.py --cfg 2_yamls/quantized_fedit/fedit-quantized.yaml
```

### With Parameter Overrides

```bash
python federatedscope/main.py --cfg 2_yamls/quantized_fedit/fedit-quantized.yaml \
    device 0 \
    federate.client_num 12 \
    glue.adapter.max_rank 64 \
    llm.adapter.max_rank 64
```

**Note:** When overriding `max_rank`, override it in both `glue.adapter` and `llm.adapter`.

### Using Scripts

The provided script tests multiple rank values:

```bash
bash 1_scripts/fedit-quantized.sh
```

This runs FedIT with ranks [4, 64, 80, 100] to find the optimal fixed rank.

### Systematic Rank Search

To find the best fixed rank:

```bash
for rank in 8 16 32 64 128; do
    python federatedscope/main.py --cfg 2_yamls/quantized_fedit/fedit-quantized.yaml \
        glue.adapter.max_rank $rank \
        llm.adapter.max_rank $rank \
        outdir exp/fedit_rank${rank}
done
```

## Output Structure

Results are saved to the `outdir` specified in config:

```
exp/quantized_fedit/
├── config.yaml                 # Saved configuration
├── exp_print.log              # Training logs
├── eval_results.log           # Evaluation results
└── eval_results.raw           # Raw evaluation data
```

## Theoretical Background

### Why Fixed Ranks?

FedIT uses fixed ranks because:
1. **Simplicity:** Easier to implement and debug
2. **Predictability:** Communication and computation costs are constant
3. **Compatibility:** Works with standard FedAvg infrastructure
4. **Baseline:** Provides comparison point for adaptive methods

### Limitations

Fixed ranks have drawbacks:
1. **Inefficiency:** Cannot adapt to heterogeneous resources
2. **Suboptimality:** Single rank may not be optimal for all clients
3. **Communication:** Cannot reduce communication when bandwidth is limited
4. **Rigidity:** Cannot exploit available resources dynamically

## Quick Comparison Table

| Aspect | FedIT | FAH-QLoRA |
|--------|-------|-----------|
| Rank Selection | Fixed `max_rank` | Adaptive via FAH scheduler |
| Aggregation | FedAvg | HeteroLoRA aggregator |
| Network-Aware | ❌ No | ✅ Yes |
| Computation-Aware | ❌ No | ✅ Yes |
| Warmup Required | ❌ No | ✅ Yes (1-3 rounds) |
| Configuration Complexity | Low | High |
| Communication Efficiency | Fixed | Adaptive (potentially better) |
| Best For | Homogeneous settings, baselines | Heterogeneous settings, optimization |
