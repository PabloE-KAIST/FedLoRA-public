# FAH-QLoRA Configuration Files

This directory contains YAML configuration files for **FAH-QLoRA (Federated Adaptive Heterogeneous QLoRA)** experiments. FAH-QLoRA implements a two-stage adaptive rank selection algorithm with support for heterogeneous base model quantization across federated clients.

## Overview

FAH-QLoRA is designed for resource-constrained federated learning environments where:
- Clients have heterogeneous computational capabilities (different quantization levels)
- Network bandwidth is limited and variable
- Both computation time and communication time significantly impact training

The framework dynamically adapts LoRA ranks to optimize the trade-off between model performance and training efficiency.

## Core Implementation Components

### 1. Two-Stage Rank Adaptation Algorithm

**Stage 1: Global Rank Update**
- Adapts the average LoRA rank across rounds by maximizing loss decrease rate
- Uses gradient sign approximation to determine rank increase/decrease
- Balances exploration vs. exploitation with `lambda_inc` and `lambda_dec` parameters

**Stage 2: Per-Device Rank Assignment**
- Optimizes individual client ranks to minimize round completion time
- Considers both computation time and communication time
- Subject to network bandwidth constraints

**Implementation:** `federatedscope/llm/fah_rank_scheduler.py`

```python
class FahRankScheduler:
    """
    Scheduler for FAH-QLoRA dynamic rank adaptation.
    
    Time Modeling (equations 12-14):
      - Computation time: t_cmp^n(r) = alpha_n + (r / r_max) * t_lora_n
      - Communication time: t_com^n(r) = L0 / b_dn_n + L(r) / b_up_n
      - Round time: T = max_n(t_cmp^n + t_com^n)
    """
```

### 2. HeteroLoRA Aggregator

**Purpose:** Handle LoRA adapters with varying ranks across clients

**Strategy:**
- Zero-padding: Smaller ranks padded to maximum rank
- Truncation: Larger ranks truncated to maximum rank
- Weighted averaging: Based on client sample sizes

**Implementation:** `federatedscope/core/aggregators/heterolora_aggregator.py`

```python
class HeteroLoRAAggregator(Aggregator):
    """
    Aggregator for heterogeneous LoRA adapters with varying ranks.
    
    Handles LoRA weights with different ranks by:
    1. Zero-padding smaller ranks to a maximum rank
    2. Truncating larger ranks to the maximum rank
    3. Weighted averaging based on sample sizes
    """
```

### 3. Heterogeneous Quantization Support

**Base Model Quantization:** Different clients can use different quantization levels

**Supported Levels:**
- 4-bit: NF4 QLoRA (lowest memory, bf16 compute)
- 8-bit: LLM.int8 QLoRA (moderate memory, fp16 compute)
- 16-bit: Half precision (higher memory, bf16 compute)
- 32-bit: Full precision (highest memory, fp32 compute)

**Implementation:** Configured via `base_quant` settings in YAML

**Key Utility Functions:** `federatedscope/llm/utils/heterolora_utils.py`
- `fah_resolve_client_compute_dtype()`: Determine compute dtype per client
- `fah_cast_trainable_params_for_quantization()`: Cast LoRA and trainable parameters
- `modify_lora_adapter_rank()`: Dynamically modify adapter ranks
- `distribute_aggregated_to_heterogeneous_clients()`: Distribute server weights to clients

## Configuration Files

### `fah_qlora-quantized.yaml`

**Purpose:** Standard FAH-QLoRA with 16-bit and 32-bit quantization

**Key Settings:**
```yaml
federate:
  method: heterolora        # Uses HeteroLoRA aggregation
  client_num: 12
  total_round_num: 20

glue.adapter:
  base_quant:
    enabled: True
    distribution:
      '16': 0.5            # 50% clients at 16-bit
      '32': 0.5            # 50% clients at 32-bit
    lora_dtype: fp32
  
  fah:
    enabled: True          # Enable adaptive rank selection
    init_rank: 16          # Starting rank
    r_min: 4              # Minimum allowed rank
    r_max: 100            # Maximum allowed rank
    lambda_dec: 1         # Rank decrease learning rate
    lambda_inc: 1         # Rank increase learning rate
    warmup_rounds: 1      # Profiling rounds before adaptation
    bandwidth_mode: "realistic"  # Time-based bandwidth sampling
```

**Use Case:** Standard experiments with moderate heterogeneity

---

### `fah_qlora-quantized_allQuantTypes.yaml`

**Purpose:** FAH-QLoRA with all four quantization levels (extreme heterogeneity)

**Key Settings:**
```yaml
glue.adapter.base_quant:
  distribution:
    '4': 0.25             # 25% clients at 4-bit
    '8': 0.25             # 25% clients at 8-bit
    '16': 0.25            # 25% clients at 16-bit
    '32': 0.25            # 25% clients at 32-bit
```

**Use Case:** Testing robustness under extreme device heterogeneity

---

### `fah_qlora-quantized_Homog.yaml`

**Purpose:** FAH-QLoRA with homogeneous bandwidth conditions

**Key Settings:**
```yaml
glue.adapter.fah:
  network_trace_path: "data/4Gnetwork_trace_selection/A_2017.12.18_04.44.30_trimmed_Easy.csv"
  bandwidth_mode: "homogeneous"  # All clients share same bandwidth
```

**Use Case:** Controlled experiments isolating the effect of computation heterogeneity from network heterogeneity

---

## Key Configuration Parameters

### Federate Settings

```yaml
federate:
  mode: standalone              # Simulation mode
  method: heterolora            # MUST be "heterolora" for FAH-QLoRA
  client_num: 12                # Number of clients
  total_round_num: 20           # Training rounds
  online_aggr: False            # Use batch aggregation
  ignore_weight: False          # Use weighted averaging
  share_local_model: False      # Keep models separate
```

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

**Important Notes:**
- Distribution values must sum to approximately 1.0
- With 12 clients and distribution `{'16': 0.5, '32': 0.5}`, you get 6 clients at 16-bit and 6 at 32-bit
- The `lora_dtype` should typically be `fp32` for stability

### FAH Rank Scheduler Parameters

```yaml
glue.adapter.fah:
  enabled: True                 # MUST be True for FAH-QLoRA
  
  # Rank bounds
  init_rank: 16                 # Initial rank (warm-start)
  r_min: 4                      # Minimum rank (lower bound)
  r_max: 100                    # Maximum rank (upper bound)
  
  # Adaptation parameters
  lambda_dec: 1                 # Rank decrease factor (higher = faster decrease)
  lambda_inc: 1                 # Rank increase factor (higher = faster increase)
  warmup_rounds: 1              # Rounds for profiling before adaptation starts
  
  # Validation parameters
  alpha_fraction: 0.3           # Fraction of training steps for alpha estimation
  validation_fraction: 0.2      # Fraction of data for validation
  validation_steps: 15          # Steps per validation during profiling
```

- **warmup_rounds**: Usually 1-3 rounds. More rounds improve profiling accuracy but delay adaptation.

### Network Configuration

```yaml
glue.adapter.fah:
  # Network bandwidth parameters (for synthetic/fallback)
  uplink_min_mbps: 5            # Minimum uplink bandwidth
  uplink_max_mbps: 20           # Maximum uplink bandwidth
  downlink_mbps: 50             # Downlink bandwidth
  
  # Network trace settings
  network_trace_path: "data/4Gnetwork_trace/"  # Path to trace files
  bandwidth_mode: "realistic"   # Sampling mode
  
  # Trace distribution (for realistic/dynamic modes)
  network_trace_distribution:
    static: 0.0                 # Stationary clients (home/office)
    pedestrian: 0.0             # Walking speed
    bus: 0.0                    # Vehicle speed
    static_extended: 50.0       # Extended stationary
    pedestrian_extended: 50.0   # Extended pedestrian
```

**Bandwidth Modes:**

1. **static**: Sample once at initialization, stays fixed
   - Use for: Baseline experiments with fixed network conditions
   
2. **dynamic**: Each client samples from their trace, updates per round
   - Use for: Testing robustness to per-client network variability
   
3. **homogeneous**: All clients share the same bandwidth that changes per round
   - Use for: Isolating computation heterogeneity effects
   
4. **realistic**: Time-based sampling (one sample per second)
   - Use for: Most realistic simulations
   - Each trace file assigned to only one client
   - Index advances by elapsed seconds per round

**Network Trace Distribution:**
- Values represent percentage of clients in each mobility category
- Only used for `realistic` and `dynamic` bandwidth modes
- Must sum to 100.0 for realistic mode
- Files are selected from subdirectories matching the mobility type

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
      r: 8                      # Initial static rank (overridden by FAH)
      lora_alpha: 16            # LoRA scaling factor
      lora_dropout: 0.05        # Dropout rate
  
  max_rank: 100                 # Must match fah.r_max
  hetero_strategy: homo         # Heterogeneity strategy
```

**Target Modules:**
- For DeBERTa: attention and MLP layers
- For other models: Adjust based on architecture
- More modules = higher memory but potentially better performance

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
- `sst2@glue`: Stanford Sentiment Treebank (binary classification)
- `qnli@glue`: Question Natural Language Inference
- Other GLUE tasks as needed

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

Both `glue.adapter` and `llm.adapter` sections must be present and match:

```yaml
glue.adapter:
  base_quant:
    enabled: True
    distribution: { '16': 0.5, '32': 0.5 }
  fah:
    enabled: True
    init_rank: 16
    # ... other settings

llm.adapter:
  base_quant:
    enabled: True
    distribution: { '16': 0.5, '32': 0.5 }  # MUST MATCH
  fah:
    enabled: True
    init_rank: 16                            # MUST MATCH
    # ... other settings (MUST MATCH)
```

**Why?** The framework uses both for compatibility and validation.

### 2. Federate Method

```yaml
federate:
  method: heterolora  # CRITICAL: Must be "heterolora", not "FedAvg"
```

Do NOT use `FedAvg` for FAH-QLoRA. The `heterolora` method activates the HeteroLoRA aggregator which handles variable-rank adapters.

### 3. Max Rank Consistency

```yaml
glue.adapter:
  max_rank: 100      # Must match fah.r_max
  fah:
    r_max: 100       # Must match max_rank
```

These must be identical for proper aggregation.

## Debugging and Monitoring

### Debug Mode

```yaml
debug:
  heterolora: True   # Enable HeteroLoRA-specific debug logging
```

Enables detailed logging of:
- Rank assignments per client per round
- Aggregation shape verification
- Time breakdown (computation vs communication)

### System Metrics

```yaml
monitor:
  system_metrics_mode: "fah_extended"
```

Tracks:
- Per-client ranks over time
- Per-client bandwidths
- Computation time breakdown
- Communication time breakdown
- Loss progression


## Running Experiments

### Basic Experiment

```bash
cd ${FEDLORA_ROOT}
python federatedscope/main.py --cfg 2_yamls/quantized_fahqlora/fah_qlora-quantized.yaml
```

### With Parameter Overrides

```bash
python federatedscope/main.py --cfg 2_yamls/quantized_fahqlora/fah_qlora-quantized.yaml \
    device 0 \
    federate.client_num 12 \
    glue.adapter.fah.init_rank 32 \
    glue.adapter.fah.bandwidth_mode "realistic"
```

### Using Scripts

```bash
bash 1_scripts/fah_qlora-quantized.sh
```

## Output Structure

After running, results are saved to the `outdir` specified in config:

```
exp/quantized_fahqlora/
├── config.yaml                 # Saved configuration
├── exp_print.log              # Training logs
├── eval_results.log           # Evaluation results
├── eval_results.raw           # Raw evaluation data
└── fah_bandwidth_history.txt  # Bandwidth history (if enabled)
```

## Key Differences across baselines

| Feature | FedIT | FAH-QLoRA | HetLoRA (complete) |
|---------|-------|-----------|---------------------|
| **Federate method** | `FedAvg` | `heterolora` | `hetlora` |
| **Rank selection** | Fixed (via `max_rank`) | Adaptive (FAH scheduler) | Client self-pruning (tail regularizer + prune rule) |
| **FAH enabled** | `False` | `True` | `False` |
| **Aggregation** | Sample-size weighted | HeteroLoRA (sample-size) | Sparsity-weighted or sample-size |
| **Network-aware** | No | Yes | No |
| **Complexity** | Low | High | Medium |


## Advanced Topics

Note: FAH-QLoRA typically uses `hetero_strategy: homo` and lets the scheduler determine ranks.