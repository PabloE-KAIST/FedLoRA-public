# FedLoRA: Federated Fine-Tuning of Language Models with Heterogeneous LoRA

Federated learning with Low-Rank Adaptation (LoRA) for language models, supporting
heterogeneous clients — different LoRA ranks, quantization levels, and adaptive
rank selection — in both **simulation** and on a **real edge-device fleet**.

Built as a fork of [FederatedScope](https://github.com/alibaba/FederatedScope)
(Apache-2.0). See [NOTICE](NOTICE) for attribution.

## Quick start

```bash
conda env create -f requirements/environment.yml
conda activate fedlora
pip install -e .

# one simulated federated run: FedIT on MRPC, 6 clients, 20 rounds
python federatedscope/main.py --cfg 2_yamls/fedit/fedit-NO_quantized.yaml \
    data.type mrpc@glue federate.client_num 6 federate.total_round_num 20 \
    outdir exp/fedit_mrpc device 0
```

Datasets and models are fetched on demand — see [docs/reproduction.md](docs/reproduction.md).

## Methods

| Method | `federate.method` | Idea |
| --- | --- | --- |
| **FedIT** | `FedAvg` + homogeneous rank | FedAvg over fixed-rank LoRA adapters |
| **HetLoRA** | `hetlora` | heterogeneous per-client LoRA ranks with zero-padded aggregation |
| **FAH-QLoRA** | `fah_qlora` | adaptive heterogeneous rank + heterogeneous quantization |
| **AdaSparse-LoRA** | `adasparse_lorav1/v2/v3` | adaptive rank-1 component selection; v2 adds personalized communication budgets, v3 treats components per-layer |

## Two runtime stacks

- **`federatedscope/`** — simulated federated learning; all clients in one process.
  This is the entry point for reproducing baselines.
- **`distributed/`** — real multi-device deployment: protobuf control plane, a
  device-side agent, and containerized workers on NVIDIA Jetson hardware.

The split of responsibility is deliberate: FedLoRA owns all FL semantics (round
logic, aggregation, client selection); the device agent owns only device
lifecycle and payload relay; workers stay thin.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/reproduction.md](docs/reproduction.md) | environment, models, datasets, env vars, how to run both stacks |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | runtime architecture and file/function map |
| [docs/federation_bug.md](docs/federation_bug.md) | root-cause writeup of a silent federation failure in heterogeneous LoRA, and the guards that now prevent it |

### A correctness note worth reading

Heterogeneous LoRA methods transmit rank-annotated adapter keys. If those keys do
not map exactly onto the client model, `load_state_dict(..., strict=False)` drops
them **silently** — clients then train in isolation while appearing to federate
normally. This repository canonicalizes those keys with strict consumption and
ships two opt-in assertions:

```bash
federate.assert_download_consumed True        # raise if any key fails to map
federate.assert_download_tensor_equality True # raise if the client's LoRA != the global adapter
```

Enable both when adding a new heterogeneous method. Details in
[docs/federation_bug.md](docs/federation_bug.md).

## Tests

```bash
python -m pytest federatedscope/contrib/common/test_federation_download.py \
                 federatedscope/contrib/common/test_head_federation.py \
                 federatedscope/contrib/common/test_hetlora_restriction.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

This repository implements federated learning with Low-Rank Adaptation (LoRA) for large language models, supporting heterogeneous client configurations including different quantization levels and adaptive rank selection.

## Implemented Baselines
FAH-QLoRA, HetLoRA (naive heterolora or complete HetLoRA), FedIT, ...

### 1. FAH-QLoRA (Federated Adaptive Heterogeneous QLoRA)

**Method:** Dynamic rank adaptation with heterogeneous quantization

**Key Features:**
- Two-stage adaptive rank selection algorithm
- Heterogeneous base model quantization across clients
- Network-aware optimization considering computation and communication costs
- Supports 4-bit, 8-bit, 16-bit, and 32-bit quantization levels

**Config Location:** `2_yamls/fahqlora/`

**Key Config Settings:**
```yaml
federate:
  method: heterolora  # Uses HeteroLoRA aggregation

glue.adapter:
  base_quant:
    enabled: True
    distribution:
      '16': 0.5
      '32': 0.5
  fah:
    enabled: True
    init_rank: 16
    r_min: 4
    r_max: 100
```

**Run Script:** `1_scripts/baseline_runs/glue/fah_qlora.sh`

### 2. HetLoRA (Heterogeneous LoRA)

**Method:** Rank self-pruning on clients + sparsity-weighted aggregation on server

**Key Features:**
- **Naive baseline** (`federate.method: heterolora`): Zero-padding and truncation only; aggregation by sample size
- **Complete baseline** (`federate.method: hetlora`): Client-side tail-rank regularizer and prune decision when tail norm product after training is smaller than before; server-side aggregation weights by Frobenius norm of effective LoRA update.

**Config Location:** `2_yamls/hetlora/`

**Key Config Settings (complete HetLoRA):**
```yaml
federate:
  method: hetlora   # Full HetLoRA (use heterolora for naive baseline only)

glue.adapter:
  max_rank: 100
  hetlora:
    enabled: True
    rank_min: 4
    rank_max: 100
    init_rank: 64
    pruning:
      enabled: True
      decay: 0.99              # Tail fraction & pruning intensity (paper gamma)
      regularizer_weight: 0.01
    aggregation:
      mode: sparsity_weighted  # or sample_size
```

**Run Script:** `1_scripts/baseline_runs/glue/hetlora.sh`

### 3. FedIT (Federated Instruction Tuning)

**Method:** Standard FedAvg with fixed LoRA ranks

**Key Features:**
- Fixed LoRA rank across all rounds
- Heterogeneous base model quantization
- Standard federated averaging aggregation

**Config Location:** `2_yamls/fedit/`

**Key Config Settings:**
```yaml
federate:
  method: FedAvg  # Standard federated averaging

glue.adapter:
  base_quant:
    enabled: True
    distribution:
      '16': 0.5
      '32': 0.5
  max_rank: 100  # Fixed rank, no adaptation
  fah:
    enabled: False  # No adaptive rank selection
```

**Run Script:** `1_scripts/baseline_runs/glue/fedit.sh`

## Modifying Configuration Files

Configuration files are in YAML format and support hierarchical parameter settings. Here are the key parameters to modify:

### General Settings

```yaml
seed: 50                  # Random seed for reproducibility
use_gpu: True             # Enable GPU acceleration
device: 0                 # GPU device ID
outdir: exp/your_exp/     # Output directory for results

federate:
  mode: standalone        # Simulation mode
  method: heterolora      # Aggregation method (heterolora or FedAvg)
  client_num: 12          # Number of federated clients
  total_round_num: 20     # Total training rounds
```

### Data Configuration

```yaml
data:
  root: data/
  type: 'sst2@glue'       # Dataset (sst2@glue, qnli@glue, etc.)
  splits: [0.90, 0.10]    # Train/validation split
  splitter: 'iid'         # Data distribution (iid or non-iid)
```

### Model Configuration

```yaml
model:
  type: '/path/to/model@huggingface_llm'
  task: 'SequenceClassification'
  out_channels: 2         # Number of output classes
```

### LoRA Adapter Configuration

```yaml
glue.adapter:
  use: True
  args:
    - adapter_package: peft
      adapter_method: lora
      target_modules: ['attention.self.in_proj', 'attention.output.dense',
                       'intermediate.dense', 'output.dense']
      r: 8                # Initial LoRA rank (for fixed rank methods)
      lora_alpha: 16
      lora_dropout: 0.05
```

### Heterogeneous Quantization

```yaml
glue.adapter.base_quant:
  enabled: True           # Enable heterogeneous quantization
  distribution:           # Distribution of quantization levels
    '4': 0.25            # 25% of clients use 4-bit
    '8': 0.25            # 25% of clients use 8-bit
    '16': 0.25           # 25% of clients use 16-bit
    '32': 0.25           # 25% of clients use 32-bit
  lora_dtype: fp32       # LoRA parameter dtype
```

### FAH-QLoRA Specific Settings

```yaml
glue.adapter.fah:
  enabled: True           # Enable adaptive rank selection
  init_rank: 16           # Initial rank
  r_min: 4               # Minimum rank
  r_max: 100             # Maximum rank
  lambda_dec: 1          # Rank decrease factor
  lambda_inc: 1          # Rank increase factor
  warmup_rounds: 1       # Warmup rounds for profiling
  alpha_fraction: 0.3    # Fraction for alpha estimation
  validation_fraction: 0.2
  validation_steps: 15
```

### Network Configuration

```yaml
glue.adapter.fah:
  network_trace_path: "data/4Gnetwork_trace/"
  bandwidth_mode: "realistic"  # static/dynamic/homogeneous/realistic
  network_trace_distribution:
    static: 0.0                # Stationary clients
    pedestrian: 0.0            # Walking speed
    bus: 0.0                   # Vehicle speed
    static_extended: 50.0      # Extended stationary
    pedestrian_extended: 50.0  # Extended pedestrian
```

### Training Configuration

```yaml
train:
  local_update_steps: 30
  batch_or_epoch: batch
  optimizer:
    type: AdamW
    lr: 0.0005
    weight_decay: 0.01
  scheduler:
    type: CosineAnnealingLR
    T_max: 20
```

## Command-Line Parameter Overrides

You can override any configuration parameter from the command line:

```bash
python federatedscope/main.py --cfg <config_file> \
    device 0 \
    federate.client_num 12 \
    glue.adapter.fah.init_rank 32 \
    glue.adapter.fah.bandwidth_mode "realistic"
```

## Directory Structure

```
FedLoRA/
├── 1_scripts/              # Experiment scripts
│   └── README.md          # Documentation for scripts
├── 2_yamls/               # Configuration files
│   ├── quantized_fahqlora/  # FAH-QLoRA configs
│   ├── quantized_fedit/     # FedIT configs
│   └── quantized_hetlora/   # HetLoRA configs
├── data/                  # Datasets and network traces
├── exp/                   # Experiment outputs
├── federatedscope/        # Core implementation
└── README.md              # This file
```

## Client Precision Levels Supported

The framework supports four quantization precision levels:

1. 4-bit (NF4 QLoRA)

Base model: 4-bit NF4 quantized weights (bitsandbytes)

Compute: bf16

Trainable: LoRA adapters (and the small PEFT modules-to-save head)

Intended profile: lowest frozen-model memory and lowest overall peak

2. 8-bit (LLM.int8 QLoRA)

Base model: 8-bit quantized weights (bitsandbytes int8)

Compute: bf16

Trainable: LoRA adapters (and modules-to-save head)

Intended profile: higher frozen-model memory than 4-bit, still much smaller than full precision

3. 16-bit (true half precision, no quantization)

Base model: loaded in bf16 if supported, otherwise fp16

Compute: bf16

Trainable: LoRA adapters (and modules-to-save head)

Gradient checkpointing: enabled in the builder path (to keep activation memory reasonable)

Intended profile: clearly above 8-bit, but below fp32

4. 32-bit (fp32 full precision, no quantization)

Base model: loaded in fp32

Compute: fp32

Trainable: LoRA adapters (and modules-to-save head)

Gradient checkpointing: enabled in the builder path (to avoid the earlier activation blowup)

Intended profile: highest memory footprint among the four

## Additional Notes

### Network Bandwidth Modes

- **static**: Sample once at initialization, bandwidth stays fixed throughout training
- **dynamic**: Each client samples from their trace file, updates per round
- **homogeneous**: All clients share the same bandwidth that changes per round
- **realistic**: Time-based sampling (one sample per second). At initialization, the first sample is taken. On each update, the index advances by elapsed seconds. Each trace file is assigned to only one client.

### System Metrics

The framework supports different system monitoring modes:
- `fah_extended`: Extended metrics for FAH-QLoRA including per-client ranks, bandwidths, and time breakdowns

### Network Trace Files

Network traces should be placed in the `data/` directory. The framework supports:
- 4G network traces: `data/4Gnetwork_trace/`
- 5G network traces: `data/5Gnetwork_trace/`
- Custom traces: CSV format with bandwidth measurements

### Important Implementation Notes

- Both `glue.adapter` and `llm.adapter` sections in configs must match for compatibility
- The `heterolora` federate method is used for the naive HetLoRA baseline and for FAH-QLoRA; the `hetlora` method enables the full HetLoRA complete baseline (rank self-pruning + sparsity-weighted aggregation).

## References

- FAH-QLoRA: Federated Adaptive Heterogeneous QLoRA
- FedIT: Federated Instruction Tuning
- HetLoRA: Heterogeneous LoRA for Federated Fine-tuning of On-Device Foundation Models