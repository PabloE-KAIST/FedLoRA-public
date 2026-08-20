# 1_scripts/

Experiment launch scripts organised by lifecycle stage.

## Directory layout

```
1_scripts/
├── baseline_runs/glue/   ← per-method standalone launchers + shared _glue_lib.sh
├── sweeps/               ← multi-task sweep orchestrators (GPU scheduling)
├── distributed/          ← fleet deployment, queues, log collection
└── README.md
```

## GLUE task configuration

All scripts source `baseline_runs/glue/_glue_lib.sh` for task-aware defaults.
Settings below apply to **DeBERTa-large + LoRA**, IID split, `batch_size=32`.
Default `total_round_num=20` except QQP/MNLI which benefit from 30 rounds (see note below).

| Task | Train samples | Clients | Samples/client | `local_update_steps` | Eff. epochs/round | Canonical metric |
|------|-------------:|--------:|---------------:|--------------------:|------------------:|-----------------|
| SST-2 | 67,349 | 12 | ~5,612 | 30 | ~0.17 | Accuracy |
| MNLI | 392,702 | 12 | ~32,725 | 30 | ~0.03 | Accuracy |
| QQP | 363,846 | 12 | ~30,320 | 30 | ~0.03 | F1 |
| QNLI | 104,743 | 12 | ~8,729 | 30 | ~0.11 | Accuracy |
| CoLA | 8,551 | 6 | ~1,425 | 30 | ~0.67 | MCC |
| STS-B | 5,749 | 6 | ~958 | 30 | ~1.00 | Pearson |
| MRPC | 3,668 | 6 | ~611 | **15** | ~0.79 | F1 |
| RTE | 2,490 | 6 | ~415 | **15** | ~1.15 | Accuracy |

**Effective epochs/round** = `local_update_steps × batch_size / samples_per_client` = how many
passes over each client's data per FL round.

**Why MRPC/RTE use steps=15:** With steps=30, these tiny partitions see 1.5–2.3 epochs per
round (31–46 total over 20 rounds), causing overfitting — val_loss increases after round 14.
Reducing to 15 halves the effective epochs and eliminates the overfitting.

**Why small tasks use 6 clients:** With 12 clients and IID split, small-train tasks end up
with too few per-client samples for meaningful LoRA updates. Dropping to 6 keeps each client
at ≥400 samples.

**QQP round count:** At 20 rounds, QQP's best val_f1=0.822 is undertrained (~0.6 effective
epochs total). A 30-round experiment improved best val_f1 to **0.852** (best individual) with
diminishing returns from round 24+. Fleet experiments use 30 rounds for QQP and MNLI
(same effective-epochs bucket).

### Eval metric mapping (per Wang et al. 2018 / gluebenchmark.com)

| Task | `best_res_update_round_wise_key` |
|------|--------------------------------|
| CoLA | `val_mcc` |
| STS-B | `val_pearson` |
| MRPC, QQP | `val_f1` |
| SST-2, MNLI, QNLI, RTE | `val_acc` |

These are resolved by `glue_eval_key()` in `_glue_lib.sh`.

## Standalone baseline scripts

Located in `baseline_runs/glue/`. Each accepts `TASK` and `CUDA_VISIBLE_DEVICES` env vars:

| Script | Method | Config YAML |
|--------|--------|-------------|
| `fedit.sh` | FedIT (FedAvg + fixed LoRA) | `2_yamls/fedit/fedit-NO_quantized.yaml` |
| `hetlora.sh` | HetLoRA (heterogeneous ranks) | `2_yamls/hetlora/hetlora-NO_quantized.yaml` |
| `fah_qlora.sh` | FAH-QLoRA (adaptive rank + quant) | `2_yamls/fahqlora/fah_qlora-NO_quantized.yaml` |

## Sweep scripts

Located in `sweeps/`. These schedule multiple tasks across available GPUs:

| Script | Description |
|--------|-------------|
| `standalone_glue_all_tasks_fedit_r64.sh` | All 8 GLUE tasks, FedIT rank=64, GPUs 0+3 |
| `standalone_glue_small_tasks_fedit_r64.sh` | 4 small tasks only (CoLA/STS-B/MRPC/RTE), GPUs 0+1+3 |

## Distributed fleet scripts

Located in `distributed/`. See `distributed/README.md` for full fleet documentation.

| Subdirectory | Purpose |
|-------------|---------|
| `prep/` | Fleet sync, DA restart, partition deploy/activate |
| `queues/` | Per-method fleet queue launchers |
| `orchestrators/` | End-to-end FL run pipeline, master queue |
| `log_tools/` | Worker log collection and merging |
| `infra/` | Bandwidth generation, fleet health checks |

### Fleet campaign: all 8 GLUE tasks × 5 methods

All queue scripts accept `TASK`, `MANIFEST`, and `PORT_OFFSET` env vars.

**Hyperparameter grids per method:**

| Method | Queue script | Swept parameters | Combos/task |
|--------|-------------|-----------------|-------------|
| FedIT r64 | `fleet_queue_fedit_r64.sh` | — | 1 |
| HetLoRA | `fleet_queue_hetlora.sh` | rw=0.1, decay={0.50,0.60,0.80,0.95} | 4 |
| FAH-QLoRA | `fleet_queue_fahqlora.sh` | init_rank={32,64} × lambda={1,5,10} | 6 |
| AdaSparse v2 | `fleet_queue_v2.sh` | rw=0.1, gamma={0.50,0.60,0.80,0.95} × UL={230,460,690} | 12 |
| AdaSparse v3 | `fleet_queue_v3.sh` | same as v2 | 12 |

**Total: 280 experiments** (35 per task × 8 tasks).

**Scheduling** (`fleet_master_queue.sh`):
- Phase 1 (small tasks): 2 concurrent sub-fleets (Group A + B, 6 devices each),
  pairs of methods run simultaneously. Tasks in fast-first order: rte → mrpc → stsb → cola.
- Phase 2 (big tasks): full 12-device fleet, sequential. Order: sst2 → qnli → qqp → mnli.

**Sub-fleet manifests** (for 6-client small tasks):
- `client_manifest_group_a.json`: 2 agxorin + 1 agxavier + 2 x86 + 1 orinnx
- `client_manifest_group_b.json`: same class composition, different devices
