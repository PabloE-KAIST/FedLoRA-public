# GLUE baseline runs

Scripts in this directory wrap each method's GLUE entry point on the
standalone/simulated FL path. Default task is **SST-2**; override via the
`TASK` environment variable.

```bash
bash 1_scripts/baseline_runs/glue/fedit.sh          # SST-2, 12 clients
TASK=rte  bash 1_scripts/baseline_runs/glue/fedit.sh   # RTE,   6 clients
TASK=qqp  bash 1_scripts/baseline_runs/glue/hetlora.sh # QQP,  12 clients
```

`_glue_lib.sh` is sourced by each script and provides `glue_clients_for_task()`,
which maps the task to `federate.client_num` according to dataset size.

## Supported tasks

| Task   | Train  | client_num (IID) | ~Per-client train | Status |
|--------|-------:|-----------------:|------------------:|--------|
| MNLI   | 392,702 | 12 | 32,725 | ✓ |
| QQP    | 363,846 | 12 | 30,320 | ✓ |
| QNLI   | 104,743 | 12 |  8,728 | ✓ |
| SST-2  |  67,349 | 12 |  5,612 | ✓ (default) |
| CoLA   |   8,551 |  6 |  1,425 | ✓ |
| STS-B  |   5,749 |  6 |    958 | ✓ |
| MRPC   |   3,668 |  6 |    611 | ✓ |
| RTE    |   2,490 |  6 |    415 | ✓ |
| WNLI   |     635 |  — | — | **disabled** — too few train samples ([loader raises](../../../federatedscope/glue/dataloader/dataloader.py#L292-L296), commit `48e8aa7`) |
| AX     |       0 |  — | — | **not added** — diagnostic-only split, no train data |

`client_num=6` for RTE / MRPC / STS-B / CoLA keeps per-client train size above
~400 samples, which is the lower bound for stable LoRA updates per round under
`batch_size=32`. CoLA (8,551 train) sits closer to STS-B (5,749) than to SST-2
(67,349) — a ~1.5× gap vs. ~8× — so it belongs in the small-task cluster.
The remaining big tasks (SST-2, QNLI, QQP, MNLI) retain `client_num=12`.

## Rounds

All scripts run `federate.total_round_num=20` from the YAML configs in
`2_yamls/<method>/`. This number was chosen for SST-2 and has not been retuned
per task. If reproducing across all tasks at scale, consider per-task round
counts that hold *total epochs of training* roughly constant. The current
20-round setting trains:

- ~5 epochs on RTE / MRPC / STS-B / CoLA at `client_num=6`
- ~3 epochs on SST-2 at `client_num=12`
- < 0.4 epochs on QQP / MNLI at `client_num=12`

Re-tuning `total_round_num` per task is left as a follow-up; the task-aware
client_num here is the minimum needed to make small-task runs meaningful.

## Files

- `_glue_lib.sh` — shared `glue_clients_for_task()` helper.
- `fedit.sh`, `hetlora.sh`, `fah_qlora.sh`, `adasparse-lora.sh`,
  `adasparse-lorav2.sh`, `adasparse-lorav3.sh` — one method each; source
  `_glue_lib.sh` and accept `TASK=...` from the environment.
