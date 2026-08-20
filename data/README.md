# Datasets

Datasets are **not redistributed** in this repository. Each is fetched at run
time and remains subject to its own license.

Everything else in this directory is gitignored — do not commit dataset files,
caches, or generated partitions.

| Dataset | Used for | How it is obtained |
| --- | --- | --- |
| **GLUE** (cola, rte, mrpc, sst2, qnli, qqp, mnli, stsb) | encoder experiments (DeBERTa) | fetched automatically via `datasets` on first run (`data.type <task>@glue`) |
| **Databricks Dolly 15k** | instruction-tuning experiments | `databricks/databricks-dolly-15k` on the Hugging Face Hub |
| **CodeAlpaca 20k** | code instruction-tuning | [`sahil2801/CodeAlpaca-20k`](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k); tag categories with `python data/tag_codealpaca.py` |
| **HumanEval** | code-generation evaluation | ships with OpenAI's harness — see below |

## HumanEval

`federatedscope/llm/eval/eval_for_code/humaneval.py` depends on OpenAI's
evaluation harness, which is not vendored here:

```bash
git clone https://github.com/openai/human-eval tools/human-eval
pip install -e tools/human-eval
```

## Federated partitions

Client partitions are **generated**, not stored. Non-IID splits are produced by
the configured splitter (e.g. `data.splitter lda` with
`data.splitter_args "[{'alpha': 1.0}]"`), seeded by `cfg.seed` so a given seed
reproduces the same split.

For real-device runs, materialise per-client partition files with:

```bash
python distributed/data/prepare_partitions.py --help
```

Output lands in `distributed/data/generated_partitions/` (gitignored).
