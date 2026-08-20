# Reproduction Guide

## 1. Environment

```bash
conda env create -f requirements/environment.yml
conda activate fedlora
pip install -e .
```

## 2. Models

Model identifiers in `2_yamls/**/*.yaml` are Hugging Face Hub ids, e.g.:

```yaml
model:
  type: 'microsoft/deberta-large@huggingface_llm'
```

The string is split on `@`: the left side is passed to `from_pretrained`, the
right side selects the FederatedScope model builder. Hub ids are resolved and
cached automatically on first use.

| Used in | Hub id |
| --- | --- |
| GLUE experiments | `microsoft/deberta-large` |
| LLM experiments  | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-2-7b-hf` |
| VLM experiments  | `Qwen/Qwen2.5-VL-3B-Instruct` |

**Offline / air-gapped:** pre-download the models and point the shell scripts at
them with `FEDLORA_MODEL_ROOT` (default `~/models`), or set the standard
`HF_HOME` / `TRANSFORMERS_OFFLINE=1`.

## 3. Datasets

Not redistributed — see [`data/README.md`](../data/README.md). GLUE is fetched
automatically through `datasets` on first run.

HumanEval code evaluation additionally needs OpenAI's harness:

```bash
git clone https://github.com/openai/human-eval tools/human-eval
pip install -e tools/human-eval
```

## 4. Environment variables

Shell runners under `1_scripts/` are parameterised rather than hardcoded:

| Variable | Default | Meaning |
| --- | --- | --- |
| `FEDLORA_ROOT` | auto-detected from the script path | repository root |
| `FEDLORA_MODEL_ROOT` | `~/models` | local model cache for offline use |
| `PYTHON` | `python` | interpreter to invoke |
| `FEDLORA_CONTROLLER_IP` | *(required for real-device runs)* | FL server address reachable by the devices |
| `FEDLORA_JETSON_USER` | `ubuntu` | SSH account on the edge devices |
| `FEDLORA_JETSON_RUNTIME` | `/home/$FEDLORA_JETSON_USER/fedlora_runtime` | device-side runtime root |
| `FEDLORA_X86_HOST` / `FEDLORA_X86_USER` | *(required for x86 workers)* | optional non-Jetson worker |

## 5. Standalone (simulated) experiments

The simplest entry point — a single process simulating all clients:

```bash
python federatedscope/main.py --cfg 2_yamls/fedit/fedit-NO_quantized.yaml \
    data.type mrpc@glue federate.client_num 6 federate.total_round_num 20 \
    outdir exp/fedit_mrpc device 0
```

Method wrappers:

```bash
bash 1_scripts/baseline_runs/glue/fedit.sh
bash 1_scripts/baseline_runs/glue/hetlora.sh
bash 1_scripts/baseline_runs/glue/fah_qlora.sh
```

Available methods: `FedAvg` (FedIT, homogeneous rank), `hetlora`,
`fah_qlora`, `adasparse_lorav1|v2|v3`.

## 6. Real-device (distributed) experiments

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the runtime path. The server:

```bash
export FEDLORA_CONTROLLER_IP=<address the devices can reach>
python -m distributed.server.main \
    --config 2_yamls/fedit/fedit_distributed.yaml \
    --manifest distributed/configs/client_manifest.json
```

Workers are launched on-device through the container start path rather than
manually; the device-side agent is a separate project (see
`distributed/docker/da_x86/README.md`).

## 7. Federation correctness guards

Two assertions verify that the aggregated global adapter actually reaches each
client — the failure mode described in [`federation_bug.md`](federation_bug.md):

```bash
python federatedscope/main.py --cfg 2_yamls/hetlora/hetlora-NO_quantized.yaml \
    data.type mrpc@glue federate.client_num 2 federate.total_round_num 2 \
    federate.assert_download_consumed True \
    federate.assert_download_tensor_equality True \
    outdir exp/hetlora_guarded device 0
```

`assert_download_consumed` raises if any distributed LoRA key fails to map onto
the client model; `assert_download_tensor_equality` raises if the client's LoRA
weights do not equal the applied global adapter at round start. Both are cheap
and recommended when adding a new heterogeneous method.

## 8. Known limitation: AdaSparse stage-2 downlink

AdaSparse-LoRA v2/v3 ship with `stage2.enabled: True` (budgeted sparse
component selection). Stage-2 produces a **sparse/partial downlink** — the
server sends only a subset of each layer's LoRA rows — which the generic
download canonicalizer cannot scatter back onto the full adapter. Running with
stage-2 on therefore raises, by design:

```
NotImplementedError: [federation] partial/Stage-2 sparse LoRA downlink requires
index-based scatter (download_indices), not implemented in the generic
canonicalizer. Use full downlink (e.g. AdaS stage2.enabled=False) ...
```

This guard is deliberate: the alternative is silently mis-applying the global
adapter, which is exactly the class of failure described in
[federation_bug.md](federation_bug.md). Until index-based grouped scatter is
implemented, run the accuracy-oriented configuration:

```bash
python federatedscope/main.py \
    --cfg 2_yamls/adasparse_lora_v3/adasparse-lorav3-NO_quantized.yaml \
    data.type mrpc@glue \
    glue.adapter.adasparse_lorav3.stage2.enabled False \
    outdir exp/adas_v3 device 0
```

With stage-2 off, every stage-1 survivor is transmitted through the normal
aggregation path (no bandwidth budget, no residual accounting). Stage-1
adaptive rank selection — the core of the method — is unaffected.

## 9. Tests

```bash
python -m pytest federatedscope/contrib/common/test_federation_download.py \
                 federatedscope/contrib/common/test_head_federation.py \
                 federatedscope/contrib/common/test_hetlora_restriction.py

# real 2-round federated run (needs a GPU + deberta-large)
FEDLORA_RUN_SLOW_TESTS=1 python -m pytest \
    federatedscope/contrib/common/test_federation_download.py -k slow
```
