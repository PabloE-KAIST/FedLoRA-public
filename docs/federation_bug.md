# Federation Bug (2026-07): HetLoRA / AdaSparse standalone clients never applied the global adapter

**Status:** Root-caused and fixed 2026-07-22. Fix real-code-verified for HetLoRA-NP and AdaSparse v3.
**Severity:** Critical for accuracy claims — the affected methods trained with **no federation** at all.
**Scope of this doc:** the standalone / simulated (`federatedscope/`) path. The physical-fleet
(`distributed/`) path is being audited **separately** — no Jetson-experiment conclusions are stated here.

---

## 1. The bug (what happened)

Every client that extends `BaseRefactorClient` — which is an alias for the core
`Client` (`federatedscope/contrib/worker/base_refactor_client.py`:
`BaseRefactorClient = CoreClient`) — inherited a **no-op** implementation of
`Client._apply_client_specific_heterolora_payload` in
`federatedscope/core/workers/client.py` (was line ~494):

```python
def _apply_client_specific_heterolora_payload(self, content, ...):
    """Generic no-op hook for HeteroLoRA/FAH client-specific payload handling."""
    return content          # <-- returned the server download UNCHANGED
```

The server's aggregated global adapter is broadcast in **distributed key format**:

```
base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight.<rank>
                                                                       ^^^^^^^ trailing rank suffix
```

but the client model's `state_dict()` keys have **no rank suffix**:

```
base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight
```

Because the no-op hook passed `content` straight through, the download reached
`trainer.update(...)` still in distributed format. `trainer.update`
(`federatedscope/core/trainers/torch_trainer.py`) calls
`model.load_state_dict(state_dict, strict=False)`. With `strict=False`, **every**
distributed-format LoRA key is an unmatched key and is **silently dropped** — no
error, no warning. The aggregated global adapter therefore **never reached client
training**. Each affected client kept optimizing its own local adapter round after
round: the methods trained **independently**, i.e. there was effectively no FL.

### Root cause in one line
No-op download hook + distributed-vs-model key-format mismatch + `load_state_dict(strict=False)`
= aggregated global adapter silently dropped on every affected client, every round.

---

## 2. Which methods were affected

Affected clients inherit `BaseRefactorClient` and do **not** override the hook, so they
got the no-op:

| Method (standalone) | Client class | Overrides hook? | Federated? |
|---|---|---|---|
| **HetLoRA** (NP-SW, NP-AVG, and homogeneous) | `HetLoRAClient` | no (inherited no-op) | **NO — broken** |
| **AdaSparse-LoRA v1** | `AdaSparseLoRAClient` | no (inherited no-op) | **NO — broken** |
| **AdaSparse-LoRA v2** | `AdaSparseLoRAv2Client` | no (inherited no-op) | **NO — broken** |
| **AdaSparse-LoRA v3** | `AdaSparseLoRAv3Client` | no (inherited no-op) | **NO — broken** |

Unaffected (correct all along):

| Method | Why it was fine |
|---|---|
| **FedIT / FedAvg** | Default (non-hetero) download path; homogeneous keys already match the model state_dict. |
| **No-LoRA** (incl. head-only probe) | No LoRA download to re-key. |
| **naive HeteroLoRA** | `HeteroLoRAClient` **overrides** the hook (`heterolora_client.py`:40) with a real distributed→model converter. |
| **FAH-QLoRA** | `FahQLoRAClient(FahQLoRAClientMixin, HeteroLoRAClient)` inherits the working `HeteroLoRAClient` override. |

So the split is: the two subclasses that define/inherit the **real** converter
(`HeteroLoRAClient`, `FahQLoRAClient`) federated correctly; the four that fell back to
the base `Client` no-op (`HetLoRA`, AdaSparse `v1/v2/v3`) did not.

---

## 3. Evidence (this session)

Three independent lines of evidence, all consistent with "the global adapter never
reached client training":

1. **Aggregation-mode invariance.** Two runs that differ in *nothing* except
   `aggregation.mode` produced **byte-identical** client uploads and validation
   metrics on **every round**, even though the server-side aggregated checkpoints
   differed between the two runs. If clients had consumed the aggregate, changing how
   the server aggregates would necessarily change what clients receive and therefore
   what they upload next — it did not. The clients were provably ignoring the server output.

2. **Runtime probe `matched_in_model == 0`.** A probe placed on `trainer.update`
   counted how many keys in the incoming download matched a key in the model's
   state_dict. For the affected methods it reported **`matched_in_model = 0`** — i.e.
   `load_state_dict(strict=False)` matched zero LoRA parameters and dropped the entire
   distributed-format download.

3. **Homogeneous HetLoRA did not collapse to matched FedIT.** With homogeneous ranks,
   HetLoRA aggregation is mathematically identical to FedIT, so a correctly-federating
   HetLoRA-homogeneous run must reproduce FedIT accuracy. It did **not**:
   HetLoRA-homogeneous on **mrpc r200** landed **-0.65** off the matched FedIT point.
   That residual gap is the signature of clients training locally instead of on the
   shared aggregate.

---

## 4. The fix (3 changes, already applied)

### (1) New canonicalizer util
`federatedscope/contrib/common/heterolora_utils.py`:
`apply_distributed_lora_download(content, model, strict=True, debug=False)`
(plus helper `_adapt_lora_rank_shape`).

- Detects whether the payload is in distributed format (a `lora_A`/`lora_B` key whose
  last dotted segment is an integer rank).
- For each distributed LoRA key it **infers the rank from the key itself** (via
  `_canonical_lora_key`, which strips the trailing `.<rank>` and any leading `model.`),
  matches it to the model's trainable LoRA param that shares the same canonical key, and
  **pad/truncates the rank dimension** to fit the local param shape.
  Because the rank is read off each key, **both** per-module HetLoRA formats **and**
  per-layer AdaSparse grouped/pruned formats are covered.
- Non-LoRA keys and non-distributed payloads **pass through unchanged**.
- With `strict=True` it **raises `KeyError`** if any distributed LoRA key cannot be
  matched — the exact silent-drop condition that caused this bug can therefore never
  recur unnoticed (strict key-consumption).

### (2) Client hook + round-start guard
`federatedscope/core/workers/client.py`:

- `Client._apply_client_specific_heterolora_payload` — the no-op stub is **replaced**
  with a call to `apply_distributed_lora_download` (skipped when `context == 'finish'`),
  gated by `federate.assert_download_consumed`. It stashes the applied LoRA subset in
  `self._pending_download_lora`.
- `Client._maybe_assert_download_applied` — new round-start tensor-equality check:
  after the download is loaded, the model's LoRA params must equal the applied download
  (`torch.allclose`, atol 1e-4). Off by default (per-round cost); enabled by
  `federate.assert_download_tensor_equality`. It is invoked right after the train-branch
  `trainer.update(...)` call.
- `HeteroLoRAClient` / `FahQLoRAClient` still **override** the hook (their real
  rank-resize converter is unchanged) — the fix only changes the base-class fallback.

### (3) Config flags
`federatedscope/core/configs/cfg_fl_setting.py`:

- `federate.assert_download_consumed = True` (default) — raise if any distributed LoRA
  download key cannot be matched to a trainable model param.
- `federate.assert_download_tensor_equality = False` (default; turn on for tests) —
  additionally verify at round start that model LoRA == applied download.

All new behavior is opt-in/guarded and byte-identical for the already-correct paths
(FedIT, No-LoRA, naive HeteroLoRA, FAH-QLoRA).

---

## 5. INVALIDATED RESULTS

Read this section before citing any historical accuracy number.

**INVALID — must be revalidated under the fixed path:**

- **All historical STANDALONE AdaSparse (v1/v2/v3) accuracy results**, across **M1–M9**.
  These runs never applied the aggregated global adapter, so their reported accuracies do
  not reflect federated training and cannot be used to compare methods or select
  operating points.
- **All historical STANDALONE HetLoRA accuracy results** (NP-SW, NP-AVG, and homogeneous),
  same reason.
- **The just-run Gate-1 heterogeneous cells** — specifically the
  `HetLoRA-NP-SW` and `HetLoRA-NP-AVG` arms × `{deployment_grounded, heavy_tail}` from
  the accuracy-gap campaign. These were produced on the broken path and are **invalid**.

**STILL VALID (accuracy stands):**

- **FedIT / FedAvg** accuracy results (including Gate-1 `FedIT r∈{64,87,142,200}`).
- **No-LoRA** results, including the Gate-1 head-only frozen-backbone probe.
- **FAH-QLoRA** accuracy results (working converter via `HeteroLoRAClient`).
- **naive HeteroLoRA** accuracy results (working override).

**Nuance — server-side diagnostics are real:**
Server-side aggregation diagnostics (aggregated-checkpoint contents, aggregation-mode
comparisons, heterogeneous-error state math, etc.) **did measure real aggregation** on the
server. The defect is strictly on the **client download-apply** side: the clients ignored
that (correct) aggregate. So server-side aggregation analyses are not themselves wrong —
but any conclusion of the form "method X's *accuracy* benefits from aggregation Y" that was
drawn from an affected method's client-side metrics is invalid.

**Physical fleet:** the `distributed/` (Jetson) path uses its own client/worker code and is
being audited **separately**. This document makes **no claims** about which physical-fleet
runs are or are not affected; treat that as an open, separately-tracked question.

---

## 6. Pointers

- Fix util: `federatedscope/contrib/common/heterolora_utils.py`
  (`apply_distributed_lora_download`, `_adapt_lora_rank_shape`; and existing
  `_canonical_lora_key`).
- Client hook + guard: `federatedscope/core/workers/client.py`
  (`_apply_client_specific_heterolora_payload`, `_maybe_assert_download_applied`).
- Config: `federatedscope/core/configs/cfg_fl_setting.py`
  (`federate.assert_download_consumed`, `federate.assert_download_tensor_equality`).
- Silent-drop site: `federatedscope/core/trainers/torch_trainer.py`
  (`load_state_dict(..., strict=False)` inside `trainer.update`).
