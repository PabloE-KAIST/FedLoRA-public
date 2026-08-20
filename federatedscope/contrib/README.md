# Contrib refactor overview

## Purpose

The current design keeps:

- `federatedscope/core/workers/client.py` and `server.py` as shared lifecycle and hook runners
- `federatedscope/glue/trainer/trainer.py` as the stable `gluetrainer` entrypoint
- method- or capability-specific logic inside `federatedscope/contrib/...`

## High-level structure

### `contrib/worker/`

This folder contains worker registration and extracted worker logic.

#### Registration files

These files are imported at startup so FederatedScope can register custom workers before builder lookup:

- `adasparse_lora_worker.py`
- `adasparse_lorav2_worker.py`
- `hetlora_worker.py`
- `heterolora_worker.py`
- `fah_qlora_worker.py`

These files should stay thin. Their job is to wire the correct client/server classes into the registry.

#### Base refactor layers

- `base_refactor_client.py`
- `base_refactor_server.py`

These provide the thin seam between the generic core workers and the extracted method-specific workers. They are intentionally lightweight and should not accumulate method logic again.

#### Extracted method/capability files

Located under:

- `contrib/worker/methods/`

Current extracted files include:

- `hetlora_client.py`
- `hetlora_server.py`
- `adasparse_lora_client.py`
- `adasparse_lora_server.py`
- `adasparse_lorav2_client.py`
- `adasparse_lorav2_server.py`
- `heterolora_client.py`
- `heterolora_server.py`
- `fah_qlora_client.py`
- `fah_qlora_server.py`

### `contrib/trainer/`

This folder contains split trainer logic used by the shared GLUE trainer entrypoint.

Current trainer files include:

- `glue_base_trainer.py`
- `glue_hetlora_trainer.py`
- `glue_adasparse_trainer.py`
- `glue_adasparse_v2_trainer.py`
- `glue_fah_trainer.py`
- `glue_heterolora_trainer.py`

The public trainer entrypoint remains:

- `trainer.type: gluetrainer`

The shared `GLUETrainer` delegates to the split trainer modules internally, so experiment configs do not need method-specific trainer names.

### `contrib/common/`

Shared utility logic extracted from the monolithic files lives here.

Typical contents:

- `config_resolver.py`
- `payload_utils.py`
- `rank_utils.py`

These modules centralize configuration lookup and repeated helper logic so the shared worker/trainer files stay smaller and more consistent.

## Current design principles

### 1. Shared core files are hook runners

The shared `client.py` and `server.py` mainly contain:

- generic lifecycle flow
- shared communication logic
- stable callback entrypoints
- default hook methods

They should not regain large blocks of method-specific logic.

### 2. Extracted files hold the real method logic

If a piece of logic is genuinely specific to:

- HetLoRA
- AdaSparse-LoRA
- AdaSparse-LoRAv2
- HeteroLoRA capability
- FAH-QLoRA capability

then it should live in the corresponding file under `contrib/worker/methods/` or `contrib/trainer/`.

### 3. Callback entrypoints remain shared

The `callback_funcs_for_xxx` methods remain in the shared core worker files as the stable message-handling interface.

The refactor intentionally keeps those callback entrypoints in the shared workers. The extracted method files should provide the helper and hook logic that those callbacks call.

## Method-specific notes

### HetLoRA

HetLoRA client and server logic are extracted into their own method files.

HetLoRA uses HeteroLoRA server capability underneath for client-specific distributed payload handling, while keeping its own concrete client/server classes and method-specific rank-update behavior in the `hetlora_*` files.

### AdaSparse-LoRA

AdaSparse-LoRA client and server logic are extracted into their own method files.

### AdaSparse-LoRAv2

AdaSparse-LoRAv2 client and server logic are extracted into their own method files.

### HeteroLoRA

HeteroLoRA is best thought of as a capability around client-specific rank configuration and personalized/distributed payload handling.

Its client/server files own that capability logic, including neutral hetero-rank configuration synchronization and heterogeneous payload preparation/broadcast behavior.

### FAH-QLoRA

FAH builds on HeteroLoRA capability underneath, but the concrete FAH worker classes live in the FAH method files.


## Startup wiring

The codebase relies on explicit imports in `main.py` to register custom workers before builder lookup.

Typical startup imports are:

```python
import federatedscope.contrib.worker.adasparse_lora_worker
import federatedscope.contrib.worker.adasparse_lorav2_worker
import federatedscope.contrib.worker.hetlora_worker
import federatedscope.contrib.worker.heterolora_worker
import federatedscope.contrib.worker.fah_qlora_worker
```

## Config expectations

### Trainer selection

Use:

```yaml
trainer:
  type: gluetrainer
```

The split trainer modules are used internally by the shared GLUE trainer.

### Method selection

The following explicit methods are expected to work as extracted worker paths:

- `hetlora`
- `adasparse_lora`
- `adasparse_lorav2`
- `heterolora`
- `fah_qlora`


## What not to do

- Do not reintroduce large method-specific blocks into shared `client.py` or `server.py`
- Do not reintroduce an extra monolithic trainer-dispatch layer when `gluetrainer` already handles delegation
- Do not move `callback_funcs_for_xxx` out of the shared workers unless you are intentionally doing a deeper architectural redesign

## Maintenance rule of thumb

When adding or changing logic, ask:

1. Is this generic worker/trainer lifecycle behavior?
   - Keep it in the shared core file.
2. Is this specific to one method or one capability?
   - Put it in the corresponding extracted method file.
3. Is this a repeated helper used across several places?
   - Put it in `contrib/common/`.