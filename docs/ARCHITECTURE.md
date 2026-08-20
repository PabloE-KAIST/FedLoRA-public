# FedLoRA Architecture Guide

## Purpose

This document explains the **current working architecture** of the real-device distributed FedLoRA deployment so that a new agent or contributor can quickly understand:

- what runs on the **server**
- what runs on the **Jetsons**
- how the codebase is split across **FedLoRA**, **AegisGov-master**, and **fedlora_runtime**
- which files/functions are important
- how control-plane and FL payload flow through the system
- where to look first when navigating the repo

This document is written as a **repo navigation guide** and **architecture map** for continued work.

---

## 1. High-Level System Picture

The current working system has three major layers:

1. **FedLoRA codebase**
   - owns FL semantics
   - owns the Python control plane
   - owns the Python comm managers
   - owns the server and worker entry points

2. **AegisGov-master**
   - owns the real C++ `DeviceAgent`
   - owns protobuf control-plane message definitions
   - owns compose-based container launch on the Jetsons

3. **fedlora_runtime** on each Jetson
   - holds runtime data needed by the worker containers
   - model directory
   - partition artifacts
   - configs
   - logs
   - optionally a source overlay for `distributed/`

The most important architectural rule is:

> **FedLoRA still owns FL logic. AegisGov owns device-side lifecycle and relay.**

That means:
- the Python `Server` and `Client` remain the real FL actors
- the C++ `DeviceAgent` does not implement FL round logic
- worker containers remain thin wrappers around the existing FedLoRA client path

---

## 2. Physical Topology

## Server
- Host: `gpu-host-a`
- IP: `<device-ip>`

## Jetsons
- `agxorin1`
  - Jetson AGX Orin
- `agxavier1`
  - Jetson AGX Xavier

## Current proven runtime modes
These have all already been proven:

- real control-plane registration
- real `CONT_START` launch through C++ `DeviceAgent`
- real worker container startup through compose
- direct FL payload mode
- relayed FL payload mode through `DeviceAgent`

The most current proven architecture is **relay mode**:
- worker talks to local `DeviceAgent`
- `DeviceAgent` relays FL payload to the Python server
- no direct worker↔server payload sockets remain in the runtime path

---

## 3. Port Ownership and Meaning

There are two distinct planes:

## A. Control Plane
Used for:
- device registration
- `CONTAINER_START`
- `CONTAINER_STOP`
- orchestration/lifecycle

Server-side ports:
- `60101`: control-plane REP
- `60102`: control-plane PUB

Jetson-side device-agent ports with system offset:
- `60111`
- `60112`

## B. FL Payload Plane
Used for:
- `join_in`
- `join_in_info`
- `model_para`
- `metrics`
- `evaluate`
- `finish`

Server-side FL payload ports:
- `60001` / `60002` in the earlier direct-mode design
- with relay/system offset handling, relay-mode reports refer to the worker-facing local agent ports and shifted payload bindings

The key conceptual rule is:
- **Control plane** decides *which worker runs where*
- **FL payload plane** carries FedLoRA `Message` traffic

---

## 4. Repo Split: What Lives Where

## 4.1 FedLoRA repo

This is the main repo you are working in.

Key areas:

### `federatedscope/`
This is the real FL engine.
It contains:
- existing `Server` logic
- existing `Client` logic
- trainer/model/data logic
- method logic (FedIT, HetLoRA, FAH-QLoRA, AdaSparse)

Important point:
- the distributed deployment work was intentionally designed **not** to replace this logic

### `distributed/`
This is the deployment/orchestration layer added around the real FedLoRA code.

It contains:
- control-plane code
- comm managers
- worker/server entry points
- distributed configs
- proto bridge
- docker-compose templates
- utilities/scripts
- older gate/harness-era validation artifacts

### `1_scripts/`
Legacy or baseline experiment scripts used for simulated/server-only runs.
These are important for:
- baseline comparison
- repeatability
- matching real-device runs against previous simulated runs

### `2_yamls/`
Legacy or baseline experiment YAML files used in the simulated/server-only path.
These are important for:
- matching method/task settings
- comparing the `/distributed` path against established experiments
- Milestone 5 baseline comparison and Milestone 6 controlled expansion

### `exp/`
Experiment outputs and run-specific configs.
Examples include:
- `exp/distributed_fedit/config.yaml`

This is often the practical config entry point for the server orchestrator.

---

## 4.2 AegisGov-master repo

This is the device-side orchestration and relay system.

It matters because the real C++ `DeviceAgent` comes from here.

Important areas:

### `build/` or `build_host/`
Contains the built `DeviceAgent` binary used on the Jetsons.

### `libs/device_agent/`
Contains the main C++ implementation of the device agent.

Important source files:
- `device_agent.cpp`
- `device_agent.h`

### `libs/misc/`
Contains shared helpers and message-type mappings.

Important source file:
- `misc.cpp`

### protobuf definitions
Important file:
- `controlmessages.proto`

This defines the real message schema used between Python control plane and C++ `DeviceAgent`.

### `dockerfiles/`
Contains the compose template path used by AegisGov:
- `docker-compose.jetson.yml`

In the working FedLoRA integration, this template is replaced or extended with a FedLoRA-compatible compose file.

### `jsons/`
Contains device-agent-related configuration and experiment JSONs.
Useful when diagnosing:
- device naming
- metrics configuration
- experiment selection
- bandwidth setup

---

## 4.3 fedlora_runtime on each Jetson

This is not the main source repo.
It is the runtime data root used by the worker containers.

Current runtime root:
- `~/pablo/fedlora_runtime`
- concretely: `~/fedlora_runtime`

Important subdirectories:

### `models/`
Contains staged models used by the worker containers.

Current important path:
- `~/fedlora_runtime/models/deberta-large`

Inside the container, this is mounted as:
- `/workspace/models/deberta-large`

### `partitions/`
Contains partition artifacts copied/generated for each client.

Examples:
- `client_1.pkl`
- `client_2.pkl`

### `configs/`
Contains configs that worker containers consume at runtime.

### `logs/`
Runtime logs.

### `distributed/`
In the final relay-mode work, a host overlay mount was added so containers can pick up updated Python `distributed/` source without rebuilding the image every time.

---

## 5. Main Runtime Flow

The current working real-device path can be described in three layers.

## Layer 1: Registration and orchestration
1. Python control plane starts on the server
2. C++ `DeviceAgent` starts on each Jetson
3. `DeviceAgent` sends `DEVICE_ADVERTISEMENT`
4. Python side returns raw `SystemInfo`
5. Python control plane publishes `CONT_START`
6. `DeviceAgent` launches worker container via compose

## Layer 2: Worker startup
1. Worker container starts
2. mounted runtime files are visible inside container
3. worker entry point runs
4. partition loads
5. model loads
6. comm manager is constructed
7. real FedLoRA `Client` is instantiated with injected `comm_manager`

## Layer 3: FL payload
1. worker sends `join_in`
2. server processes join
3. server broadcasts `model_para`
4. client trains and uploads `model_para`
5. server aggregates
6. server evaluates / repeats round
7. server sends `finish`
8. workers exit cleanly

In relay mode, the FL payload path is:
- worker -> local `DeviceAgent`
- local `DeviceAgent` -> server
- server -> `DeviceAgent`
- `DeviceAgent` -> worker

---

## 6. FedLoRA File / Function Map

This section maps the most important Python files and what they are responsible for.

## 6.1 `federatedscope/core/workers/server.py`
This is still the real FL server implementation.

Key responsibility:
- handle `join_in`
- maintain round state
- broadcast model parameters
- aggregate client updates
- run eval
- terminate

Important architectural change:
- supports injected `comm_manager`

What this means:
- deployment code wraps around this class
- FL semantics still live here

## 6.2 `federatedscope/core/workers/client.py`
This is still the real FL client implementation.

Key responsibility:
- receive model parameters
- update trainer/model
- train locally
- send back `(sample_count, model_para)` or method-specific variants
- handle eval and finish

Important architectural change:
- supports injected `comm_manager`
- supports injected `local_address`

This is the class instantiated inside the worker container.

## 6.3 `distributed/server/main.py`
This is the server-side orchestrator entry point.

Role:
- parse config/manifest/runtime options
- start control plane when needed
- start or coordinate the FL server
- build the injected comm manager
- bridge orchestration to the real FedLoRA `Server`

This is the first file to inspect when asking:
- how is the distributed server actually started?
- what flags are used in direct mode vs proto mode?
- how is control-plane startup coupled to FL server startup?

## 6.4 `distributed/worker/main.py`
This is the worker entry point inside the container.

Role:
- parse worker CLI flags
- load partition artifact
- load model/config
- construct `ZMQClientCommManager`
- instantiate the real FedLoRA `Client`
- run `join_in()` and `run()`

This is the first file to inspect when asking:
- what exact runtime args does the worker need?
- how does it switch between direct mode and relay mode?
- where are partition/model/config paths consumed?

## 6.5 `distributed/comm/zmq_client_comm.py`
Client-side communication manager.

Role:
- connect worker-side message flow to either direct server transport or relay mode
- send FedLoRA `Message` objects outward
- receive relayed or direct messages inward
- in relay mode, deal with relay-envelope specifics

This is the file to inspect when asking:
- how does the worker actually talk in relay mode?
- what local ports does it connect to?
- how are topic prefixes or payload wrappers handled?

The latest relay milestone specifically changed this file to:
- auto-detect and decode legacy base64 relay payloads after topic stripping (raw bytes is the default)

## 6.6 `distributed/comm/zmq_server_comm.py`
Server-side communication manager.

Role:
- receive FedLoRA `Message` traffic from clients
- publish/broadcast payloads outward
- support direct mode and relay-compatible routing behavior

This is the file to inspect when asking:
- how does the server distinguish direct mode vs relay mode?
- how are recipients/topics encoded for broadcast?
- how does server-side REP/PUB behavior work?

## 6.7 `distributed/control_plane/service.py`
The Python control-plane service.

Role:
- accept real or reduced-mode device registration
- validate devices against manifest
- build/send container lifecycle commands
- in the current mature version, speak protobuf-compatible control-plane behavior

This is the main file to inspect when asking:
- how are `DEVICE_ADVERTISEMENT`, `SystemInfo`, and `CONT_START` handled?
- how is proto mode enabled?
- where does manifest-driven container config get built?

## 6.8 `distributed/control_plane/proto_bridge.py`
This file exists specifically to translate Python-side concepts into the real AegisGov protobuf wire format.

Role:
- map Python control-plane objects to protobuf messages
- build `ContainerConfig`
- encode/decode control-plane messages
- handle wire-format constraints

This became especially important in Milestone 2b and 2c.

Known key behavior:
- `PackagedMsg.payload` is `bytes` — raw binary is the default encoding (legacy base64 can be re-enabled via `FEDLORA_RELAY_BASE64=1`)

This is the first file to inspect when asking:
- how does Python adapt to the real C++ `DeviceAgent` protocol?
- where are CLI args packaged into container launch config?
- where is relay payload wrapping handled?

## 6.9 `distributed/proto/controlmessages.proto`
Python-side copy of the AegisGov-compatible protobuf schema.

Role:
- define `ContainerConfig`, `DeviceInfo`, `SystemInfo`, and related messages

This is the first file to inspect when asking:
- what exact fields are available in protobuf messages?
- what types are expected?
- where are mismatches with AegisGov likely to happen?

## 6.10 `distributed/docker/docker-compose.aegisgov.jetson.yml`
The compose template used when workers are launched by AegisGov/device_agent.

Role:
- map AegisGov environment variables into the actual worker launch
- define mounts
- define image name/runtime behavior
- make the worker container start the correct entrypoint

This is the first file to inspect when asking:
- how does `CONT_START` end up becoming a running worker?
- where do mounts come from?
- which env vars from AegisGov are consumed?
- how is the worker CLI ultimately formed?

## 6.11 `distributed/docker/launch_worker.sh`
Helper used to avoid AegisGov `EXECUTABLE` word-splitting issues.

Role:
- consume quoted start information
- reconstruct the worker CLI safely
- act as a space-free executable path for compose/AegisGov launch

This is especially important because AegisGov `runCompose()` does not safely pass a long CLI string as an executable.

## 6.12 `distributed/configs/client_manifest.json`
The manifest is the source of truth for:
- static client IDs
- device names
- container names
- partition paths
- device class / image mapping, depending on current version

This is the first file to inspect when asking:
- what is the intended identity and placement of each client?
- how do `agxorin1` and `agxavier1` map to partitions and containers?

## 6.13 `2_yamls/fedit/fedit_distributed.yaml`
Worker-side / distributed runtime config.

Role:
- FedIT-specific distributed config
- model path
- trainer type
- eval split
- transport/runtime assumptions used by worker path

This is the first file to inspect when asking:
- what config is actually used inside worker containers?
- what task/model/trainer assumptions are active?

---

## 7. AegisGov-master File / Function Map

This section maps the most important C++ side components.

## 7.1 `controlmessages.proto`
Authoritative control-plane message schema.

Important messages include:
- `ContainerConfig`
- `DeviceInfo`
- `SystemInfo`
- forwarding container messages / packaged messages

Why it matters:
- Python control-plane must match this exactly in proto mode

## 7.2 `device_agent.cpp` / `device_agent.h`
Core device-agent implementation.

Important conceptual responsibilities:
- connect to controller
- send registration
- subscribe to control-plane PUB
- parse and react to `CONT_START`
- run compose-based container launch
- forward payload between worker and controller/server
- connect to metrics infrastructure

Important functions/behaviors to understand:
- startup and registration
- control-command handling
- relay/forwarding
- metrics initialization
- compose launch path

These are the first files to inspect when asking:
- why did the device agent fail to start?
- how is registration really encoded?
- how does `CONT_START` actually launch a worker?
- what does the worker-facing relay path expect?

## 7.3 `misc.cpp`
Contains important shared mappings and helpers.

Known importance:
- maps long control message types to abbreviations like:
  - `CONTAINER_START` -> `CONT_START`
  - `DEVICE_ADVERTISEMENT` -> `DEV_AD`

This is the first file to inspect when wire-format message names don’t match expectations.

## 7.4 `runCompose()` path
Whether located in `device_agent.h` or nearby helper code, this behavior is crucial.

Role:
- receives env vars like `DOCKER_NAME`, `CONTAINER_NAME`, `EXECUTABLE`, `START_STRING`
- runs `docker compose -f ... up`
- uses compose template to create worker container

This is the first behavior to inspect when:
- worker did not launch
- command line was malformed
- compose env vars are not flowing correctly

## 7.5 Metrics / PostgreSQL integration
AegisGov `DeviceAgent` expects metrics database connectivity.

This matters because:
- startup can fail if DB/schema/permissions are wrong
- several real integration bugs were caused by this

This is the first area to inspect when:
- `DeviceAgent` exits early
- logs mention SQL or schema issues
- startup succeeds only partially

---

## 8. fedlora_runtime Map

This section explains what the runtime root contributes.

## 8.1 `models/`
Purpose:
- store local model checkpoint/directory used by worker containers

Current key path:
- `~/fedlora_runtime/models/deberta-large`

Container path:
- `/workspace/models/deberta-large`

## 8.2 `partitions/`
Purpose:
- store client-specific partition artifacts

Examples:
- `client_1.pkl`
- `client_2.pkl`

These are loaded by `distributed/worker/main.py`.

## 8.3 `configs/`
Purpose:
- store configs copied/staged for workers and possibly runtime variants

## 8.4 `logs/`
Purpose:
- capture runtime logs and outputs for real-device runs

## 8.5 `distributed/` overlay
Purpose:
- allow containers to pick up updated Python source from the host without rebuilding the image

This became particularly useful during relay-mode debugging and iteration.

---

## 9. Current End-to-End Runtime Chain

The currently proven relay-mode runtime chain is:

1. Server starts Python control plane and FL server
2. Jetson `DeviceAgent`s start
3. `DeviceAgent` sends protobuf `DEV_AD`
4. Python control plane replies with raw `SystemInfo`
5. Python control plane publishes `CONT_START`
6. `DeviceAgent` receives it and runs compose
7. compose starts worker container
8. `launch_worker.sh` reconstructs worker CLI
9. `distributed/worker/main.py` loads config + partition + model
10. worker builds `ZMQClientCommManager`
11. real FedLoRA `Client` starts
12. worker FL payload goes to local `DeviceAgent`
13. `DeviceAgent` relays to server payload ports
14. real FedLoRA `Server` handles messages
15. server broadcast goes back through `DeviceAgent`
16. course completes, `finish` propagates, workers exit

---

## 10. How to Navigate the Repo Quickly

If a future Claude session needs to understand the repo fast, the recommended reading order is:

### A. Architecture and current state
1. latest milestone report
2. this architecture guide
3. `DISTRIBUTED_DEPLOYMENT_PLAN_v2.3_patched.md`

### B. Python runtime entry points
4. `distributed/server/main.py`
5. `distributed/worker/main.py`

### C. Communication path
6. `distributed/comm/zmq_server_comm.py`
7. `distributed/comm/zmq_client_comm.py`
8. `distributed/control_plane/service.py`
9. `distributed/control_plane/proto_bridge.py`

### D. Real FL semantics
10. `federatedscope/core/workers/server.py`
11. `federatedscope/core/workers/client.py`

### E. Device-side launch and relay
12. AegisGov `controlmessages.proto`
13. AegisGov `device_agent.cpp` / `device_agent.h`
14. AegisGov `misc.cpp`
15. `distributed/docker/docker-compose.aegisgov.jetson.yml`
16. `distributed/docker/launch_worker.sh`

### F. Runtime data and experiment baseline
17. `distributed/configs/client_manifest.json`
18. `2_yamls/fedit/fedit_distributed.yaml`
19. `exp/distributed_fedit/config.yaml`
20. baseline `1_scripts/` and `2_yamls/`

---

## 11. What Is Legacy / Transitional / Still Worth Reviewing

Not everything in the repo is equally current now.

Potentially legacy or transitional areas include:
- gate-era harness tests
- early reduced-mode control-plane validation helpers
- direct-mode-only transitional paths
- stale runbooks from pre-protobuf or pre-relay stages

These are still useful for historical debugging, but they are not the primary runtime path anymore.

---

## 12. Most Important Current Open Technical Themes

A future Claude session should know these are the most relevant next topics:

1. **LoRA-adapter-only payloads**
   - LoRA is active in training
   - payload still appears too large
   - communication path likely still needs correction

2. **Cleanup**
   - gates and transitional scaffolding should be reviewed and possibly removed/archived

3. **Milestone 5 baseline comparison**
   - compare real-device distributed path against established simulated/server-only runs using `1_scripts/` and `2_yamls/`

4. **Milestone 6 method expansion**
   - reuse the same architecture for HetLoRA, FAH-QLoRA, and AdaSparse variants

---

## 13. Bottom Line

The repo should now be understood as a layered system:

- **FedLoRA** = real FL semantics
- **distributed/** = Python deployment/orchestration/bridge layer
- **AegisGov-master** = real C++ device-side lifecycle and relay
- **fedlora_runtime** = per-device runtime data root

If you remember only one thing, remember this:

> **The FL logic lives in FedLoRA. The orchestration and relay layer exists only to let that same logic run on real Jetsons through the real device-agent path.**
