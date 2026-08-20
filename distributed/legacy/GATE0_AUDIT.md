# Gate 0: CommManager Contract Audit (v2.3)

**Status:** COMPLETE (documentation)  
**Date:** 2026-04-11  
**Governing contract:** `docs/ARCHITECTURE.md`  
**Audited FedLoRA sources:**
- `federatedscope/core/workers/server.py`
- `federatedscope/core/workers/client.py`

**Audited distributed transport:**
- `distributed/comm/zmq_client_comm.py`
- `distributed/comm/zmq_server_comm.py`

---

## 1. FedLoRA Server: comm_manager surface area

| Surface | Relied on? | Notes |
|--------|------------|--------|
| `comm_manager.host` | **No** | Server code does not read `host` on the comm manager. `ZMQServerCommManager` exposes `self.host` for binding only. |
| `comm_manager.port` | **No** | Server does not read a single `port`; `ZMQServerCommManager` uses `rep_port` / `pub_port`. |
| `comm_manager.neighbors` | **Yes** | `list(self.comm_manager.neighbors.keys())` used as broadcast receiver lists; `add_neighbors` on join. |
| `self.local_address` | **N/A** | `local_address` is a **Client** attribute, not Server. |
| `close()` / cleanup | **Not invoked** | Base `Server`/`Client` do not call `close()` on shutdown. Injected managers **should** implement `close()` for process hygiene (ZMQ managers do). |
| `receive()` blocking | **Yes** | Main loop: `msg = self.comm_manager.receive()`; blocking is assumed. No timeout argument in base code (optional `Timeout` context elsewhere). |
| `send()` with `receiver is None` | **Yes** | When `message.receiver` is unset, server code passes `receiver=list(self.comm_manager.neighbors.keys())` in several paths — broadcast = “all registered neighbors”. |
| `send()` with `receiver` a `list` | **Yes** | Standard broadcast / targeted send. |
| Monitor hooks / byte counts on send/receive | **No** | Injected path bypasses `StandaloneCommManager`/`gRPCCommManager` monitor wiring. ZMQ managers do **not** call `monitor` today. |

---

## 2. FedLoRA Client: comm_manager surface area

| Surface | Relied on? | Notes |
|--------|------------|--------|
| `comm_manager.host` | **Yes** | `callback_funcs_for_assign_id` logs `self.comm_manager.host` (dynamic ID path). **Static-ID Phase 1** avoids `assign_client_id` messages; logging still valid if that path runs. |
| `comm_manager.port` | **Yes** | Same as `host` — assign-ID logging and **construction of `local_address`** in gRPC path. Injected `ZMQClientCommManager` sets `port` to the worker-facing device_agent REP port (60011+offset) so the attribute exists and is stable. |
| `comm_manager.neighbors` | **Yes** | Iterated for secret-sharing sends when `use_ss` is enabled; `add_neighbors` for server. |
| `self.local_address` | **Yes** | Set from `kwargs.get('local_address', None)` when `comm_manager` is injected; otherwise derived from gRPC comm manager. Used in join / address exchange paths. |
| `close()` / cleanup | **Not invoked** | Same as Server; ZMQ client manager implements `close()` for tests and manual shutdown. |
| `receive()` blocking | **Yes** | `run()` loop: blocking `receive()` expected. |
| `send()` receivers | **Yes** | Client sends with `receiver=[self.server_id]` or neighbor lists for SS; **expects list or int-compatible** usage consistent with `Message`. |
| Monitor hooks / byte counts | **No** | Not used on injected ZMQ path. |

---

## 3. ZeroMQ REQ/REP discipline (v2.3)

### 3.1 `ZMQClientCommManager`

| Topic | Policy |
|-------|--------|
| **Socket roles** | `REQ` → device_agent REP (outbound FL wrapped in `TO_CONTROLLER` envelope). `SUB` → device_agent PUB (inbound FL after topic strip). |
| **Thread ownership** | **Main thread:** `REQ` — all `send()` calls run on the caller thread and are guarded by `_send_lock`. **Background thread (`_receive_loop`):** `SUB` only — recv and push `Message` into `_recv_queue`. |
| **Serialization** | All `REQ` send + blocking `REQ` recv for ACK are inside `_send_lock` so REQ socket is not interleaved across threads. |
| **Timeouts** | `SNDTIMEO` / `RCVTIMEO` set on REQ; SUB uses `RCVTIMEO` in recv loop (`zmq.Again` → continue). `receive()` for FedLoRA blocks on `_recv_queue.get` (1s timeout) until an FL message arrives. |
| **Recovery after timeout / broken REQ** | **Not implemented.** A timed-out or half-open REQ can leave the socket in an invalid strict REQ/REP state. **TODO (v2.3):** on `REQ` failure, close and recreate REQ socket (and optionally resync with device_agent) before the next send. |

### 3.2 `ZMQServerCommManager`

| Topic | Policy |
|-------|--------|
| **Socket roles** | `REP` ← device_agent (inbound worker traffic). `PUB` → device_agents (outbound FL to routed topics). |
| **Thread ownership** | **Background thread (`_receive_loop`):** `REP` only. **Main / caller thread:** `PUB` — `send()` uses `_send_lock` around `PUB` operations. |
| **Serialization** | REP recv → ACK → deserialize must stay on the receive thread; PUB sends serialized under `_send_lock`. |
| **Routing table** | **Mandatory for correct delivery:** `client_routing_table` / `_client_routing` maps `client_id` → `{device_name, container_name}` for PUB topic `"{device_name}|{container_name}|"`. Without it, `send()` falls back to placeholder topic names. |
| **Timeouts** | REP uses `RCVTIMEO`; on `zmq.Again`, loop continues. |
| **Recovery after timeout / broken REP** | **Partial:** receive loop catches some errors and attempts `send(b"ERROR")` on REP — this can still fail if the socket is out of FSM. **TODO (v2.3):** formal recovery (recreate REP, resubscribe peers policy) aligned with device_agent expectations. |

---

## 4. Broadcast / `receiver` semantics vs FedLoRA usage

- **Server:** Uses explicit `list(self.comm_manager.neighbors.keys())` when broadcasting; `ZMQServerCommManager.send` also treats `receiver is None` as “all neighbor IDs”. Both align with “all clients” semantics.
- **Client:** Sends primarily to `[server_id]`; secret-sharing paths iterate `neighbors`. `ZMQClientCommManager.send` relays a single message envelope per call (no multi-client fan-out on worker); matches Client usage.

---

## 5. Conclusion (provisional contract)

- The **injected comm_manager contract** documented in this file is **sufficient for Phase 1** FedIT-style flows **provided**:
  - **Client:** `host`, `port`, `neighbors`, `send`, `receive`, `add_neighbors`, `get_neighbors`, and **`close()` for tests** are available.
  - **Server:** `neighbors`, `send`, `receive`, `add_neighbors`, `get_neighbors`, **`close()`**, and **`client_routing_table` semantics** for device-agent PUB routing are available.
- **Not mandatory for correctness today but recommended:** REQ/REP **socket recovery after timeout** (called out as TODO above).
- **Not required:** `comm_manager.host` / `port` on the **server** manager; monitor byte hooks on the ZMQ plane.

## 6. Constructor injection patches

Constructor injection patches in `server.py` / `client.py` remain the approved integration point; **do not expand** unless a new FedLoRA code path introduces additional comm_manager attributes.

---

## 7. Gate 5 validation (v2.3)

Gate 5 exercises the **Server-side** injection path end-to-end:

- `distributed/tests/test_gate5_server.py` instantiates the real `Server` with `ZMQServerCommManager`
- Uses existing `FedAvg` aggregator (no custom round manager / aggregator)
- Proves:
  1. `join_in` → `add_neighbors` → `trigger_for_start` → `broadcast_model_para`
  2. Client upload received → `msg_buffer['train']` populated
  3. `check_and_move_on` → `_perform_federated_aggregation` → `state += 1`
  4. Round 1 `broadcast_model_para` proves rebroadcast

No additional `comm_manager` surface beyond what is documented above is required.

---

## 8. Gate 6 validation (v2.3)

Gate 6 exercises multi-round training with multiple clients:

- `distributed/tests/test_gate6_multiround.py` uses `MultiClientTestHarness` with 2 clients
- `test_two_clients_five_rounds`: proves 2 devices can complete 5 FL rounds
- `test_stable_manifest_mapping_across_reruns`: verifies static ID mapping is deterministic
- `test_loss_trajectory_consistency`: validates aggregation produces bounded model updates

Pass criteria met:
1. **2 devices complete 5 rounds** — `server.state` reaches `total_round_num`
2. **Loss trajectory comparable** — model norms do not explode across rounds
3. **Static manifest stable** — same client IDs join in all reruns
4. **No custom aggregator** — uses existing `FedAvg` path throughout
