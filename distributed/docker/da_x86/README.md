# Device Agent binary — not redistributed

`Dockerfile.da.x86` (one directory up) builds a container image around the
**AegisGov `DeviceAgent`**, the C++ device-side agent that handles device
registration, container lifecycle, and FL payload relay.

That binary and its bundled shared libraries are **not part of this repository**
and are not redistributed here. They belong to a separate project with its own
licensing.

## What is expected here

To build `Dockerfile.da.x86`, place the agent build output in this directory:

```
distributed/docker/da_x86/
├── DeviceAgent              # the compiled agent binary
└── lib*.so*                 # any shared libraries it links against
```

Obtain or build these from the AegisGov project, then build as usual.

## The integration contract

FedLoRA does not depend on the agent's internals — only on the protobuf control
plane. The schema this repository speaks is versioned in:

- `distributed/proto/` — the protobuf schema and generated Python bindings

The division of responsibility (unchanged by this omission):

- **FedLoRA** owns all federated-learning semantics — round logic, aggregation,
  client selection — in `federatedscope/core/workers/`.
- **The device agent** owns device-side lifecycle and relay only. It carries no
  FL round logic.
- **Workers** stay thin: `distributed/worker/main.py` is a small wrapper around
  the standard FedLoRA client path.

Because the coupling is limited to the protobuf schema, the agent can be
substituted with any implementation that speaks it. See
[`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md).
