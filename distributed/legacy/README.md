# distributed/legacy/

Archived artifacts from the gate/milestone-1/milestone-2 proof phase.

Nothing in this tree is on the active runtime path. It is kept in the
working tree (not deleted) so that:

- historical gate tests can still be read when debugging wire-format issues
- the Python ``DeviceAgentStub`` is available as a reference for the
  relay contract that the real C++ ``device_agent`` now implements
- Milestone 1 preflight scripts can be reread when bringing up a new device

Do **not** import from ``distributed.legacy.*`` in active runtime code.
The canonical runtime path is documented in
``docs/reproduction.md`` and
``docs/ARCHITECTURE.md``.

Contents:

- ``tests/`` — Gate 0-6 test harness + integration harness
  (uses the in-process DeviceAgentStub; superseded by real C++ ``device_agent``)
- ``comm/device_agent_stub.py`` — Python stub of the relay boundary
- ``configs/cluster_info.json`` — pre-manifest device/port mapping
- ``configs/fedit_partition_prep.yaml`` — harness-only FedIT config
- ``docker/docker-compose.jetson.yml`` — superseded by
  ``distributed/docker/docker-compose.aegisgov.jetson.yml``
- ``scripts/remote_probe_*.sh`` — Milestone 1 on-device probes
- ``scripts/launch_worker_dry_run.sh`` — preflight dry-run
- ``scripts/print_remote_runtime_commands.sh`` — preflight scp/ssh template
- ``GATE0_AUDIT.md`` — Gate 0 comm-manager contract audit
