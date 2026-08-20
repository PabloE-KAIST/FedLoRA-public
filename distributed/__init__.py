"""
FedLoRA Distributed Deployment Package

This package provides the infrastructure for running FedLoRA across real
distributed devices. The AegisGov C++ ``device_agent`` owns container
lifecycle and relays FL payload between workers and the FL server; ZeroMQ
is used for all transports.

Architecture (with ``--dev_system_port_offset`` ``K``):
- Control plane: server REP/PUB on ``60101+K`` / ``60102+K``;
  device-agent worker-facing REP/PUB on ``60111+K`` / ``60112+K``.
- FL payload plane (relay mode, current default):
  worker -> local device_agent -> FL server, and back.

Canonical reproduction: ``docs/reproduction.md``.
Architecture map: ``docs/ARCHITECTURE.md``.

Minimal wiring (server side):

    from distributed.control_plane import ControlPlaneService
    from distributed.comm import ZMQServerCommManager

    cp = ControlPlaneService(manifest_path="distributed/configs/client_manifest.json")
    routing_table = cp.build_client_routing_table()
    server_comm = ZMQServerCommManager(
        host="0.0.0.0",
        port_offset=0,
        client_routing_table=routing_table,
    )
    server = Server(..., comm_manager=server_comm)
"""

__version__ = "0.1.0"

