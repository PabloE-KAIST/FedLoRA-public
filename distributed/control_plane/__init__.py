"""
Control Plane Service for distributed FedLoRA deployment.

Responsibilities:
- Receive DEVICE_ADVERTISEMENT from device agents
- Maintain device registry and manifest validation
- Send CONTAINER_START and CONTAINER_STOP commands
- Never participate in FL message interpretation or aggregation
"""

from distributed.control_plane.service import (
    ControlPlaneService,
    ContainerConfig,
    container_config_from_manifest_client,
)

__all__ = [
    'ControlPlaneService',
    'ContainerConfig',
    'container_config_from_manifest_client',
]
