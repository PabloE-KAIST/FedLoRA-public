"""
Device Agent Stub for testing and development.

This is a Python implementation of the device_agent relay behavior,
intended for testing and local development. It mimics the C++ device_agent's
message relay functionality without the container lifecycle management.

IMPORTANT: This is a TEST STUB, not production code.
Production deployment should use the actual C++ device_agent from AegisGov.

Architecture:
    Worker Container                Device Agent Stub              FL Server / Control Plane
          │                               │                               │
          │ ──── TO_CONTROLLER ──────────>│                               │
          │                               │ ──── relay ──────────────────>│
          │                               │                               │
          │                               │<───── FL_PAYLOAD ─────────────│
          │<────── FL_PAYLOAD ────────────│                               │

Port assignments (per v2.3):
    - 60011: Device agent REP (receives from worker)
    - 60012: Device agent PUB (broadcasts to worker)
    - 60001: Server REP (receives from device agent)
    - 60002: Server PUB (broadcasts to device agents)
    - 60101: Control plane REP
    - 60102: Control plane PUB
"""

import logging
import threading
import time
from typing import Dict, Optional, Set
from dataclasses import dataclass, field

from distributed.comm.serialization import (
    MessageType,
    pack_relay_envelope,
    unpack_relay_envelope,
)

logger = logging.getLogger(__name__)


@dataclass
class ContainerInfo:
    """Information about a managed container."""
    name: str
    client_id: int
    status: str = "running"


class DeviceAgentStub:
    """
    Device Agent Stub for testing FL message relay.
    
    This stub provides:
    1. REP socket (60011) for receiving messages from local workers
    2. PUB socket (60012) for broadcasting messages to local workers
    3. REQ socket for forwarding to server's FL payload plane (60001)
    4. SUB socket for receiving from server's FL payload plane (60002)
    5. REQ socket for control plane communication (60101)
    6. SUB socket for control plane broadcasts (60102)
    
    NOTE: This is a test stub. It does not:
    - Manage actual Docker containers
    - Implement full protobuf protocol
    - Handle all device_agent message types
    """
    
    # Device-side ports (worker-facing)
    DEFAULT_WORKER_REP_PORT = 60011
    DEFAULT_WORKER_PUB_PORT = 60012
    
    # Server-side ports (FL payload)
    DEFAULT_SERVER_REP_PORT = 60001
    DEFAULT_SERVER_PUB_PORT = 60002
    
    # Control plane ports
    DEFAULT_CONTROL_REP_PORT = 60101
    DEFAULT_CONTROL_PUB_PORT = 60102
    
    def __init__(
        self,
        device_name: str,
        server_host: str,
        control_plane_host: str,
        local_host: str = "0.0.0.0",
        port_offset: int = 0,
    ):
        """
        Initialize Device Agent Stub.
        
        Args:
            device_name: Name of this device agent
            server_host: FL server host address
            control_plane_host: Control plane service host address
            local_host: Local host to bind worker-facing sockets
            port_offset: Port offset for all ports
        """
        self.device_name = device_name
        self.server_host = server_host
        self.control_plane_host = control_plane_host
        self.local_host = local_host
        self.port_offset = port_offset
        
        # Compute ports
        self.worker_rep_port = self.DEFAULT_WORKER_REP_PORT + port_offset
        self.worker_pub_port = self.DEFAULT_WORKER_PUB_PORT + port_offset
        self.server_rep_port = self.DEFAULT_SERVER_REP_PORT + port_offset
        self.server_pub_port = self.DEFAULT_SERVER_PUB_PORT + port_offset
        self.control_rep_port = self.DEFAULT_CONTROL_REP_PORT + port_offset
        self.control_pub_port = self.DEFAULT_CONTROL_PUB_PORT + port_offset
        
        # Container registry
        self.containers: Dict[str, ContainerInfo] = {}
        self._containers_lock = threading.Lock()
        
        # ZMQ state
        self._context = None
        self._worker_rep_socket = None
        self._worker_pub_socket = None
        self._server_req_socket = None
        self._server_sub_socket = None
        self._control_req_socket = None
        self._control_sub_socket = None
        
        self._running = False
        self._threads = []
        
        logger.info(
            f"DeviceAgentStub created: device={device_name}, "
            f"server={server_host}, control_plane={control_plane_host}"
        )
    
    def register_container(self, container_name: str, client_id: int) -> None:
        """
        Register a container for routing.
        
        Args:
            container_name: Container name
            client_id: FL client ID
        """
        with self._containers_lock:
            self.containers[container_name] = ContainerInfo(
                name=container_name,
                client_id=client_id,
            )
        logger.info(f"Registered container: {container_name} (client {client_id})")
    
    def start(self) -> None:
        """Start the device agent stub."""
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required. Install with: pip install pyzmq")
        
        self._context = zmq.Context()
        
        # Worker-facing sockets
        self._worker_rep_socket = self._context.socket(zmq.REP)
        self._worker_rep_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._worker_rep_socket.bind(f"tcp://{self.local_host}:{self.worker_rep_port}")
        logger.info(f"Worker REP bound to tcp://{self.local_host}:{self.worker_rep_port}")
        
        self._worker_pub_socket = self._context.socket(zmq.PUB)
        self._worker_pub_socket.bind(f"tcp://{self.local_host}:{self.worker_pub_port}")
        logger.info(f"Worker PUB bound to tcp://{self.local_host}:{self.worker_pub_port}")
        
        # Server-facing sockets (FL payload plane)
        self._server_req_socket = self._context.socket(zmq.REQ)
        self._server_req_socket.setsockopt(zmq.SNDTIMEO, 5000)
        self._server_req_socket.setsockopt(zmq.RCVTIMEO, 10000)
        self._server_req_socket.connect(f"tcp://{self.server_host}:{self.server_rep_port}")
        logger.info(f"Server REQ connected to tcp://{self.server_host}:{self.server_rep_port}")
        
        self._server_sub_socket = self._context.socket(zmq.SUB)
        self._server_sub_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._server_sub_socket.connect(f"tcp://{self.server_host}:{self.server_pub_port}")
        # Subscribe to messages for this device
        topic = f"{self.device_name}|"
        self._server_sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        logger.info(f"Server SUB connected, subscribed to '{topic}'")
        
        self._running = True
        
        # Start relay threads
        self._threads = [
            threading.Thread(target=self._worker_to_server_loop, daemon=True, name="W2S"),
            threading.Thread(target=self._server_to_worker_loop, daemon=True, name="S2W"),
        ]
        for t in self._threads:
            t.start()
        
        logger.info(f"DeviceAgentStub {self.device_name} started")
    
    def stop(self) -> None:
        """Stop the device agent stub."""
        self._running = False
        
        for t in self._threads:
            t.join(timeout=2.0)
        
        for sock in [
            self._worker_rep_socket,
            self._worker_pub_socket,
            self._server_req_socket,
            self._server_sub_socket,
            self._control_req_socket,
            self._control_sub_socket,
        ]:
            if sock:
                sock.close()
        
        if self._context:
            self._context.term()
        
        logger.info(f"DeviceAgentStub {self.device_name} stopped")
    
    def _worker_to_server_loop(self) -> None:
        """Relay messages from worker to server."""
        import zmq
        
        while self._running:
            try:
                # Receive from worker
                envelope = self._worker_rep_socket.recv()
                
                # Unpack to verify it's TO_CONTROLLER
                msg_type, payload = unpack_relay_envelope(envelope)
                
                if msg_type != MessageType.TO_CONTROLLER:
                    logger.warning(f"Unexpected message type from worker: {msg_type}")
                    self._worker_rep_socket.send(b"ERROR")
                    continue
                
                # Forward to server (re-pack with device info if needed)
                self._server_req_socket.send(envelope)
                
                # Get server ACK
                server_ack = self._server_req_socket.recv()
                
                # Send ACK back to worker
                self._worker_rep_socket.send(server_ack)
                
                logger.debug(f"Relayed worker->server message")
                
            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error in worker->server relay: {e}")
                    try:
                        self._worker_rep_socket.send(b"ERROR")
                    except:
                        pass
    
    def _server_to_worker_loop(self) -> None:
        """Relay messages from server to worker."""
        import zmq
        
        while self._running:
            try:
                # Receive from server PUB
                # Format: "device_name|container_name|<envelope>"
                raw_msg = self._server_sub_socket.recv()
                
                # Parse routing prefix
                first_pipe = raw_msg.find(b'|')
                if first_pipe < 0:
                    continue
                    
                second_pipe = raw_msg.find(b'|', first_pipe + 1)
                if second_pipe < 0:
                    continue
                
                # device_name = raw_msg[:first_pipe].decode('utf-8')
                container_name = raw_msg[first_pipe + 1:second_pipe].decode('utf-8')
                envelope = raw_msg[second_pipe + 1:]
                
                # Forward to worker with container topic
                worker_msg = f"{container_name}|".encode('utf-8') + envelope
                self._worker_pub_socket.send(worker_msg)
                
                logger.debug(f"Relayed server->worker message to {container_name}")
                
            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error in server->worker relay: {e}")


def run_stub(
    device_name: str,
    server_host: str,
    control_plane_host: str,
    containers: Dict[str, int],
):
    """
    Run a device agent stub for testing.
    
    Args:
        device_name: Device name
        server_host: Server host
        control_plane_host: Control plane host
        containers: Dict mapping container_name -> client_id
    """
    stub = DeviceAgentStub(
        device_name=device_name,
        server_host=server_host,
        control_plane_host=control_plane_host,
    )
    
    for container_name, client_id in containers.items():
        stub.register_container(container_name, client_id)
    
    stub.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stub.stop()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Device Agent Stub')
    parser.add_argument('--device-name', required=True, help='Device name')
    parser.add_argument('--server-host', default='localhost', help='Server host')
    parser.add_argument('--control-plane-host', default='localhost', help='Control plane host')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    run_stub(
        device_name=args.device_name,
        server_host=args.server_host,
        control_plane_host=args.control_plane_host,
        containers={},
    )
