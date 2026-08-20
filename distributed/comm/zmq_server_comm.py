"""
ZeroMQ-based Communication Manager for FedLoRA Server.

Operates in one of two modes:

**Direct mode** (``direct_mode=True``):
  Binds its own REP/PUB sockets (60001/60002 + offset).  Workers connect
  directly to these ports.  PUB topics use ``container_name|``.

**Relay mode** (``direct_mode=False``, default for production):
  The C++ device_agent uses the *same* controller ports for both control
  and FL data relay.  In relay mode this class does NOT bind its own
  sockets.  Instead:
    - ``receive()`` reads relay envelopes from
      ``ControlPlaneService._fl_relay_queue``
    - ``send()`` publishes through ``ControlPlaneService.publish_fl_relay()``
  The ``ctrl_service`` constructor argument wires the bridge.

---------------------------------------------------------------------------
ZeroMQ discipline (v2.3) — thread ownership, locking, timeouts, recovery
---------------------------------------------------------------------------

**Direct mode**
  Same as before: REP in background thread, PUB under ``_send_lock``.

**Relay mode**
  No ZMQ sockets owned by this class.  Thread-safety is delegated to the
  ControlPlaneService (PUB lock, queue).
"""

import logging
import threading
from typing import Dict, Any, Optional
from queue import Queue, Empty

from federatedscope.core.message import Message
from distributed.comm.serialization import (
    serialize_message,
    deserialize_message,
    pack_relay_envelope,
    unpack_relay_envelope,
    MessageType,
)
from distributed.comm import lora_payload

logger = logging.getLogger(__name__)


class ZMQServerCommManager:
    """
    ZeroMQ-based communication manager for FedLoRA Server.

    ZMQ pattern (direct mode):
    - REP (default 60001 + offset) — inbound from workers
    - PUB (default 60002 + offset) — outbound to workers

    Relay mode: uses ControlPlaneService's sockets via queues.
    """

    DEFAULT_REP_PORT = 60001
    DEFAULT_PUB_PORT = 60002

    def __init__(
        self,
        host: str = "0.0.0.0",
        port_offset: int = 0,
        recv_timeout_ms: int = 1000,
        send_timeout_ms: int = 5000,
        client_routing_table: Optional[Dict[int, Dict[str, str]]] = None,
        direct_mode: bool = False,
        ctrl_service: Optional[Any] = None,
    ):
        """
        Initialize ZMQ Server Comm Manager.

        Args:
            host: Host to bind REP/PUB sockets (direct mode only).
            port_offset: Added to default FL payload ports (direct mode only).
            recv_timeout_ms: REP ``RCVTIMEO`` (direct mode only).
            send_timeout_ms: PUB ``SNDTIMEO`` (direct mode only).
            client_routing_table: Maps each ``client_id`` to
                ``{"device_name": ..., "container_name": ...}`` for routing.
            direct_mode: When True, bind own sockets; workers connect directly.
            ctrl_service: ControlPlaneService instance for relay mode.
                Required when ``direct_mode=False``.
        """
        self.host = host
        self.rep_port = self.DEFAULT_REP_PORT + port_offset
        self.pub_port = self.DEFAULT_PUB_PORT + port_offset
        self.direct_mode = direct_mode
        self._ctrl_service = ctrl_service

        # Required attribute for FedLoRA Server compatibility
        self.neighbors: Dict[int, Any] = {}

        # Routing table: client_id -> {device_name, container_name}
        self._client_routing: Dict[int, Dict[str, str]] = client_routing_table or {}

        # Timeouts (direct mode)
        self.recv_timeout_ms = recv_timeout_ms
        self.send_timeout_ms = send_timeout_ms

        # ZMQ state (direct mode only)
        self._context = None
        self._rep_socket = None
        self._pub_socket = None
        self._initialized = False

        # Message queue for received messages
        self._recv_queue: Queue = Queue()
        self._recv_thread = None
        self._running = False

        # Thread safety
        self._send_lock = threading.Lock()

        mode_str = "DIRECT" if direct_mode else "RELAY"
        if direct_mode:
            logger.info(
                f"ZMQServerCommManager created ({mode_str}): "
                f"REP={host}:{self.rep_port}, PUB={host}:{self.pub_port}"
            )
        else:
            logger.info(
                f"ZMQServerCommManager created ({mode_str}): "
                f"bridged through ControlPlaneService"
            )

    def set_client_routing(self, routing_table: Dict[int, Dict[str, str]]) -> None:
        self._client_routing = routing_table
        logger.info(f"Client routing table set with {len(routing_table)} entries")

    def add_client_route(
        self,
        client_id: int,
        device_name: str,
        container_name: str,
    ) -> None:
        self._client_routing[client_id] = {
            'device_name': device_name,
            'container_name': container_name,
        }
        logger.debug(f"Added route: client {client_id} -> {device_name}/{container_name}")

    def _initialize(self) -> None:
        """Initialize sockets (direct) or relay bridge, start receive thread."""
        if self._initialized:
            return

        self._running = True

        if self.direct_mode:
            self._initialize_direct()
        else:
            self._initialize_relay()

        self._initialized = True

    def _initialize_direct(self) -> None:
        """Bind own REP/PUB sockets for direct worker connections."""
        import zmq
        import time

        self._context = zmq.Context()

        self._rep_socket = self._context.socket(zmq.REP)
        self._rep_socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._rep_socket.setsockopt(zmq.LINGER, 1000)
        rep_addr = f"tcp://{self.host}:{self.rep_port}"
        self._rep_socket.bind(rep_addr)
        logger.info(f"FL Payload REP socket bound to {rep_addr}")

        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
        self._pub_socket.setsockopt(zmq.LINGER, 1000)
        self._pub_socket.setsockopt(zmq.SNDHWM, 1000)
        pub_addr = f"tcp://{self.host}:{self.pub_port}"
        self._pub_socket.bind(pub_addr)
        logger.info(f"FL Payload PUB socket bound to {pub_addr}")

        self._recv_thread = threading.Thread(
            target=self._receive_loop_direct,
            daemon=True,
            name="ZMQServer-Recv-Direct"
        )
        self._recv_thread.start()
        time.sleep(0.5)

    def _initialize_relay(self) -> None:
        """Set up relay bridge through ControlPlaneService."""
        if self._ctrl_service is None:
            raise RuntimeError(
                "ZMQServerCommManager in relay mode requires ctrl_service"
            )
        self._relay_queue = self._ctrl_service.get_fl_relay_queue()
        self._recv_thread = threading.Thread(
            target=self._receive_loop_relay,
            daemon=True,
            name="ZMQServer-Recv-Relay"
        )
        self._recv_thread.start()
        logger.info("FL relay bridge initialized via ControlPlaneService")

    def _receive_loop_direct(self) -> None:
        """Background thread: receive from own REP socket (direct mode)."""
        import zmq

        while self._running:
            try:
                raw_msg = self._rep_socket.recv()

                msg_type, fl_payload = unpack_relay_envelope(raw_msg)
                self._rep_socket.send(b"ACK")

                if msg_type != MessageType.TO_CONTROLLER:
                    logger.warning(f"Unexpected message type on FL plane: {msg_type}")
                    continue

                message = deserialize_message(fl_payload)
                self._recv_queue.put(message)

                logger.debug(
                    f"Server received FL message (direct): "
                    f"type={message.msg_type}, from={message.sender}"
                )

            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error in direct receive loop: {e}")
                    try:
                        self._rep_socket.send(b"ERROR")
                    except:
                        pass

    def _receive_loop_relay(self) -> None:
        """Background thread: read relay envelopes from ControlPlaneService queue."""
        from queue import Empty as QEmpty

        while self._running:
            try:
                raw_envelope = self._relay_queue.get(timeout=1.0)
            except QEmpty:
                continue

            try:
                msg_type, fl_payload = unpack_relay_envelope(raw_envelope)

                if msg_type != MessageType.TO_CONTROLLER:
                    logger.warning(
                        f"Unexpected relay msg_type: {msg_type} "
                        f"(expected TO_CONTROLLER)"
                    )
                    continue

                message = deserialize_message(fl_payload)
                self._recv_queue.put(message)

                logger.debug(
                    f"Server received FL message (relay): "
                    f"type={message.msg_type}, from={message.sender}"
                )
            except Exception as e:
                if self._running:
                    logger.error(f"Error unpacking relay envelope: {e}")

    def send(self, message: Message) -> None:
        """
        Send a FedLoRA Message to client(s).

        Direct mode: publishes via own PUB socket.
        Relay mode: publishes through ControlPlaneService PUB socket.
        """
        self._initialize()

        receivers = message.receiver
        if receivers is None:
            receivers = list(self.neighbors.keys())
        elif not isinstance(receivers, list):
            receivers = [receivers]

        message = lora_payload.apply_if_enabled(message, role="server")
        fl_payload = serialize_message(message)

        for receiver_id in receivers:
            try:
                route = self._client_routing.get(receiver_id)
                if route is None:
                    logger.warning(
                        f"No routing info for client {receiver_id}, "
                        f"falling back to neighbor address"
                    )
                    neighbor_info = self.neighbors.get(receiver_id, {})
                    route = {
                        'device_name': neighbor_info.get('device_name', f'device_{receiver_id}'),
                        'container_name': neighbor_info.get('container_name', f'fl_worker_{receiver_id}'),
                    }

                device_name = route['device_name']
                container_name = route['container_name']

                envelope = pack_relay_envelope(
                    msg_type=MessageType.FL_PAYLOAD,
                    source_container="server",
                    payload=fl_payload,
                )

                if self.direct_mode:
                    topic = f"{container_name}|"
                    full_msg = topic.encode('utf-8') + envelope
                    with self._send_lock:
                        self._pub_socket.send(full_msg)
                else:
                    # Relay mode: publish through ControlPlaneService
                    self._ctrl_service.publish_fl_relay(
                        device_name, container_name, envelope,
                    )

                logger.debug(
                    f"Server sent FL message to client {receiver_id} "
                    f"via {device_name}/{container_name}: type={message.msg_type}"
                )

            except Exception as e:
                logger.error(f"Failed to send to client {receiver_id}: {e}")

    def receive(self) -> Message:
        """Receive the next FedLoRA Message (blocking)."""
        self._initialize()

        while True:
            try:
                message = self._recv_queue.get(timeout=1.0)
                return message
            except Empty:
                if not self._running:
                    raise RuntimeError("Comm manager stopped")

    def add_neighbors(self, neighbor_id: int, address: Any = None) -> None:
        """Add a neighbor (client) to the neighbors dict."""
        self.neighbors[neighbor_id] = address

        if isinstance(address, dict):
            if 'device_name' in address and 'container_name' in address:
                self.add_client_route(
                    client_id=neighbor_id,
                    device_name=address['device_name'],
                    container_name=address['container_name'],
                )

        logger.info(f"Added neighbor (client) {neighbor_id}")

    def get_neighbors(self, neighbor_id: int = None) -> Any:
        """Get neighbor information."""
        if neighbor_id is None:
            return self.neighbors
        return self.neighbors.get(neighbor_id)

    def close(self) -> None:
        """Clean up resources."""
        self._running = False

        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)

        if self._rep_socket:
            self._rep_socket.close()
        if self._pub_socket:
            self._pub_socket.close()
        if self._context:
            self._context.term()

        logger.info("ZMQServerCommManager closed")
