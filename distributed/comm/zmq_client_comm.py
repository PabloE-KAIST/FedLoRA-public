"""
ZeroMQ-based Communication Manager for FedLoRA Client (Worker).

ARCHITECTURE NOTE:
This comm manager targets the device-agent relay boundary, NOT direct
connection to the FL server. The communication path is:

    Worker Container <-> Device Agent <-> FL Server

The worker sends FL messages to the local device_agent (ports 60011/60012),
which relays them to the server. The worker NEVER connects directly to
the server's FL payload ports (60001/60002).

Message Envelope:
- Outbound: TO_CONTROLLER <routing_envelope><serialized_FL_message>
- Inbound: Subscribed topic (container_name) + serialized_FL_message

This design preserves the device-agent-managed process boundary as required
by the v2.3 architecture.

---------------------------------------------------------------------------
ZeroMQ discipline (v2.3) — thread ownership, locking, timeouts, recovery
---------------------------------------------------------------------------

**Sockets**
  - ``REQ`` (main thread only): sends relay envelopes to the device_agent
    REP endpoint; each send is paired with a blocking recv for ACK (strict
    REQ/REP).
  - ``SUB`` (dedicated receive thread): subscribes to
    ``{container_name}|`` on the device_agent PUB port; pushes decoded
    ``Message`` objects into ``_recv_queue``.

**Thread ownership**
  - ``REQ``: owned and operated **only** on the thread that calls ``send()``
    (FedLoRA Client main thread in normal use). Access is serialized with
    ``_send_lock``.
  - ``SUB``: owned by the background ``_receive_loop`` thread.

**Locking**
  - ``_send_lock`` wraps the full REQ send + ACK recv sequence so REQ is never
    interleaved across threads.
  - ``receive()`` for FedLoRA blocks on ``_recv_queue`` (not on SUB directly).

**Timeouts**
  - REQ: ``SNDTIMEO`` / ``RCVTIMEO`` from constructor args.
  - SUB loop: ``RCVTIMEO``; empty polls yield ``zmq.Again`` and continue.

**Recovery after timeout or broken REQ/REP**
  - **Current limitation:** failed REQ operations can leave the REQ socket in
    an undefined strict REQ/REP state; there is **no** automatic socket
    teardown/reconnect yet.
  - **TODO (align with v2.3):** on send/recv failure or persistent
    ``Again``/``ETERM``, close and recreate the REQ socket (and optionally
    resynchronize with the device_agent) before the next FL send.

**FedLoRA ``send`` / ``receiver`` expectations**
  - The Client sends one logical FL message per ``send()`` call with
    ``receiver`` typically ``[server_id]`` (or neighbor lists for secret
    sharing). This manager forwards the serialized ``Message`` as-is; it does
    not interpret ``receiver`` for fan-out (device_agent handles routing).

**``host`` / ``port`` (FedLoRA compatibility)**
  - ``host`` = ``device_agent_host``; ``port`` = worker-facing REP port
    (60011 + offset). Matches ``GATE0_AUDIT.md`` conclusions for Client-side
    logging and assign-ID paths.
"""

import logging
import threading
import time
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


class ZMQClientCommManager:
    """
    ZeroMQ-based communication manager for FedLoRA Client (Worker).

    See module docstring for architecture, thread ownership, locking,
    timeout behavior, and REQ recovery limitations (TODO).

    Contract summary (FedLoRA Client):
    - ``send(message: Message) -> None``
    - ``receive() -> Message``
    - ``add_neighbors`` / ``get_neighbors``
    - ``neighbors``, ``host``, ``port``
    - ``close()`` for tests and clean process shutdown (not called by base
      FedLoRA Client code).

    ZMQ pattern (device_agent-facing):
    - REQ → device_agent REP (default 60011 + offset)
    - SUB → device_agent PUB (default 60012 + offset)
    """

    # Device agent ports (worker-facing)
    DEFAULT_DEVICE_AGENT_REP_PORT = 60011
    DEFAULT_DEVICE_AGENT_PUB_PORT = 60012

    DEFAULT_FL_SERVER_REP_PORT = 60001
    DEFAULT_FL_SERVER_PUB_PORT = 60002

    def __init__(
        self,
        client_id: int,
        container_name: str,
        device_agent_host: str = "localhost",
        device_agent_port_offset: int = 0,
        recv_timeout_ms: int = 1000,
        send_timeout_ms: int = 5000,
        fl_server_host: Optional[str] = None,
        fl_server_port_offset: int = 0,
        nic: str = "",
        shared_nic: bool = False,
    ):
        """
        Initialize ZMQ Client Comm Manager.

        When ``fl_server_host`` is provided the client connects directly to the
        FL server's payload ports (60001/60002 + offset) instead of the local
        device_agent.  The device_agent still handles control-plane lifecycle;
        only FL payload traffic takes the direct path.
        """
        self.client_id = client_id
        self.container_name = container_name
        self.device_agent_host = device_agent_host

        # Determine target endpoints
        if fl_server_host:
            self._target_host = fl_server_host
            self._req_port = self.DEFAULT_FL_SERVER_REP_PORT + fl_server_port_offset
            self._sub_port = self.DEFAULT_FL_SERVER_PUB_PORT + fl_server_port_offset
            self._direct_mode = True
        else:
            self._target_host = device_agent_host
            self._req_port = self.DEFAULT_DEVICE_AGENT_REP_PORT + device_agent_port_offset
            self._sub_port = self.DEFAULT_DEVICE_AGENT_PUB_PORT + device_agent_port_offset
            self._direct_mode = False

        # Compat aliases
        self.device_agent_req_port = self._req_port
        self.device_agent_sub_port = self._sub_port

        # Required attributes for FedLoRA Client compatibility
        self.host = self._target_host
        self.port = self._req_port
        self.neighbors: Dict[int, Any] = {}
        
        # Timeouts
        self.recv_timeout_ms = recv_timeout_ms
        self.send_timeout_ms = send_timeout_ms
        
        # ZMQ state
        self._context = None
        self._req_socket = None  # For sending to device_agent
        self._sub_socket = None  # For receiving from device_agent
        self._initialized = False
        
        # Message queue for received messages
        self._recv_queue: Queue = Queue()
        self._recv_thread = None
        self._running = False
        
        # Thread safety for REQ/REP sequencing
        self._send_lock = threading.Lock()
        
        # Bandwidth measurement
        from distributed.worker.bandwidth_measure import BandwidthMeasure
        self.bandwidth_measure = BandwidthMeasure(nic=nic, shared_nic=shared_nic)
        self._last_recv_payload_size: int = 0

        target_label = (
            f"FL server (direct) {fl_server_host}:{self._req_port}"
            if self._direct_mode
            else f"device_agent {device_agent_host}:{self._req_port}"
        )
        logger.info(
            f"ZMQClientCommManager created: client_id={client_id}, "
            f"container={container_name}, target={target_label}"
        )
    
    def _initialize(self) -> None:
        """Initialize ZMQ sockets and start receive thread."""
        if self._initialized:
            return
        
        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required. Install with: pip install pyzmq")
        
        self._context = zmq.Context()
        
        # REQ socket for sending FL messages
        self._req_socket = self._context.socket(zmq.REQ)
        self._req_socket.setsockopt(zmq.SNDTIMEO, self.send_timeout_ms)
        self._req_socket.setsockopt(zmq.RCVTIMEO, self.send_timeout_ms * 2)
        self._req_socket.setsockopt(zmq.LINGER, 1000)
        req_addr = f"tcp://{self._target_host}:{self._req_port}"
        self._req_socket.connect(req_addr)

        # SUB socket for receiving FL broadcasts
        self._sub_socket = self._context.socket(zmq.SUB)
        self._sub_socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout_ms)
        self._sub_socket.setsockopt(zmq.LINGER, 1000)
        sub_addr = f"tcp://{self._target_host}:{self._sub_port}"
        self._sub_socket.connect(sub_addr)

        container_topic = f"{self.container_name}|"
        self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, container_topic)

        target_label = "FL server (direct)" if self._direct_mode else "device_agent"
        logger.info(
            f"REQ connected to {target_label} at {req_addr}, "
            f"SUB connected at {sub_addr}, subscribed to [{container_topic}]"
        )
        
        # Start receive thread
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name=f"ZMQClient-{self.container_name}-Recv"
        )
        self._recv_thread.start()
        
        self._initialized = True
    
    def _receive_loop(self) -> None:
        """Background thread to receive messages from device_agent or server."""
        import zmq

        while self._running:
            try:
                raw_msg = self._sub_socket.recv()

                # Parse container topic and payload.
                # Direct mode format: "container_name|<relay_envelope>"
                # Relay mode format:  "container_name| <relay_envelope>"
                #   (device_agent's sendMessageToContainer inserts a space).
                # Relay payload is raw bytes by default.  Auto-detect
                # legacy base64 by first-byte range: raw envelopes start
                # with a MessageType byte (< 0x20), base64 output is
                # ASCII (>= 0x2B).  See ``proto_bridge.packaged_msg_to_proto``.
                pipe_idx = raw_msg.find(b'|')
                if pipe_idx < 0:
                    logger.warning("Received message without topic separator")
                    continue

                offset = pipe_idx + 1
                # Skip the space that device_agent inserts after "|"
                if offset < len(raw_msg) and raw_msg[offset:offset + 1] == b' ':
                    offset += 1
                envelope = raw_msg[offset:]

                if not self._direct_mode and envelope and envelope[0] >= 0x20:
                    import base64
                    envelope = base64.b64decode(envelope)

                # Unpack relay envelope and deserialize FL message
                msg_type, fl_payload = unpack_relay_envelope(envelope)

                if msg_type != MessageType.FL_PAYLOAD:
                    logger.debug(f"Received non-FL message type: {msg_type}")
                    continue

                message = deserialize_message(fl_payload)

                self._last_recv_payload_size = len(envelope)
                self._recv_queue.put(message)

                logger.debug(
                    f"Client {self.client_id} received FL message: "
                    f"type={message.msg_type}, from={message.sender}"
                )

            except zmq.Again:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"Error in receive loop: {e}")
    
    def send(self, message: Message) -> None:
        """
        Send a FedLoRA Message through the device_agent relay or directly.

        In relay mode (through device_agent):
          The message is prefixed with ``"FWD_CTRL "`` so the device_agent's
          HandleDeviceMessages dispatches to ForwardToController, which strips
          the prefix and forwards the raw relay envelope to the server.

        In direct mode:
          The raw relay envelope is sent directly to the server REP socket.
        """
        self._initialize()

        with self._send_lock:
            try:
                message = lora_payload.apply_if_enabled(message, role="client")
                fl_payload = serialize_message(message)

                envelope = pack_relay_envelope(
                    msg_type=MessageType.TO_CONTROLLER,
                    source_container=self.container_name,
                    payload=fl_payload,
                )

                payload_size = len(envelope)
                self.bandwidth_measure.mark_upload_start()

                if self._direct_mode:
                    self._req_socket.send(envelope)
                else:
                    wire_msg = b"FWD_CTRL " + envelope
                    self._req_socket.send(wire_msg)

                # Wait for ACK (REQ/REP discipline)
                ack = self._req_socket.recv()

                self.bandwidth_measure.mark_upload_end(payload_bytes=payload_size)
                self.bandwidth_measure.mark_download_start()

                target = "server (direct)" if self._direct_mode else "device_agent"
                logger.debug(
                    f"Client {self.client_id} sent FL message via {target}: "
                    f"type={message.msg_type}, to={message.receiver}"
                )

            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                raise
    
    def receive(self) -> Message:
        """
        Receive the next FedLoRA Message (blocking).

        Also marks download end for bandwidth measurement when a message
        arrives. Download start is marked after the preceding send().
        """
        self._initialize()

        while True:
            try:
                message = self._recv_queue.get(timeout=1.0)
                self.bandwidth_measure.mark_download_end(
                    payload_bytes=self._last_recv_payload_size,
                )
                return message
            except Empty:
                if not self._running:
                    raise RuntimeError("Comm manager stopped")
    
    def add_neighbors(self, neighbor_id: int, address: Any = None) -> None:
        """Add a neighbor to the neighbors dict."""
        self.neighbors[neighbor_id] = address
        logger.debug(f"Added neighbor {neighbor_id}: {address}")
    
    def get_neighbors(self, neighbor_id: int = None) -> Any:
        """Get neighbor information."""
        if neighbor_id is None:
            return self.neighbors
        return self.neighbors.get(neighbor_id)
    
    def close(self) -> None:
        """Clean up ZMQ resources."""
        self._running = False
        
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        
        if self._req_socket:
            self._req_socket.close()
        if self._sub_socket:
            self._sub_socket.close()
        if self._context:
            self._context.term()
        
        logger.info(f"ZMQClientCommManager closed for {self.container_name}")
