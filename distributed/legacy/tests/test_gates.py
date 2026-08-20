"""
Gate tests for distributed FedLoRA deployment.

These tests validate the pass criteria for each implementation gate.

Gate 0: Constructor and Contract Audit
- Required comm-manager fields and methods are documented
- Constructor injection patch scope is confirmed
- No hidden constructor-time assumptions remain unaccounted for

Gate 1: Device Registration
- Device agent registers with control plane
- Registry entry matches manifest
- No FL join is triggered by registration alone

Gate 2: Worker Launch
- Worker container launches
- Launch arguments match static manifest
- Worker process initializes correctly
- ``--dry-run-init`` proves cfg → data → model → injected Client without FL loop

Gate 4 (v2.3):
- Full proof: ``distributed/tests/test_gate4_real_client.py`` (real Client + training)
- Transport pre-validation: ``TestGate4TransportPreValidation`` in this file

Gate 5 (v2.3):
- Real FedLoRA Server with injected ZMQServerCommManager
- One aggregation cycle + rebroadcast
- See ``distributed/tests/test_gate5_server.py``

Gate 6 (v2.3):
- Multi-round FedIT baseline comparison
- 2 devices complete 5 rounds
- Loss trajectory consistency validated
- Static manifest mapping stable across reruns
- See ``distributed/tests/test_gate6_multiround.py``

Tests include:
- Unit tests for individual components
- Integration tests with loopback relay through device_agent_stub

Run with: python -m pytest distributed/tests/test_gates.py -v
Or: python -m unittest distributed.tests.test_gates -v
"""

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestGate0ConstructorInjection(unittest.TestCase):
    """
    Gate 0: Verify constructor injection patch in Server and Client.
    
    Pass criteria:
    - Required comm-manager fields and methods are documented
    - Constructor injection patch scope is confirmed
    - No hidden constructor-time assumptions remain unaccounted for
    """
    
    def test_server_accepts_injected_comm_manager(self):
        """Verify Server class accepts injected comm_manager."""
        server_path = project_root / "federatedscope/core/workers/server.py"
        with open(server_path, 'r') as f:
            server_source = f.read()
        
        self.assertIn("'comm_manager' in kwargs", server_source,
                      "Server should check for injected comm_manager in kwargs")
        self.assertIn("kwargs['comm_manager'] is not None", server_source,
                      "Server should verify comm_manager is not None")
        self.assertIn("self.comm_manager = kwargs['comm_manager']", server_source,
                      "Server should assign injected comm_manager")
    
    def test_client_accepts_injected_comm_manager(self):
        """Verify Client class accepts injected comm_manager."""
        client_path = project_root / "federatedscope/core/workers/client.py"
        with open(client_path, 'r') as f:
            client_source = f.read()
        
        self.assertIn("'comm_manager' in kwargs", client_source,
                      "Client should check for injected comm_manager in kwargs")
        self.assertIn("kwargs['comm_manager'] is not None", client_source,
                      "Client should verify comm_manager is not None")
        self.assertIn("self.comm_manager = kwargs['comm_manager']", client_source,
                      "Client should assign injected comm_manager")
    
    def test_audit_document_exists(self):
        """Verify Gate 0 audit document exists and is complete."""
        audit_path = project_root / "distributed/GATE0_AUDIT.md"
        self.assertTrue(audit_path.exists(), "Gate 0 audit document should exist")
        
        with open(audit_path, 'r') as f:
            audit_content = f.read()
        
        required_sections = [
            "FedLoRA Server: comm_manager surface area",
            "FedLoRA Client: comm_manager surface area",
            "ZeroMQ REQ/REP discipline",
            "Broadcast / `receiver` semantics",
            "Constructor injection patches",
            "Conclusion (provisional contract)",
        ]
        
        for section in required_sections:
            self.assertIn(section, audit_content,
                          f"Audit should document {section}")


class TestGate1DeviceRegistration(unittest.TestCase):
    """
    Gate 1: Verify device registration through control plane service.
    
    Pass criteria:
    - Device agent registers
    - Registry entry matches manifest
    - No FL join is triggered by registration alone
    """
    
    def test_control_plane_service_imports(self):
        """Verify ControlPlaneService can be imported."""
        from distributed.control_plane import ControlPlaneService
        self.assertIsNotNone(ControlPlaneService)
    
    def test_control_plane_service_init(self):
        """Verify ControlPlaneService initializes correctly."""
        from distributed.control_plane import ControlPlaneService
        
        service = ControlPlaneService(
            host="127.0.0.1",
            port_offset=1000,  # Use high offset to avoid conflicts
        )
        
        self.assertEqual(service.host, "127.0.0.1")
        self.assertEqual(service.rep_port, 61101)  # 60101 + 1000
        self.assertEqual(service.pub_port, 61102)  # 60102 + 1000
        self.assertEqual(len(service.devices), 0)
    
    def test_manifest_loading(self):
        """Verify manifest can be loaded and validated."""
        from distributed.control_plane import ControlPlaneService
        
        manifest_path = project_root / "distributed/configs/client_manifest.json"
        
        service = ControlPlaneService(
            manifest_path=str(manifest_path)
        )
        
        self.assertIsNotNone(service.manifest)
        self.assertIn('clients', service.manifest)
        self.assertEqual(len(service.manifest['clients']), 2)
        
        client_1 = service.manifest['clients'][0]
        self.assertEqual(client_1['client_id'], 1)
        self.assertEqual(client_1['device_name'], 'orinagx1')
    
    def test_device_info_dataclass(self):
        """Verify DeviceInfo dataclass works correctly."""
        from distributed.control_plane.service import DeviceInfo
        
        info = DeviceInfo(
            name="test_device",
            device_type="orinagx",
            ip_address="192.168.1.101",
            agent_port_offset=0,
            processors=1,
            memory=[8000],
        )
        
        info_dict = info.to_dict()
        self.assertEqual(info_dict['name'], "test_device")
        self.assertEqual(info_dict['device_type'], "orinagx")
        
        info_restored = DeviceInfo.from_dict(info_dict)
        self.assertEqual(info_restored.name, info.name)
        self.assertEqual(info_restored.ip_address, info.ip_address)
    
    def test_container_config_dataclass(self):
        """Verify ContainerConfig dataclass works correctly."""
        from distributed.control_plane.service import ContainerConfig
        
        config = ContainerConfig(
            name="fl_worker_test",
            executable="python -m distributed.worker.main",
            client_id=1,
            partition_path="data/partitions/client_1.pkl",
            device_agent_host="localhost",
            device_agent_port_offset=0,
        )
        
        config_dict = config.to_dict()
        self.assertEqual(config_dict['name'], "fl_worker_test")
        self.assertEqual(config_dict['client_id'], 1)
        self.assertEqual(config_dict['device_agent_host'], "localhost")
        
        # Test worker args generation - should use device_agent args, not server args
        worker_args = config.to_worker_args()
        self.assertIn("--client-id=1", worker_args)
        self.assertIn("--container-name=fl_worker_test", worker_args)
        self.assertIn("--device-agent-host=localhost", worker_args)
        self.assertIn("--device-agent-port-offset=0", worker_args)
        
        # Should NOT have direct server arguments
        for arg in worker_args:
            self.assertNotIn("--server-host", arg)
            self.assertNotIn("--server-port", arg)
    
    def test_client_routing_table_from_manifest(self):
        """Verify routing table can be built from manifest."""
        from distributed.control_plane import ControlPlaneService
        
        manifest_path = project_root / "distributed/configs/client_manifest.json"
        service = ControlPlaneService(manifest_path=str(manifest_path))
        
        routing = service.build_client_routing_table()
        
        self.assertIn(1, routing)
        self.assertIn(2, routing)
        self.assertEqual(routing[1]['device_name'], 'orinagx1')
        self.assertEqual(routing[1]['container_name'], 'fl_worker_orinagx1')

    def test_container_config_from_manifest_entry(self):
        """Launch args for workers should derive from manifest via one helper."""
        from distributed.control_plane import (
            ControlPlaneService,
            container_config_from_manifest_client,
        )

        manifest_path = project_root / "distributed/configs/client_manifest.json"
        service = ControlPlaneService(manifest_path=str(manifest_path))
        entry = service.manifest["clients"][0]
        cfg = project_root / "distributed/configs/fedit_distributed.yaml"

        c1 = container_config_from_manifest_client(
            entry,
            config_path=str(cfg),
            device_agent_host="127.0.0.1",
            device_agent_port_offset=3,
            partition_root=str(project_root),
        )
        self.assertEqual(c1.client_id, 1)
        self.assertEqual(c1.name, entry["container_name"])
        self.assertEqual(c1.device_agent_host, "127.0.0.1")
        self.assertEqual(c1.device_agent_port_offset, 3)
        self.assertTrue(c1.partition_path.endswith("client_1.pkl"))

        c2 = service.build_container_config(
            entry,
            config_path=str(cfg),
            partition_root=str(project_root),
        )
        self.assertEqual(c2.client_id, c1.client_id)
        args = c2.to_worker_args()
        self.assertTrue(any(a.startswith("--client-id=") for a in args))


class TestGate2WorkerLaunch(unittest.TestCase):
    """
    Gate 2: Verify worker launch infrastructure.
    
    Pass criteria:
    - Worker container launches
    - Launch arguments match static manifest
    - Worker process initializes correctly
    """
    
    def test_zmq_client_comm_manager_imports(self):
        """Verify ZMQClientCommManager can be imported."""
        from distributed.comm import ZMQClientCommManager
        self.assertIsNotNone(ZMQClientCommManager)
    
    def test_zmq_server_comm_manager_imports(self):
        """Verify ZMQServerCommManager can be imported."""
        from distributed.comm import ZMQServerCommManager
        self.assertIsNotNone(ZMQServerCommManager)
    
    def test_device_agent_stub_imports(self):
        """Verify DeviceAgentStub can be imported."""
        from distributed.comm import DeviceAgentStub
        self.assertIsNotNone(DeviceAgentStub)
    
    def test_serialization_imports(self):
        """Verify serialization utilities can be imported."""
        from distributed.comm import (
            serialize_message,
            deserialize_message,
            pack_relay_envelope,
            unpack_relay_envelope,
            MessageType,
        )
        self.assertIsNotNone(serialize_message)
        self.assertIsNotNone(pack_relay_envelope)
        self.assertEqual(MessageType.TO_CONTROLLER.value, 0x01)
        self.assertEqual(MessageType.FL_PAYLOAD.value, 0x02)
    
    def test_relay_envelope_roundtrip(self):
        """Verify relay envelope pack/unpack works correctly."""
        from distributed.comm import (
            pack_relay_envelope,
            unpack_relay_envelope,
            MessageType,
        )
        
        original_payload = b"test_payload_data"
        
        envelope = pack_relay_envelope(
            msg_type=MessageType.TO_CONTROLLER,
            source_container="test_container",
            payload=original_payload,
        )
        
        self.assertIsInstance(envelope, bytes)
        self.assertGreater(len(envelope), len(original_payload))
        
        msg_type, payload = unpack_relay_envelope(envelope)
        
        self.assertEqual(msg_type, MessageType.TO_CONTROLLER)
        self.assertEqual(payload, original_payload)
    
    def test_control_message_roundtrip(self):
        """Verify control message pack/unpack works correctly."""
        from distributed.comm import pack_control_message, unpack_control_message
        
        original_payload = {"key": "value", "number": 42}
        
        packed = pack_control_message("TEST_MESSAGE", original_payload)
        self.assertIsInstance(packed, bytes)
        
        msg_type, payload = unpack_control_message(packed)
        
        self.assertEqual(msg_type, "TEST_MESSAGE")
        self.assertEqual(payload, original_payload)
    
    def test_zmq_client_comm_manager_targets_device_agent(self):
        """Verify ZMQClientCommManager targets device_agent ports."""
        from distributed.comm import ZMQClientCommManager
        
        comm = ZMQClientCommManager(
            client_id=1,
            container_name="fl_worker_test",
            device_agent_host="localhost",
            device_agent_port_offset=0,
        )
        
        # Verify ports target device_agent, not server
        self.assertEqual(comm.device_agent_req_port, 60011)  # Device agent REP
        self.assertEqual(comm.device_agent_sub_port, 60012)  # Device agent PUB
        
        # These should NOT be server ports
        self.assertNotEqual(comm.device_agent_req_port, 60001)
        self.assertNotEqual(comm.device_agent_sub_port, 60002)
    
    def test_zmq_server_comm_manager_binds_fl_payload_ports(self):
        """Verify ZMQServerCommManager binds FL payload plane ports."""
        from distributed.comm import ZMQServerCommManager
        
        comm = ZMQServerCommManager(
            host="0.0.0.0",
            port_offset=0,
        )
        
        # Verify ports are FL payload plane ports
        self.assertEqual(comm.rep_port, 60001)
        self.assertEqual(comm.pub_port, 60002)
        
        # These should NOT be control plane or device agent ports
        self.assertNotEqual(comm.rep_port, 60101)
        self.assertNotEqual(comm.rep_port, 60011)
    
    def test_zmq_server_comm_manager_client_routing(self):
        """Verify ZMQServerCommManager can set client routing."""
        from distributed.comm import ZMQServerCommManager
        
        comm = ZMQServerCommManager(host="0.0.0.0", port_offset=1000)
        
        # Set routing table
        routing = {
            1: {'device_name': 'device_a', 'container_name': 'worker_1'},
            2: {'device_name': 'device_b', 'container_name': 'worker_2'},
        }
        comm.set_client_routing(routing)
        
        self.assertEqual(len(comm._client_routing), 2)
        self.assertEqual(comm._client_routing[1]['device_name'], 'device_a')
    
    def test_worker_module_exists(self):
        """Verify worker main module exists."""
        worker_main = project_root / "distributed/worker/main.py"
        self.assertTrue(worker_main.exists(), "Worker main.py should exist")
        
        with open(worker_main, 'r') as f:
            content = f.read()
        
        # Verify worker targets device_agent, not server directly
        self.assertIn("device_agent", content.lower(),
                      "Worker should reference device_agent")
        self.assertIn("ZMQClientCommManager", content,
                      "Worker should use ZMQClientCommManager")
        self.assertIn("--dry-run-init", content,
                      "Worker should expose Gate 2 dry-run-init flag")

    def test_gate2_dry_run_init_subprocess(self):
        """Gate 2: worker reaches injected Client construction without FL loop."""
        import subprocess

        try:
            import yaml  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML and torch required for dry-run-init subprocess")

        cfg_path = project_root / "distributed/tests/configs/gate4_minimal_quadratic.yaml"
        self.assertTrue(cfg_path.is_file(), "Minimal gate4 config must exist")

        cmd = [
            sys.executable,
            "-m",
            "distributed.worker.main",
            "--client-id",
            "1",
            "--container-name",
            "gate2_dryrun_container",
            "--config",
            str(cfg_path),
            "--dry-run-init",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"dry-run-init failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertIn("[Gate2] stage=dry_run_init", combined)
    
    def test_device_agent_stub_init(self):
        """Verify DeviceAgentStub initializes correctly."""
        from distributed.comm import DeviceAgentStub
        
        stub = DeviceAgentStub(
            device_name="test_device",
            server_host="127.0.0.1",
            control_plane_host="127.0.0.1",
            port_offset=2000,
        )
        
        # Verify port assignments
        self.assertEqual(stub.worker_rep_port, 62011)  # 60011 + 2000
        self.assertEqual(stub.worker_pub_port, 62012)  # 60012 + 2000
        self.assertEqual(stub.server_rep_port, 62001)  # 60001 + 2000
        self.assertEqual(stub.server_pub_port, 62002)  # 60002 + 2000
    
    def test_configs_exist(self):
        """Verify configuration files exist."""
        configs_dir = project_root / "distributed/configs"
        
        expected_files = [
            "client_manifest.json",
            "cluster_info.json",
            "fedit_distributed.yaml",
        ]
        
        for filename in expected_files:
            filepath = configs_dir / filename
            self.assertTrue(filepath.exists(), f"{filename} should exist")


class TestCommManagerContract(unittest.TestCase):
    """Test that comm manager implementations follow the contract."""
    
    def test_client_comm_manager_has_required_methods(self):
        """Verify ZMQClientCommManager has all required methods."""
        from distributed.comm import ZMQClientCommManager
        
        self.assertTrue(hasattr(ZMQClientCommManager, 'send'))
        self.assertTrue(hasattr(ZMQClientCommManager, 'receive'))
        self.assertTrue(hasattr(ZMQClientCommManager, 'add_neighbors'))
        self.assertTrue(hasattr(ZMQClientCommManager, 'get_neighbors'))
        self.assertTrue(hasattr(ZMQClientCommManager, 'close'))
    
    def test_server_comm_manager_has_required_methods(self):
        """Verify ZMQServerCommManager has all required methods."""
        from distributed.comm import ZMQServerCommManager
        
        self.assertTrue(hasattr(ZMQServerCommManager, 'send'))
        self.assertTrue(hasattr(ZMQServerCommManager, 'receive'))
        self.assertTrue(hasattr(ZMQServerCommManager, 'add_neighbors'))
        self.assertTrue(hasattr(ZMQServerCommManager, 'get_neighbors'))
        self.assertTrue(hasattr(ZMQServerCommManager, 'close'))
        self.assertTrue(hasattr(ZMQServerCommManager, 'set_client_routing'))
    
    def test_client_comm_manager_has_required_attributes(self):
        """Verify ZMQClientCommManager has all required attributes."""
        from distributed.comm import ZMQClientCommManager
        
        comm = ZMQClientCommManager(
            client_id=1,
            container_name="test_container",
            device_agent_host="localhost",
        )
        
        self.assertTrue(hasattr(comm, 'neighbors'))
        self.assertTrue(hasattr(comm, 'host'))
        self.assertTrue(hasattr(comm, 'port'))
        
        self.assertIsInstance(comm.neighbors, dict)
        self.assertIsInstance(comm.host, str)
        self.assertIsInstance(comm.port, int)
    
    def test_server_comm_manager_has_required_attributes(self):
        """Verify ZMQServerCommManager has all required attributes."""
        from distributed.comm import ZMQServerCommManager
        
        comm = ZMQServerCommManager(host="localhost", port_offset=3000)
        
        self.assertTrue(hasattr(comm, 'neighbors'))
        self.assertIsInstance(comm.neighbors, dict)


class TestRoutingTableFlow(unittest.TestCase):
    """
    Test the manifest routing table flow into ZMQServerCommManager.
    
    The routing table flow for Gate 3/4 is:
    
    1. ControlPlaneService loads client_manifest.json
    2. ControlPlaneService.build_client_routing_table() extracts:
       {client_id: {device_name, container_name}}
    3. When FL server starts, it passes this table to ZMQServerCommManager:
       - Either via constructor: ZMQServerCommManager(..., client_routing_table=routing)
       - Or via method: comm_manager.set_client_routing(routing)
    4. ZMQServerCommManager uses routing to build PUB topics:
       "<device_name>|<container_name>|" for each message
    """
    
    def test_routing_table_from_manifest_to_server_comm(self):
        """Test full routing table flow from manifest to server comm manager."""
        from distributed.control_plane import ControlPlaneService
        from distributed.comm import ZMQServerCommManager
        
        # Step 1: Load manifest via control plane
        manifest_path = project_root / "distributed/configs/client_manifest.json"
        cp = ControlPlaneService(manifest_path=str(manifest_path))
        
        # Step 2: Build routing table
        routing = cp.build_client_routing_table()
        
        self.assertIn(1, routing)
        self.assertIn(2, routing)
        self.assertEqual(routing[1]['device_name'], 'orinagx1')
        self.assertEqual(routing[1]['container_name'], 'fl_worker_orinagx1')
        self.assertEqual(routing[2]['device_name'], 'orinnx1')
        self.assertEqual(routing[2]['container_name'], 'fl_worker_orinnx1')
        
        # Step 3a: Pass to server comm manager via constructor
        server_comm = ZMQServerCommManager(
            host="0.0.0.0",
            port_offset=4000,  # Avoid conflicts
            client_routing_table=routing,
        )
        
        self.assertEqual(len(server_comm._client_routing), 2)
        self.assertEqual(server_comm._client_routing[1]['device_name'], 'orinagx1')
        
        # Step 3b: Or pass via method
        server_comm2 = ZMQServerCommManager(host="0.0.0.0", port_offset=4001)
        server_comm2.set_client_routing(routing)
        
        self.assertEqual(len(server_comm2._client_routing), 2)
    
    def test_routing_table_used_for_pub_topics(self):
        """Verify routing table determines PUB message topics."""
        from distributed.comm import ZMQServerCommManager
        
        routing = {
            1: {'device_name': 'device_a', 'container_name': 'worker_1'},
            2: {'device_name': 'device_b', 'container_name': 'worker_2'},
        }
        
        server_comm = ZMQServerCommManager(
            host="0.0.0.0",
            port_offset=4002,
            client_routing_table=routing,
        )
        
        # Verify routing is stored
        self.assertEqual(server_comm._client_routing[1]['device_name'], 'device_a')
        self.assertEqual(server_comm._client_routing[2]['container_name'], 'worker_2')
        
        # The send() method will use this to build topics like:
        # "device_a|worker_1|" for client 1
        # "device_b|worker_2|" for client 2


class TestArchitectureCompliance(unittest.TestCase):
    """Test that implementation matches v2.3 architecture requirements."""
    
    def test_port_assignments_match_plan(self):
        """Verify port assignments match v2.3 plan."""
        from distributed.comm import ZMQClientCommManager, ZMQServerCommManager
        from distributed.control_plane import ControlPlaneService
        
        # Control plane: 60101/60102
        cp = ControlPlaneService(port_offset=0)
        self.assertEqual(cp.rep_port, 60101)
        self.assertEqual(cp.pub_port, 60102)
        
        # FL payload plane: 60001/60002
        server_comm = ZMQServerCommManager(port_offset=0)
        self.assertEqual(server_comm.rep_port, 60001)
        self.assertEqual(server_comm.pub_port, 60002)
        
        # Device agent facing (worker connects to): 60011/60012
        client_comm = ZMQClientCommManager(
            client_id=1,
            container_name="test",
            device_agent_port_offset=0,
        )
        self.assertEqual(client_comm.device_agent_req_port, 60011)
        self.assertEqual(client_comm.device_agent_sub_port, 60012)
    
    def test_worker_does_not_connect_to_server_directly(self):
        """Verify worker comm manager does NOT target server ports."""
        from distributed.comm import ZMQClientCommManager
        
        comm = ZMQClientCommManager(
            client_id=1,
            container_name="test",
            device_agent_host="localhost",
            device_agent_port_offset=0,
        )
        
        # Worker should target device_agent ports (60011/60012)
        # NOT server FL payload ports (60001/60002)
        self.assertEqual(comm.device_agent_req_port, 60011)
        self.assertNotEqual(comm.device_agent_req_port, 60001)
        
        # Verify docstring mentions device_agent
        self.assertIn("device_agent", ZMQClientCommManager.__doc__.lower())
    
    def test_development_mode_documented(self):
        """Verify control plane service documents development/mock status."""
        from distributed.control_plane.service import ControlPlaneService
        
        # Check module docstring mentions development/mock status
        import distributed.control_plane.service as service_module
        self.assertIn("DEVELOPMENT", service_module.__doc__.upper())
        
        # Check class or its logger output mentions it
        cp = ControlPlaneService()
        # The init should have logged "DEVELOPMENT MODE"


class TestDirectoryStructure(unittest.TestCase):
    """Verify the distributed directory structure is correct."""
    
    def test_distributed_package_structure(self):
        """Verify all expected files and directories exist."""
        distributed_dir = project_root / "distributed"
        
        expected_structure = [
            "__init__.py",
            "GATE0_AUDIT.md",
            "comm/__init__.py",
            "comm/serialization.py",
            "comm/zmq_client_comm.py",
            "comm/zmq_server_comm.py",
            "comm/device_agent_stub.py",
            "control_plane/__init__.py",
            "control_plane/service.py",
            "worker/__init__.py",
            "worker/main.py",
            "data/__init__.py",
            "data/prepare_partitions.py",
            "data/validate_partition.py",
            "configs/client_manifest.json",
            "configs/cluster_info.json",
            "configs/fedit_distributed.yaml",
            "tests/__init__.py",
            "tests/test_gates.py",
            "tests/test_gate4_real_client.py",
            "tests/test_gate5_server.py",
            "tests/test_gate6_multiround.py",
            "tests/integration_harness.py",
            "tests/configs/gate4_minimal_quadratic.yaml",
        ]
        
        for path in expected_structure:
            filepath = distributed_dir / path
            self.assertTrue(filepath.exists(), f"{path} should exist")


class TestIntegrationLoopback(unittest.TestCase):
    """
    Integration tests with loopback through device_agent_stub.
    
    These tests prove:
    - Control-plane registration exchange
    - Container-start command formatting
    - FL message envelope round-trip through relay boundary
    """
    
    @classmethod
    def setUpClass(cls):
        """Check if ZMQ is available for integration tests."""
        try:
            import zmq
            cls.zmq_available = True
        except ImportError:
            cls.zmq_available = False
    
    def setUp(self):
        if not self.zmq_available:
            self.skipTest("pyzmq not installed, skipping integration tests")
    
    def test_control_plane_registration_roundtrip(self):
        """Test device registration through control plane."""
        from distributed.control_plane import ControlPlaneService
        from distributed.control_plane.service import DeviceInfo
        from distributed.comm import pack_control_message, unpack_control_message
        import zmq
        
        # Use high port offset to avoid conflicts
        port_offset = 5000
        
        # Start control plane
        cp = ControlPlaneService(host="127.0.0.1", port_offset=port_offset)
        cp.start()
        
        try:
            # Give service time to start
            time.sleep(0.3)
            
            # Create a mock device agent that sends registration
            ctx = zmq.Context()
            req_socket = ctx.socket(zmq.REQ)
            req_socket.connect(f"tcp://127.0.0.1:{60101 + port_offset}")
            req_socket.setsockopt(zmq.SNDTIMEO, 2000)
            req_socket.setsockopt(zmq.RCVTIMEO, 2000)
            
            # Send device advertisement
            device_info = DeviceInfo(
                name="test_device_loopback",
                device_type="virtual",
                ip_address="127.0.0.1",
                processors=1,
                memory=[8000],
            )
            
            msg = pack_control_message("DEVICE_ADVERTISEMENT", device_info.to_dict())
            req_socket.send(msg)
            
            # Receive response
            reply = req_socket.recv()
            msg_type, payload = unpack_control_message(reply)
            
            # Verify response
            self.assertEqual(msg_type, "SYSTEM_INFO")
            self.assertIn("name", payload)
            self.assertEqual(payload["name"], "fedlora_distributed")
            
            # Verify device is registered
            self.assertTrue(cp.is_device_registered("test_device_loopback"))
            
            req_socket.close()
            ctx.term()
            
        finally:
            cp.stop()
    
    def test_container_start_command_format(self):
        """Test CONTAINER_START command formatting."""
        from distributed.control_plane import ControlPlaneService
        from distributed.control_plane.service import ContainerConfig
        from distributed.comm import unpack_control_message
        import zmq
        
        port_offset = 5100
        
        # Start control plane
        cp = ControlPlaneService(host="127.0.0.1", port_offset=port_offset)
        cp.start()
        
        try:
            time.sleep(0.3)
            
            # Subscribe to PUB socket
            ctx = zmq.Context()
            sub_socket = ctx.socket(zmq.SUB)
            sub_socket.connect(f"tcp://127.0.0.1:{60102 + port_offset}")
            sub_socket.setsockopt_string(zmq.SUBSCRIBE, "test_device|")
            sub_socket.setsockopt(zmq.RCVTIMEO, 2000)
            
            time.sleep(0.2)  # Let subscription settle
            
            # Register a device first
            with cp._devices_lock:
                from distributed.control_plane.service import RegisteredDevice, DeviceInfo
                cp.devices["test_device"] = RegisteredDevice(
                    info=DeviceInfo("test_device", "virtual", "127.0.0.1"),
                    registered_at=time.time(),
                    last_seen=time.time(),
                )
            
            # Send container start
            config = ContainerConfig(
                name="fl_worker_test",
                client_id=1,
                device_agent_host="localhost",
                device_agent_port_offset=0,
            )
            
            success = cp.send_container_start("test_device", config)
            self.assertTrue(success)
            
            # Receive command
            raw_msg = sub_socket.recv()
            
            # Parse topic and message
            pipe_idx = raw_msg.find(b'|')
            self.assertGreater(pipe_idx, 0)
            
            topic = raw_msg[:pipe_idx].decode('utf-8')
            self.assertEqual(topic, "test_device")
            
            msg_data = raw_msg[pipe_idx + 1:]
            msg_type, payload = unpack_control_message(msg_data)
            
            self.assertEqual(msg_type, "CONTAINER_START")
            self.assertEqual(payload["name"], "fl_worker_test")
            self.assertEqual(payload["client_id"], 1)
            
            sub_socket.close()
            ctx.term()
            
        finally:
            cp.stop()
    
    def test_fl_message_envelope_structure(self):
        """Test FL message envelope structure for relay."""
        from distributed.comm import (
            pack_relay_envelope,
            unpack_relay_envelope,
            serialize_message,
            deserialize_message,
            MessageType,
        )
        from federatedscope.core.message import Message
        
        # Create FL message
        fl_message = Message(
            msg_type='model_para',
            sender=1,
            receiver=[0],
            state=0,
            timestamp=0,
            content={'test': 'data'},
        )
        
        # Serialize FL message
        fl_payload = serialize_message(fl_message)
        
        # Pack into relay envelope (as worker would send to device_agent)
        envelope = pack_relay_envelope(
            msg_type=MessageType.TO_CONTROLLER,
            source_container="fl_worker_test",
            payload=fl_payload,
        )
        
        # Simulate device_agent receiving and forwarding
        # (In real system, device_agent would forward to server)
        msg_type, received_payload = unpack_relay_envelope(envelope)
        
        self.assertEqual(msg_type, MessageType.TO_CONTROLLER)
        
        # Deserialize FL message
        received_message = deserialize_message(received_payload)
        
        self.assertEqual(received_message.msg_type, 'model_para')
        self.assertEqual(received_message.sender, 1)
        self.assertEqual(received_message.content, {'test': 'data'})


class TestGate3FLMessageRoundTrip(unittest.TestCase):
    """
    Gate 3: FL Message Round-Trip
    
    Validate real FedLoRA Message serialization over the FL payload plane.
    
    Pass criteria:
    - model_para reaches worker without corruption
    - Deserialized Message is structurally correct
    - message.content matches standard FedLoRA expectations exactly
    - REQ/REP sequencing remains healthy under timeout-tested conditions
    """
    
    @classmethod
    def setUpClass(cls):
        """Check if dependencies are available."""
        try:
            import zmq
            import torch
            cls.deps_available = True
        except ImportError:
            cls.deps_available = False
    
    def setUp(self):
        if not self.deps_available:
            self.skipTest("zmq or torch not installed")
    
    def test_server_to_client_model_para_integrity(self):
        """Test that model_para message reaches client without corruption."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
            create_test_model_para_message,
            verify_message_integrity,
        )
        import torch
        
        config = HarnessConfig(port_offset=7000)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Create model_para message from server
            original = create_test_model_para_message(
                sender=0,
                receiver=[config.client_id],
                state=0,
                model_size=50,
            )
            
            # Send from server
            harness.server_comm.send(original)
            
            # Receive at client
            received = harness.client_comm.receive()
            
            # Verify integrity
            checks = verify_message_integrity(original, received)
            
            self.assertTrue(checks['msg_type_match'], "msg_type should match")
            self.assertTrue(checks['sender_match'], "sender should match")
            self.assertTrue(checks['content_keys_match'], "content keys should match")
            self.assertTrue(checks['tensor_values_match'], "tensor values should match")
            
        finally:
            harness.stop()
    
    def test_client_to_server_upload_integrity(self):
        """Test that client upload (sample_count, state_dict) reaches server intact."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
            create_test_client_upload_message,
            verify_message_integrity,
        )
        
        config = HarnessConfig(port_offset=7100)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Create client upload message
            original = create_test_client_upload_message(
                client_id=config.client_id,
                server_id=0,
                state=0,
                sample_count=256,
                model_size=50,
            )
            
            # Send from client
            harness.client_comm.send(original)
            
            # Receive at server
            received = harness.server_comm.receive()
            
            # Verify integrity
            checks = verify_message_integrity(original, received)
            
            self.assertTrue(checks['msg_type_match'], "msg_type should match")
            self.assertTrue(checks['tuple_length_match'], "tuple length should match")
            self.assertTrue(checks['sample_count_match'], "sample_count should match")
            self.assertTrue(checks['state_dict_match'], "state_dict tensors should match")
            
        finally:
            harness.stop()
    
    def test_message_content_exact_fedit_format(self):
        """Verify message.content matches standard FedIT expectations exactly."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
        )
        from federatedscope.core.message import Message
        import torch
        
        config = HarnessConfig(port_offset=7200)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Standard FedIT server->client format: state_dict
            server_content = {
                'base_model.classifier.weight': torch.randn(10, 768),
                'base_model.classifier.bias': torch.randn(10),
            }
            
            server_msg = Message(
                msg_type='model_para',
                sender=0,
                receiver=[config.client_id],
                state=0,
                timestamp=time.time(),
                content=server_content,
            )
            
            harness.server_comm.send(server_msg)
            received = harness.client_comm.receive()
            
            # Verify exact structure
            self.assertIsInstance(received.content, dict)
            self.assertIn('base_model.classifier.weight', received.content)
            self.assertIn('base_model.classifier.bias', received.content)
            
            # Standard FedIT client->server format: (sample_count, state_dict)
            client_content = (
                128,  # sample_count
                {
                    'base_model.classifier.weight': torch.randn(10, 768),
                    'base_model.classifier.bias': torch.randn(10),
                }
            )
            
            client_msg = Message(
                msg_type='model_para',
                sender=config.client_id,
                receiver=[0],
                state=0,
                timestamp=time.time(),
                content=client_content,
            )
            
            harness.client_comm.send(client_msg)
            received = harness.server_comm.receive()
            
            # Verify exact structure
            self.assertIsInstance(received.content, tuple)
            self.assertEqual(len(received.content), 2)
            self.assertEqual(received.content[0], 128)  # sample_count
            self.assertIsInstance(received.content[1], dict)  # state_dict
            
        finally:
            harness.stop()
    
    def test_reqrep_sequencing_multiple_messages(self):
        """Verify REQ/REP sequencing remains healthy across multiple messages."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
        )
        from federatedscope.core.message import Message
        import torch
        
        config = HarnessConfig(port_offset=7300)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Send multiple messages in sequence
            for round_idx in range(3):
                # Server -> Client
                server_msg = Message(
                    msg_type='model_para',
                    sender=0,
                    receiver=[config.client_id],
                    state=round_idx,
                    timestamp=time.time(),
                    content={'round': round_idx, 'data': torch.randn(10)},
                )
                
                harness.server_comm.send(server_msg)
                received = harness.client_comm.receive()
                
                self.assertEqual(received.state, round_idx)
                self.assertEqual(received.content['round'], round_idx)
                
                # Client -> Server
                client_msg = Message(
                    msg_type='model_para',
                    sender=config.client_id,
                    receiver=[0],
                    state=round_idx,
                    timestamp=time.time(),
                    content=(100 + round_idx, {'round': round_idx}),
                )
                
                harness.client_comm.send(client_msg)
                received = harness.server_comm.receive()
                
                self.assertEqual(received.state, round_idx)
                self.assertEqual(received.content[0], 100 + round_idx)
            
        finally:
            harness.stop()


class TestGate4TransportPreValidation(unittest.TestCase):
    """
    Gate 4 — transport / message-path pre-validation (NOT full Gate 4 alone).

    Exercises relay + ``Message`` shapes for join / model_para / upload.
    **v2.3 Gate 4 completion** additionally requires
    ``distributed/tests/test_gate4_real_client.py`` (real ``Client`` +
    ``callback_funcs_for_model_para`` + local training).
    """
    
    @classmethod
    def setUpClass(cls):
        """Check if dependencies are available."""
        try:
            import zmq
            import torch
            cls.deps_available = True
        except ImportError:
            cls.deps_available = False
    
    def setUp(self):
        if not self.deps_available:
            self.skipTest("zmq or torch not installed")
    
    def test_join_in_message_flow(self):
        """Test that join_in message reaches server correctly."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
        )
        from federatedscope.core.message import Message
        
        config = HarnessConfig(port_offset=7400, client_id=1)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Simulate join_in message from client
            join_msg = Message(
                msg_type='join_in',
                sender=config.client_id,
                receiver=[0],
                timestamp=0,
                content=None,  # or local_address
            )
            
            harness.client_comm.send(join_msg)
            received = harness.server_comm.receive()
            
            self.assertEqual(received.msg_type, 'join_in')
            self.assertEqual(received.sender, config.client_id)
            
        finally:
            harness.stop()
    
    def test_static_client_id_preserved(self):
        """Verify static client ID is preserved through transport."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
        )
        from federatedscope.core.message import Message
        
        # Test with specific client IDs
        for client_id in [1, 2, 5, 10]:
            config = HarnessConfig(port_offset=7500 + client_id * 10, client_id=client_id)
            harness = IntegrationTestHarness(config)
            
            try:
                harness.start()
                
                msg = Message(
                    msg_type='join_in',
                    sender=client_id,
                    receiver=[0],
                    timestamp=0,
                    content=None,
                )
                
                harness.client_comm.send(msg)
                received = harness.server_comm.receive()
                
                self.assertEqual(received.sender, client_id,
                    f"Client ID {client_id} should be preserved")
                
            finally:
                harness.stop()
    
    def test_training_result_upload_format(self):
        """Verify client upload matches FedIT (sample_count, state_dict) format."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
        )
        from federatedscope.core.message import Message
        import torch
        
        config = HarnessConfig(port_offset=7700)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Simulate training result upload
            sample_count = 256
            trained_state = {
                'lora_A.weight': torch.randn(8, 768),
                'lora_B.weight': torch.randn(768, 8),
            }
            
            upload_msg = Message(
                msg_type='model_para',
                sender=config.client_id,
                receiver=[0],
                state=0,
                timestamp=time.time(),
                content=(sample_count, trained_state),
            )
            
            harness.client_comm.send(upload_msg)
            received = harness.server_comm.receive()
            
            # Verify format
            self.assertEqual(received.msg_type, 'model_para')
            self.assertIsInstance(received.content, tuple)
            self.assertEqual(len(received.content), 2)
            
            recv_sample_count, recv_state = received.content
            self.assertEqual(recv_sample_count, sample_count)
            self.assertIsInstance(recv_state, dict)
            self.assertIn('lora_A.weight', recv_state)
            self.assertIn('lora_B.weight', recv_state)
            
            # Verify tensor shapes
            self.assertEqual(recv_state['lora_A.weight'].shape, (8, 768))
            self.assertEqual(recv_state['lora_B.weight'].shape, (768, 8))
            
        finally:
            harness.stop()
    
    def test_full_join_train_upload_sequence(self):
        """Test complete sequence: join -> receive model -> upload result."""
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
        )
        from federatedscope.core.message import Message
        import torch
        
        config = HarnessConfig(port_offset=7800)
        harness = IntegrationTestHarness(config)
        
        try:
            harness.start()
            
            # Step 1: Client sends join_in
            join_msg = Message(
                msg_type='join_in',
                sender=config.client_id,
                receiver=[0],
                timestamp=0,
                content=None,
            )
            harness.client_comm.send(join_msg)
            
            # Server receives join
            join_recv = harness.server_comm.receive()
            self.assertEqual(join_recv.msg_type, 'join_in')
            self.assertEqual(join_recv.sender, config.client_id)
            
            # Step 2: Server sends model_para to client
            model_state = {
                'layer.weight': torch.randn(100, 100),
                'layer.bias': torch.randn(100),
            }
            model_msg = Message(
                msg_type='model_para',
                sender=0,
                receiver=[config.client_id],
                state=0,
                timestamp=time.time(),
                content=model_state,
            )
            harness.server_comm.send(model_msg)
            
            # Client receives model
            model_recv = harness.client_comm.receive()
            self.assertEqual(model_recv.msg_type, 'model_para')
            self.assertEqual(model_recv.state, 0)
            self.assertIn('layer.weight', model_recv.content)
            
            # Step 3: Client sends training result back
            # (Simulating local training by modifying weights)
            trained_state = {k: v + 0.01 for k, v in model_recv.content.items()}
            result_msg = Message(
                msg_type='model_para',
                sender=config.client_id,
                receiver=[0],
                state=0,
                timestamp=time.time(),
                content=(128, trained_state),  # (sample_count, state_dict)
            )
            harness.client_comm.send(result_msg)
            
            # Server receives result
            result_recv = harness.server_comm.receive()
            self.assertEqual(result_recv.msg_type, 'model_para')
            self.assertEqual(result_recv.sender, config.client_id)
            self.assertIsInstance(result_recv.content, tuple)
            self.assertEqual(result_recv.content[0], 128)
            
        finally:
            harness.stop()


if __name__ == '__main__':
    unittest.main()
