"""
Gate 5 (v2.3): real FedLoRA Server + injected ZMQServerCommManager proof.

Pass criteria:
  - Real Server instantiated with injected ZMQServerCommManager
  - Existing aggregation path exercised (no custom round manager / aggregator)
  - One aggregation cycle completed
  - One rebroadcast sent after aggregation

Scope is intentionally narrow per v2.3: only standard FedAvg-like flow.
"""

import sys
import time
import threading
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestGate5RealFedLoRAServer(unittest.TestCase):
    """Gate 5: real Server with injected comm manager, one aggregation cycle."""

    @classmethod
    def setUpClass(cls):
        try:
            import zmq  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            cls.deps = False
            return
        try:
            import federatedscope  # noqa: F401
        except ImportError:
            cls.deps = False
            return
        cls.deps = True

    def setUp(self):
        if not self.deps:
            self.skipTest("zmq, torch, or federatedscope not available")

    def test_server_receives_join_and_broadcasts_model(self):
        """Server handles join_in and broadcasts model_para to client."""
        import torch
        from federatedscope.core.message import Message
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
            build_injected_server_for_harness,
        )

        static_client_id = 1
        port_offset = 9100
        harness = IntegrationTestHarness(
            HarnessConfig(
                port_offset=port_offset,
                client_id=static_client_id,
                container_name="gate5_client",
                device_name="gate5_dev",
            )
        )

        try:
            harness.start()

            server, cfg, model = build_injected_server_for_harness(
                harness, client_num=1
            )
            self.assertEqual(server.ID, 0)
            self.assertEqual(server.client_num, 1)

            join_msg = Message(
                msg_type="join_in",
                sender=static_client_id,
                receiver=[0],
                timestamp=time.time(),
                state=0,
                content={"host": "gate5_test", "port": 0},
            )
            harness.client_comm.send(join_msg)

            recv_join = harness.server_comm.receive()
            self.assertEqual(recv_join.msg_type, "join_in")
            self.assertEqual(recv_join.sender, static_client_id)

            server.callback_funcs_for_join_in(recv_join)
            self.assertEqual(server.join_in_client_num, 1)
            self.assertIn(static_client_id, server.comm_manager.neighbors)

            broadcast_msg = harness.client_comm.receive()
            self.assertEqual(broadcast_msg.msg_type, "model_para")
            self.assertEqual(broadcast_msg.sender, 0)
            self.assertIsInstance(broadcast_msg.content, dict)

            if "model_para" in broadcast_msg.content:
                model_state = broadcast_msg.content["model_para"]
            else:
                model_state = broadcast_msg.content
            self.assertIsInstance(model_state, dict)
            self.assertGreater(len(model_state), 0)

        finally:
            harness.stop()

    def test_server_aggregates_and_rebroadcasts(self):
        """Server aggregates client upload and rebroadcasts next round model."""
        import torch
        from federatedscope.core.message import Message
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
            build_injected_server_for_harness,
        )

        static_client_id = 1
        port_offset = 9200
        harness = IntegrationTestHarness(
            HarnessConfig(
                port_offset=port_offset,
                client_id=static_client_id,
                container_name="gate5_client_agg",
                device_name="gate5_dev_agg",
            )
        )

        try:
            harness.start()

            server, cfg, model = build_injected_server_for_harness(
                harness, client_num=1
            )

            join_msg = Message(
                msg_type="join_in",
                sender=static_client_id,
                receiver=[0],
                timestamp=time.time(),
                state=0,
                content={"host": "gate5_agg_test", "port": 0},
            )
            harness.client_comm.send(join_msg)
            recv_join = harness.server_comm.receive()
            server.callback_funcs_for_join_in(recv_join)

            round0_model = harness.client_comm.receive()
            self.assertEqual(round0_model.msg_type, "model_para")
            self.assertEqual(round0_model.state, 0)

            model_content = round0_model.content
            if isinstance(model_content, dict) and "model_para" in model_content:
                model_state = model_content["model_para"]
            else:
                model_state = model_content

            sample_count = 64
            trained_state = {}
            for k, v in model_state.items():
                if isinstance(v, torch.Tensor):
                    trained_state[k] = v + 0.01
                else:
                    trained_state[k] = v
            upload_msg = Message(
                msg_type="model_para",
                sender=static_client_id,
                receiver=[0],
                timestamp=time.time(),
                state=0,
                content=(sample_count, trained_state),
            )
            harness.client_comm.send(upload_msg)

            recv_upload = harness.server_comm.receive()
            self.assertEqual(recv_upload.msg_type, "model_para")
            self.assertEqual(recv_upload.sender, static_client_id)

            initial_state = server.state
            move_on = server.callback_funcs_model_para(recv_upload)
            self.assertTrue(move_on, "Server should move on after receiving enough messages")
            self.assertEqual(server.state, initial_state + 1)

            round1_model = harness.client_comm.receive()
            self.assertEqual(round1_model.msg_type, "model_para")
            self.assertEqual(round1_model.state, 1)
            self.assertIsInstance(round1_model.content, dict)

            if "model_para" in round1_model.content:
                round1_state = round1_model.content["model_para"]
            else:
                round1_state = round1_model.content
            self.assertIsInstance(round1_state, dict)
            self.assertGreater(len(round1_state), 0)

        finally:
            harness.stop()

    def test_full_one_round_cycle(self):
        """Complete cycle: join -> broadcast -> train -> upload -> aggregate -> rebroadcast."""
        import torch
        from federatedscope.core.message import Message
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
            build_injected_server_for_harness,
        )

        static_client_id = 1
        port_offset = 9300
        harness = IntegrationTestHarness(
            HarnessConfig(
                port_offset=port_offset,
                client_id=static_client_id,
                container_name="gate5_full_cycle",
                device_name="gate5_full_dev",
            )
        )

        try:
            harness.start()

            server, cfg, model = build_injected_server_for_harness(
                harness, client_num=1
            )

            join_msg = Message(
                msg_type="join_in",
                sender=static_client_id,
                receiver=[0],
                timestamp=time.time(),
                state=0,
                content={"host": "cycle_test", "port": 0},
            )
            harness.client_comm.send(join_msg)
            recv_join = harness.server_comm.receive()
            server.callback_funcs_for_join_in(recv_join)

            round0_broadcast = harness.client_comm.receive()
            self.assertEqual(round0_broadcast.msg_type, "model_para")
            self.assertEqual(server.state, 0)

            broadcast_content = round0_broadcast.content
            if isinstance(broadcast_content, dict) and "model_para" in broadcast_content:
                model_state = broadcast_content["model_para"]
            else:
                model_state = broadcast_content

            sample_count = 100
            updated_state = {}
            for k, v in model_state.items():
                if isinstance(v, torch.Tensor):
                    updated_state[k] = v + torch.randn_like(v) * 0.01
                else:
                    updated_state[k] = v

            upload = Message(
                msg_type="model_para",
                sender=static_client_id,
                receiver=[0],
                timestamp=time.time(),
                state=0,
                content=(sample_count, updated_state),
            )
            harness.client_comm.send(upload)

            recv_upload = harness.server_comm.receive()
            self.assertEqual(recv_upload.msg_type, "model_para")

            initial_state = server.state
            move_on = server.callback_funcs_model_para(recv_upload)

            self.assertTrue(move_on)
            self.assertEqual(server.state, initial_state + 1)

            round1_broadcast = harness.client_comm.receive()
            self.assertEqual(round1_broadcast.msg_type, "model_para")
            self.assertEqual(round1_broadcast.sender, 0)
            self.assertEqual(round1_broadcast.state, 1)

            self.assertIsInstance(round1_broadcast.content, dict)
            if "model_para" in round1_broadcast.content:
                round1_state = round1_broadcast.content["model_para"]
            else:
                round1_state = round1_broadcast.content

            for k in model_state.keys():
                self.assertIn(k, round1_state)
                self.assertIsInstance(round1_state[k], torch.Tensor)

        finally:
            harness.stop()


if __name__ == "__main__":
    unittest.main()
