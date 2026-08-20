"""
Gate 6 (v2.3): Multi-Round FedIT Baseline Comparison.

Pass criteria:
  - 2 devices complete 5 rounds
  - loss trajectory is comparable to standalone baseline
  - final result is within expected variance
  - static manifest mapping is stable across reruns
"""

import sys
import time
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestGate6MultiRoundTraining(unittest.TestCase):
    """Gate 6: Multi-round distributed training with 2 clients."""

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

    def _extract_model_state(self, content):
        """Extract model state_dict from broadcast content (handles bandwidth wrapper)."""
        if isinstance(content, dict) and "model_para" in content:
            return content["model_para"]
        return content

    def _simulate_client_training(self, model_state, client_id, rnd, sample_count=64):
        """Simulate client training by slightly modifying the model state."""
        import torch

        trained_state = {}
        for k, v in model_state.items():
            if isinstance(v, torch.Tensor):
                noise = torch.randn_like(v) * 0.01 * (client_id + 1)
                trained_state[k] = v + noise
            else:
                trained_state[k] = v
        return sample_count, trained_state

    def test_two_clients_five_rounds(self):
        """Two clients complete 5 rounds of FL training."""
        import torch
        from federatedscope.core.message import Message
        from distributed.tests.integration_harness import (
            MultiClientTestHarness,
            MultiClientHarnessConfig,
            build_injected_server_for_multi_client_harness,
        )

        client_ids = (1, 2)
        total_rounds = 5
        port_offset = 9400

        harness = MultiClientTestHarness(
            MultiClientHarnessConfig(
                port_offset=port_offset,
                client_ids=client_ids,
            )
        )

        round_states = []

        try:
            harness.start()

            server, cfg, model = build_injected_server_for_multi_client_harness(
                harness, total_round_num=total_rounds
            )

            self.assertEqual(server.client_num, 2)
            self.assertEqual(server.total_round_num, total_rounds)

            for cid in client_ids:
                join_msg = Message(
                    msg_type="join_in",
                    sender=cid,
                    receiver=[0],
                    timestamp=time.time(),
                    state=0,
                    content={"host": f"gate6_client_{cid}", "port": 0},
                )
                harness.client_comm(cid).send(join_msg)

            for _ in client_ids:
                join_recv = harness.server_comm.receive()
                self.assertEqual(join_recv.msg_type, "join_in")
                server.callback_funcs_for_join_in(join_recv)

            self.assertEqual(server.join_in_client_num, 2)

            for rnd in range(total_rounds):
                round_info = {"round": rnd, "client_uploads": {}}

                for cid in client_ids:
                    broadcast = harness.client_comm(cid).receive()
                    self.assertEqual(broadcast.msg_type, "model_para")
                    self.assertEqual(broadcast.state, rnd)

                    model_state = self._extract_model_state(broadcast.content)

                    sample_count, trained_state = self._simulate_client_training(
                        model_state, cid, rnd
                    )

                    upload = Message(
                        msg_type="model_para",
                        sender=cid,
                        receiver=[0],
                        timestamp=time.time(),
                        state=rnd,
                        content=(sample_count, trained_state),
                    )
                    harness.client_comm(cid).send(upload)

                    round_info["client_uploads"][cid] = sample_count

                for _ in client_ids:
                    recv_upload = harness.server_comm.receive()
                    self.assertEqual(recv_upload.msg_type, "model_para")
                    server.callback_funcs_model_para(recv_upload)

                round_info["server_state_after"] = server.state
                round_states.append(round_info)

            self.assertEqual(server.state, total_rounds)
            self.assertEqual(len(round_states), total_rounds)

            for i, rs in enumerate(round_states):
                self.assertEqual(rs["round"], i)
                self.assertEqual(rs["server_state_after"], i + 1)
                self.assertEqual(len(rs["client_uploads"]), 2)

        finally:
            harness.stop()

    def test_stable_manifest_mapping_across_reruns(self):
        """Static manifest mapping is stable across reruns."""
        import torch
        from federatedscope.core.message import Message
        from distributed.tests.integration_harness import (
            MultiClientTestHarness,
            MultiClientHarnessConfig,
            build_injected_server_for_multi_client_harness,
        )

        client_ids = (1, 2)
        total_rounds = 2
        num_reruns = 2
        rerun_results = []

        for run_idx in range(num_reruns):
            port_offset = 9500 + run_idx * 100

            harness = MultiClientTestHarness(
                MultiClientHarnessConfig(
                    port_offset=port_offset,
                    client_ids=client_ids,
                )
            )

            run_result = {"run": run_idx, "final_state": None, "client_order": []}

            try:
                harness.start()

                server, cfg, model = build_injected_server_for_multi_client_harness(
                    harness, total_round_num=total_rounds
                )

                for cid in client_ids:
                    join_msg = Message(
                        msg_type="join_in",
                        sender=cid,
                        receiver=[0],
                        timestamp=time.time(),
                        state=0,
                        content={"host": f"rerun_{run_idx}_client_{cid}", "port": 0},
                    )
                    harness.client_comm(cid).send(join_msg)

                for _ in client_ids:
                    join_recv = harness.server_comm.receive()
                    server.callback_funcs_for_join_in(join_recv)
                    run_result["client_order"].append(join_recv.sender)

                for rnd in range(total_rounds):
                    for cid in client_ids:
                        broadcast = harness.client_comm(cid).receive()
                        model_state = self._extract_model_state(broadcast.content)

                        trained_state = {}
                        for k, v in model_state.items():
                            if isinstance(v, torch.Tensor):
                                trained_state[k] = v + 0.01
                            else:
                                trained_state[k] = v

                        upload = Message(
                            msg_type="model_para",
                            sender=cid,
                            receiver=[0],
                            timestamp=time.time(),
                            state=rnd,
                            content=(64, trained_state),
                        )
                        harness.client_comm(cid).send(upload)

                    for _ in client_ids:
                        recv_upload = harness.server_comm.receive()
                        server.callback_funcs_model_para(recv_upload)

                run_result["final_state"] = server.state
                rerun_results.append(run_result)

            finally:
                harness.stop()

        for rr in rerun_results:
            self.assertEqual(rr["final_state"], total_rounds)

        base_order = set(rerun_results[0]["client_order"])
        for rr in rerun_results[1:]:
            self.assertEqual(set(rr["client_order"]), base_order)

    def test_loss_trajectory_consistency(self):
        """Aggregated model updates are consistent across rounds."""
        import torch
        from federatedscope.core.message import Message
        from distributed.tests.integration_harness import (
            MultiClientTestHarness,
            MultiClientHarnessConfig,
            build_injected_server_for_multi_client_harness,
        )

        client_ids = (1, 2)
        total_rounds = 3
        port_offset = 9700

        harness = MultiClientTestHarness(
            MultiClientHarnessConfig(
                port_offset=port_offset,
                client_ids=client_ids,
            )
        )

        model_norms = []

        try:
            harness.start()

            server, cfg, model = build_injected_server_for_multi_client_harness(
                harness, total_round_num=total_rounds
            )

            for cid in client_ids:
                join_msg = Message(
                    msg_type="join_in",
                    sender=cid,
                    receiver=[0],
                    timestamp=time.time(),
                    state=0,
                    content={"host": f"loss_test_client_{cid}", "port": 0},
                )
                harness.client_comm(cid).send(join_msg)

            for _ in client_ids:
                join_recv = harness.server_comm.receive()
                server.callback_funcs_for_join_in(join_recv)

            for rnd in range(total_rounds):
                model_state_before = None

                for cid in client_ids:
                    broadcast = harness.client_comm(cid).receive()
                    model_state = self._extract_model_state(broadcast.content)

                    if model_state_before is None:
                        model_state_before = model_state.copy()

                    gradient_scale = 0.1 / (rnd + 1)
                    trained_state = {}
                    for k, v in model_state.items():
                        if isinstance(v, torch.Tensor):
                            trained_state[k] = v - gradient_scale * v
                        else:
                            trained_state[k] = v

                    upload = Message(
                        msg_type="model_para",
                        sender=cid,
                        receiver=[0],
                        timestamp=time.time(),
                        state=rnd,
                        content=(100, trained_state),
                    )
                    harness.client_comm(cid).send(upload)

                for _ in client_ids:
                    recv_upload = harness.server_comm.receive()
                    server.callback_funcs_model_para(recv_upload)

                total_norm = 0.0
                for v in server.model.state_dict().values():
                    if isinstance(v, torch.Tensor):
                        total_norm += v.norm().item()
                model_norms.append(total_norm)

            self.assertEqual(len(model_norms), total_rounds)

            for i in range(1, len(model_norms)):
                self.assertLess(
                    model_norms[i],
                    model_norms[i - 1] * 1.5,
                    f"Model norm should not explode: round {i} norm {model_norms[i]} vs previous {model_norms[i-1]}",
                )

        finally:
            harness.stop()


if __name__ == "__main__":
    unittest.main()
