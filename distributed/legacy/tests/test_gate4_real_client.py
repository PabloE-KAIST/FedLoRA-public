"""
Gate 4 (v2.3): real FedLoRA Client + injected ZMQClientCommManager proof.

Pre-validation transport tests live in ``test_gates.py`` under
``TestGate4TransportPreValidation``. This module is the **completion** test:
it exercises ``callback_funcs_for_model_para`` with local training and
asserts the upload wire format.
"""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


class TestGate4RealFedLoRAClient(unittest.TestCase):
    """Gate 4 full proof: real Client, real trainer step, ZMQ upload."""

    @classmethod
    def setUpClass(cls):
        try:
            import zmq  # noqa: F401
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
            self.skipTest("zmq or federatedscope not available")

    def test_real_client_training_and_model_para_upload(self):
        import torch
        from distributed.tests.integration_harness import (
            IntegrationTestHarness,
            HarnessConfig,
            build_injected_client_for_harness,
            capture_model_para_after_training,
            gate4_generated_partition_path,
            gate4_real_glue_config_path,
        )

        # Static FL client ID must match routing + harness; use 1 with data_idx 1 fixture
        static_id = 1
        port_offset = 8900
        harness = IntegrationTestHarness(
            HarnessConfig(
                port_offset=port_offset,
                client_id=static_id,
                container_name="gate4_real_client",
                device_name="gate4_dev",
            )
        )
        try:
            harness.start()
            partition_path = gate4_generated_partition_path(static_id)
            self.assertTrue(
                partition_path.exists(),
                f"Expected validated Gate 4 partition artifact at {partition_path}",
            )

            client = build_injected_client_for_harness(
                harness,
                client_id=static_id,
                yaml_path=gate4_real_glue_config_path(),
                partition_path=str(partition_path),
            )
            self.assertEqual(
                client.ID,
                static_id,
                "Phase 1 uses static client ID; no assign_client_id in this path",
            )

            upload = capture_model_para_after_training(harness, client, training_round=0)

            self.assertEqual(upload.msg_type, "model_para")
            self.assertEqual(upload.sender, static_id)
            self.assertIsInstance(upload.content, tuple)
            self.assertEqual(len(upload.content), 2)
            sample_count, state_dict = upload.content
            self.assertIsInstance(sample_count, int)
            self.assertGreater(sample_count, 0)
            self.assertIsInstance(state_dict, dict)
            self.assertGreater(len(state_dict), 0)
            self.assertTrue(
                any(isinstance(v, torch.Tensor) for v in state_dict.values()),
                "Uploaded model_para payload should contain tensor-valued parameters",
            )
        finally:
            harness.stop()


if __name__ == "__main__":
    unittest.main()
