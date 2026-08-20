"""HeteroLoRA placeholder trainer.

HeteroLoRA currently reuses the generic GLUE local training behavior.
"""

from federatedscope.contrib.trainer.glue_base_trainer import GLUEBaseTrainer


class GLUEHeteroLoRATrainer(GLUEBaseTrainer):
    METHOD_NAME = 'heterolora'