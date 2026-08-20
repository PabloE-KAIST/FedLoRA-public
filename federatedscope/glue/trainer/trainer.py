"""Shared GLUE trainer entrypoint."""

import logging

from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.glue_base_trainer import GLUEBaseTrainer
from federatedscope.contrib.trainer.glue_hetlora_trainer import compute_hetlora_regularizer
from federatedscope.contrib.trainer.glue_adasparse_trainer import compute_adasparse_regularizer
from federatedscope.contrib.trainer.glue_adasparse_v2_trainer import compute_adasparse_v2_regularizer
from federatedscope.contrib.trainer.glue_adasparse_v3_trainer import compute_adasparse_v3_regularizer
from federatedscope.contrib.trainer.glue_fah_trainer import evaluate_for_fah_impl

logger = logging.getLogger(__name__)


class GLUETrainer(GLUEBaseTrainer):
    def _get_method_regularizer(self, ctx):
        terms = []
        for fn in (
            compute_hetlora_regularizer,
            compute_adasparse_regularizer,
            compute_adasparse_v2_regularizer,
            compute_adasparse_v3_regularizer,
        ):
            value = fn(ctx)
            if value is not None:
                terms.append(value)

        if not terms:
            return None

        total = terms[0]
        for value in terms[1:]:
            total = total + value
        return total

    def evaluate_for_fah(self, rank: int, validation_fraction: float = 0.1, validation_steps: int = 10):
        return evaluate_for_fah_impl(
            self,
            rank=rank,
            validation_fraction=validation_fraction,
            validation_steps=validation_steps,
        )


def call_glue_trainer(trainer_type):
    if trainer_type == 'gluetrainer':
        return GLUETrainer
    return None


register_trainer('gluetrainer', call_glue_trainer)