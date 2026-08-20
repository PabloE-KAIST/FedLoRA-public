import torch
import logging
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import MODE, LIFECYCLE
from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.optimizer_builder import get_optimizer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.vlm.model.model_builder import VLMAdapterModel
from federatedscope.contrib.trainer.glue_hetlora_trainer import compute_hetlora_regularizer
from federatedscope.contrib.trainer.glue_adasparse_trainer import compute_adasparse_regularizer
from federatedscope.contrib.trainer.glue_adasparse_v2_trainer import compute_adasparse_v2_regularizer
from federatedscope.contrib.trainer.glue_adasparse_v3_trainer import compute_adasparse_v3_regularizer
from federatedscope.contrib.trainer.glue_fah_trainer import evaluate_for_fah_impl

logger = logging.getLogger(__name__)


class VLMTrainer(GeneralTorchTrainer):
    def _hook_on_fit_start_numerical_precision(self, ctx):
        if self.cfg.train.is_enable_half:
            ctx.model = ctx.model.half()

    def _hook_on_fit_start_init(self, ctx):
        ctx.model.to(ctx.device)
        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:
            ctx.model.model.enable_input_require_grads()
            ctx.model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            ctx.optimizer = get_optimizer(
                ctx.model, **ctx.cfg[ctx.cur_mode].optimizer)
            ctx.scheduler = get_scheduler(
                ctx.optimizer, **ctx.cfg[ctx.cur_mode].scheduler)

        ctx.loss_batch_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.loss_regular_total = CtxVar(0., LIFECYCLE.ROUTINE)
        ctx.num_samples = CtxVar(0, LIFECYCLE.ROUTINE)
        ctx.ys_true = CtxVar([], LIFECYCLE.ROUTINE)
        ctx.ys_prob = CtxVar([], LIFECYCLE.ROUTINE)

    def _hook_on_batch_forward(self, ctx):
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)
        pixel_values = ctx.data_batch['pixel_values'].to(
            device=ctx.device, dtype=torch.bfloat16)
        image_grid_thw = ctx.data_batch['image_grid_thw'].to(ctx.device)

        outputs = ctx.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        logits = outputs.logits
        loss = outputs.loss

        if torch.isnan(loss):
            ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
            logger.warning('Skip the batch due to NaN loss.')
        else:
            ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

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

    def evaluate_for_fah(self, rank, validation_fraction=0.1,
                         validation_steps=10):
        return evaluate_for_fah_impl(
            self, rank=rank,
            validation_fraction=validation_fraction,
            validation_steps=validation_steps,
        )

    def _hook_on_batch_forward_regularizer(self, ctx):
        ctx.loss_regular = CtxVar(
            self.cfg.regularizer.mu * ctx.regularizer(ctx), LIFECYCLE.BATCH)
        total_loss = ctx.loss_batch + ctx.loss_regular

        method_regularizer = self._get_method_regularizer(ctx)
        if method_regularizer is not None:
            total_loss = total_loss + method_regularizer

        ctx.loss_task = CtxVar(total_loss, LIFECYCLE.BATCH)

    def _hook_on_batch_backward(self, ctx):
        if ctx.skip_this_batch:
            return

        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                ctx.model.parameters(), ctx.grad_clip)

        ctx.optimizer.step()
        if ctx.scheduler is not None:
            ctx.scheduler.step()

    def _hook_on_batch_end(self, ctx):
        if ctx.skip_this_batch:
            if ctx.cfg.vlm.retry_on_nan_loss:
                if ctx.cur_mode == MODE.TRAIN:
                    self._run_batch(self.hooks_in_train, run_step=1)
                elif ctx.cur_mode == MODE.FINETUNE:
                    self._run_batch(self.hooks_in_ft, run_step=1)
            return

        ctx.num_samples += ctx.batch_size
        ctx.loss_batch_total += ctx.loss_batch.item() * ctx.batch_size
        ctx.loss_regular_total += float(ctx.get("loss_regular", 0.))

    def _hook_on_fit_end(self, ctx):
        avg_loss = 0 if float(
            ctx.num_samples) == 0 else ctx.loss_batch_total / float(
                ctx.num_samples)
        eval_results = {
            f'{ctx.cur_split}_loss': ctx.loss_batch_total,
            f'{ctx.cur_split}_total': ctx.num_samples,
            f'{ctx.cur_split}_avg_loss': avg_loss,
        }
        setattr(ctx, 'eval_metrics', eval_results)

        if ctx.cfg.vlm.adapter.mv_to_cpu:
            for p in ctx.model.parameters():
                if p.requires_grad:
                    p.data = p.to('cpu')
                    if p.grad is not None:
                        p.grad.data = p.grad.to('cpu')


def call_vlm_trainer(trainer_type):
    if trainer_type == 'vlmtrainer':
        trainer_builder = VLMTrainer
        return trainer_builder


register_trainer('vlmtrainer', call_vlm_trainer)
