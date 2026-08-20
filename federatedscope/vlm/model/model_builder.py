import logging
import torch
import torch.nn as nn
from collections import OrderedDict

logger = logging.getLogger(__name__)


def enable_vlm_adapter(model, package, adapter, **kwargs):
    adapter = adapter.lower()
    if package == 'peft':
        from peft import get_peft_model, TaskType

        if adapter == 'lora':
            from peft import LoraConfig
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **kwargs)
            model = get_peft_model(model, peft_config)
        elif adapter == 'qlora':
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=kwargs['r'],
                lora_alpha=kwargs['lora_alpha'],
                target_modules=kwargs['target_modules'],
                lora_dropout=kwargs['lora_dropout'],
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
        else:
            raise NotImplementedError(f'Adapter {adapter} not supported')
        model.print_trainable_parameters()
    else:
        raise NotImplementedError(f'Package {package} not supported')
    return model


class VLMAdapterModel(nn.Module):
    def __init__(self, model, use_adapter=False, *args, **kwargs):
        super().__init__()
        self.model = None
        if use_adapter:
            adapter_package = kwargs.pop('adapter_package', 'peft')
            adapter_method = kwargs.pop('adapter_method', 'lora')
            self.model = enable_vlm_adapter(
                model, adapter_package, adapter_method, **kwargs)
        else:
            self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def state_dict(self, return_trainable=True, *args, **kwargs):
        if return_trainable:
            return self.get_trainable_state_dict()
        else:
            return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict=False)

    def get_trainable_state_dict(self):
        grad_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                grad_params.append(name)
        model_state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k in grad_params:
                new_state_dict[k] = v
        return new_state_dict

    def save_model(self, path, state=0):
        ckpt = {'cur_round': state, 'model': self.model.state_dict()}
        torch.save(ckpt, path)


def get_vlm_from_huggingface(model_name, config):
    from transformers import Qwen2_5_VLForConditionalGeneration

    kwargs = {}
    if len(config.vlm.cache.model):
        kwargs['cache_dir'] = config.vlm.cache.model

    quant_method = getattr(config.computation_quantization, 'method', 'none')
    nbits = getattr(config.computation_quantization, 'nbits', 4)

    if quant_method == 'qlora' and nbits in [4, 8]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        if nbits == 4:
            logger.info(
                f'Loading {model_name} with 4-bit quantization (NF4)')
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
        else:
            logger.info(
                f'Loading {model_name} with 8-bit quantization (LLM.int8)')
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=True,
            )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        )
        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    elif nbits == 16 or quant_method == 'none':
        if nbits == 16:
            logger.info(f'Loading {model_name} in 16-bit (bfloat16) mode')
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )

    else:
        raise ValueError(
            f'Invalid quantization config: method={quant_method}, '
            f'nbits={nbits}.')


def get_vlm_model(config):
    model_config = config.model
    model_name, model_hub = model_config.type.split('@')

    if model_hub == 'huggingface_vlm':
        model = get_vlm_from_huggingface(model_name, config)
    else:
        raise NotImplementedError(
            f'Not supported VLM hub: {model_hub}')

    adapter_cfg = config.vlm.adapter
    base_args = adapter_cfg.args[0] if adapter_cfg.args else {}
    args = dict(base_args)

    if adapter_cfg.use and config.computation_quantization.method == 'qlora':
        args['nbits'] = config.computation_quantization.nbits

    fah_enabled = hasattr(adapter_cfg, 'fah') and adapter_cfg.fah.enabled
    max_rank = getattr(adapter_cfg, 'max_rank', None)

    if adapter_cfg.use and max_rank is not None:
        original_r = args.get('r', None)
        if original_r != max_rank:
            logger.warning(
                f'Overriding LoRA rank from {original_r} to {max_rank} '
                f'to match global max_rank.')
        args['r'] = max_rank

    model = VLMAdapterModel(model, use_adapter=adapter_cfg.use, **args)

    if config.federate.freeze_A:
        for name, param in model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False

    return model
