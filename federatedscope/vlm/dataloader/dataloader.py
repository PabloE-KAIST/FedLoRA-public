import os
import json
import logging
import torch

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def get_vlm_processor(model_name, cache_dir, min_pixels=None, max_pixels=None):
    from transformers import AutoProcessor

    kwargs = {}
    if cache_dir:
        kwargs['cache_dir'] = cache_dir
    if min_pixels is not None:
        kwargs['min_pixels'] = min_pixels
    if max_pixels is not None:
        kwargs['max_pixels'] = max_pixels

    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    return processor


def convert_llava_to_qwen_messages(conversations, image):
    messages = []
    for turn in conversations:
        role = 'user' if turn['from'] == 'human' else 'assistant'
        value = turn['value']

        content = []
        if role == 'user' and '<image>' in value:
            content.append({'type': 'image', 'image': image})
            value = value.replace('<image>', '').strip()
            if value.startswith('\n'):
                value = value[1:]

        if value:
            content.append({'type': 'text', 'text': value})

        messages.append({'role': role, 'content': content})
    return messages


class VLMDataset(Dataset):
    def __init__(self, list_data_dict, processor, image_dir, max_length=512):
        self.data = list_data_dict
        self.processor = processor
        self.image_dir = image_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_file = entry['image']
        image_path = os.path.join(self.image_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        messages = convert_llava_to_qwen_messages(
            entry['conversations'], image)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values']
        image_grid_thw = inputs['image_grid_thw']

        labels = self._create_labels(input_ids, messages)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }

    def _create_labels(self, input_ids, messages):
        labels = input_ids.clone()
        assistant_token = self.processor.tokenizer.encode(
            'assistant', add_special_tokens=False)

        found_assistant = False
        for i in range(len(input_ids) - len(assistant_token)):
            if input_ids[i:i + len(assistant_token)].tolist() == assistant_token:
                labels[:i + len(assistant_token)] = IGNORE_INDEX
                found_assistant = True
                break

        if not found_assistant:
            half = len(input_ids) // 2
            labels[:half] = IGNORE_INDEX

        return labels


@dataclass
class VLMDataCollator:
    processor: object

    def __call__(self, instances):
        input_ids = [inst['input_ids'] for inst in instances]
        labels = [inst['labels'] for inst in instances]

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(pad_token_id)

        pixel_values = torch.cat(
            [inst['pixel_values'] for inst in instances], dim=0)
        image_grid_thw = torch.cat(
            [inst['image_grid_thw'] for inst in instances], dim=0)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
        }


def load_vlm_dataset(config=None, **kwargs):
    model_name, _ = config.model.type.split('@')
    dataset_name, _ = config.data.type.split('@')

    min_pixels = getattr(config.vlm, 'min_pixels', None)
    max_pixels = getattr(config.vlm, 'max_pixels', None)
    processor = get_vlm_processor(
        model_name, config.data.root,
        min_pixels=min_pixels, max_pixels=max_pixels)

    image_dir = config.vlm.image_dir
    if not os.path.isabs(image_dir):
        image_dir = os.path.join(config.data.root, image_dir)

    if dataset_name.lower() == 'llava-instruct-150k':
        json_path = os.path.join(
            config.data.root, 'llava_instruct_150k',
            'llava_instruct_150k.json')
        if not os.path.exists(json_path):
            alt_path = os.path.join(
                config.data.root, 'llava_instruct_150k.json')
            if os.path.exists(alt_path):
                json_path = alt_path
            else:
                raise FileNotFoundError(
                    f'LLaVA-Instruct-150K JSON not found at {json_path}. '
                    f'Download with: huggingface-cli download '
                    f'liuhaotian/LLaVA-Instruct-150K '
                    f'--local-dir {config.data.root}/llava_instruct_150k')

        logger.info(f'Loading LLaVA-Instruct-150K from {json_path}')
        with open(json_path, 'r') as f:
            list_data_dict = json.load(f)
        logger.info(f'Loaded {len(list_data_dict)} samples')
    else:
        raise ValueError(f'Unsupported VLM dataset: {dataset_name}')

    dataset = VLMDataset(
        list_data_dict, processor, image_dir,
        max_length=config.vlm.tok_len)

    return dataset, config
