"""Worker wiring for AdaSparse-LoRA."""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.adasparse_lora_client import AdaSparseLoRAClient
from federatedscope.contrib.worker.methods.adasparse_lora_server import AdaSparseLoRAServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'adasparse_lora', 'adasparse-lora'}


def call_adasparse_lora_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using AdaSparse-LoRA extracted client + server.')
        return {
            'client': AdaSparseLoRAClient,
            'server': AdaSparseLoRAServer,
        }
    return None


register_worker('adasparse_lora', call_adasparse_lora_worker)