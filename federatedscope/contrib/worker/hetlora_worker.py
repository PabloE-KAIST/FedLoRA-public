"""Worker wiring for HetLoRA.

This worker now points the hetlora method to the extracted method-specific
HetLoRA client and server.
"""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.hetlora_client import HetLoRAClient
from federatedscope.contrib.worker.methods.hetlora_server import HetLoRAServer

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'hetlora', 'het-lora'}


def call_hetlora_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using extracted HetLoRA client + server')
        return {
            'client': HetLoRAClient,
            'server': HetLoRAServer,
        }
    return None


register_worker('hetlora', call_hetlora_worker)
