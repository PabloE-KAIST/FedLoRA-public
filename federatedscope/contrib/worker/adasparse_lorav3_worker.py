"""Worker wiring for AdaSparse-LoRAv3."""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.adasparse_lorav3_client import AdaSparseLoRAv3Client
from federatedscope.contrib.worker.methods.adasparse_lorav3_server import AdaSparseLoRAv3Server

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'adasparse_lorav3', 'adasparse-lorav3'}


def call_adasparse_lorav3_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using AdaSparse-LoRAv3 extracted client + server')
        return {
            'client': AdaSparseLoRAv3Client,
            'server': AdaSparseLoRAv3Server,
        }
    return None


register_worker('adasparse_lorav3', call_adasparse_lorav3_worker)
