"""Wworker wiring for AdaSparse-LoRAv2."""

import logging

from federatedscope.register import register_worker
from federatedscope.contrib.worker.methods.adasparse_lorav2_client import AdaSparseLoRAv2Client
from federatedscope.contrib.worker.methods.adasparse_lorav2_server import AdaSparseLoRAv2Server

logger = logging.getLogger(__name__)

_METHOD_ALIASES = {'adasparse_lorav2', 'adasparse-lorav2'}


def call_adasparse_lorav2_worker(method):
    if method is None:
        return None
    if method.lower() in _METHOD_ALIASES:
        logger.info('Using AdaSparse-LoRAv2 extracted client + server')
        return {
            'client': AdaSparseLoRAv2Client,
            'server': AdaSparseLoRAv2Server,
        }
    return None


register_worker('adasparse_lorav2', call_adasparse_lorav2_worker)
