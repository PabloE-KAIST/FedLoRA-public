"""
Compatibility shim for BaseRefactorServer.

The shared bandwidth management functionality has been merged directly into
the core Server class in federatedscope.core.workers.server.

This module re-exports the core Server as BaseRefactorServer for backward
compatibility with method-specific server classes that extend it.

NOTE: New code should extend federatedscope.core.workers.Server directly.
"""
import logging

from federatedscope.core.workers import Server as CoreServer

logger = logging.getLogger(__name__)

# Re-export core Server as BaseRefactorServer for backward compatibility
BaseRefactorServer = CoreServer

logger.debug(
    "BaseRefactorServer is now an alias for core Server. "
    "Bandwidth management is built into the shared Server class."
)
