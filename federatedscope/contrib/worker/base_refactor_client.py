"""
Compatibility shim for BaseRefactorClient.

The shared bandwidth tracking functionality has been merged directly into
the core Client class in federatedscope.core.workers.client.

This module re-exports the core Client as BaseRefactorClient for backward
compatibility with method-specific client classes that extend it.

NOTE: New code should extend federatedscope.core.workers.Client directly.
"""
import logging

from federatedscope.core.workers import Client as CoreClient

logger = logging.getLogger(__name__)

# Re-export core Client as BaseRefactorClient for backward compatibility
BaseRefactorClient = CoreClient

logger.debug(
    "BaseRefactorClient is now an alias for core Client. "
    "Bandwidth tracking is built into the shared Client class."
)
