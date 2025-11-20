"""
GUI Worker Threads

Background threads for running SENTINEL system processing without blocking the GUI.
"""

import logging

logger = logging.getLogger(__name__)

# Auto-generated exports
from .sentinel_worker import SentinelWorker

__all__ = ['SentinelWorker']

logger.debug("GUI workers module initialized")
