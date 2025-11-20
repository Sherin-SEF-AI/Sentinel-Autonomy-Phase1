"""SENTINEL - Contextual Safety Intelligence Platform."""

import logging

logger = logging.getLogger(__name__)

# Auto-generated exports
try:
    logger.debug("Importing SENTINEL core modules...")
    from .gui_main import main as gui_main
    from .main import SentinelSystem, main
    
    __all__ = ['SentinelSystem', 'gui_main', 'main']
    logger.info("SENTINEL package initialized successfully: modules=['SentinelSystem', 'gui_main', 'main']")
except ImportError as e:
    # Allow imports to work even if dependencies are missing
    logger.warning(f"SENTINEL package initialization incomplete: missing_dependencies={e}")
    __all__ = []
