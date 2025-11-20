"""
PyQt6 GUI Application for SENTINEL System

This module provides a professional desktop GUI for monitoring and controlling
the SENTINEL contextual safety intelligence platform.
"""

import logging

logger = logging.getLogger(__name__)

from .main_window import SENTINELMainWindow

__all__ = ['SENTINELMainWindow']

logger.debug("GUI module initialized")
