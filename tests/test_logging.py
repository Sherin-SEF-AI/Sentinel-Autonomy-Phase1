"""Tests for logging infrastructure."""

import pytest
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.logging import LoggerSetup


def test_logger_setup():
    """Test logger initialization."""
    logger = LoggerSetup.setup(log_level='INFO', log_dir='logs')
    
    assert logger is not None
    assert logger.level == logging.INFO


def test_logger_levels():
    """Test different logging levels."""
    for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        logger = LoggerSetup.setup(log_level=level, log_dir='logs')
        assert logger.level == getattr(logging, level)


def test_get_logger():
    """Test getting module-specific logger."""
    logger = LoggerSetup.get_logger('test_module')
    
    assert logger is not None
    assert logger.name == 'test_module'


def test_set_level():
    """Test changing log level at runtime."""
    LoggerSetup.setup(log_level='INFO', log_dir='logs')
    LoggerSetup.set_level('DEBUG')
    
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
