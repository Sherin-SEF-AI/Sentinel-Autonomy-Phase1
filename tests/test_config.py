"""Tests for configuration management."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.config import ConfigManager


def test_config_loading():
    """Test configuration file loading."""
    config = ConfigManager('configs/default.yaml')
    
    assert config.config is not None
    assert 'system' in config.config
    assert 'cameras' in config.config
    assert 'models' in config.config


def test_config_get():
    """Test getting configuration values."""
    config = ConfigManager('configs/default.yaml')
    
    # Test simple key
    assert config.get('system.name') == 'SENTINEL'
    assert config.get('system.version') == '1.0'
    
    # Test nested key
    assert config.get('cameras.interior.device') == 0
    assert config.get('cameras.interior.fps') == 30
    
    # Test default value
    assert config.get('nonexistent.key', 'default') == 'default'


def test_config_get_section():
    """Test getting configuration sections."""
    config = ConfigManager('configs/default.yaml')
    
    cameras = config.get_section('cameras')
    assert 'interior' in cameras
    assert 'front_left' in cameras
    assert 'front_right' in cameras


def test_config_set():
    """Test setting configuration values."""
    config = ConfigManager('configs/default.yaml')
    
    config.set('system.test_value', 42)
    assert config.get('system.test_value') == 42


def test_config_validation():
    """Test configuration validation."""
    config = ConfigManager('configs/default.yaml')
    
    assert config.validate() is True


def test_config_file_not_found():
    """Test handling of missing configuration file."""
    with pytest.raises(FileNotFoundError):
        ConfigManager('nonexistent.yaml')
