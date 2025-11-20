"""
Simple unit tests for Configuration Dock Widget
"""

import pytest
import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def qapp():
    """Create QApplication instance"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def temp_config_file():
    """Create temporary config file"""
    config_data = {
        'system': {
            'name': 'SENTINEL',
            'version': '1.0'
        },
        'cameras': {
            'interior': {
                'device': 0,
                'fps': 30
            }
        },
        'models': {
            'detection': {
                'confidence_threshold': 0.5,
                'nms_threshold': 0.4
            },
            'segmentation': {
                'smoothing_alpha': 0.7
            }
        },
        'risk_assessment': {
            'ttc_calculation': {
                'safety_margin': 1.5
            },
            'thresholds': {
                'critical': 0.9
            }
        },
        'alerts': {
            'suppression': {
                'duplicate_window': 5.0
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    backup_path = f"{temp_path}.backup"
    if os.path.exists(backup_path):
        os.unlink(backup_path)


def test_labeled_slider_import(qapp):
    """Test importing LabeledSlider"""
    from gui.widgets.configuration_dock import LabeledSlider
    assert LabeledSlider is not None


def test_labeled_slider_creation(qapp):
    """Test creating a labeled slider"""
    from gui.widgets.configuration_dock import LabeledSlider
    
    slider = LabeledSlider(
        "Test Parameter",
        min_value=0.0,
        max_value=1.0,
        current_value=0.5,
        step=0.1,
        decimals=2,
        units="units",
        tooltip="Test tooltip"
    )
    
    assert slider is not None
    assert slider.get_value() == 0.5
    assert slider.min_value == 0.0
    assert slider.max_value == 1.0


def test_labeled_slider_value_change(qapp):
    """Test slider value change"""
    from gui.widgets.configuration_dock import LabeledSlider
    
    slider = LabeledSlider("Test", 0.0, 1.0, 0.5)
    
    # Track value changes
    values = []
    slider.value_changed.connect(lambda v: values.append(v))
    
    # Change value
    slider.set_value(0.7)
    
    assert slider.get_value() == 0.7
    assert len(values) > 0


def test_configuration_dock_import(qapp):
    """Test importing ConfigurationDockWidget"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    assert ConfigurationDockWidget is not None


def test_configuration_dock_creation(qapp, temp_config_file):
    """Test creating configuration dock widget"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    assert dock is not None
    assert dock.windowTitle() == "Configuration"
    assert dock.config_path == temp_config_file
    assert len(dock.config) > 0


def test_load_config(qapp, temp_config_file):
    """Test loading configuration"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    assert 'system' in dock.config
    assert 'cameras' in dock.config
    assert dock.config['system']['name'] == 'SENTINEL'


def test_tabs_created(qapp, temp_config_file):
    """Test that all tabs are created"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    tab_widget = dock.tab_widget
    assert tab_widget.count() == 5
    
    # Check tab names
    tab_names = [tab_widget.tabText(i) for i in range(tab_widget.count())]
    assert 'Cameras' in tab_names
    assert 'Detection' in tab_names
    assert 'DMS' in tab_names
    assert 'Risk' in tab_names
    assert 'Alerts' in tab_names


def test_parameter_change(qapp, temp_config_file):
    """Test parameter value change"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    # Track config changes
    configs = []
    dock.config_changed.connect(lambda c: configs.append(c))
    
    # Change a parameter
    dock._on_param_changed('models.detection.confidence_threshold', 0.6)
    
    assert dock.config['models']['detection']['confidence_threshold'] == 0.6
    assert dock.has_unsaved_changes
    assert len(configs) > 0


def test_unsaved_changes_indicator(qapp, temp_config_file):
    """Test unsaved changes indicator"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    # Initially no changes
    assert not dock.has_unsaved_changes
    assert not dock.save_button.isEnabled()
    
    # Make a change
    dock._on_param_changed('models.detection.confidence_threshold', 0.6)
    
    # Should show unsaved changes
    assert dock.has_unsaved_changes
    assert dock.save_button.isEnabled()
    assert "Unsaved" in dock.changes_label.text()


def test_get_config(qapp, temp_config_file):
    """Test getting current configuration"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    config = dock.get_config()
    
    assert isinstance(config, dict)
    assert 'system' in config
    assert config == dock.config


def test_has_changes(qapp, temp_config_file):
    """Test checking for unsaved changes"""
    from gui.widgets.configuration_dock import ConfigurationDockWidget
    
    dock = ConfigurationDockWidget(temp_config_file)
    
    assert not dock.has_changes()
    
    dock._on_param_changed('models.detection.confidence_threshold', 0.7)
    
    assert dock.has_changes()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
