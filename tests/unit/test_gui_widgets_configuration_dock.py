"""
Unit tests for Configuration Dock Widget
"""

import pytest
import sys
import os
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui.widgets import ConfigurationDockWidget, LabeledSlider


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


class TestLabeledSlider:
    """Test LabeledSlider widget"""
    
    def test_labeled_slider_creation(self, qapp):
        """Test creating a labeled slider"""
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
    
    def test_labeled_slider_value_change(self, qapp):
        """Test slider value change"""
        slider = LabeledSlider("Test", 0.0, 1.0, 0.5)
        
        # Track value changes
        values = []
        slider.value_changed.connect(lambda v: values.append(v))
        
        # Change value
        slider.set_value(0.7)
        
        assert slider.get_value() == 0.7
        assert len(values) > 0
    
    def test_labeled_slider_integer_mode(self, qapp):
        """Test slider in integer mode"""
        slider = LabeledSlider(
            "Integer Param",
            min_value=10,
            max_value=60,
            current_value=30,
            step=1,
            decimals=0
        )
        
        assert slider.get_value() == 30
        assert not slider.use_float


class TestConfigurationDockWidget:
    """Test ConfigurationDockWidget"""
    
    def test_widget_creation(self, qapp, temp_config_file):
        """Test creating configuration dock widget"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        assert dock is not None
        assert dock.windowTitle() == "Configuration"
        assert dock.config_path == temp_config_file
        assert len(dock.config) > 0
    
    def test_load_config(self, qapp, temp_config_file):
        """Test loading configuration"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        assert 'system' in dock.config
        assert 'cameras' in dock.config
        assert dock.config['system']['name'] == 'SENTINEL'
    
    def test_tabs_created(self, qapp, temp_config_file):
        """Test that all tabs are created"""
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
    
    def test_parameter_change(self, qapp, temp_config_file):
        """Test parameter value change"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        # Track config changes
        configs = []
        dock.config_changed.connect(lambda c: configs.append(c))
        
        # Change a parameter
        dock._on_param_changed('models.detection.confidence_threshold', 0.6)
        
        assert dock.config['models']['detection']['confidence_threshold'] == 0.6
        assert dock.has_unsaved_changes
        assert len(configs) > 0
    
    def test_unsaved_changes_indicator(self, qapp, temp_config_file):
        """Test unsaved changes indicator"""
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
    
    def test_save_config(self, qapp, temp_config_file):
        """Test saving configuration"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        # Make a change
        dock._on_param_changed('models.detection.confidence_threshold', 0.6)
        
        # Mock the message box
        with patch('gui.widgets.configuration_dock.QMessageBox.question', return_value=MagicMock()):
            with patch('gui.widgets.configuration_dock.QMessageBox.information'):
                # Save config
                dock._save_config()
        
        # Check that backup was created
        backup_path = f"{temp_config_file}.backup"
        assert os.path.exists(backup_path)
        
        # Check that changes are cleared
        assert not dock.has_unsaved_changes
    
    def test_reset_config(self, qapp, temp_config_file):
        """Test resetting configuration"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        original_value = dock.config['models']['detection']['confidence_threshold']
        
        # Make a change
        dock._on_param_changed('models.detection.confidence_threshold', 0.8)
        assert dock.config['models']['detection']['confidence_threshold'] == 0.8
        
        # Reset
        with patch('gui.widgets.configuration_dock.QMessageBox.information'):
            dock._reset_config()
        
        # Should be back to original
        assert dock.config['models']['detection']['confidence_threshold'] == original_value
        assert not dock.has_unsaved_changes
    
    def test_export_profile(self, qapp, temp_config_file):
        """Test exporting configuration profile"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        # Create temp export file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            export_path = f.name
        
        try:
            # Export
            with patch('gui.widgets.configuration_dock.QMessageBox.information'):
                dock._export_profile(export_path)
            
            # Check file exists and contains data
            assert os.path.exists(export_path)
            
            with open(export_path, 'r') as f:
                exported_config = yaml.safe_load(f)
            
            assert 'system' in exported_config
            assert exported_config['system']['name'] == 'SENTINEL'
            
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_import_profile(self, qapp, temp_config_file):
        """Test importing configuration profile"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        # Create a different config to import
        import_config = {
            'system': {'name': 'IMPORTED'},
            'models': {
                'detection': {
                    'confidence_threshold': 0.9
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(import_config, f)
            import_path = f.name
        
        try:
            # Import
            with patch('gui.widgets.configuration_dock.QMessageBox.information'):
                dock._import_profile(import_path)
            
            # Check config was updated
            assert dock.config['system']['name'] == 'IMPORTED'
            assert dock.has_unsaved_changes
            
        finally:
            if os.path.exists(import_path):
                os.unlink(import_path)
    
    def test_get_config(self, qapp, temp_config_file):
        """Test getting current configuration"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        config = dock.get_config()
        
        assert isinstance(config, dict)
        assert 'system' in config
        assert config == dock.config
    
    def test_has_changes(self, qapp, temp_config_file):
        """Test checking for unsaved changes"""
        dock = ConfigurationDockWidget(temp_config_file)
        
        assert not dock.has_changes()
        
        dock._on_param_changed('models.detection.confidence_threshold', 0.7)
        
        assert dock.has_changes()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
