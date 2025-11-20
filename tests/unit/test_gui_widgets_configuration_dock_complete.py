"""Test suite for ConfigurationDockWidget module."""

import pytest
import yaml
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Ensure QApplication exists for widget testing
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def sample_config():
    """Fixture providing sample configuration for testing."""
    return {
        'system': {
            'name': 'SENTINEL',
            'version': '1.0',
            'log_level': 'INFO'
        },
        'cameras': {
            'interior': {
                'device': 0,
                'resolution': [640, 480],
                'fps': 30,
                'calibration': 'configs/calibration/interior.yaml'
            },
            'front_left': {
                'device': 1,
                'resolution': [1280, 720],
                'fps': 30,
                'calibration': 'configs/calibration/front_left.yaml'
            },
            'front_right': {
                'device': 2,
                'resolution': [1280, 720],
                'fps': 30,
                'calibration': 'configs/calibration/front_right.yaml'
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
        'fusion': {
            'iou_threshold_3d': 0.3
        },
        'tracking': {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3
        },
        'risk_assessment': {
            'ttc_calculation': {
                'safety_margin': 1.5
            },
            'trajectory_prediction': {
                'horizon': 3.0,
                'dt': 0.1
            },
            'base_risk_weights': {
                'ttc': 0.4,
                'trajectory_conflict': 0.3,
                'vulnerability': 0.2,
                'relative_speed': 0.1
            },
            'thresholds': {
                'hazard_detection': 0.3,
                'intervention': 0.7,
                'critical': 0.9
            }
        },
        'alerts': {
            'suppression': {
                'duplicate_window': 5.0,
                'max_simultaneous': 2
            },
            'escalation': {
                'critical_threshold': 0.9,
                'high_threshold': 0.7,
                'medium_threshold': 0.5
            },
            'modalities': {
                'visual': {
                    'display_duration': 3.0,
                    'flash_rate': 2
                },
                'audio': {
                    'volume': 0.8
                }
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    backup_path = f"{temp_path}.backup"
    if os.path.exists(backup_path):
        os.unlink(backup_path)


@pytest.fixture
def config_dock(qapp, temp_config_file):
    """Fixture creating an instance of ConfigurationDockWidget for testing."""
    from src.gui.widgets.configuration_dock import ConfigurationDockWidget
    
    widget = ConfigurationDockWidget(config_path=temp_config_file)
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.fixture
def labeled_slider(qapp):
    """Fixture creating a LabeledSlider for testing."""
    from src.gui.widgets.configuration_dock import LabeledSlider
    
    slider = LabeledSlider(
        label="Test Parameter",
        min_value=0.0,
        max_value=1.0,
        current_value=0.5,
        step=0.1,
        decimals=2,
        units="units",
        tooltip="Test tooltip"
    )
    yield slider
    slider.deleteLater()


class TestLabeledSlider:
    """Test suite for LabeledSlider class."""
    
    def test_initialization_float(self, labeled_slider):
        """Test that LabeledSlider initializes correctly with float values."""
        assert labeled_slider is not None
        assert labeled_slider.min_value == 0.0
        assert labeled_slider.max_value == 1.0
        assert labeled_slider.step == 0.1
        assert labeled_slider.decimals == 2
        assert labeled_slider.units == "units"
        assert labeled_slider.use_float is True
    
    def test_initialization_int(self, qapp):
        """Test that LabeledSlider initializes correctly with integer values."""
        from src.gui.widgets.configuration_dock import LabeledSlider
        
        slider = LabeledSlider(
            label="Int Parameter",
            min_value=0,
            max_value=100,
            current_value=50,
            step=1,
            decimals=0
        )
        
        assert slider.use_float is False
        assert slider.get_value() == 50
        slider.deleteLater()
    
    def test_get_value(self, labeled_slider):
        """Test getting current value from slider."""
        value = labeled_slider.get_value()
        assert value == 0.5
    
    def test_set_value(self, labeled_slider):
        """Test setting value programmatically."""
        labeled_slider.set_value(0.75)
        assert labeled_slider.get_value() == 0.75
    
    def test_value_changed_signal(self, labeled_slider, qtbot):
        """Test that value_changed signal is emitted on value change."""
        with qtbot.waitSignal(labeled_slider.value_changed, timeout=1000) as blocker:
            labeled_slider.set_value(0.8)
        
        assert blocker.args[0] == 0.8
    
    def test_slider_spinbox_sync(self, labeled_slider):
        """Test that slider and spinbox stay synchronized."""
        # Change slider
        labeled_slider.slider.setValue(750)  # 75% of range
        assert abs(labeled_slider.spinbox.value() - 0.75) < 0.01
        
        # Change spinbox
        labeled_slider.spinbox.setValue(0.25)
        # Slider should update (approximately)
        assert labeled_slider.slider.value() >= 200  # ~25% of 1000 steps
    
    def test_edge_cases_min_value(self, labeled_slider):
        """Test setting minimum value."""
        labeled_slider.set_value(0.0)
        assert labeled_slider.get_value() == 0.0
    
    def test_edge_cases_max_value(self, labeled_slider):
        """Test setting maximum value."""
        labeled_slider.set_value(1.0)
        assert labeled_slider.get_value() == 1.0


class TestConfigurationDockWidget:
    """Test suite for ConfigurationDockWidget class."""
    
    def test_initialization(self, config_dock, temp_config_file):
        """Test that ConfigurationDockWidget initializes correctly."""
        assert config_dock is not None
        assert config_dock.config_path == temp_config_file
        assert isinstance(config_dock.config, dict)
        assert len(config_dock.config) > 0
        assert config_dock.has_unsaved_changes is False
    
    def test_load_config_success(self, config_dock, sample_config):
        """Test successful configuration loading."""
        assert 'cameras' in config_dock.config
        assert 'models' in config_dock.config
        assert 'risk_assessment' in config_dock.config
        assert config_dock.config['cameras']['interior']['fps'] == 30
    
    def test_load_config_file_not_found(self, qapp):
        """Test handling of missing configuration file."""
        from src.gui.widgets.configuration_dock import ConfigurationDockWidget
        
        with patch('src.gui.widgets.configuration_dock.QMessageBox.critical'):
            widget = ConfigurationDockWidget(config_path='nonexistent.yaml')
            assert widget.config == {}
            assert widget.original_config == {}
            widget.deleteLater()
    
    def test_load_config_invalid_yaml(self, qapp):
        """Test handling of invalid YAML file."""
        from src.gui.widgets.configuration_dock import ConfigurationDockWidget
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            with patch('src.gui.widgets.configuration_dock.QMessageBox.critical'):
                widget = ConfigurationDockWidget(config_path=temp_path)
                assert widget.config == {}
                widget.deleteLater()
        finally:
            os.unlink(temp_path)
    
    def test_tabs_created(self, config_dock):
        """Test that all configuration tabs are created."""
        assert config_dock.tab_widget.count() == 5
        
        tab_names = []
        for i in range(config_dock.tab_widget.count()):
            tab_names.append(config_dock.tab_widget.tabText(i))
        
        assert "Cameras" in tab_names
        assert "Detection" in tab_names
        assert "DMS" in tab_names
        assert "Risk" in tab_names
        assert "Alerts" in tab_names
    
    def test_param_widgets_registered(self, config_dock):
        """Test that parameter widgets are registered."""
        assert len(config_dock.param_widgets) > 0
        
        # Check for some expected parameters
        expected_keys = [
            'models.detection.confidence_threshold',
            'tracking.max_age',
            'risk_assessment.ttc_calculation.safety_margin'
        ]
        
        for key in expected_keys:
            assert key in config_dock.param_widgets
    
    def test_on_param_changed(self, config_dock, qtbot):
        """Test parameter change handling."""
        # Monitor config_changed signal
        with qtbot.waitSignal(config_dock.config_changed, timeout=1000):
            config_dock._on_param_changed('models.detection.confidence_threshold', 0.6)
        
        # Check config was updated
        assert config_dock.config['models']['detection']['confidence_threshold'] == 0.6
        
        # Check unsaved changes flag
        assert config_dock.has_unsaved_changes is True
        assert config_dock.save_button.isEnabled() is True
    
    def test_on_param_changed_nested_creation(self, config_dock):
        """Test parameter change creates nested dictionaries if needed."""
        config_dock._on_param_changed('new.nested.parameter', 42)
        
        assert 'new' in config_dock.config
        assert 'nested' in config_dock.config['new']
        assert config_dock.config['new']['nested']['parameter'] == 42
    
    def test_save_config_success(self, config_dock, temp_config_file, qtbot):
        """Test successful configuration save."""
        # Make a change
        config_dock._on_param_changed('models.detection.confidence_threshold', 0.7)
        
        # Mock message box
        with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
            with qtbot.waitSignal(config_dock.config_saved, timeout=1000):
                config_dock._save_config()
        
        # Check file was written
        with open(temp_config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config['models']['detection']['confidence_threshold'] == 0.7
        
        # Check unsaved changes cleared
        assert config_dock.has_unsaved_changes is False
        assert config_dock.save_button.isEnabled() is False
    
    def test_save_config_creates_backup(self, config_dock, temp_config_file):
        """Test that save creates a backup of existing config."""
        config_dock._on_param_changed('test.param', 123)
        
        with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
            config_dock._save_config()
        
        backup_path = f"{temp_config_file}.backup"
        assert os.path.exists(backup_path)
    
    def test_save_config_error_handling(self, config_dock):
        """Test error handling during save."""
        # Set invalid path
        config_dock.config_path = '/invalid/path/config.yaml'
        
        with patch('src.gui.widgets.configuration_dock.QMessageBox.critical') as mock_error:
            config_dock._save_config()
            mock_error.assert_called_once()
    
    def test_reset_config(self, config_dock):
        """Test resetting configuration to defaults."""
        # Make changes
        original_value = config_dock.config['models']['detection']['confidence_threshold']
        config_dock._on_param_changed('models.detection.confidence_threshold', 0.9)
        
        assert config_dock.has_unsaved_changes is True
        
        # Reset
        with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
            config_dock._reset_config()
        
        # Check values restored
        assert config_dock.config['models']['detection']['confidence_threshold'] == original_value
        assert config_dock.has_unsaved_changes is False
    
    def test_reset_updates_widgets(self, config_dock):
        """Test that reset updates all parameter widgets."""
        # Get a widget
        widget_key = 'models.detection.confidence_threshold'
        widget = config_dock.param_widgets[widget_key]
        original_value = widget.get_value()
        
        # Change value
        widget.set_value(0.9)
        
        # Reset
        with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
            config_dock._reset_config()
        
        # Check widget restored
        assert widget.get_value() == original_value
    
    def test_import_profile_success(self, config_dock, sample_config):
        """Test successful profile import."""
        # Create a different config file
        modified_config = sample_config.copy()
        modified_config['models']['detection']['confidence_threshold'] = 0.8
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(modified_config, f)
            import_path = f.name
        
        try:
            with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
                config_dock._import_profile(import_path)
            
            assert config_dock.config['models']['detection']['confidence_threshold'] == 0.8
            assert config_dock.has_unsaved_changes is True
        finally:
            os.unlink(import_path)
    
    def test_import_profile_invalid_format(self, config_dock):
        """Test import with invalid configuration format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("not_a_dict")
            import_path = f.name
        
        try:
            with patch('src.gui.widgets.configuration_dock.QMessageBox.critical') as mock_error:
                config_dock._import_profile(import_path)
                mock_error.assert_called_once()
        finally:
            os.unlink(import_path)
    
    def test_export_profile_success(self, config_dock):
        """Test successful profile export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            export_path = f.name
        
        try:
            with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
                config_dock._export_profile(export_path)
            
            # Verify file was created and contains config
            assert os.path.exists(export_path)
            
            with open(export_path, 'r') as f:
                exported_config = yaml.safe_load(f)
            
            assert 'cameras' in exported_config
            assert 'models' in exported_config
        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)
    
    def test_export_profile_error_handling(self, config_dock):
        """Test error handling during export."""
        with patch('src.gui.widgets.configuration_dock.QMessageBox.critical') as mock_error:
            config_dock._export_profile('/invalid/path/export.yaml')
            mock_error.assert_called_once()
    
    def test_get_config(self, config_dock):
        """Test getting current configuration."""
        config = config_dock.get_config()
        
        assert isinstance(config, dict)
        assert 'cameras' in config
        assert 'models' in config
    
    def test_has_changes(self, config_dock):
        """Test checking for unsaved changes."""
        assert config_dock.has_changes() is False
        
        config_dock._on_param_changed('test.param', 123)
        
        assert config_dock.has_changes() is True
    
    def test_changes_indicator_updates(self, config_dock):
        """Test that changes indicator updates correctly."""
        # Initially no changes
        assert config_dock.changes_label.text() == ""
        assert config_dock.save_button.isEnabled() is False
        
        # Make a change
        config_dock._on_param_changed('test.param', 123)
        
        assert "Unsaved Changes" in config_dock.changes_label.text()
        assert config_dock.save_button.isEnabled() is True
        
        # Save
        with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
            config_dock._save_config()
        
        assert config_dock.changes_label.text() == ""
        assert config_dock.save_button.isEnabled() is False
    
    @pytest.mark.performance
    def test_initialization_performance(self, qapp, temp_config_file):
        """Test that initialization completes within performance requirements."""
        import time
        from src.gui.widgets.configuration_dock import ConfigurationDockWidget
        
        start_time = time.perf_counter()
        widget = ConfigurationDockWidget(config_path=temp_config_file)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        widget.deleteLater()
        
        # GUI initialization should be fast (< 500ms)
        assert execution_time_ms < 500, f"Initialization took {execution_time_ms:.2f}ms, expected < 500ms"
    
    @pytest.mark.performance
    def test_param_change_performance(self, config_dock):
        """Test that parameter changes are processed quickly."""
        import time
        
        start_time = time.perf_counter()
        config_dock._on_param_changed('models.detection.confidence_threshold', 0.75)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Parameter updates should be near-instant (< 10ms)
        assert execution_time_ms < 10, f"Parameter change took {execution_time_ms:.2f}ms, expected < 10ms"
    
    def test_multiple_param_changes(self, config_dock):
        """Test handling multiple parameter changes."""
        changes = [
            ('models.detection.confidence_threshold', 0.6),
            ('tracking.max_age', 40),
            ('risk_assessment.ttc_calculation.safety_margin', 2.0),
            ('alerts.suppression.duplicate_window', 7.0)
        ]
        
        for key, value in changes:
            config_dock._on_param_changed(key, value)
        
        # Verify all changes applied
        assert config_dock.config['models']['detection']['confidence_threshold'] == 0.6
        assert config_dock.config['tracking']['max_age'] == 40
        assert config_dock.config['risk_assessment']['ttc_calculation']['safety_margin'] == 2.0
        assert config_dock.config['alerts']['suppression']['duplicate_window'] == 7.0
    
    def test_signal_emissions(self, config_dock, qtbot):
        """Test that appropriate signals are emitted."""
        # Test config_changed signal
        with qtbot.waitSignal(config_dock.config_changed, timeout=1000) as blocker:
            config_dock._on_param_changed('test.param', 123)
        
        assert isinstance(blocker.args[0], dict)
        
        # Test config_saved signal
        with patch('src.gui.widgets.configuration_dock.QMessageBox.information'):
            with qtbot.waitSignal(config_dock.config_saved, timeout=1000) as blocker:
                config_dock._save_config()
        
        assert isinstance(blocker.args[0], str)  # File path
