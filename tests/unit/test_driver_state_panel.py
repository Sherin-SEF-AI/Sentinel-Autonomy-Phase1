"""
Unit tests for Driver State Panel widgets

Tests the driver state panel components without requiring GUI display.
"""

import sys
import pytest

# Add src to path
sys.path.insert(0, 'src')


def test_circular_gauge_import():
    """Test that CircularGaugeWidget can be imported"""
    from gui.widgets.circular_gauge import CircularGaugeWidget
    assert CircularGaugeWidget is not None


def test_gaze_direction_import():
    """Test that GazeDirectionWidget can be imported"""
    from gui.widgets.gaze_direction import GazeDirectionWidget
    assert GazeDirectionWidget is not None


def test_metric_display_import():
    """Test that metric display widgets can be imported"""
    from gui.widgets.metric_display import (
        MetricDisplayWidget,
        MetricsGridWidget,
        DriverMetricsPanel
    )
    assert MetricDisplayWidget is not None
    assert MetricsGridWidget is not None
    assert DriverMetricsPanel is not None


def test_status_indicator_import():
    """Test that status indicator widgets can be imported"""
    from gui.widgets.status_indicator import (
        StatusIndicatorWidget,
        DriverStatusPanel
    )
    assert StatusIndicatorWidget is not None
    assert DriverStatusPanel is not None


def test_warning_animations_import():
    """Test that warning animation classes can be imported"""
    from gui.widgets.warning_animations import (
        WarningAnimationManager,
        ThresholdMonitor
    )
    assert WarningAnimationManager is not None
    assert ThresholdMonitor is not None


def test_driver_state_panel_import():
    """Test that DriverStatePanel can be imported"""
    from gui.widgets.driver_state_panel import DriverStatePanel
    assert DriverStatePanel is not None


def test_gaze_zone_calculation():
    """Test attention zone calculation from yaw angle (without widget instantiation)"""
    from gui.widgets.gaze_direction import GazeDirectionWidget
    
    # Test the zone definitions
    zones = GazeDirectionWidget.ZONES
    assert 'front' in zones
    assert 'front_left' in zones
    assert 'left' in zones
    assert 'rear_left' in zones
    assert 'rear' in zones
    assert 'rear_right' in zones
    assert 'right' in zones
    assert 'front_right' in zones
    
    # Verify zone structure
    assert len(zones) == 8
    for zone_name, zone_data in zones.items():
        assert len(zone_data) == 3  # (zone_id, yaw_min, yaw_max)


def test_threshold_monitor():
    """Test threshold monitoring logic"""
    from gui.widgets.warning_animations import WarningAnimationManager, ThresholdMonitor
    
    manager = WarningAnimationManager()
    monitor = ThresholdMonitor(manager)
    
    # Register a metric (higher is better)
    monitor.register_metric(
        'test_metric',
        warning_threshold=50.0,
        critical_threshold=30.0,
        reverse=False
    )
    
    # Test OK state
    state = monitor.update_metric('test_metric', 80.0)
    assert state == 'ok'
    
    # Test warning state
    state = monitor.update_metric('test_metric', 45.0)
    assert state == 'warning'
    
    # Test critical state
    state = monitor.update_metric('test_metric', 25.0)
    assert state == 'critical'
    
    # Test back to OK
    state = monitor.update_metric('test_metric', 75.0)
    assert state == 'ok'


def test_threshold_monitor_reverse():
    """Test threshold monitoring with reverse logic (lower is better)"""
    from gui.widgets.warning_animations import WarningAnimationManager, ThresholdMonitor
    
    manager = WarningAnimationManager()
    monitor = ThresholdMonitor(manager)
    
    # Register a metric (lower is better, e.g., blink rate)
    monitor.register_metric(
        'blink_rate',
        warning_threshold=30.0,
        critical_threshold=40.0,
        reverse=True
    )
    
    # Test OK state (low value)
    state = monitor.update_metric('blink_rate', 20.0)
    assert state == 'ok'
    
    # Test warning state
    state = monitor.update_metric('blink_rate', 35.0)
    assert state == 'warning'
    
    # Test critical state
    state = monitor.update_metric('blink_rate', 45.0)
    assert state == 'critical'


def test_warning_animation_manager():
    """Test warning animation manager"""
    from gui.widgets.warning_animations import WarningAnimationManager
    
    manager = WarningAnimationManager()
    
    # Test sound enable/disable
    manager.set_sounds_enabled(True)
    manager.set_sounds_enabled(False)
    
    # Test that manager can be created
    assert manager is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
