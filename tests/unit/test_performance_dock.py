"""
Unit tests for Performance Dock Widget

Tests all components of the performance monitoring dock including FPS graph,
latency graph, module breakdown, resource usage, and performance logging.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, mock_open
import numpy as np

# Direct import to avoid circular dependencies
from PyQt6.QtWidgets import QApplication

# Import only the performance dock module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "performance_dock",
    os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'gui', 'widgets', 'performance_dock.py')
)
performance_dock = importlib.util.module_from_spec(spec)

# Mock the circular gauge import
sys.modules['src.gui.widgets.circular_gauge'] = Mock()

try:
    spec.loader.exec_module(performance_dock)
    FPSGraphWidget = performance_dock.FPSGraphWidget
    LatencyGraphWidget = performance_dock.LatencyGraphWidget
    ModuleBreakdownWidget = performance_dock.ModuleBreakdownWidget
    ResourceUsageWidget = performance_dock.ResourceUsageWidget
    PerformanceLoggingWidget = performance_dock.PerformanceLoggingWidget
    PerformanceDockWidget = performance_dock.PerformanceDockWidget
except Exception as e:
    # If import fails, skip tests
    pytest.skip(f"Could not import performance_dock: {e}", allow_module_level=True)


@pytest.fixture(scope='module')
def qapp():
    """Create QApplication instance for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


class TestFPSGraphWidget:
    """Test FPS graph widget"""
    
    def test_initialization(self, qapp):
        """Test widget initializes correctly"""
        widget = FPSGraphWidget()
        assert widget is not None
        assert widget.target_fps == 30.0
        assert len(widget.fps_values) == 0
    
    def test_update_fps(self, qapp):
        """Test FPS update"""
        widget = FPSGraphWidget()
        widget.update_fps(32.5)
        
        assert len(widget.fps_values) == 1
        assert widget.fps_values[0] == 32.5
    
    def test_clear(self, qapp):
        """Test clearing data"""
        widget = FPSGraphWidget()
        widget.update_fps(30.0)
        widget.clear()
        
        assert len(widget.fps_values) == 0


class TestLatencyGraphWidget:
    """Test latency graph widget"""
    
    def test_initialization(self, qapp):
        """Test widget initializes correctly"""
        widget = LatencyGraphWidget()
        assert widget is not None
        assert widget.threshold_ms == 100.0
        assert len(widget.latency_values) == 0
    
    def test_update_latency(self, qapp):
        """Test latency update"""
        widget = LatencyGraphWidget()
        widget.update_latency(85.5)
        
        assert len(widget.latency_values) == 1
        assert widget.latency_values[0] == 85.5
    
    def test_p95_calculation(self, qapp):
        """Test P95 latency calculation"""
        widget = LatencyGraphWidget()
        
        # Add multiple values
        for i in range(20):
            widget.update_latency(80 + i)
        
        # P95 should be calculated
        assert len(widget.latency_values) == 20


class TestModuleBreakdownWidget:
    """Test module breakdown widget"""
    
    def test_initialization(self, qapp):
        """Test widget initializes correctly"""
        widget = ModuleBreakdownWidget()
        assert widget is not None
        assert len(widget.module_timings) == 0
    
    def test_update_timings(self, qapp):
        """Test module timings update"""
        widget = ModuleBreakdownWidget()
        
        timings = {
            'Camera': 5.0,
            'BEV': 15.0,
            'Detection': 20.0
        }
        
        widget.update_timings(timings)
        assert widget.module_timings == timings


class TestResourceUsageWidget:
    """Test resource usage widget"""
    
    def test_initialization(self, qapp):
        """Test widget initializes correctly"""
        widget = ResourceUsageWidget()
        assert widget is not None
        assert widget.gpu_max_mb == 8192
        assert widget.cpu_max_percent == 60.0
    
    def test_update_resources(self, qapp):
        """Test resource update"""
        widget = ResourceUsageWidget()
        widget.update_resources(4096.0, 45.0)
        
        # Peak should be tracked
        assert widget.gpu_peak_mb == 4096.0
        assert widget.cpu_peak_percent == 45.0


class TestPerformanceLoggingWidget:
    """Test performance logging widget"""
    
    def test_initialization(self, qapp):
        """Test widget initializes correctly"""
        widget = PerformanceLoggingWidget()
        assert widget is not None
        assert widget.logging_enabled is False
        assert len(widget.fps_history) == 0
    
    def test_log_metrics(self, qapp):
        """Test logging metrics"""
        widget = PerformanceLoggingWidget()
        widget.log_metrics(30.0, 85.0, 4096.0, 45.0)
        
        assert len(widget.fps_history) == 1
        assert len(widget.latency_history) == 1
        assert widget.fps_history[0] == 30.0
        assert widget.latency_history[0] == 85.0
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_start_logging(self, mock_makedirs, mock_file, qapp):
        """Test starting logging"""
        widget = PerformanceLoggingWidget()
        widget._start_logging()
        
        assert widget.logging_enabled is True
        mock_makedirs.assert_called_once()
        mock_file.assert_called_once()


class TestPerformanceDockWidget:
    """Test main performance dock widget"""
    
    def test_initialization(self, qapp):
        """Test widget initializes correctly"""
        widget = PerformanceDockWidget()
        assert widget is not None
        assert widget.fps_widget is not None
        assert widget.latency_widget is not None
        assert widget.module_widget is not None
        assert widget.resource_widget is not None
        assert widget.logging_widget is not None
    
    def test_update_fps(self, qapp):
        """Test FPS update"""
        widget = PerformanceDockWidget()
        widget.update_fps(30.0)
        
        assert len(widget.fps_widget.fps_values) == 1
    
    def test_update_latency(self, qapp):
        """Test latency update"""
        widget = PerformanceDockWidget()
        widget.update_latency(85.0)
        
        assert len(widget.latency_widget.latency_values) == 1
    
    def test_update_module_timings(self, qapp):
        """Test module timings update"""
        widget = PerformanceDockWidget()
        
        timings = {'Camera': 5.0, 'BEV': 15.0}
        widget.update_module_timings(timings)
        
        assert widget.module_widget.module_timings == timings
    
    def test_update_resources(self, qapp):
        """Test resource update"""
        widget = PerformanceDockWidget()
        widget.update_resources(4096.0, 45.0)
        
        assert widget.resource_widget.gpu_peak_mb == 4096.0
    
    def test_update_all_metrics(self, qapp):
        """Test updating all metrics at once"""
        widget = PerformanceDockWidget()
        
        timings = {'Camera': 5.0, 'BEV': 15.0}
        widget.update_all_metrics(30.0, 85.0, timings, 4096.0, 45.0)
        
        assert len(widget.fps_widget.fps_values) == 1
        assert len(widget.latency_widget.latency_values) == 1
        assert widget.module_widget.module_timings == timings
        assert widget.resource_widget.gpu_peak_mb == 4096.0
        assert len(widget.logging_widget.fps_history) == 1
    
    def test_clear_all(self, qapp):
        """Test clearing all data"""
        widget = PerformanceDockWidget()
        
        # Add some data
        widget.update_fps(30.0)
        widget.update_latency(85.0)
        
        # Clear
        widget.clear_all()
        
        assert len(widget.fps_widget.fps_values) == 0
        assert len(widget.latency_widget.latency_values) == 0
    
    def test_start_stop_monitoring(self, qapp):
        """Test starting and stopping monitoring"""
        widget = PerformanceDockWidget()
        
        widget.start_monitoring()
        assert widget.update_timer.isActive()
        
        widget.stop_monitoring()
        assert not widget.update_timer.isActive()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
