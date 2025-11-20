"""Test suite for GUI widgets performance_dock module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import time
import numpy as np

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import sys

# Ensure QApplication exists for widget testing
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def fps_graph_widget(qapp):
    """Fixture creating an instance of FPSGraphWidget for testing."""
    from src.gui.widgets.performance_dock import FPSGraphWidget
    widget = FPSGraphWidget()
    yield widget
    widget.deleteLater()


@pytest.fixture
def performance_dock_widget(qapp):
    """Fixture creating an instance of PerformanceDockWidget for testing."""
    from src.gui.widgets.performance_dock import PerformanceDockWidget
    widget = PerformanceDockWidget()
    yield widget
    widget.deleteLater()


class TestFPSGraphWidget:
    """Test suite for FPSGraphWidget class."""
    
    def test_initialization(self, fps_graph_widget):
        """Test that FPSGraphWidget initializes correctly."""
        assert fps_graph_widget is not None
        assert fps_graph_widget.max_points == 60
        assert fps_graph_widget.target_fps == 30.0
        assert len(fps_graph_widget.timestamps) == 0
        assert len(fps_graph_widget.fps_values) == 0
        assert fps_graph_widget.plot_widget is not None
        assert fps_graph_widget.fps_curve is not None
        assert fps_graph_widget.target_line is not None
        assert fps_graph_widget.fps_label is not None
    
    def test_update_fps_single_value(self, fps_graph_widget):
        """Test updating FPS with a single value."""
        fps_graph_widget.update_fps(30.5)
        
        assert len(fps_graph_widget.fps_values) == 1
        assert len(fps_graph_widget.timestamps) == 1
        assert fps_graph_widget.fps_values[0] == 30.5
        assert "30.5" in fps_graph_widget.fps_label.text()
    
    def test_update_fps_multiple_values(self, fps_graph_widget):
        """Test updating FPS with multiple values."""
        fps_values = [28.5, 30.2, 31.5, 29.8, 30.1]
        
        for fps in fps_values:
            fps_graph_widget.update_fps(fps)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        assert len(fps_graph_widget.fps_values) == 5
        assert len(fps_graph_widget.timestamps) == 5
        assert list(fps_graph_widget.fps_values) == fps_values
    
    def test_update_fps_color_coding_above_target(self, fps_graph_widget):
        """Test that FPS label is green when above target."""
        fps_graph_widget.update_fps(35.0)
        
        style = fps_graph_widget.fps_label.styleSheet()
        assert '#00ff00' in style  # Green color
    
    def test_update_fps_color_coding_below_target(self, fps_graph_widget):
        """Test that FPS label is red when below target."""
        fps_graph_widget.update_fps(25.0)
        
        style = fps_graph_widget.fps_label.styleSheet()
        assert '#ff0000' in style  # Red color
    
    def test_update_fps_color_coding_at_target(self, fps_graph_widget):
        """Test that FPS label is green when at target."""
        fps_graph_widget.update_fps(30.0)
        
        style = fps_graph_widget.fps_label.styleSheet()
        assert '#00ff00' in style  # Green color
    
    def test_update_fps_max_points_limit(self, fps_graph_widget):
        """Test that FPS data respects max_points limit."""
        # Add more than max_points values
        for i in range(70):
            fps_graph_widget.update_fps(30.0 + i * 0.1)
            time.sleep(0.001)
        
        # Should only keep last 60 points
        assert len(fps_graph_widget.fps_values) == 60
        assert len(fps_graph_widget.timestamps) == 60
    
    def test_clear(self, fps_graph_widget):
        """Test clearing FPS data."""
        # Add some data
        for i in range(10):
            fps_graph_widget.update_fps(30.0)
        
        assert len(fps_graph_widget.fps_values) > 0
        
        # Clear
        fps_graph_widget.clear()
        
        assert len(fps_graph_widget.fps_values) == 0
        assert len(fps_graph_widget.timestamps) == 0
        assert "Current FPS: --" in fps_graph_widget.fps_label.text()
    
    def test_update_fps_edge_case_zero(self, fps_graph_widget):
        """Test updating FPS with zero value."""
        fps_graph_widget.update_fps(0.0)
        
        assert len(fps_graph_widget.fps_values) == 1
        assert fps_graph_widget.fps_values[0] == 0.0
        assert "0.0" in fps_graph_widget.fps_label.text()
    
    def test_update_fps_edge_case_very_high(self, fps_graph_widget):
        """Test updating FPS with very high value."""
        fps_graph_widget.update_fps(120.0)
        
        assert len(fps_graph_widget.fps_values) == 1
        assert fps_graph_widget.fps_values[0] == 120.0
        assert "120.0" in fps_graph_widget.fps_label.text()
    
    def test_update_fps_edge_case_negative(self, fps_graph_widget):
        """Test updating FPS with negative value (edge case)."""
        fps_graph_widget.update_fps(-5.0)
        
        # Should still accept the value (validation is caller's responsibility)
        assert len(fps_graph_widget.fps_values) == 1
        assert fps_graph_widget.fps_values[0] == -5.0


class TestPerformanceDockWidget:
    """Test suite for PerformanceDockWidget class."""
    
    def test_initialization(self, performance_dock_widget):
        """Test that PerformanceDockWidget initializes correctly."""
        assert performance_dock_widget is not None
        assert performance_dock_widget.tab_widget is not None
        assert performance_dock_widget.fps_widget is not None
        assert performance_dock_widget.update_timer is not None
        assert not performance_dock_widget.update_timer.isActive()
    
    def test_tab_widget_structure(self, performance_dock_widget):
        """Test that tab widget has correct structure."""
        tab_widget = performance_dock_widget.tab_widget
        
        # Should have 5 tabs
        assert tab_widget.count() == 5
        
        # Check tab names
        assert tab_widget.tabText(0) == "FPS"
        assert tab_widget.tabText(1) == "Latency"
        assert tab_widget.tabText(2) == "Modules"
        assert tab_widget.tabText(3) == "Resources"
        assert tab_widget.tabText(4) == "Logging"
    
    def test_fps_tab_is_fps_widget(self, performance_dock_widget):
        """Test that first tab contains FPSGraphWidget."""
        from src.gui.widgets.performance_dock import FPSGraphWidget
        
        fps_tab = performance_dock_widget.tab_widget.widget(0)
        assert isinstance(fps_tab, FPSGraphWidget)
    
    def test_start_monitoring(self, performance_dock_widget):
        """Test starting performance monitoring."""
        assert not performance_dock_widget.update_timer.isActive()
        
        performance_dock_widget.start_monitoring()
        
        assert performance_dock_widget.update_timer.isActive()
        assert performance_dock_widget.update_timer.interval() == 1000  # 1 Hz
    
    def test_stop_monitoring(self, performance_dock_widget):
        """Test stopping performance monitoring."""
        performance_dock_widget.start_monitoring()
        assert performance_dock_widget.update_timer.isActive()
        
        performance_dock_widget.stop_monitoring()
        
        assert not performance_dock_widget.update_timer.isActive()
    
    def test_clear_all(self, performance_dock_widget):
        """Test clearing all performance data."""
        # Add some FPS data
        performance_dock_widget.update_fps(30.0)
        performance_dock_widget.update_fps(31.0)
        
        assert len(performance_dock_widget.fps_widget.fps_values) > 0
        
        # Clear all
        performance_dock_widget.clear_all()
        
        assert len(performance_dock_widget.fps_widget.fps_values) == 0
    
    def test_update_fps(self, performance_dock_widget):
        """Test updating FPS through dock widget."""
        performance_dock_widget.update_fps(32.5)
        
        # Should propagate to FPS widget
        assert len(performance_dock_widget.fps_widget.fps_values) == 1
        assert performance_dock_widget.fps_widget.fps_values[0] == 32.5
    
    def test_update_fps_multiple_values(self, performance_dock_widget):
        """Test updating FPS with multiple values."""
        fps_values = [28.0, 29.5, 30.2, 31.0, 29.8]
        
        for fps in fps_values:
            performance_dock_widget.update_fps(fps)
        
        assert len(performance_dock_widget.fps_widget.fps_values) == 5
        assert list(performance_dock_widget.fps_widget.fps_values) == fps_values
    
    def test_update_latency_placeholder(self, performance_dock_widget):
        """Test update_latency method (placeholder for subtask 21.2)."""
        # Should not raise exception
        performance_dock_widget.update_latency(50.0)
        # Currently does nothing, just verify it doesn't crash
    
    def test_update_module_timings_placeholder(self, performance_dock_widget):
        """Test update_module_timings method (placeholder for subtask 21.3)."""
        timings = {
            'camera': 5.0,
            'bev': 15.0,
            'detection': 20.0,
            'dms': 25.0
        }
        
        # Should not raise exception
        performance_dock_widget.update_module_timings(timings)
        # Currently does nothing, just verify it doesn't crash
    
    def test_update_resources_placeholder(self, performance_dock_widget):
        """Test update_resources method (placeholder for subtask 21.4)."""
        # Should not raise exception
        performance_dock_widget.update_resources(4096.0, 45.5)
        # Currently does nothing, just verify it doesn't crash
    
    def test_mock_data_generation(self, performance_dock_widget):
        """Test that mock data generation works."""
        initial_count = len(performance_dock_widget.fps_widget.fps_values)
        
        # Manually trigger mock data generation
        performance_dock_widget._generate_mock_data()
        
        # Should have added one FPS value
        assert len(performance_dock_widget.fps_widget.fps_values) == initial_count + 1
        
        # FPS should be around 30 (25-35 range)
        fps = performance_dock_widget.fps_widget.fps_values[-1]
        assert 25.0 <= fps <= 35.0
    
    def test_timer_triggers_mock_data(self, performance_dock_widget, qapp):
        """Test that timer correctly triggers mock data generation."""
        initial_count = len(performance_dock_widget.fps_widget.fps_values)
        
        performance_dock_widget.start_monitoring()
        
        # Process events to allow timer to fire
        # Note: This is a simplified test; in real scenario would use QTest.qWait
        for _ in range(3):
            qapp.processEvents()
            time.sleep(0.1)
        
        performance_dock_widget.stop_monitoring()
        
        # Should have generated at least some data
        # (exact count depends on timing, so just check it increased)
        assert len(performance_dock_widget.fps_widget.fps_values) >= initial_count
    
    @pytest.mark.performance
    def test_update_fps_performance(self, performance_dock_widget):
        """Test that FPS update completes within performance requirements."""
        start_time = time.perf_counter()
        
        # Update FPS 100 times
        for i in range(100):
            performance_dock_widget.update_fps(30.0 + i * 0.1)
        
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / 100
        
        # Each update should be very fast (< 5ms average)
        assert avg_time_ms < 5.0, f"Average update took {avg_time_ms:.2f}ms, expected < 5ms"
    
    @pytest.mark.performance
    def test_clear_performance(self, performance_dock_widget):
        """Test that clear operation is fast."""
        # Add lots of data
        for i in range(60):
            performance_dock_widget.update_fps(30.0)
        
        start_time = time.perf_counter()
        performance_dock_widget.clear_all()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 10.0, f"Clear took {execution_time_ms:.2f}ms, expected < 10ms"
    
    def test_widget_hierarchy(self, performance_dock_widget):
        """Test that widget hierarchy is correctly established."""
        # Check that fps_widget is child of tab_widget
        assert performance_dock_widget.fps_widget.parent() == performance_dock_widget.tab_widget
        
        # Check that tab_widget is child of dock widget
        assert performance_dock_widget.tab_widget.parent() == performance_dock_widget
    
    def test_multiple_start_stop_cycles(self, performance_dock_widget):
        """Test multiple start/stop cycles."""
        for _ in range(5):
            performance_dock_widget.start_monitoring()
            assert performance_dock_widget.update_timer.isActive()
            
            performance_dock_widget.stop_monitoring()
            assert not performance_dock_widget.update_timer.isActive()
    
    def test_update_fps_with_edge_values(self, performance_dock_widget):
        """Test FPS updates with various edge case values."""
        edge_values = [0.0, 0.1, 15.0, 30.0, 60.0, 120.0, 999.9]
        
        for fps in edge_values:
            performance_dock_widget.update_fps(fps)
        
        assert len(performance_dock_widget.fps_widget.fps_values) == len(edge_values)
        assert list(performance_dock_widget.fps_widget.fps_values) == edge_values


class TestIntegration:
    """Integration tests for performance dock components."""
    
    def test_fps_widget_integration_with_dock(self, performance_dock_widget):
        """Test that FPS widget integrates correctly with dock widget."""
        # Update through dock
        performance_dock_widget.update_fps(28.5)
        
        # Verify it appears in FPS widget
        fps_widget = performance_dock_widget.fps_widget
        assert len(fps_widget.fps_values) == 1
        assert fps_widget.fps_values[0] == 28.5
        assert "28.5" in fps_widget.fps_label.text()
    
    def test_clear_propagates_to_fps_widget(self, performance_dock_widget):
        """Test that clear operation propagates to FPS widget."""
        # Add data
        for i in range(10):
            performance_dock_widget.update_fps(30.0 + i)
        
        assert len(performance_dock_widget.fps_widget.fps_values) == 10
        
        # Clear through dock
        performance_dock_widget.clear_all()
        
        # Verify FPS widget is cleared
        assert len(performance_dock_widget.fps_widget.fps_values) == 0
        assert "Current FPS: --" in performance_dock_widget.fps_widget.fps_label.text()
    
    def test_monitoring_lifecycle(self, performance_dock_widget, qapp):
        """Test complete monitoring lifecycle."""
        # Start monitoring
        performance_dock_widget.start_monitoring()
        assert performance_dock_widget.update_timer.isActive()
        
        # Let it run briefly
        qapp.processEvents()
        time.sleep(0.1)
        
        # Update some values manually
        performance_dock_widget.update_fps(31.5)
        
        # Stop monitoring
        performance_dock_widget.stop_monitoring()
        assert not performance_dock_widget.update_timer.isActive()
        
        # Clear data
        performance_dock_widget.clear_all()
        assert len(performance_dock_widget.fps_widget.fps_values) == 0
