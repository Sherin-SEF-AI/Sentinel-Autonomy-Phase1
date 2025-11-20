"""
Comprehensive GUI component tests for SENTINEL system.
Tests all custom widgets, signal/slot connections, thread safety, and rendering performance.
"""

import pytest
import sys
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtTest import QTest

# Ensure QApplication exists for all tests
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestCircularGauge:
    """Test CircularGaugeWidget."""
    
    def test_gauge_creation(self, qapp):
        """Test gauge widget creation."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge = CircularGaugeWidget(min_value=0, max_value=100, title="Test")
        assert gauge is not None
        assert gauge.min_value == 0
        assert gauge.max_value == 100
    
    def test_gauge_value_update(self, qapp):
        """Test gauge value updates."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge = CircularGaugeWidget(min_value=0, max_value=100)
        gauge.set_value(50)
        assert gauge.value == 50
        
        # Test clamping
        gauge.set_value(150)
        assert gauge.value == 100
        
        gauge.set_value(-10)
        assert gauge.value == 0
    
    def test_gauge_rendering(self, qapp):
        """Test gauge renders without errors."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge = CircularGaugeWidget(min_value=0, max_value=100)
        gauge.set_value(75)
        gauge.show()
        
        # Force paint event
        gauge.repaint()
        QTest.qWait(10)


class TestGazeDirection:
    """Test GazeDirectionWidget."""
    
    def test_gaze_widget_creation(self, qapp):
        """Test gaze direction widget creation."""
        from src.gui.widgets.gaze_direction import GazeDirectionWidget
        
        widget = GazeDirectionWidget()
        assert widget is not None
    
    def test_gaze_update(self, qapp):
        """Test gaze direction updates."""
        from src.gui.widgets.gaze_direction import GazeDirectionWidget
        
        widget = GazeDirectionWidget()
        widget.update_gaze(pitch=10.0, yaw=15.0, zone="front")
        
        assert widget.pitch == 10.0
        assert widget.yaw == 15.0
        assert widget.current_zone == "front"


class TestStatusIndicator:
    """Test StatusIndicator widget."""
    
    def test_indicator_creation(self, qapp):
        """Test status indicator creation."""
        from src.gui.widgets.status_indicator import StatusIndicator
        
        indicator = StatusIndicator(label="Test Status")
        assert indicator is not None
    
    def test_indicator_states(self, qapp):
        """Test status indicator state changes."""
        from src.gui.widgets.status_indicator import StatusIndicator
        
        indicator = StatusIndicator(label="Test")
        
        indicator.set_status("ok")
        assert indicator.status == "ok"
        
        indicator.set_status("warning")
        assert indicator.status == "warning"
        
        indicator.set_status("critical")
        assert indicator.status == "critical"


class TestMetricDisplay:
    """Test MetricDisplay widget."""
    
    def test_metric_creation(self, qapp):
        """Test metric display creation."""
        from src.gui.widgets.metric_display import MetricDisplay
        
        metric = MetricDisplay(label="Speed", unit="m/s")
        assert metric is not None
    
    def test_metric_value_update(self, qapp):
        """Test metric value updates."""
        from src.gui.widgets.metric_display import MetricDisplay
        
        metric = MetricDisplay(label="Speed", unit="m/s")
        metric.set_value(25.5)
        
        # Value should be formatted
        assert "25.5" in metric.value_label.text()


class TestVideoDisplay:
    """Test VideoDisplayWidget."""
    
    def test_video_display_creation(self, qapp):
        """Test video display widget creation."""
        from src.gui.widgets.video_display import VideoDisplayWidget
        
        display = VideoDisplayWidget(title="Test Camera")
        assert display is not None
    
    def test_video_frame_update(self, qapp):
        """Test video frame updates."""
        from src.gui.widgets.video_display import VideoDisplayWidget
        
        display = VideoDisplayWidget(title="Test")
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        display.update_frame(frame)
        
        # Should not crash
        QTest.qWait(10)


class TestBEVCanvas:
    """Test BEVCanvas widget."""
    
    def test_bev_canvas_creation(self, qapp):
        """Test BEV canvas creation."""
        from src.gui.widgets.bev_canvas import BEVCanvas
        
        canvas = BEVCanvas()
        assert canvas is not None
        assert canvas.scene is not None
        assert canvas.view is not None
    
    def test_bev_image_update(self, qapp):
        """Test BEV image updates."""
        from src.gui.widgets.bev_canvas import BEVCanvas
        
        canvas = BEVCanvas()
        
        # Create test BEV image
        bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        canvas.update_bev(bev_image)
        
        QTest.qWait(10)
    
    def test_bev_detections_overlay(self, qapp):
        """Test detection overlays on BEV."""
        from src.gui.widgets.bev_canvas import BEVCanvas
        from src.core.data_structures import Detection3D
        
        canvas = BEVCanvas()
        
        # Create test detection
        detection = Detection3D(
            bbox_3d=(10.0, 5.0, 0.0, 2.0, 1.5, 1.0, 0.0),
            class_name="vehicle",
            confidence=0.95,
            velocity=(5.0, 0.0, 0.0),
            track_id=1
        )
        
        canvas.update_detections([detection])
        QTest.qWait(10)


class TestDriverStatePanel:
    """Test DriverStatePanel widget."""
    
    def test_panel_creation(self, qapp):
        """Test driver state panel creation."""
        from src.gui.widgets.driver_state_panel import DriverStatePanel
        
        panel = DriverStatePanel()
        assert panel is not None
    
    def test_driver_state_update(self, qapp):
        """Test driver state updates."""
        from src.gui.widgets.driver_state_panel import DriverStatePanel
        from src.core.data_structures import DriverState
        
        panel = DriverStatePanel()
        
        # Create test driver state
        driver_state = DriverState(
            face_detected=True,
            landmarks=np.zeros((68, 2)),
            head_pose={'roll': 0.0, 'pitch': 5.0, 'yaw': 10.0},
            gaze={'pitch': 5.0, 'yaw': 10.0, 'attention_zone': 'front'},
            eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
            drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
            distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
            readiness_score=85.0
        )
        
        panel.update_driver_state(driver_state)
        QTest.qWait(10)


class TestRiskPanel:
    """Test RiskPanel widget."""
    
    def test_risk_panel_creation(self, qapp):
        """Test risk panel creation."""
        from src.gui.widgets.risk_panel import RiskPanel
        
        panel = RiskPanel()
        assert panel is not None
    
    def test_risk_assessment_update(self, qapp):
        """Test risk assessment updates."""
        from src.gui.widgets.risk_panel import RiskPanel
        from src.core.data_structures import RiskAssessment, Risk, Hazard
        
        panel = RiskPanel()
        
        # Create test hazard
        hazard = Hazard(
            object_id=1,
            type="vehicle",
            position=(10.0, 5.0, 0.0),
            velocity=(5.0, 0.0, 0.0),
            trajectory=[(10.0, 5.0, 0.0), (15.0, 5.0, 0.0)],
            ttc=2.0,
            zone="front",
            base_risk=0.7
        )
        
        risk = Risk(
            hazard=hazard,
            contextual_score=0.8,
            driver_aware=False,
            urgency="high",
            intervention_needed=True
        )
        
        assessment = RiskAssessment(
            scene_graph={},
            hazards=[hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        panel.update_risk_assessment(assessment)
        QTest.qWait(10)


class TestAlertsPanel:
    """Test AlertsPanel widget."""
    
    def test_alerts_panel_creation(self, qapp):
        """Test alerts panel creation."""
        from src.gui.widgets.alerts_panel import AlertsPanel
        
        panel = AlertsPanel()
        assert panel is not None
    
    def test_alert_display(self, qapp):
        """Test alert display."""
        from src.gui.widgets.alerts_panel import AlertsPanel
        from src.core.data_structures import Alert
        
        panel = AlertsPanel()
        
        # Create test alert
        alert = Alert(
            timestamp=time.time(),
            urgency="critical",
            modalities=["visual", "audio"],
            message="Vehicle ahead - brake now!",
            hazard_id=1,
            dismissed=False
        )
        
        panel.add_alert(alert)
        QTest.qWait(10)
        
        # Check alert was added
        assert panel.alert_count > 0


class TestPerformanceDock:
    """Test PerformanceDock widget."""
    
    def test_performance_dock_creation(self, qapp):
        """Test performance dock creation."""
        from src.gui.widgets.performance_dock import PerformanceDock
        
        dock = PerformanceDock()
        assert dock is not None
    
    def test_performance_metrics_update(self, qapp):
        """Test performance metrics updates."""
        from src.gui.widgets.performance_dock import PerformanceDock
        
        dock = PerformanceDock()
        
        metrics = {
            'fps': 30.5,
            'latency': 85.2,
            'gpu_memory': 4500.0,
            'cpu_usage': 45.0,
            'module_timings': {
                'camera': 5.0,
                'bev': 15.0,
                'detection': 20.0,
                'dms': 25.0,
                'intelligence': 10.0
            }
        }
        
        dock.update_metrics(metrics)
        QTest.qWait(10)


class TestScenariosDock:
    """Test ScenariosDock widget."""
    
    def test_scenarios_dock_creation(self, qapp):
        """Test scenarios dock creation."""
        from src.gui.widgets.scenarios_dock import ScenariosDock
        
        dock = ScenariosDock()
        assert dock is not None
    
    def test_scenario_list_update(self, qapp):
        """Test scenario list updates."""
        from src.gui.widgets.scenarios_dock import ScenariosDock
        
        dock = ScenariosDock()
        
        scenarios = [
            {
                'timestamp': '2024-11-18T10:30:00',
                'duration': 15.5,
                'trigger': 'high_risk',
                'path': 'scenarios/test1'
            }
        ]
        
        dock.update_scenarios(scenarios)
        QTest.qWait(10)


class TestConfigurationDock:
    """Test ConfigurationDock widget."""
    
    def test_configuration_dock_creation(self, qapp):
        """Test configuration dock creation."""
        from src.gui.widgets.configuration_dock import ConfigurationDock
        
        dock = ConfigurationDock()
        assert dock is not None
    
    def test_configuration_load(self, qapp):
        """Test configuration loading."""
        from src.gui.widgets.configuration_dock import ConfigurationDock
        
        dock = ConfigurationDock()
        
        config = {
            'risk_assessment': {
                'thresholds': {
                    'critical': 0.9,
                    'high': 0.7,
                    'medium': 0.5
                }
            }
        }
        
        dock.load_configuration(config)
        QTest.qWait(10)


class TestSignalSlotConnections:
    """Test signal/slot connections between components."""
    
    def test_worker_to_gui_signals(self, qapp):
        """Test worker thread signals connect to GUI slots."""
        from src.gui.workers.sentinel_worker import SentinelWorker
        
        worker = SentinelWorker(config={})
        
        # Check signals exist
        assert hasattr(worker, 'frame_ready')
        assert hasattr(worker, 'bev_ready')
        assert hasattr(worker, 'detections_ready')
        assert hasattr(worker, 'driver_state_ready')
        assert hasattr(worker, 'risks_ready')
        assert hasattr(worker, 'alerts_ready')
        assert hasattr(worker, 'performance_ready')
        assert hasattr(worker, 'error_occurred')
    
    def test_signal_emission(self, qapp):
        """Test signals can be emitted."""
        from src.gui.workers.sentinel_worker import SentinelWorker
        
        worker = SentinelWorker(config={})
        
        # Create mock slot
        mock_slot = Mock()
        worker.performance_ready.connect(mock_slot)
        
        # Emit signal
        test_data = {'fps': 30.0}
        worker.performance_ready.emit(test_data)
        
        # Verify slot was called
        QTest.qWait(10)
        mock_slot.assert_called_once_with(test_data)


class TestThreadSafety:
    """Test thread safety of GUI components."""
    
    def test_worker_thread_lifecycle(self, qapp):
        """Test worker thread start/stop."""
        from src.gui.workers.sentinel_worker import SentinelWorker
        
        worker = SentinelWorker(config={})
        
        # Start worker
        worker.start()
        assert worker.isRunning()
        
        # Stop worker
        worker.stop()
        worker.wait(1000)  # Wait up to 1 second
        assert not worker.isRunning()
    
    def test_cross_thread_signal_delivery(self, qapp):
        """Test signals work across threads."""
        
        class TestWorker(QThread):
            test_signal = pyqtSignal(int)
            
            def run(self):
                for i in range(5):
                    self.test_signal.emit(i)
                    self.msleep(10)
        
        worker = TestWorker()
        received_values = []
        
        def slot(value):
            received_values.append(value)
        
        worker.test_signal.connect(slot)
        worker.start()
        worker.wait(1000)
        
        # Should have received all values
        assert len(received_values) == 5
        assert received_values == [0, 1, 2, 3, 4]


class TestRenderingPerformance:
    """Test GUI rendering performance."""
    
    def test_video_display_fps(self, qapp):
        """Test video display can handle 30 FPS."""
        from src.gui.widgets.video_display import VideoDisplayWidget
        
        display = VideoDisplayWidget(title="FPS Test")
        display.show()
        
        # Simulate 30 FPS updates
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        start_time = time.time()
        frame_count = 30
        
        for _ in range(frame_count):
            display.update_frame(frame)
            QTest.qWait(33)  # ~30 FPS
        
        elapsed = time.time() - start_time
        actual_fps = frame_count / elapsed
        
        # Should achieve close to 30 FPS
        assert actual_fps >= 25.0, f"FPS too low: {actual_fps}"
    
    def test_bev_canvas_update_performance(self, qapp):
        """Test BEV canvas update performance."""
        from src.gui.widgets.bev_canvas import BEVCanvas
        
        canvas = BEVCanvas()
        canvas.show()
        
        bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Measure update time
        start_time = time.time()
        iterations = 30
        
        for _ in range(iterations):
            canvas.update_bev(bev_image)
            QTest.qWait(10)
        
        elapsed = time.time() - start_time
        avg_time = (elapsed / iterations) * 1000  # ms
        
        # Should update quickly
        assert avg_time < 50, f"Update too slow: {avg_time}ms"
    
    def test_gauge_animation_smoothness(self, qapp):
        """Test gauge animations are smooth."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge = CircularGaugeWidget(min_value=0, max_value=100)
        gauge.show()
        
        # Animate value changes
        for value in range(0, 101, 10):
            gauge.set_value(value)
            QTest.qWait(50)
        
        # Should complete without errors
        assert gauge.value == 100


class TestMainWindow:
    """Test main window integration."""
    
    def test_main_window_creation(self, qapp):
        """Test main window can be created."""
        from src.gui.main_window import SENTINELMainWindow
        
        window = SENTINELMainWindow()
        assert window is not None
    
    def test_main_window_components(self, qapp):
        """Test main window has all required components."""
        from src.gui.main_window import SENTINELMainWindow
        
        window = SENTINELMainWindow()
        
        # Check menu bar
        assert window.menuBar() is not None
        
        # Check toolbar
        assert window.toolBar is not None
        
        # Check status bar
        assert window.statusBar() is not None
        
        # Check central widget
        assert window.centralWidget() is not None
    
    def test_keyboard_shortcuts(self, qapp):
        """Test keyboard shortcuts are registered."""
        from src.gui.main_window import SENTINELMainWindow
        
        window = SENTINELMainWindow()
        
        # Check for key actions
        actions = window.findChildren(type(window.menuBar().actions()[0]))
        
        # Should have actions with shortcuts
        shortcuts_found = any(action.shortcut() for action in actions if hasattr(action, 'shortcut'))
        assert shortcuts_found


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
