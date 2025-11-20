"""
Unit tests for SentinelWorker thread.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from PyQt6.QtCore import QCoreApplication, QTimer
from PyQt6.QtTest import QSignalSpy

from gui.workers import SentinelWorker
from core.config import ConfigManager
from core.data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput,
    Detection3D, DriverState, RiskAssessment, Alert
)


@pytest.fixture
def qapp():
    """Create QApplication for tests."""
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock(spec=ConfigManager)
    config.get = Mock(return_value={})
    config.validate = Mock(return_value=True)
    return config


def test_sentinel_worker_initialization(qapp, mock_config):
    """Test SentinelWorker initialization."""
    worker = SentinelWorker(mock_config)
    
    assert worker is not None
    assert worker.config == mock_config
    assert worker._running is False
    assert worker._stop_requested is False
    assert worker.frame_count == 0
    assert worker.camera_manager is None


def test_sentinel_worker_signals_defined(qapp, mock_config):
    """Test that all required signals are defined."""
    worker = SentinelWorker(mock_config)
    
    # Check that all signals exist
    assert hasattr(worker, 'frame_ready')
    assert hasattr(worker, 'bev_ready')
    assert hasattr(worker, 'detections_ready')
    assert hasattr(worker, 'driver_state_ready')
    assert hasattr(worker, 'risks_ready')
    assert hasattr(worker, 'alerts_ready')
    assert hasattr(worker, 'performance_ready')
    assert hasattr(worker, 'error_occurred')
    assert hasattr(worker, 'status_changed')


def test_sentinel_worker_stop_method(qapp, mock_config):
    """Test worker stop method."""
    worker = SentinelWorker(mock_config)
    
    # Initially not stopped
    assert worker._stop_requested is False
    assert worker._running is False
    
    # Call stop
    worker.stop()
    
    # Should be marked as stopped
    assert worker._stop_requested is True
    assert worker._running is False


def test_sentinel_worker_copy_camera_bundle(qapp, mock_config):
    """Test thread-safe copying of camera bundle."""
    worker = SentinelWorker(mock_config)
    
    # Create test camera bundle
    bundle = CameraBundle(
        timestamp=1.0,
        interior=np.zeros((480, 640, 3), dtype=np.uint8),
        front_left=np.zeros((720, 1280, 3), dtype=np.uint8),
        front_right=np.zeros((720, 1280, 3), dtype=np.uint8)
    )
    
    # Copy bundle
    copied = worker._copy_camera_bundle(bundle)
    
    # Check structure
    assert 'timestamp' in copied
    assert 'interior' in copied
    assert 'front_left' in copied
    assert 'front_right' in copied
    
    # Check that arrays are copied (not same object)
    assert copied['interior'] is not bundle.interior
    assert copied['front_left'] is not bundle.front_left
    assert copied['front_right'] is not bundle.front_right
    
    # Check values are equal
    assert copied['timestamp'] == bundle.timestamp
    assert np.array_equal(copied['interior'], bundle.interior)


def test_sentinel_worker_copy_driver_state(qapp, mock_config):
    """Test thread-safe copying of driver state."""
    worker = SentinelWorker(mock_config)
    
    # Create test driver state
    state = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.0, 'yawn_detected': False},
        distraction={'type': 'none', 'confidence': 0.0},
        readiness_score=85.0
    )
    
    # Copy state
    copied = worker._copy_driver_state(state)
    
    # Check that it's a different object
    assert copied is not state
    
    # Check values are equal
    assert copied.face_detected == state.face_detected
    assert copied.readiness_score == state.readiness_score


def test_sentinel_worker_copy_detections(qapp, mock_config):
    """Test thread-safe copying of detections."""
    worker = SentinelWorker(mock_config)
    
    # Create test detections
    detections = [
        Detection3D(
            bbox_3d=(1.0, 2.0, 0.0, 1.5, 1.8, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.95,
            velocity=(5.0, 0.0, 0.0),
            track_id=1
        )
    ]
    
    # Copy detections
    copied = worker._copy_detections(detections)
    
    # Check that it's a different list
    assert copied is not detections
    assert copied[0] is not detections[0]
    
    # Check values are equal
    assert copied[0].class_name == detections[0].class_name
    assert copied[0].confidence == detections[0].confidence


def test_sentinel_worker_calculate_performance_metrics(qapp, mock_config):
    """Test performance metrics calculation."""
    worker = SentinelWorker(mock_config)
    
    # Set up test data
    import time
    worker.start_time = time.time() - 1.0  # 1 second ago
    worker.frame_count = 30  # 30 frames in 1 second
    
    # Add some latency data
    worker.module_latencies['camera'] = [0.005] * 10
    worker.module_latencies['bev'] = [0.015] * 10
    worker.module_latencies['dms'] = [0.025] * 10
    
    # Calculate metrics
    metrics = worker._calculate_performance_metrics(0.033)
    
    # Check metrics structure
    assert 'fps' in metrics
    assert 'frame_count' in metrics
    assert 'loop_time_ms' in metrics
    assert 'module_latencies' in metrics
    assert 'total_latency_ms' in metrics
    
    # Check values
    assert metrics['frame_count'] == 30
    assert metrics['fps'] > 0
    assert 'camera' in metrics['module_latencies']
    assert 'bev' in metrics['module_latencies']


@patch('gui.workers.sentinel_worker.CameraManager')
@patch('gui.workers.sentinel_worker.BEVGenerator')
@patch('gui.workers.sentinel_worker.SemanticSegmentor')
@patch('gui.workers.sentinel_worker.ObjectDetector')
@patch('gui.workers.sentinel_worker.DriverMonitor')
@patch('gui.workers.sentinel_worker.ContextualIntelligence')
@patch('gui.workers.sentinel_worker.AlertSystem')
@patch('gui.workers.sentinel_worker.ScenarioRecorder')
def test_sentinel_worker_initialization_modules(
    mock_recorder, mock_alerts, mock_intelligence, mock_dms,
    mock_detector, mock_segmentor, mock_bev, mock_camera,
    qapp, mock_config
):
    """Test module initialization in worker."""
    worker = SentinelWorker(mock_config)
    
    # Call initialization
    result = worker._initialize_modules()
    
    # Check that all modules were created
    assert result is True
    mock_camera.assert_called_once_with(mock_config)
    mock_bev.assert_called_once_with(mock_config)
    mock_segmentor.assert_called_once_with(mock_config)
    mock_detector.assert_called_once_with(mock_config)
    mock_dms.assert_called_once_with(mock_config)
    mock_intelligence.assert_called_once_with(mock_config)
    mock_alerts.assert_called_once_with(mock_config)
    mock_recorder.assert_called_once_with(mock_config)


def test_sentinel_worker_signal_emission(qapp, mock_config):
    """Test that signals can be emitted."""
    worker = SentinelWorker(mock_config)
    
    # Create signal spies
    status_spy = QSignalSpy(worker.status_changed)
    error_spy = QSignalSpy(worker.error_occurred)
    
    # Emit signals
    worker.status_changed.emit("Test status")
    worker.error_occurred.emit("TestError", "Test error message")
    
    # Check signals were emitted
    assert len(status_spy) == 1
    assert status_spy[0][0] == "Test status"
    
    assert len(error_spy) == 1
    assert error_spy[0][0] == "TestError"
    assert error_spy[0][1] == "Test error message"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
