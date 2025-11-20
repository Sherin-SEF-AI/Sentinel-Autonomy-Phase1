"""Test suite for camera capture module."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock, call
from collections import deque
import threading

from src.camera.capture import CameraCapture


@pytest.fixture
def mock_video_capture():
    """Fixture providing mock cv2.VideoCapture."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_cap.get.side_effect = lambda prop: {
        3: 1280,  # CAP_PROP_FRAME_WIDTH
        4: 720,   # CAP_PROP_FRAME_HEIGHT
        5: 30     # CAP_PROP_FPS
    }.get(prop, 0)
    return mock_cap


@pytest.fixture
def camera_capture_params():
    """Fixture providing standard camera capture parameters."""
    return {
        'camera_id': 1,
        'device': 1,
        'resolution': (1280, 720),
        'fps': 30,
        'buffer_size': 5
    }


@pytest.fixture
def camera_capture(camera_capture_params):
    """Fixture creating a CameraCapture instance for testing."""
    return CameraCapture(**camera_capture_params)


class TestCameraCaptureInitialization:
    """Test suite for CameraCapture initialization."""
    
    def test_initialization_with_valid_parameters(self, camera_capture, camera_capture_params):
        """Test that CameraCapture initializes correctly with valid parameters."""
        assert camera_capture.camera_id == camera_capture_params['camera_id']
        assert camera_capture.device == camera_capture_params['device']
        assert camera_capture.resolution == camera_capture_params['resolution']
        assert camera_capture.fps == camera_capture_params['fps']
        assert camera_capture.buffer_size == camera_capture_params['buffer_size']
        assert camera_capture.is_running is False
        assert camera_capture.is_connected is False
        assert camera_capture.frame_count == 0
        assert len(camera_capture.frame_buffer) == 0
    
    def test_initialization_creates_circular_buffer(self, camera_capture):
        """Test that initialization creates a deque with correct max length."""
        assert isinstance(camera_capture.frame_buffer, deque)
        assert camera_capture.frame_buffer.maxlen == 5
    
    def test_initialization_creates_thread_lock(self, camera_capture):
        """Test that initialization creates a threading lock."""
        assert isinstance(camera_capture.buffer_lock, type(threading.Lock()))
    
    def test_initialization_with_custom_buffer_size(self):
        """Test initialization with custom buffer size."""
        capture = CameraCapture(
            camera_id=0,
            device=0,
            resolution=(640, 480),
            fps=30,
            buffer_size=10
        )
        assert capture.frame_buffer.maxlen == 10


class TestCameraCaptureStart:
    """Test suite for camera capture start functionality."""
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_start_success(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test successful camera start."""
        mock_cv2_capture.return_value = mock_video_capture
        
        result = camera_capture.start()
        
        assert result is True
        assert camera_capture.is_running is True
        assert camera_capture.is_connected is True
        assert camera_capture.capture_thread is not None
        assert camera_capture.capture_thread.daemon is True
        mock_cv2_capture.assert_called_once_with(camera_capture.device)
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_start_sets_camera_properties(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test that start sets correct camera properties."""
        mock_cv2_capture.return_value = mock_video_capture
        
        camera_capture.start()
        
        # Verify camera properties were set
        calls = mock_video_capture.set.call_args_list
        assert any(call[0][0] == 3 and call[0][1] == 1280 for call in calls)  # Width
        assert any(call[0][0] == 4 and call[0][1] == 720 for call in calls)   # Height
        assert any(call[0][0] == 5 and call[0][1] == 30 for call in calls)    # FPS
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_start_failure_camera_not_opened(self, mock_cv2_capture, camera_capture):
        """Test start failure when camera cannot be opened."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        result = camera_capture.start()
        
        assert result is False
        assert camera_capture.is_running is False
        assert camera_capture.is_connected is False
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_start_when_already_running(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test start when camera is already running."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture.start()
        
        # Try to start again
        result = camera_capture.start()
        
        assert result is True
        # Should only be called once (not twice)
        assert mock_cv2_capture.call_count == 1
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_start_exception_handling(self, mock_cv2_capture, camera_capture):
        """Test start handles exceptions gracefully."""
        mock_cv2_capture.side_effect = Exception("Camera error")
        
        result = camera_capture.start()
        
        assert result is False
        assert camera_capture.is_connected is False


class TestCameraCaptureStop:
    """Test suite for camera capture stop functionality."""
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_stop_success(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test successful camera stop."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture.start()
        
        camera_capture.stop()
        
        assert camera_capture.is_running is False
        assert camera_capture.is_connected is False
        mock_video_capture.release.assert_called_once()
    
    def test_stop_when_not_running(self, camera_capture):
        """Test stop when camera is not running."""
        # Should not raise exception
        camera_capture.stop()
        assert camera_capture.is_running is False
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_stop_waits_for_thread(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test that stop waits for capture thread to finish."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture.start()
        
        # Mock the thread
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        camera_capture.capture_thread = mock_thread
        
        camera_capture.stop()
        
        mock_thread.join.assert_called_once_with(timeout=2.0)


class TestCameraCaptureFrameRetrieval:
    """Test suite for frame retrieval functionality."""
    
    def test_get_latest_frame_empty_buffer(self, camera_capture):
        """Test getting frame from empty buffer returns None."""
        result = camera_capture.get_latest_frame()
        assert result is None
    
    def test_get_latest_frame_with_data(self, camera_capture):
        """Test getting latest frame from buffer with data."""
        test_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        test_timestamp = time.time()
        
        camera_capture.frame_buffer.append((test_frame, test_timestamp))
        
        result = camera_capture.get_latest_frame()
        
        assert result is not None
        frame, timestamp = result
        assert np.array_equal(frame, test_frame)
        assert timestamp == test_timestamp
    
    def test_get_latest_frame_returns_most_recent(self, camera_capture):
        """Test that get_latest_frame returns the most recent frame."""
        # Add multiple frames
        for i in range(3):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * i
            timestamp = time.time() + i
            camera_capture.frame_buffer.append((frame, timestamp))
        
        result = camera_capture.get_latest_frame()
        frame, timestamp = result
        
        # Should return the last frame (value 2)
        assert np.all(frame == 2)
    
    def test_get_latest_frame_thread_safety(self, camera_capture):
        """Test that get_latest_frame is thread-safe."""
        test_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        test_timestamp = time.time()
        
        # Simulate concurrent access
        def add_frames():
            for _ in range(10):
                camera_capture.frame_buffer.append((test_frame, test_timestamp))
        
        def get_frames():
            for _ in range(10):
                camera_capture.get_latest_frame()
        
        thread1 = threading.Thread(target=add_frames)
        thread2 = threading.Thread(target=get_frames)
        
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        
        # Should not raise any exceptions


class TestCameraCaptureHealth:
    """Test suite for camera health monitoring."""
    
    def test_is_healthy_when_not_running(self, camera_capture):
        """Test health check when camera is not running."""
        assert camera_capture.is_healthy() is False
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_is_healthy_when_running_and_connected(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test health check when camera is running and connected."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture.start()
        camera_capture.last_frame_time = time.time()
        
        assert camera_capture.is_healthy() is True
    
    def test_is_healthy_no_recent_frames(self, camera_capture):
        """Test health check fails when no recent frames."""
        camera_capture.is_running = True
        camera_capture.is_connected = True
        camera_capture.last_frame_time = time.time() - 3.0  # 3 seconds ago
        
        assert camera_capture.is_healthy() is False
    
    def test_is_healthy_recent_frames(self, camera_capture):
        """Test health check passes with recent frames."""
        camera_capture.is_running = True
        camera_capture.is_connected = True
        camera_capture.last_frame_time = time.time() - 0.5  # 0.5 seconds ago
        
        assert camera_capture.is_healthy() is True


class TestCameraCaptureFrameCapture:
    """Test suite for internal frame capture functionality."""
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_capture_frame_success(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test successful frame capture."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture._initialize_camera()
        
        result = camera_capture._capture_frame()
        
        assert result is True
        assert camera_capture.frame_count == 1
        assert len(camera_capture.frame_buffer) == 1
        assert camera_capture.last_frame_time > 0
    
    def test_capture_frame_no_camera(self, camera_capture):
        """Test frame capture when camera is not initialized."""
        result = camera_capture._capture_frame()
        assert result is False
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_capture_frame_read_failure(self, mock_cv2_capture, camera_capture):
        """Test frame capture when read fails."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv2_capture.return_value = mock_cap
        
        camera_capture._initialize_camera()
        result = camera_capture._capture_frame()
        
        assert result is False
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_capture_frame_exception_handling(self, mock_cv2_capture, camera_capture):
        """Test frame capture handles exceptions."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = Exception("Read error")
        mock_cv2_capture.return_value = mock_cap
        
        camera_capture._initialize_camera()
        result = camera_capture._capture_frame()
        
        assert result is False


class TestCameraCaptureReconnection:
    """Test suite for camera reconnection functionality."""
    
    @patch('src.camera.capture.cv2.VideoCapture')
    @patch('src.camera.capture.time.sleep')
    def test_attempt_reconnection_success(self, mock_sleep, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test successful camera reconnection."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture._initialize_camera()
        
        # Simulate disconnection
        camera_capture.is_connected = False
        
        camera_capture._attempt_reconnection()
        
        assert camera_capture.is_connected is True
        mock_sleep.assert_called_once_with(1.0)
    
    @patch('src.camera.capture.cv2.VideoCapture')
    @patch('src.camera.capture.time.sleep')
    def test_attempt_reconnection_failure(self, mock_sleep, mock_cv2_capture, camera_capture):
        """Test failed camera reconnection."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2_capture.return_value = mock_cap
        
        camera_capture._attempt_reconnection()
        
        assert camera_capture.is_connected is False


class TestCameraCaptureBufferManagement:
    """Test suite for circular buffer management."""
    
    def test_buffer_circular_behavior(self, camera_capture):
        """Test that buffer maintains circular behavior at max capacity."""
        # Fill buffer beyond capacity
        for i in range(10):
            frame = np.ones((720, 1280, 3), dtype=np.uint8) * i
            camera_capture.frame_buffer.append((frame, float(i)))
        
        # Buffer should only contain last 5 frames
        assert len(camera_capture.frame_buffer) == 5
        
        # Verify it contains frames 5-9
        frames = list(camera_capture.frame_buffer)
        for i, (frame, timestamp) in enumerate(frames):
            expected_value = i + 5
            assert np.all(frame == expected_value)
            assert timestamp == float(expected_value)


class TestCameraCapturePerformance:
    """Test suite for performance requirements."""
    
    @patch('src.camera.capture.cv2.VideoCapture')
    @pytest.mark.performance
    def test_capture_frame_performance(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test that frame capture completes within 5ms target."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture._initialize_camera()
        
        # Measure capture time
        start_time = time.perf_counter()
        camera_capture._capture_frame()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 5, f"Frame capture took {execution_time_ms:.2f}ms, expected < 5ms"
    
    @patch('src.camera.capture.cv2.VideoCapture')
    @pytest.mark.performance
    def test_get_latest_frame_performance(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test that frame retrieval is fast (< 1ms)."""
        mock_cv2_capture.return_value = mock_video_capture
        camera_capture._initialize_camera()
        camera_capture._capture_frame()
        
        start_time = time.perf_counter()
        camera_capture.get_latest_frame()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 1, f"Frame retrieval took {execution_time_ms:.2f}ms, expected < 1ms"


class TestCameraCaptureEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_initialization_with_zero_buffer_size(self):
        """Test initialization with zero buffer size."""
        capture = CameraCapture(
            camera_id=0,
            device=0,
            resolution=(640, 480),
            fps=30,
            buffer_size=0
        )
        # Should handle gracefully
        assert capture.frame_buffer.maxlen == 0
    
    def test_initialization_with_extreme_resolution(self):
        """Test initialization with extreme resolution values."""
        capture = CameraCapture(
            camera_id=0,
            device=0,
            resolution=(4096, 2160),  # 4K
            fps=60,
            buffer_size=5
        )
        assert capture.resolution == (4096, 2160)
        assert capture.fps == 60
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_multiple_start_stop_cycles(self, mock_cv2_capture, camera_capture, mock_video_capture):
        """Test multiple start/stop cycles."""
        mock_cv2_capture.return_value = mock_video_capture
        
        for _ in range(3):
            assert camera_capture.start() is True
            camera_capture.stop()
            assert camera_capture.is_running is False
