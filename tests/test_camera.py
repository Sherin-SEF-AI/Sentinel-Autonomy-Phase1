"""Tests for camera management module."""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from src.camera.sync import TimestampSync
from src.camera.calibration import CalibrationLoader, CameraIntrinsics, CameraExtrinsics
from src.camera.manager import CameraManager


class TestTimestampSync:
    """Test timestamp synchronization."""
    
    def test_sync_within_tolerance(self):
        """Test synchronization with frames within tolerance."""
        sync = TimestampSync(tolerance_ms=5.0)
        
        # Create frames with timestamps within 5ms
        base_time = time.time()
        frames = {
            0: (np.zeros((480, 640, 3), dtype=np.uint8), base_time),
            1: (np.zeros((720, 1280, 3), dtype=np.uint8), base_time + 0.003),  # 3ms
            2: (np.zeros((720, 1280, 3), dtype=np.uint8), base_time + 0.004),  # 4ms
        }
        
        result = sync.synchronize(frames)
        
        assert result is not None
        synchronized_frames, timestamp = result
        assert len(synchronized_frames) == 3
        assert 0 in synchronized_frames
        assert 1 in synchronized_frames
        assert 2 in synchronized_frames
    
    def test_sync_outside_tolerance(self):
        """Test synchronization with frames outside tolerance."""
        sync = TimestampSync(tolerance_ms=5.0)
        
        # Create frames with timestamps outside 5ms
        base_time = time.time()
        frames = {
            0: (np.zeros((480, 640, 3), dtype=np.uint8), base_time),
            1: (np.zeros((720, 1280, 3), dtype=np.uint8), base_time + 0.010),  # 10ms
            2: (np.zeros((720, 1280, 3), dtype=np.uint8), base_time + 0.015),  # 15ms
        }
        
        result = sync.synchronize(frames)
        
        # Should fail due to frames outside tolerance
        assert result is None
    
    def test_sync_statistics(self):
        """Test synchronization statistics tracking."""
        sync = TimestampSync(tolerance_ms=5.0)
        
        base_time = time.time()
        frames = {
            0: (np.zeros((480, 640, 3), dtype=np.uint8), base_time),
            1: (np.zeros((720, 1280, 3), dtype=np.uint8), base_time + 0.002),
        }
        
        sync.synchronize(frames)
        
        stats = sync.get_statistics()
        assert stats['sync_attempts'] == 1
        assert stats['sync_successes'] == 1


class TestCalibrationLoader:
    """Test calibration loading."""
    
    def test_load_calibration(self, tmp_path):
        """Test loading calibration from YAML file."""
        # Create temporary calibration file
        calib_file = tmp_path / "test_calib.yaml"
        calib_file.write_text("""
intrinsics:
  fx: 800.0
  fy: 800.0
  cx: 640.0
  cy: 360.0
  distortion: [0.1, -0.2, 0.0, 0.0, 0.0]

extrinsics:
  translation: [2.0, 0.5, 1.0]
  rotation: [0.0, 0.0, -0.785]

homography:
  matrix: [[1.0, 0.0, 0.0],
           [0.0, 1.0, 0.0],
           [0.0, 0.0, 1.0]]
""")
        
        loader = CalibrationLoader()
        calibration = loader.load(0, str(calib_file))
        
        assert calibration is not None
        assert calibration.intrinsics.fx == 800.0
        assert calibration.intrinsics.fy == 800.0
        assert calibration.intrinsics.cx == 640.0
        assert calibration.intrinsics.cy == 360.0
        assert len(calibration.intrinsics.distortion) == 5
        assert len(calibration.extrinsics.translation) == 3
        assert len(calibration.extrinsics.rotation) == 3
        assert calibration.homography.shape == (3, 3)
    
    def test_intrinsics_to_matrix(self):
        """Test conversion of intrinsics to camera matrix."""
        intrinsics = CameraIntrinsics(
            fx=800.0,
            fy=800.0,
            cx=640.0,
            cy=360.0,
            distortion=np.zeros(5)
        )
        
        matrix = intrinsics.to_matrix()
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 800.0  # fx
        assert matrix[1, 1] == 800.0  # fy
        assert matrix[0, 2] == 640.0  # cx
        assert matrix[1, 2] == 360.0  # cy
    
    def test_extrinsics_to_rotation_matrix(self):
        """Test conversion of extrinsics to rotation matrix."""
        extrinsics = CameraExtrinsics(
            translation=np.array([0.0, 0.0, 0.0]),
            rotation=np.array([0.0, 0.0, 0.0])  # Identity rotation
        )
        
        rotation_matrix = extrinsics.to_rotation_matrix()
        
        assert rotation_matrix.shape == (3, 3)
        # Should be close to identity matrix for zero rotation
        np.testing.assert_array_almost_equal(rotation_matrix, np.eye(3), decimal=5)


class TestCameraManager:
    """Test camera manager."""
    
    @patch('src.camera.capture.cv2.VideoCapture')
    def test_camera_manager_initialization(self, mock_video_capture):
        """Test camera manager initialization."""
        # Mock camera capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 30
        mock_video_capture.return_value = mock_cap
        
        config = {
            'cameras': {
                'interior': {
                    'device': 0,
                    'resolution': [640, 480],
                    'fps': 30,
                    'calibration': 'configs/calibration/interior.yaml'
                }
            }
        }
        
        manager = CameraManager(config)
        
        assert manager is not None
        assert not manager.is_running
    
    def test_camera_manager_statistics(self):
        """Test camera manager statistics."""
        config = {'cameras': {}}
        manager = CameraManager(config)
        
        stats = manager.get_statistics()
        
        assert 'is_running' in stats
        assert 'frame_count' in stats
        assert 'active_cameras' in stats
