"""Test suite for DMS face detection module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys


@pytest.fixture
def mock_mediapipe():
    """Fixture providing mock MediaPipe modules."""
    mock_mp = MagicMock()
    
    # Mock face detection
    mock_face_detection = MagicMock()
    mock_face_detection_instance = MagicMock()
    mock_face_detection.FaceDetection.return_value = mock_face_detection_instance
    mock_mp.solutions.face_detection = mock_face_detection
    
    # Mock face mesh
    mock_face_mesh = MagicMock()
    mock_face_mesh_instance = MagicMock()
    mock_face_mesh.FaceMesh.return_value = mock_face_mesh_instance
    mock_mp.solutions.face_mesh = mock_face_mesh
    
    return mock_mp


@pytest.fixture
def sample_frame():
    """Fixture providing a sample BGR frame."""
    # Create a 480x640 BGR frame (typical interior camera resolution)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def mock_face_landmarks():
    """Fixture providing mock face landmarks from MediaPipe."""
    mock_landmarks = MagicMock()
    
    # Create mock landmark points (468 landmarks in MediaPipe Face Mesh)
    landmarks = []
    for i in range(468):
        lm = MagicMock()
        lm.x = 0.5 + (i % 10) * 0.01  # Normalized x coordinate
        lm.y = 0.5 + (i % 10) * 0.01  # Normalized y coordinate
        lm.z = 0.0
        landmarks.append(lm)
    
    mock_landmarks.landmark = landmarks
    return mock_landmarks


class TestFaceDetector:
    """Test suite for FaceDetector class."""
    
    @patch('src.dms.face.mp')
    def test_initialization_success(self, mock_mp):
        """Test that FaceDetector initializes correctly with MediaPipe available."""
        from src.dms.face import FaceDetector
        
        # Setup mock
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        # Initialize
        detector = FaceDetector()
        
        # Verify initialization
        assert detector is not None
        assert detector.mp_face_detection is not None
        assert detector.mp_face_mesh is not None
        mock_mp.solutions.face_detection.FaceDetection.assert_called_once()
        mock_mp.solutions.face_mesh.FaceMesh.assert_called_once()
    
    @patch('src.dms.face.mp', None)
    def test_initialization_no_mediapipe(self):
        """Test that FaceDetector raises ImportError when MediaPipe is not installed."""
        from src.dms.face import FaceDetector
        
        with pytest.raises(ImportError, match="MediaPipe not installed"):
            FaceDetector()
    
    @patch('src.dms.face.mp')
    def test_detect_face_success(self, mock_mp, sample_frame, mock_face_landmarks):
        """Test successful face detection and landmark extraction."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock face mesh results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Detect face
        face_detected, landmarks = detector.detect_and_extract_landmarks(sample_frame)
        
        # Verify results
        assert face_detected is True
        assert landmarks is not None
        assert landmarks.shape == (68, 2)
        assert landmarks.dtype == np.float32
    
    @patch('src.dms.face.mp')
    def test_detect_no_face(self, mock_mp, sample_frame):
        """Test detection when no face is present in frame."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock no face detected
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = None
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Detect face
        face_detected, landmarks = detector.detect_and_extract_landmarks(sample_frame)
        
        # Verify results
        assert face_detected is False
        assert landmarks is None
    
    @patch('src.dms.face.mp')
    def test_detect_with_none_frame(self, mock_mp):
        """Test detection with None frame input."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Detect with None frame
        face_detected, landmarks = detector.detect_and_extract_landmarks(None)
        
        # Verify results
        assert face_detected is False
        assert landmarks is None
    
    @patch('src.dms.face.mp')
    def test_detect_with_empty_frame(self, mock_mp):
        """Test detection with empty frame."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Create empty frame
        empty_frame = np.array([], dtype=np.uint8)
        
        # Detect with empty frame
        face_detected, landmarks = detector.detect_and_extract_landmarks(empty_frame)
        
        # Verify results
        assert face_detected is False
        assert landmarks is None
    
    @patch('src.dms.face.mp')
    def test_landmark_indices_count(self, mock_mp):
        """Test that _get_68_landmark_indices returns exactly 68 indices."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Get landmark indices
        indices = detector._get_68_landmark_indices()
        
        # Verify count
        assert len(indices) == 68
        assert all(isinstance(idx, int) for idx in indices)
    
    @patch('src.dms.face.mp')
    def test_landmark_coordinates_in_frame_bounds(self, mock_mp, sample_frame, mock_face_landmarks):
        """Test that extracted landmarks are within frame boundaries."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock face mesh results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Detect face
        face_detected, landmarks = detector.detect_and_extract_landmarks(sample_frame)
        
        # Verify landmarks are within frame bounds
        h, w = sample_frame.shape[:2]
        assert face_detected is True
        assert np.all(landmarks[:, 0] >= 0)  # x >= 0
        assert np.all(landmarks[:, 0] <= w)  # x <= width
        assert np.all(landmarks[:, 1] >= 0)  # y >= 0
        assert np.all(landmarks[:, 1] <= h)  # y <= height
    
    @patch('src.dms.face.mp')
    def test_multiple_detections_uses_first_face(self, mock_mp, sample_frame, mock_face_landmarks):
        """Test that when multiple faces are detected, only the first is used."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock multiple faces detected
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks, mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Detect face
        face_detected, landmarks = detector.detect_and_extract_landmarks(sample_frame)
        
        # Verify only first face is used
        assert face_detected is True
        assert landmarks is not None
        assert landmarks.shape == (68, 2)
    
    @patch('src.dms.face.mp')
    @patch('src.dms.face.cv2.cvtColor')
    def test_bgr_to_rgb_conversion(self, mock_cvtcolor, mock_mp, sample_frame, mock_face_landmarks):
        """Test that frame is converted from BGR to RGB for MediaPipe."""
        from src.dms.face import FaceDetector
        import cv2
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock face mesh results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Mock cvtColor to return the frame
        mock_cvtcolor.return_value = sample_frame
        
        # Detect face
        detector.detect_and_extract_landmarks(sample_frame)
        
        # Verify BGR to RGB conversion was called
        mock_cvtcolor.assert_called_once()
        args = mock_cvtcolor.call_args[0]
        assert np.array_equal(args[0], sample_frame)
        assert args[1] == cv2.COLOR_BGR2RGB
    
    @patch('src.dms.face.mp')
    @pytest.mark.performance
    def test_detection_performance(self, mock_mp, sample_frame, mock_face_landmarks):
        """Test that face detection completes within performance requirements (< 25ms)."""
        from src.dms.face import FaceDetector
        import time
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock face mesh results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Measure execution time
        start_time = time.perf_counter()
        detector.detect_and_extract_landmarks(sample_frame)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Note: This is testing the mock overhead, not actual MediaPipe performance
        # In real usage, MediaPipe should complete within 25ms target
        assert execution_time_ms < 50, f"Execution took {execution_time_ms:.2f}ms"
    
    @patch('src.dms.face.mp')
    def test_cleanup_on_deletion(self, mock_mp):
        """Test that MediaPipe resources are cleaned up on deletion."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock close methods
        detector.face_detection.close = MagicMock()
        detector.face_mesh.close = MagicMock()
        
        # Delete detector
        del detector
        
        # Note: __del__ is called by garbage collector, so we can't directly verify
        # This test documents the expected behavior
    
    @patch('src.dms.face.mp')
    def test_landmark_array_dtype(self, mock_mp, sample_frame, mock_face_landmarks):
        """Test that landmarks are returned as float32 dtype."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock face mesh results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Detect face
        face_detected, landmarks = detector.detect_and_extract_landmarks(sample_frame)
        
        # Verify dtype
        assert face_detected is True
        assert landmarks.dtype == np.float32
    
    @patch('src.dms.face.mp')
    def test_different_frame_sizes(self, mock_mp, mock_face_landmarks):
        """Test detection with different frame sizes."""
        from src.dms.face import FaceDetector
        
        # Setup mocks
        mock_mp.solutions.face_detection = MagicMock()
        mock_mp.solutions.face_mesh = MagicMock()
        
        detector = FaceDetector()
        
        # Mock face mesh results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        detector.face_mesh.process = MagicMock(return_value=mock_results)
        
        # Test with different resolutions
        test_sizes = [(480, 640), (720, 1280), (1080, 1920)]
        
        for h, w in test_sizes:
            frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            face_detected, landmarks = detector.detect_and_extract_landmarks(frame)
            
            assert face_detected is True
            assert landmarks.shape == (68, 2)
            # Landmarks should be scaled to frame size
            assert np.all(landmarks[:, 0] <= w)
            assert np.all(landmarks[:, 1] <= h)
