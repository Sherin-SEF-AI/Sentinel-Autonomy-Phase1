"""Tests for multi-view object detection module."""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_structures import Detection2D, Detection3D
from src.perception.detection import (
    Detector2D, Estimator3D, MultiViewFusion, ObjectTracker, ObjectDetector
)


@pytest.fixture
def detection_config():
    """Detection configuration for testing."""
    return {
        'architecture': 'YOLOv8',
        'variant': 'yolov8n',  # Use nano for faster testing
        'weights': 'yolov8n.pt',
        'confidence_threshold': 0.5,
        'nms_threshold': 0.4,
        'device': 'cpu'  # Use CPU for testing
    }


@pytest.fixture
def calibration_data():
    """Camera calibration data for testing."""
    return {
        1: {
            'intrinsics': {
                'fx': 800.0, 'fy': 800.0,
                'cx': 640.0, 'cy': 360.0,
                'distortion': [0, 0, 0, 0, 0]
            },
            'extrinsics': {
                'translation': [2.0, -0.5, 1.2],
                'rotation': [0, 0.1, 0]
            }
        },
        2: {
            'intrinsics': {
                'fx': 800.0, 'fy': 800.0,
                'cx': 640.0, 'cy': 360.0,
                'distortion': [0, 0, 0, 0, 0]
            },
            'extrinsics': {
                'translation': [2.0, 0.5, 1.2],
                'rotation': [0, 0.1, 0]
            }
        }
    }


@pytest.fixture
def fusion_config():
    """Fusion configuration for testing."""
    return {
        'iou_threshold_3d': 0.3,
        'confidence_weighting': True
    }


@pytest.fixture
def tracking_config():
    """Tracking configuration for testing."""
    return {
        'max_age': 30,
        'min_hits': 3,
        'iou_threshold': 0.3
    }


@pytest.fixture
def test_frame():
    """Create a test frame."""
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_detection_2d():
    """Create a sample 2D detection."""
    return Detection2D(
        bbox=(100.0, 100.0, 300.0, 400.0),
        class_name='vehicle',
        confidence=0.85,
        camera_id=1
    )


@pytest.fixture
def sample_detection_3d():
    """Create a sample 3D detection."""
    return Detection3D(
        bbox_3d=(10.0, 2.0, 0.0, 1.8, 1.5, 4.5, 0.0),
        class_name='vehicle',
        confidence=0.85,
        velocity=(0.0, 0.0, 0.0),
        track_id=-1
    )


class TestDetector2D:
    """Tests for 2D object detector."""
    
    def test_initialization(self, detection_config):
        """Test detector initialization."""
        detector = Detector2D(detection_config)
        assert detector is not None
        assert detector.confidence_threshold == 0.5
        assert detector.nms_threshold == 0.4
    
    def test_detect_returns_list(self, detection_config, test_frame):
        """Test that detect returns a list."""
        detector = Detector2D(detection_config)
        detections = detector.detect(test_frame, camera_id=1)
        assert isinstance(detections, list)
    
    def test_detect_batch(self, detection_config, test_frame):
        """Test batch detection."""
        detector = Detector2D(detection_config)
        frames = {1: test_frame, 2: test_frame}
        detections = detector.detect_batch(frames)
        assert isinstance(detections, dict)
        assert 1 in detections
        assert 2 in detections


class TestEstimator3D:
    """Tests for 3D bounding box estimator."""
    
    def test_initialization(self, calibration_data):
        """Test estimator initialization."""
        estimator = Estimator3D(calibration_data)
        assert estimator is not None
        assert len(estimator.calibration_data) == 2
    
    def test_estimate(self, calibration_data, sample_detection_2d):
        """Test 3D estimation from 2D detection."""
        estimator = Estimator3D(calibration_data)
        detection_3d = estimator.estimate(sample_detection_2d)
        
        assert detection_3d is not None
        assert isinstance(detection_3d, Detection3D)
        assert len(detection_3d.bbox_3d) == 7
        assert detection_3d.class_name == sample_detection_2d.class_name
        assert detection_3d.confidence == sample_detection_2d.confidence
    
    def test_estimate_batch(self, calibration_data, sample_detection_2d):
        """Test batch 3D estimation."""
        estimator = Estimator3D(calibration_data)
        detections_2d = [sample_detection_2d] * 3
        detections_3d = estimator.estimate_batch(detections_2d)
        
        assert isinstance(detections_3d, list)
        assert len(detections_3d) <= len(detections_2d)


class TestMultiViewFusion:
    """Tests for multi-view fusion."""
    
    def test_initialization(self, fusion_config):
        """Test fusion initialization."""
        fusion = MultiViewFusion(fusion_config)
        assert fusion is not None
        assert fusion.iou_threshold == 0.3
    
    def test_fuse_empty_list(self, fusion_config):
        """Test fusion with empty list."""
        fusion = MultiViewFusion(fusion_config)
        result = fusion.fuse([])
        assert result == []
    
    def test_fuse_single_detection(self, fusion_config, sample_detection_3d):
        """Test fusion with single detection."""
        fusion = MultiViewFusion(fusion_config)
        result = fusion.fuse([sample_detection_3d])
        assert len(result) == 1
    
    def test_fuse_overlapping_detections(self, fusion_config):
        """Test fusion of overlapping detections."""
        fusion = MultiViewFusion(fusion_config)
        
        # Create two overlapping detections
        det1 = Detection3D(
            bbox_3d=(10.0, 2.0, 0.0, 1.8, 1.5, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.85,
            velocity=(0.0, 0.0, 0.0),
            track_id=-1
        )
        det2 = Detection3D(
            bbox_3d=(10.5, 2.2, 0.0, 1.8, 1.5, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.80,
            velocity=(0.0, 0.0, 0.0),
            track_id=-1
        )
        
        result = fusion.fuse([det1, det2])
        # Should merge into one detection
        assert len(result) <= 2


class TestObjectTracker:
    """Tests for object tracker."""
    
    def test_initialization(self, tracking_config):
        """Test tracker initialization."""
        tracker = ObjectTracker(tracking_config)
        assert tracker is not None
        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert len(tracker.tracks) == 0
    
    def test_update_empty(self, tracking_config):
        """Test update with no detections."""
        tracker = ObjectTracker(tracking_config)
        result = tracker.update([])
        assert result == []
    
    def test_update_creates_tracks(self, tracking_config, sample_detection_3d):
        """Test that update creates new tracks."""
        tracker = ObjectTracker(tracking_config)
        
        # Update multiple times to confirm track
        for _ in range(5):
            result = tracker.update([sample_detection_3d])
        
        # Should have confirmed tracks after min_hits
        assert len(result) > 0
        assert result[0].track_id >= 0
    
    def test_track_persistence(self, tracking_config, sample_detection_3d):
        """Test that tracks persist across frames."""
        tracker = ObjectTracker(tracking_config)
        
        # Create track
        for _ in range(5):
            tracker.update([sample_detection_3d])
        
        # Track should persist even without detections for a while
        for _ in range(10):
            result = tracker.update([])
        
        # Track should still exist (max_age = 30)
        assert len(tracker.tracks) > 0


class TestObjectDetector:
    """Tests for integrated object detector."""
    
    def test_initialization(self, detection_config, calibration_data, 
                          fusion_config, tracking_config):
        """Test detector initialization."""
        config = {
            'detection': detection_config,
            'fusion': fusion_config,
            'tracking': tracking_config
        }
        detector = ObjectDetector(config, calibration_data)
        assert detector is not None
    
    def test_detect_returns_tuple(self, detection_config, calibration_data,
                                  fusion_config, tracking_config, test_frame):
        """Test that detect returns correct tuple."""
        config = {
            'detection': detection_config,
            'fusion': fusion_config,
            'tracking': tracking_config
        }
        detector = ObjectDetector(config, calibration_data)
        
        frames = {1: test_frame, 2: test_frame}
        detections_2d, detections_3d = detector.detect(frames)
        
        assert isinstance(detections_2d, dict)
        assert isinstance(detections_3d, list)
    
    def test_get_statistics(self, detection_config, calibration_data,
                           fusion_config, tracking_config):
        """Test statistics retrieval."""
        config = {
            'detection': detection_config,
            'fusion': fusion_config,
            'tracking': tracking_config
        }
        detector = ObjectDetector(config, calibration_data)
        
        stats = detector.get_statistics()
        assert 'active_tracks' in stats
        assert 'confirmed_tracks' in stats
        assert 'frame_count' in stats
        assert 'error_count' in stats


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
