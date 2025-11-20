"""Tests for Driver Monitoring System (DMS)."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dms import (
    DriverMonitor,
    FaceDetector,
    GazeEstimator,
    HeadPoseEstimator,
    DrowsinessDetector,
    DistractionClassifier,
    ReadinessCalculator
)
from src.core.config import ConfigManager


@pytest.fixture
def config():
    """Load test configuration."""
    config_manager = ConfigManager('configs/default.yaml')
    return config_manager.config


@pytest.fixture
def test_frame():
    """Create test frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_landmarks():
    """Create test landmarks."""
    # Generate random landmarks in reasonable positions
    landmarks = np.random.rand(68, 2) * np.array([640, 480])
    return landmarks.astype(np.float32)


class TestHeadPoseEstimator:
    """Tests for HeadPoseEstimator."""
    
    def test_initialization(self):
        """Test head pose estimator initialization."""
        estimator = HeadPoseEstimator()
        assert estimator is not None
        assert estimator.model_points.shape == (6, 3)
    
    def test_estimate_head_pose(self, test_landmarks):
        """Test head pose estimation."""
        estimator = HeadPoseEstimator()
        head_pose = estimator.estimate_head_pose(test_landmarks, (480, 640))
        
        assert 'roll' in head_pose
        assert 'pitch' in head_pose
        assert 'yaw' in head_pose
        assert isinstance(head_pose['roll'], float)
        assert isinstance(head_pose['pitch'], float)
        assert isinstance(head_pose['yaw'], float)
    
    def test_default_pose_on_invalid_input(self):
        """Test default pose returned on invalid input."""
        estimator = HeadPoseEstimator()
        head_pose = estimator.estimate_head_pose(None, (480, 640))
        
        assert head_pose['roll'] == 0.0
        assert head_pose['pitch'] == 0.0
        assert head_pose['yaw'] == 0.0


class TestGazeEstimator:
    """Tests for GazeEstimator."""
    
    def test_initialization(self):
        """Test gaze estimator initialization."""
        estimator = GazeEstimator()
        assert estimator is not None
        assert len(estimator.ATTENTION_ZONES) == 8
    
    def test_estimate_gaze(self, test_frame, test_landmarks):
        """Test gaze estimation."""
        estimator = GazeEstimator()
        gaze = estimator.estimate_gaze(test_frame, test_landmarks)
        
        assert 'pitch' in gaze
        assert 'yaw' in gaze
        assert 'attention_zone' in gaze
        assert isinstance(gaze['pitch'], float)
        assert isinstance(gaze['yaw'], float)
        assert gaze['attention_zone'] in estimator.ATTENTION_ZONES.keys()
    
    def test_attention_zone_mapping(self):
        """Test attention zone mapping."""
        estimator = GazeEstimator()
        
        # Test front zone
        assert estimator._map_to_attention_zone(0.0) == 'front'
        assert estimator._map_to_attention_zone(15.0) == 'front'
        
        # Test left zones
        assert estimator._map_to_attention_zone(45.0) == 'front_left'
        assert estimator._map_to_attention_zone(90.0) == 'left'
        
        # Test right zones
        assert estimator._map_to_attention_zone(-45.0) == 'front_right'
        assert estimator._map_to_attention_zone(-90.0) == 'right'


class TestDrowsinessDetector:
    """Tests for DrowsinessDetector."""
    
    def test_initialization(self):
        """Test drowsiness detector initialization."""
        detector = DrowsinessDetector(fps=30)
        assert detector is not None
        assert detector.fps == 30
        assert detector.perclos_window == 60
    
    def test_detect_drowsiness(self, test_landmarks):
        """Test drowsiness detection."""
        detector = DrowsinessDetector(fps=30)
        head_pose = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        
        drowsiness = detector.detect_drowsiness(test_landmarks, head_pose)
        
        assert 'score' in drowsiness
        assert 'yawn_detected' in drowsiness
        assert 'micro_sleep' in drowsiness
        assert 'head_nod' in drowsiness
        assert 'perclos' in drowsiness
        assert 0.0 <= drowsiness['score'] <= 1.0
        assert 0.0 <= drowsiness['perclos'] <= 1.0
    
    def test_ear_calculation(self, test_landmarks):
        """Test Eye Aspect Ratio calculation."""
        detector = DrowsinessDetector(fps=30)
        
        left_ear = detector._calculate_ear(test_landmarks, 'left')
        right_ear = detector._calculate_ear(test_landmarks, 'right')
        
        assert isinstance(left_ear, float)
        assert isinstance(right_ear, float)
        assert left_ear >= 0.0
        assert right_ear >= 0.0


class TestDistractionClassifier:
    """Tests for DistractionClassifier."""
    
    def test_initialization(self):
        """Test distraction classifier initialization."""
        classifier = DistractionClassifier()
        assert classifier is not None
        assert len(classifier.DISTRACTION_TYPES) == 6
    
    def test_classify_distraction(self, test_frame):
        """Test distraction classification."""
        classifier = DistractionClassifier()
        gaze = {'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'}
        head_pose = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        
        distraction = classifier.classify_distraction(test_frame, gaze, head_pose)
        
        assert 'type' in distraction
        assert 'confidence' in distraction
        assert 'duration' in distraction
        assert 'eyes_off_road' in distraction
        assert distraction['type'] in classifier.DISTRACTION_TYPES
        assert 0.0 <= distraction['confidence'] <= 1.0
    
    def test_rule_based_classification(self):
        """Test rule-based distraction classification."""
        classifier = DistractionClassifier()
        
        # Test safe driving
        gaze = {'attention_zone': 'front'}
        head_pose = {'yaw': 0.0, 'pitch': 0.0}
        dist_type, conf = classifier._rule_based_classification(gaze, head_pose)
        assert dist_type == 'safe_driving'
        
        # Test looking down (phone)
        head_pose = {'yaw': 0.0, 'pitch': -25.0}
        dist_type, conf = classifier._rule_based_classification(gaze, head_pose)
        assert dist_type == 'phone_usage'


class TestReadinessCalculator:
    """Tests for ReadinessCalculator."""
    
    def test_initialization(self):
        """Test readiness calculator initialization."""
        calculator = ReadinessCalculator()
        assert calculator is not None
        assert calculator.ALERTNESS_WEIGHT == 0.4
        assert calculator.ATTENTION_WEIGHT == 0.3
        assert calculator.DISTRACTION_WEIGHT == 0.3
    
    def test_calculate_readiness(self):
        """Test readiness score calculation."""
        calculator = ReadinessCalculator()
        
        drowsiness = {'score': 0.2}
        gaze = {'attention_zone': 'front'}
        distraction = {'type': 'safe_driving', 'duration': 0.0, 'eyes_off_road': False}
        
        readiness = calculator.calculate_readiness(drowsiness, gaze, distraction)
        
        assert isinstance(readiness, float)
        assert 0.0 <= readiness <= 100.0
    
    def test_readiness_with_high_drowsiness(self):
        """Test readiness with high drowsiness."""
        calculator = ReadinessCalculator()
        
        drowsiness = {'score': 0.9}  # Very drowsy
        gaze = {'attention_zone': 'front'}
        distraction = {'type': 'safe_driving', 'duration': 0.0, 'eyes_off_road': False}
        
        readiness = calculator.calculate_readiness(drowsiness, gaze, distraction)
        
        # Should be low due to high drowsiness
        assert readiness < 50.0
    
    def test_readiness_with_distraction(self):
        """Test readiness with distraction."""
        calculator = ReadinessCalculator()
        
        drowsiness = {'score': 0.1}
        gaze = {'attention_zone': 'left'}
        distraction = {'type': 'phone_usage', 'duration': 5.0, 'eyes_off_road': True}
        
        readiness = calculator.calculate_readiness(drowsiness, gaze, distraction)
        
        # Should be low due to distraction
        assert readiness < 50.0


class TestDriverMonitor:
    """Tests for DriverMonitor (main DMS class)."""
    
    def test_initialization(self, config):
        """Test driver monitor initialization."""
        dms = DriverMonitor(config)
        assert dms is not None
        assert dms.gaze_estimator is not None
        assert dms.pose_estimator is not None
        assert dms.drowsiness_detector is not None
        assert dms.distraction_classifier is not None
        assert dms.readiness_calculator is not None
    
    def test_analyze_with_test_frame(self, config, test_frame):
        """Test analyze method with test frame."""
        dms = DriverMonitor(config)
        driver_state = dms.analyze(test_frame)
        
        # Check all required fields are present
        assert hasattr(driver_state, 'face_detected')
        assert hasattr(driver_state, 'landmarks')
        assert hasattr(driver_state, 'head_pose')
        assert hasattr(driver_state, 'gaze')
        assert hasattr(driver_state, 'eye_state')
        assert hasattr(driver_state, 'drowsiness')
        assert hasattr(driver_state, 'distraction')
        assert hasattr(driver_state, 'readiness_score')
        
        # Check readiness score is in valid range
        assert 0.0 <= driver_state.readiness_score <= 100.0
    
    def test_default_state_creation(self, config):
        """Test default state creation."""
        dms = DriverMonitor(config)
        default_state = dms._create_default_state()
        
        assert default_state.face_detected is False
        assert default_state.landmarks.shape == (68, 2)
        assert default_state.readiness_score == 50.0
    
    def test_error_recovery(self, config):
        """Test error recovery mechanism."""
        dms = DriverMonitor(config)
        
        # Simulate error by passing None
        driver_state = dms.analyze(None)
        
        # Should return default state
        assert driver_state is not None
        assert driver_state.readiness_score >= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
