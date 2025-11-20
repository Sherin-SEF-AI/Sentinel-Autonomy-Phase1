"""
Unit tests for driver behavior profiling module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.profiling.face_recognition import FaceRecognitionSystem
from src.profiling.metrics_tracker import MetricsTracker
from src.profiling.style_classifier import DrivingStyle, DrivingStyleClassifier
from src.profiling.threshold_adapter import ThresholdAdapter
from src.profiling.report_generator import DriverReportGenerator
from src.profiling.profile_manager import ProfileManager, DriverProfile


class TestFaceRecognitionSystem:
    """Tests for FaceRecognitionSystem."""
    
    def test_initialization(self):
        """Test face recognition system initialization."""
        config = {'recognition_threshold': 0.6, 'embedding_size': 128}
        system = FaceRecognitionSystem(config)
        
        assert system.recognition_threshold == 0.6
        assert system.embedding_size == 128
    
    def test_generate_driver_id(self):
        """Test driver ID generation."""
        config = {'recognition_threshold': 0.6, 'embedding_size': 128}
        system = FaceRecognitionSystem(config)
        
        embedding = np.random.randn(128).astype(np.float32)
        driver_id = system.generate_driver_id(embedding)
        
        assert driver_id.startswith('driver_')
        assert len(driver_id) > 7
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        config = {'recognition_threshold': 0.6, 'embedding_size': 128}
        system = FaceRecognitionSystem(config)
        
        # Identical embeddings
        emb1 = np.random.randn(128).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        similarity = system._cosine_similarity(emb1, emb1)
        assert similarity > 0.99
        
        # Different embeddings
        emb2 = np.random.randn(128).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)
        similarity = system._cosine_similarity(emb1, emb2)
        assert 0.0 <= similarity <= 1.0


class TestMetricsTracker:
    """Tests for MetricsTracker."""
    
    def test_initialization(self):
        """Test metrics tracker initialization."""
        tracker = MetricsTracker({})
        
        assert len(tracker.reaction_times) == 0
        assert len(tracker.following_distances) == 0
        assert tracker.session_start is None
    
    def test_session_tracking(self):
        """Test session start and end."""
        tracker = MetricsTracker({})
        
        tracker.start_session(100.0)
        assert tracker.session_start == 100.0
        
        tracker.end_session(200.0)
        assert tracker.session_duration == 100.0
    
    def test_metrics_update(self):
        """Test metrics update."""
        tracker = MetricsTracker({})
        tracker.start_session(0.0)
        
        tracker.update(1.0, speed=25.0, following_distance=30.0, lane_id=2, risk_score=0.3)
        
        assert len(tracker.speeds) == 1
        assert len(tracker.following_distances) == 1
        assert len(tracker.risk_scores) == 1
    
    def test_reaction_time_tracking(self):
        """Test reaction time tracking."""
        tracker = MetricsTracker({})
        
        # Record alert
        tracker.record_alert(1, 100.0)
        assert 1 in tracker.pending_alerts
        
        # Record action
        tracker.record_driver_action(1, 101.5, 'brake')
        assert 1 not in tracker.pending_alerts
        assert len(tracker.reaction_times) == 1
        assert tracker.reaction_times[0] == 1.5
    
    def test_lane_change_detection(self):
        """Test lane change detection."""
        tracker = MetricsTracker({})
        tracker.start_session(0.0)
        
        tracker.update(1.0, lane_id=2)
        tracker.update(2.0, lane_id=3)  # Lane change
        
        assert len(tracker.lane_changes) == 1


class TestDrivingStyleClassifier:
    """Tests for DrivingStyleClassifier."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        classifier = DrivingStyleClassifier({})
        
        assert 'reaction_time' in classifier.thresholds
        assert 'following_distance' in classifier.thresholds
    
    def test_aggressive_classification(self):
        """Test aggressive driving classification."""
        classifier = DrivingStyleClassifier({})
        
        # Aggressive metrics
        metrics = {
            'reaction_time': {'mean': 0.6, 'count': 10},
            'following_distance': {'mean': 12.0, 'count': 100},
            'lane_change_frequency': 10.0,
            'speed_profile': {'std': 6.0},
            'risk_tolerance': 0.7
        }
        
        style = classifier.classify(metrics)
        assert style == DrivingStyle.AGGRESSIVE
    
    def test_cautious_classification(self):
        """Test cautious driving classification."""
        classifier = DrivingStyleClassifier({})
        
        # Cautious metrics
        metrics = {
            'reaction_time': {'mean': 1.8, 'count': 10},
            'following_distance': {'mean': 40.0, 'count': 100},
            'lane_change_frequency': 1.5,
            'speed_profile': {'std': 1.5},
            'risk_tolerance': 0.2
        }
        
        style = classifier.classify(metrics)
        assert style == DrivingStyle.CAUTIOUS
    
    def test_insufficient_data(self):
        """Test classification with insufficient data."""
        classifier = DrivingStyleClassifier({})
        
        metrics = {
            'reaction_time': {'mean': 1.0, 'count': 1},
            'following_distance': {'mean': 25.0, 'count': 5}
        }
        
        style = classifier.classify(metrics)
        assert style == DrivingStyle.UNKNOWN


class TestThresholdAdapter:
    """Tests for ThresholdAdapter."""
    
    def test_initialization(self):
        """Test threshold adapter initialization."""
        config = {
            'base_ttc_threshold': 2.0,
            'base_following_distance': 25.0,
            'base_alert_sensitivity': 0.7
        }
        adapter = ThresholdAdapter(config)
        
        assert adapter.base_ttc_threshold == 2.0
        assert adapter.safety_margin == 1.5
    
    def test_ttc_adaptation(self):
        """Test TTC threshold adaptation."""
        config = {'base_ttc_threshold': 2.0}
        adapter = ThresholdAdapter(config)
        
        # Fast reaction time
        adapted_ttc = adapter._adapt_ttc_threshold(0.8)
        assert adapted_ttc == 0.8 * 1.5  # 1.2s
        
        # Slow reaction time
        adapted_ttc = adapter._adapt_ttc_threshold(2.5)
        assert adapted_ttc == 2.5 * 1.5  # 3.75s
    
    def test_following_distance_adaptation(self):
        """Test following distance adaptation."""
        config = {'base_following_distance': 25.0}
        adapter = ThresholdAdapter(config)
        
        # Aggressive style
        distance = adapter._adapt_following_distance(DrivingStyle.AGGRESSIVE)
        assert distance < 25.0
        
        # Cautious style
        distance = adapter._adapt_following_distance(DrivingStyle.CAUTIOUS)
        assert distance > 25.0
    
    def test_safety_margin_applied(self):
        """Test that safety margin is always applied."""
        config = {'base_ttc_threshold': 2.0}
        adapter = ThresholdAdapter(config)
        
        metrics = {'reaction_time': {'mean': 1.0}}
        adapted = adapter.adapt_thresholds(metrics, DrivingStyle.NORMAL)
        
        # TTC should be at least reaction_time * 1.5
        assert adapted['ttc_threshold'] >= 1.0 * 1.5


class TestReportGenerator:
    """Tests for DriverReportGenerator."""
    
    def test_initialization(self):
        """Test report generator initialization."""
        generator = DriverReportGenerator({})
        
        assert len(generator.report_history) == 0
    
    def test_report_generation(self):
        """Test report generation."""
        generator = DriverReportGenerator({})
        
        metrics = {
            'session_duration': 3600.0,
            'total_distance': 50000.0,
            'reaction_time': {'mean': 1.2, 'std': 0.3, 'count': 10},
            'following_distance': {'mean': 28.0, 'std': 5.0, 'count': 100},
            'lane_change_frequency': 4.5,
            'speed_profile': {'mean': 22.0, 'std': 3.0},
            'risk_tolerance': 0.4,
            'near_miss_count': 1
        }
        
        report = generator.generate_report(metrics, DrivingStyle.NORMAL, 'driver_test')
        
        assert 'driver_id' in report
        assert 'scores' in report
        assert 'safety' in report['scores']
        assert 'attention' in report['scores']
        assert 'eco_driving' in report['scores']
        assert 0 <= report['scores']['safety'] <= 100
    
    def test_score_calculation(self):
        """Test score calculations."""
        generator = DriverReportGenerator({})
        
        # Good metrics
        good_metrics = {
            'session_duration': 3600.0,
            'total_distance': 50000.0,
            'reaction_time': {'mean': 1.0, 'std': 0.2, 'count': 10},
            'following_distance': {'mean': 30.0, 'std': 3.0, 'count': 100},
            'lane_change_frequency': 3.0,
            'speed_profile': {'mean': 22.0, 'std': 2.0},
            'risk_tolerance': 0.3,
            'near_miss_count': 0
        }
        
        safety_score = generator._calculate_safety_score(good_metrics, DrivingStyle.CAUTIOUS)
        assert safety_score > 80
    
    def test_export_text(self):
        """Test text export."""
        generator = DriverReportGenerator({})
        
        metrics = {
            'session_duration': 3600.0,
            'total_distance': 50000.0,
            'reaction_time': {'mean': 1.2, 'count': 10},
            'following_distance': {'mean': 28.0, 'count': 100},
            'lane_change_frequency': 4.5,
            'speed_profile': {'mean': 22.0},
            'risk_tolerance': 0.4,
            'near_miss_count': 1
        }
        
        report = generator.generate_report(metrics, DrivingStyle.NORMAL, 'driver_test')
        text = generator.export_report_text(report)
        
        assert 'DRIVER BEHAVIOR REPORT' in text
        assert 'driver_test' in text
        assert 'SCORES:' in text


class TestProfileManager:
    """Tests for ProfileManager."""
    
    @pytest.fixture
    def temp_profiles_dir(self):
        """Create temporary profiles directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_profiles_dir):
        """Test profile manager initialization."""
        config = {'profiles_dir': temp_profiles_dir}
        manager = ProfileManager(config)
        
        assert manager.profiles_dir == Path(temp_profiles_dir)
        assert len(manager.profiles) == 0
    
    def test_profile_creation(self, temp_profiles_dir):
        """Test new profile creation."""
        config = {'profiles_dir': temp_profiles_dir, 'auto_save': False}
        manager = ProfileManager(config)
        
        # Create embedding
        embedding = np.random.randn(128).astype(np.float32)
        driver_id = manager.face_recognition.generate_driver_id(embedding)
        
        # Create profile
        manager._create_new_profile(driver_id, embedding)
        
        assert driver_id in manager.profiles
        profile = manager.profiles[driver_id]
        assert profile.driver_id == driver_id
        assert len(profile.face_embedding) == 128
    
    def test_profile_persistence(self, temp_profiles_dir):
        """Test profile save and load."""
        config = {'profiles_dir': temp_profiles_dir, 'auto_save': True}
        manager = ProfileManager(config)
        
        # Create and save profile
        embedding = np.random.randn(128).astype(np.float32)
        driver_id = manager.face_recognition.generate_driver_id(embedding)
        manager._create_new_profile(driver_id, embedding)
        manager.save_profile(driver_id)
        
        # Create new manager and load
        new_manager = ProfileManager(config)
        loaded_profile = new_manager.get_profile(driver_id)
        
        assert loaded_profile is not None
        assert loaded_profile.driver_id == driver_id
    
    def test_session_workflow(self, temp_profiles_dir):
        """Test complete session workflow."""
        config = {'profiles_dir': temp_profiles_dir, 'auto_save': False}
        manager = ProfileManager(config)
        
        # Create profile
        embedding = np.random.randn(128).astype(np.float32)
        driver_id = manager.face_recognition.generate_driver_id(embedding)
        manager._create_new_profile(driver_id, embedding)
        
        # Start session
        manager.start_session(driver_id, 0.0)
        
        # Update metrics
        tracker = manager.get_metrics_tracker()
        tracker.update(1.0, speed=25.0, following_distance=30.0, risk_score=0.3)
        
        # End session
        manager.end_session(10.0)
        
        # Check profile updated
        profile = manager.get_profile(driver_id)
        assert profile.session_count == 1
        assert profile.total_time > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
