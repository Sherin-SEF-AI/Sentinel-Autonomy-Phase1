"""Test suite for profiling module initialization and exports."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Test that all exports are available
from src.profiling import (
    DriverProfile,
    DriverReportGenerator,
    DrivingStyle,
    DrivingStyleClassifier,
    FaceRecognitionSystem,
    MetricsSnapshot,
    MetricsTracker,
    ProfileManager,
    ThresholdAdapter,
)


class TestProfilingModuleExports:
    """Test that all expected classes are exported from the profiling module."""
    
    def test_driver_profile_export(self):
        """Test that DriverProfile is exported and can be instantiated."""
        profile = DriverProfile(
            driver_id="test_driver",
            face_embedding=[0.1, 0.2, 0.3],
            total_distance=1000.0,
            total_time=3600.0
        )
        assert profile.driver_id == "test_driver"
        assert profile.total_distance == 1000.0
    
    def test_metrics_snapshot_export(self):
        """Test that MetricsSnapshot is exported and can be instantiated."""
        snapshot = MetricsSnapshot(
            timestamp=1234567890.0,
            reaction_time=0.8,
            following_distance=25.0,
            speed=20.0
        )
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.reaction_time == 0.8
    
    def test_driving_style_export(self):
        """Test that DrivingStyle enum is exported."""
        assert hasattr(DrivingStyle, 'AGGRESSIVE')
        assert hasattr(DrivingStyle, 'NORMAL')
        assert hasattr(DrivingStyle, 'CAUTIOUS')
        assert hasattr(DrivingStyle, 'UNKNOWN')
        
        assert DrivingStyle.AGGRESSIVE.value == "aggressive"
        assert DrivingStyle.NORMAL.value == "normal"
    
    def test_face_recognition_system_export(self):
        """Test that FaceRecognitionSystem is exported and can be instantiated."""
        config = {
            'recognition_threshold': 0.6,
            'embedding_size': 128
        }
        system = FaceRecognitionSystem(config)
        assert system.recognition_threshold == 0.6
        assert system.embedding_size == 128
    
    def test_metrics_tracker_export(self):
        """Test that MetricsTracker is exported and can be instantiated."""
        config = {}
        tracker = MetricsTracker(config)
        assert tracker is not None
        assert hasattr(tracker, 'start_session')
        assert hasattr(tracker, 'update')
    
    def test_driving_style_classifier_export(self):
        """Test that DrivingStyleClassifier is exported and can be instantiated."""
        config = {}
        classifier = DrivingStyleClassifier(config)
        assert classifier is not None
        assert hasattr(classifier, 'classify')
    
    def test_threshold_adapter_export(self):
        """Test that ThresholdAdapter is exported and can be instantiated."""
        config = {
            'base_ttc_threshold': 2.0,
            'base_following_distance': 25.0,
            'base_alert_sensitivity': 0.7
        }
        adapter = ThresholdAdapter(config)
        assert adapter.base_ttc_threshold == 2.0
        assert adapter.base_following_distance == 25.0
    
    def test_driver_report_generator_export(self):
        """Test that DriverReportGenerator is exported and can be instantiated."""
        config = {}
        generator = DriverReportGenerator(config)
        assert generator is not None
        assert hasattr(generator, 'generate_report')
    
    @patch('src.profiling.profile_manager.FaceRecognitionSystem')
    @patch('src.profiling.profile_manager.MetricsTracker')
    @patch('src.profiling.profile_manager.DrivingStyleClassifier')
    @patch('src.profiling.profile_manager.ThresholdAdapter')
    @patch('src.profiling.profile_manager.DriverReportGenerator')
    def test_profile_manager_export(self, mock_report, mock_adapter, mock_classifier, 
                                   mock_tracker, mock_face_rec):
        """Test that ProfileManager is exported and can be instantiated."""
        config = {
            'profiles_dir': 'test_profiles',
            'auto_save': False
        }
        manager = ProfileManager(config)
        assert manager is not None
        assert hasattr(manager, 'identify_driver')
        assert hasattr(manager, 'start_session')


class TestProfilingModuleIntegration:
    """Test integration between profiling module components."""
    
    @patch('src.profiling.profile_manager.FaceRecognitionSystem')
    @patch('src.profiling.profile_manager.MetricsTracker')
    @patch('src.profiling.profile_manager.DrivingStyleClassifier')
    @patch('src.profiling.profile_manager.ThresholdAdapter')
    @patch('src.profiling.profile_manager.DriverReportGenerator')
    def test_profile_manager_uses_all_components(self, mock_report, mock_adapter, 
                                                mock_classifier, mock_tracker, mock_face_rec):
        """Test that ProfileManager integrates all profiling components."""
        config = {
            'profiles_dir': 'test_profiles',
            'auto_save': False,
            'face_recognition': {},
            'metrics_tracker': {},
            'style_classifier': {},
            'threshold_adapter': {},
            'report_generator': {}
        }
        
        manager = ProfileManager(config)
        
        # Verify all components were initialized
        mock_face_rec.assert_called_once()
        mock_tracker.assert_called_once()
        mock_classifier.assert_called_once()
        mock_adapter.assert_called_once()
        mock_report.assert_called_once()
    
    def test_metrics_tracker_and_classifier_integration(self):
        """Test that MetricsTracker output can be used by DrivingStyleClassifier."""
        # Create tracker and collect some metrics
        tracker = MetricsTracker({})
        tracker.start_session(1000.0)
        
        # Simulate some driving behavior
        for i in range(20):
            tracker.update(
                timestamp=1000.0 + i,
                speed=25.0,
                following_distance=30.0,
                lane_id=1,
                risk_score=0.3
            )
        
        # Record some reaction times
        tracker.record_alert(1, 1005.0)
        tracker.record_driver_action(1, 1006.2, 'brake')
        
        tracker.end_session(1020.0)
        
        # Get metrics summary
        metrics = tracker.get_summary()
        
        # Classify driving style
        classifier = DrivingStyleClassifier({})
        style = classifier.classify(metrics)
        
        # Should return a valid style (may be UNKNOWN due to limited data)
        assert isinstance(style, DrivingStyle)
    
    def test_classifier_and_adapter_integration(self):
        """Test that DrivingStyleClassifier output can be used by ThresholdAdapter."""
        # Create mock metrics with sufficient data
        metrics = {
            'reaction_time': {'mean': 1.2, 'count': 5},
            'following_distance': {'mean': 28.0, 'count': 20},
            'lane_change_frequency': 4.0,
            'speed_profile': {'mean': 22.0, 'std': 3.0},
            'risk_tolerance': 0.4
        }
        
        # Classify style
        classifier = DrivingStyleClassifier({})
        style = classifier.classify(metrics)
        
        # Adapt thresholds based on style
        adapter = ThresholdAdapter({
            'base_ttc_threshold': 2.0,
            'base_following_distance': 25.0,
            'base_alert_sensitivity': 0.7
        })
        
        adapted = adapter.adapt_thresholds(metrics, style)
        
        # Verify adapted thresholds are returned
        assert 'ttc_threshold' in adapted
        assert 'following_distance' in adapted
        assert 'alert_sensitivity' in adapted
        assert all(isinstance(v, float) for v in adapted.values())
    
    def test_metrics_and_report_generator_integration(self):
        """Test that MetricsTracker output can be used by DriverReportGenerator."""
        # Create tracker with some data
        tracker = MetricsTracker({})
        tracker.start_session(1000.0)
        
        for i in range(30):
            tracker.update(
                timestamp=1000.0 + i,
                speed=25.0,
                following_distance=28.0,
                lane_id=1,
                risk_score=0.35
            )
        
        tracker.end_session(1030.0)
        metrics = tracker.get_summary()
        
        # Classify style
        classifier = DrivingStyleClassifier({})
        style = classifier.classify(metrics)
        
        # Generate report
        generator = DriverReportGenerator({})
        report = generator.generate_report(metrics, style, "test_driver")
        
        # Verify report structure
        assert 'driver_id' in report
        assert 'scores' in report
        assert 'safety' in report['scores']
        assert 'attention' in report['scores']
        assert 'eco_driving' in report['scores']
        assert 'recommendations' in report


class TestProfilingModuleDataStructures:
    """Test data structures used in profiling module."""
    
    def test_driver_profile_dataclass(self):
        """Test DriverProfile dataclass structure and defaults."""
        profile = DriverProfile(
            driver_id="test_123",
            face_embedding=[0.1] * 128
        )
        
        assert profile.driver_id == "test_123"
        assert len(profile.face_embedding) == 128
        assert profile.total_distance == 0.0
        assert profile.total_time == 0.0
        assert profile.driving_style == "unknown"
        assert profile.session_count == 0
    
    def test_metrics_snapshot_dataclass(self):
        """Test MetricsSnapshot dataclass structure and defaults."""
        snapshot = MetricsSnapshot(timestamp=1234567890.0)
        
        assert snapshot.timestamp == 1234567890.0
        assert snapshot.reaction_time is None
        assert snapshot.following_distance is None
        assert snapshot.speed is None
        assert snapshot.lane_change is False
        assert snapshot.near_miss is False
        assert snapshot.risk_score == 0.0
    
    def test_metrics_snapshot_with_all_fields(self):
        """Test MetricsSnapshot with all fields populated."""
        snapshot = MetricsSnapshot(
            timestamp=1234567890.0,
            reaction_time=0.85,
            following_distance=27.5,
            speed=22.3,
            lane_change=True,
            near_miss=False,
            risk_score=0.42
        )
        
        assert snapshot.reaction_time == 0.85
        assert snapshot.following_distance == 27.5
        assert snapshot.speed == 22.3
        assert snapshot.lane_change is True
        assert snapshot.risk_score == 0.42


class TestProfilingModuleEnums:
    """Test enums used in profiling module."""
    
    def test_driving_style_enum_values(self):
        """Test DrivingStyle enum has correct values."""
        assert DrivingStyle.AGGRESSIVE.value == "aggressive"
        assert DrivingStyle.NORMAL.value == "normal"
        assert DrivingStyle.CAUTIOUS.value == "cautious"
        assert DrivingStyle.UNKNOWN.value == "unknown"
    
    def test_driving_style_enum_comparison(self):
        """Test DrivingStyle enum comparison."""
        style1 = DrivingStyle.AGGRESSIVE
        style2 = DrivingStyle.AGGRESSIVE
        style3 = DrivingStyle.NORMAL
        
        assert style1 == style2
        assert style1 != style3
    
    def test_driving_style_enum_from_string(self):
        """Test creating DrivingStyle from string value."""
        style = DrivingStyle("aggressive")
        assert style == DrivingStyle.AGGRESSIVE
        
        style = DrivingStyle("normal")
        assert style == DrivingStyle.NORMAL


@pytest.mark.performance
class TestProfilingModulePerformance:
    """Test performance characteristics of profiling module."""
    
    def test_metrics_tracker_update_performance(self):
        """Test that MetricsTracker.update completes within performance requirements."""
        import time
        
        tracker = MetricsTracker({})
        tracker.start_session(1000.0)
        
        # Measure update performance
        iterations = 100
        start_time = time.perf_counter()
        
        for i in range(iterations):
            tracker.update(
                timestamp=1000.0 + i * 0.033,
                speed=25.0,
                following_distance=28.0,
                lane_id=1,
                risk_score=0.35
            )
        
        end_time = time.perf_counter()
        
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should be very fast (< 1ms per update)
        assert avg_time_ms < 1.0, f"Update took {avg_time_ms:.2f}ms, expected < 1ms"
    
    def test_style_classification_performance(self):
        """Test that style classification completes within performance requirements."""
        import time
        
        # Create metrics with sufficient data
        metrics = {
            'reaction_time': {'mean': 1.2, 'count': 10},
            'following_distance': {'mean': 28.0, 'count': 50},
            'lane_change_frequency': 4.0,
            'speed_profile': {'mean': 22.0, 'std': 3.0},
            'risk_tolerance': 0.4
        }
        
        classifier = DrivingStyleClassifier({})
        
        # Measure classification performance
        start_time = time.perf_counter()
        style = classifier.classify(metrics)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should complete quickly (< 10ms)
        assert execution_time_ms < 10.0, f"Classification took {execution_time_ms:.2f}ms, expected < 10ms"
    
    def test_threshold_adaptation_performance(self):
        """Test that threshold adaptation completes within performance requirements."""
        import time
        
        metrics = {
            'reaction_time': {'mean': 1.2},
            'following_distance': {'mean': 28.0},
            'risk_tolerance': 0.4
        }
        
        adapter = ThresholdAdapter({
            'base_ttc_threshold': 2.0,
            'base_following_distance': 25.0,
            'base_alert_sensitivity': 0.7
        })
        
        # Measure adaptation performance
        start_time = time.perf_counter()
        adapted = adapter.adapt_thresholds(metrics, DrivingStyle.NORMAL)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should be very fast (< 5ms)
        assert execution_time_ms < 5.0, f"Adaptation took {execution_time_ms:.2f}ms, expected < 5ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
