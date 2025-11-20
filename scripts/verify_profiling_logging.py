"""
Verification script for Driver Profiling Module logging.

Tests logging functionality across all profiling components.
"""

import sys
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import LoggerSetup
from src.profiling import (
    FaceRecognitionSystem,
    MetricsTracker,
    DrivingStyleClassifier,
    ThresholdAdapter,
    DriverReportGenerator,
    ProfileManager,
    DrivingStyle
)


def test_face_recognition_logging():
    """Test face recognition logging."""
    print("\n" + "="*60)
    print("Testing Face Recognition Logging")
    print("="*60)
    
    config = {
        'recognition_threshold': 0.6,
        'embedding_size': 128
    }
    
    face_rec = FaceRecognitionSystem(config)
    
    # Test face embedding extraction
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    embedding = face_rec.extract_face_embedding(test_frame)
    
    if embedding is not None:
        print(f"✓ Face embedding extracted: shape={embedding.shape}")
    
    # Test face matching
    stored_embeddings = {
        'driver_001': np.random.randn(128).astype(np.float32)
    }
    driver_id, similarity = face_rec.match_face(embedding, stored_embeddings)
    print(f"✓ Face matching completed: driver_id={driver_id}, similarity={similarity:.3f}")
    
    # Test driver ID generation
    new_id = face_rec.generate_driver_id(embedding)
    print(f"✓ Driver ID generated: {new_id}")


def test_metrics_tracker_logging():
    """Test metrics tracker logging."""
    print("\n" + "="*60)
    print("Testing Metrics Tracker Logging")
    print("="*60)
    
    config = {}
    tracker = MetricsTracker(config)
    
    # Start session
    tracker.start_session(0.0)
    
    # Update metrics
    for i in range(10):
        tracker.update(
            timestamp=i * 0.1,
            speed=25.0 + np.random.randn() * 2,
            following_distance=28.0 + np.random.randn() * 3,
            lane_id=0,
            risk_score=0.3 + np.random.rand() * 0.2
        )
    
    # Record alert and action
    tracker.record_alert(1, 0.5)
    tracker.record_driver_action(1, 1.2, 'brake')
    
    # Record near-miss
    tracker.record_near_miss(0.8, 1.5, 0.8)
    
    # Get statistics
    reaction_stats = tracker.get_reaction_time_stats()
    print(f"✓ Reaction time stats: mean={reaction_stats['mean']:.3f}s")
    
    following_stats = tracker.get_following_distance_stats()
    print(f"✓ Following distance stats: mean={following_stats['mean']:.1f}m")
    
    lane_change_freq = tracker.get_lane_change_frequency()
    print(f"✓ Lane change frequency: {lane_change_freq:.1f} per hour")
    
    risk_tolerance = tracker.get_risk_tolerance()
    print(f"✓ Risk tolerance: {risk_tolerance:.3f}")
    
    # End session
    tracker.end_session(1.0)
    
    # Get summary
    summary = tracker.get_summary()
    print(f"✓ Session summary generated: duration={summary['session_duration']:.1f}s")


def test_style_classifier_logging():
    """Test driving style classifier logging."""
    print("\n" + "="*60)
    print("Testing Driving Style Classifier Logging")
    print("="*60)
    
    config = {}
    classifier = DrivingStyleClassifier(config)
    
    # Test with aggressive metrics
    aggressive_metrics = {
        'reaction_time': {'mean': 0.7, 'count': 5},
        'following_distance': {'mean': 12.0, 'count': 20},
        'lane_change_frequency': 10.0,
        'speed_profile': {'std': 6.0},
        'risk_tolerance': 0.7
    }
    
    style = classifier.classify(aggressive_metrics)
    print(f"✓ Aggressive style classified: {style.value}")
    
    # Test with cautious metrics
    cautious_metrics = {
        'reaction_time': {'mean': 1.8, 'count': 5},
        'following_distance': {'mean': 38.0, 'count': 20},
        'lane_change_frequency': 1.5,
        'speed_profile': {'std': 1.5},
        'risk_tolerance': 0.2
    }
    
    style = classifier.classify(cautious_metrics)
    print(f"✓ Cautious style classified: {style.value}")
    
    # Test description
    description = classifier.get_style_description(style)
    print(f"✓ Style description: {description[:50]}...")


def test_threshold_adapter_logging():
    """Test threshold adapter logging."""
    print("\n" + "="*60)
    print("Testing Threshold Adapter Logging")
    print("="*60)
    
    config = {
        'base_ttc_threshold': 2.0,
        'base_following_distance': 25.0,
        'base_alert_sensitivity': 0.7
    }
    
    adapter = ThresholdAdapter(config)
    
    # Test threshold adaptation
    metrics = {
        'reaction_time': {'mean': 1.2},
        'risk_tolerance': 0.6
    }
    
    adapted = adapter.adapt_thresholds(metrics, DrivingStyle.AGGRESSIVE)
    print(f"✓ Thresholds adapted for aggressive driver:")
    print(f"  - TTC threshold: {adapted['ttc_threshold']:.2f}s")
    print(f"  - Following distance: {adapted['following_distance']:.1f}m")
    print(f"  - Alert sensitivity: {adapted['alert_sensitivity']:.2f}")
    
    # Test safety margin info
    margin_info = adapter.get_safety_margin_info()
    print(f"✓ Safety margin: {margin_info['safety_margin_multiplier']}x")
    
    # Test reset
    adapter.reset_to_defaults()
    print("✓ Thresholds reset to defaults")


def test_report_generator_logging():
    """Test report generator logging."""
    print("\n" + "="*60)
    print("Testing Report Generator Logging")
    print("="*60)
    
    config = {}
    generator = DriverReportGenerator(config)
    
    # Create test metrics
    metrics = {
        'session_duration': 1800.0,
        'total_distance': 25000.0,
        'reaction_time': {'mean': 1.1, 'std': 0.3, 'count': 8},
        'following_distance': {'mean': 26.0, 'std': 4.0, 'count': 50},
        'lane_change_frequency': 4.5,
        'speed_profile': {'mean': 22.0, 'max': 30.0, 'std': 3.5},
        'risk_tolerance': 0.45,
        'near_miss_count': 1
    }
    
    # Generate report
    report = generator.generate_report(metrics, DrivingStyle.NORMAL, 'driver_test_001')
    
    print(f"✓ Report generated for {report['driver_id']}")
    print(f"  - Safety score: {report['scores']['safety']:.1f}")
    print(f"  - Attention score: {report['scores']['attention']:.1f}")
    print(f"  - Eco score: {report['scores']['eco_driving']:.1f}")
    print(f"  - Overall score: {report['scores']['overall']:.1f}")
    print(f"  - Recommendations: {len(report['recommendations'])}")
    
    # Export as text
    text_report = generator.export_report_text(report)
    print(f"✓ Text report exported: {len(text_report)} characters")


def test_profile_manager_logging():
    """Test profile manager logging."""
    print("\n" + "="*60)
    print("Testing Profile Manager Logging")
    print("="*60)
    
    config = {
        'profiles_dir': 'test_profiles',
        'auto_save': False,
        'face_recognition': {'recognition_threshold': 0.6},
        'metrics_tracker': {},
        'style_classifier': {},
        'threshold_adapter': {
            'base_ttc_threshold': 2.0,
            'base_following_distance': 25.0,
            'base_alert_sensitivity': 0.7
        },
        'report_generator': {}
    }
    
    manager = ProfileManager(config)
    
    # Test driver identification (will create new profile)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    driver_id = manager.identify_driver(test_frame)
    print(f"✓ Driver identified/created: {driver_id}")
    
    # Start session
    manager.start_session(driver_id, 0.0)
    print(f"✓ Session started for {driver_id}")
    
    # Simulate some metrics
    tracker = manager.get_metrics_tracker()
    for i in range(20):
        tracker.update(
            timestamp=i * 0.1,
            speed=25.0,
            following_distance=28.0,
            lane_id=0,
            risk_score=0.3
        )
    
    # End session
    manager.end_session(2.0)
    print(f"✓ Session ended for {driver_id}")
    
    # Get profile
    profile = manager.get_profile(driver_id)
    if profile:
        print(f"✓ Profile retrieved:")
        print(f"  - Sessions: {profile.session_count}")
        print(f"  - Driving style: {profile.driving_style}")
        print(f"  - Safety score: {profile.safety_score:.1f}")
    
    # Get adapted thresholds
    thresholds = manager.get_adapted_thresholds(driver_id)
    print(f"✓ Adapted thresholds:")
    print(f"  - TTC: {thresholds['ttc_threshold']:.2f}s")
    print(f"  - Following distance: {thresholds['following_distance']:.1f}m")
    
    # Clean up test profiles
    import shutil
    if Path('test_profiles').exists():
        shutil.rmtree('test_profiles')
    print("✓ Test profiles cleaned up")


def main():
    """Run all profiling logging tests."""
    print("="*60)
    print("DRIVER PROFILING MODULE - LOGGING VERIFICATION")
    print("="*60)
    
    # Setup logging
    LoggerSetup.setup(log_level='DEBUG', log_dir='logs')
    
    try:
        # Run tests
        test_face_recognition_logging()
        test_metrics_tracker_logging()
        test_style_classifier_logging()
        test_threshold_adapter_logging()
        test_report_generator_logging()
        test_profile_manager_logging()
        
        print("\n" + "="*60)
        print("ALL PROFILING LOGGING TESTS PASSED")
        print("="*60)
        print("\nCheck logs/sentinel.log for detailed logging output")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
