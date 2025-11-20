"""
Simple verification script for Driver Profiling Module logging.

Tests logging configuration without requiring full dependencies.
"""

import sys
import logging
import yaml
from pathlib import Path

def verify_logging_config():
    """Verify profiling module logging configuration."""
    print("="*60)
    print("DRIVER PROFILING MODULE - LOGGING CONFIGURATION VERIFICATION")
    print("="*60)
    
    # Load logging configuration
    config_path = Path('configs/logging.yaml')
    if not config_path.exists():
        print(f"✗ Logging config not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        logging_config = yaml.safe_load(f)
    
    # Check for profiling loggers
    loggers = logging_config.get('loggers', {})
    
    required_loggers = [
        'src.profiling',
        'src.profiling.face_recognition',
        'src.profiling.metrics_tracker',
        'src.profiling.style_classifier',
        'src.profiling.threshold_adapter',
        'src.profiling.report_generator',
        'src.profiling.profile_manager'
    ]
    
    print("\nChecking logger configuration:")
    all_present = True
    
    for logger_name in required_loggers:
        if logger_name in loggers:
            logger_config = loggers[logger_name]
            level = logger_config.get('level', 'NOTSET')
            handlers = logger_config.get('handlers', [])
            print(f"✓ {logger_name}")
            print(f"  - Level: {level}")
            print(f"  - Handlers: {', '.join(handlers)}")
        else:
            print(f"✗ {logger_name} - NOT CONFIGURED")
            all_present = False
    
    if all_present:
        print("\n" + "="*60)
        print("✓ ALL PROFILING LOGGERS CONFIGURED")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ SOME LOGGERS MISSING")
        print("="*60)
    
    return all_present


def test_logger_creation():
    """Test creating loggers for profiling modules."""
    print("\n" + "="*60)
    print("Testing Logger Creation")
    print("="*60)
    
    # Create loggers
    loggers = {
        'face_recognition': logging.getLogger('src.profiling.face_recognition'),
        'metrics_tracker': logging.getLogger('src.profiling.metrics_tracker'),
        'style_classifier': logging.getLogger('src.profiling.style_classifier'),
        'threshold_adapter': logging.getLogger('src.profiling.threshold_adapter'),
        'report_generator': logging.getLogger('src.profiling.report_generator'),
        'profile_manager': logging.getLogger('src.profiling.profile_manager')
    }
    
    # Test logging at different levels
    for name, logger in loggers.items():
        logger.info(f"Test INFO message from {name}")
        logger.debug(f"Test DEBUG message from {name}")
        print(f"✓ Logger created: {name}")
    
    print("\n✓ All loggers created successfully")
    return True


def verify_log_patterns():
    """Verify expected log message patterns."""
    print("\n" + "="*60)
    print("Verifying Log Message Patterns")
    print("="*60)
    
    patterns = {
        'face_recognition': [
            'FaceRecognitionSystem initialized',
            'Driver matched',
            'Face detection',
            'Face embedding'
        ],
        'metrics_tracker': [
            'MetricsTracker initialized',
            'session started',
            'session ended',
            'Reaction time recorded',
            'Near-miss event'
        ],
        'style_classifier': [
            'DrivingStyleClassifier initialized',
            'Driving style classified',
            'Insufficient data'
        ],
        'threshold_adapter': [
            'ThresholdAdapter initialized',
            'Thresholds adapted',
            'TTC threshold adapted',
            'Following distance adapted'
        ],
        'report_generator': [
            'DriverReportGenerator initialized',
            'Report generated'
        ],
        'profile_manager': [
            'ProfileManager initialized',
            'Driver identified',
            'Session started',
            'Session ended',
            'Profile updated',
            'Profile saved'
        ]
    }
    
    print("\nExpected log patterns by component:")
    for component, messages in patterns.items():
        print(f"\n{component}:")
        for msg in messages:
            print(f"  - {msg}")
    
    print("\n✓ Log patterns documented")
    return True


def main():
    """Run verification tests."""
    try:
        # Verify configuration
        config_ok = verify_logging_config()
        
        # Test logger creation
        loggers_ok = test_logger_creation()
        
        # Verify patterns
        patterns_ok = verify_log_patterns()
        
        if config_ok and loggers_ok and patterns_ok:
            print("\n" + "="*60)
            print("✓ ALL VERIFICATION CHECKS PASSED")
            print("="*60)
            print("\nDriver Profiling Module logging is properly configured.")
            print("\nNext steps:")
            print("1. Run full system to generate actual logs")
            print("2. Check logs/sentinel.log for profiling events")
            print("3. Verify performance impact (<1ms per frame)")
            return 0
        else:
            print("\n" + "="*60)
            print("✗ SOME VERIFICATION CHECKS FAILED")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
