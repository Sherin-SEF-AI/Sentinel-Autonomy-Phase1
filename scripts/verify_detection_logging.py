"""Verify logging setup for detection module."""

import sys
import logging
import logging.config
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def verify_logging_config():
    """Verify logging configuration for detection module."""
    print("=" * 60)
    print("Detection Module Logging Verification")
    print("=" * 60)
    
    # Load logging configuration
    with open('configs/logging.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)
    
    # Check detection module loggers
    detection_modules = [
        'src.perception.detection.detector_2d',
        'src.perception.detection.estimator_3d',
        'src.perception.detection.fusion',
        'src.perception.detection.tracker',
        'src.perception.detection.detector'
    ]
    
    print("\n1. Logger Configuration Check:")
    print("-" * 60)
    for module_name in detection_modules:
        logger = logging.getLogger(module_name)
        print(f"  {module_name}:")
        print(f"    - Level: {logging.getLevelName(logger.level)}")
        print(f"    - Handlers: {len(logger.handlers)}")
        print(f"    - Propagate: {logger.propagate}")
    
    # Test logging at different levels
    print("\n2. Test Logging Output:")
    print("-" * 60)
    
    test_logger = logging.getLogger('src.perception.detection.detector')
    
    test_logger.debug("DEBUG: Detection pipeline started")
    test_logger.info("INFO: Detected 5 objects in frame")
    test_logger.warning("WARNING: Detection latency exceeded target")
    test_logger.error("ERROR: YOLOv8 inference failed")
    
    print("  ✓ Log messages sent (check logs/perception.log)")
    
    # Verify log files exist
    print("\n3. Log File Verification:")
    print("-" * 60)
    
    log_files = [
        'logs/perception.log',
        'logs/sentinel.log'
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            size = Path(log_file).stat().st_size
            print(f"  ✓ {log_file} exists ({size} bytes)")
        else:
            print(f"  ✗ {log_file} not found")
    
    print("\n" + "=" * 60)
    print("Verification Complete")
    print("=" * 60)


if __name__ == '__main__':
    verify_logging_config()
