"""Demonstration of detection module logging."""

import sys
import logging
import logging.config
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def setup_logging():
    """Setup logging from configuration."""
    with open('configs/logging.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)


def demo_detection_logging():
    """Demonstrate detection module logging at various levels."""
    
    print("=" * 70)
    print("SENTINEL Detection Module - Logging Demonstration")
    print("=" * 70)
    
    # Setup logging
    setup_logging()
    
    # Get loggers for each detection component
    detector_2d_logger = logging.getLogger('src.perception.detection.detector_2d')
    estimator_3d_logger = logging.getLogger('src.perception.detection.estimator_3d')
    fusion_logger = logging.getLogger('src.perception.detection.fusion')
    tracker_logger = logging.getLogger('src.perception.detection.tracker')
    detector_logger = logging.getLogger('src.perception.detection.detector')
    
    print("\n1. Initialization Logs (INFO level)")
    print("-" * 70)
    detector_2d_logger.info("Detector2D initialized with yolov8m")
    estimator_3d_logger.info("Estimator3D initialized")
    fusion_logger.info("MultiViewFusion initialized with IoU threshold 0.3")
    tracker_logger.info("ObjectTracker initialized with max_age=30, min_hits=3")
    detector_logger.info("ObjectDetector initialized")
    
    print("\n2. Normal Operation Logs (DEBUG level)")
    print("-" * 70)
    detector_logger.debug("Detection pipeline: 18.5ms, 3 objects tracked")
    tracker_logger.debug("Track 1 updated: position=(5.2, 2.1, 0.0), velocity=(2.5, 0.0, 0.0)")
    fusion_logger.debug("Fused 5 detections into 3 unique objects")
    
    print("\n3. Performance Monitoring (INFO level)")
    print("-" * 70)
    detector_logger.info("Frame 100: 3 vehicles, 2 pedestrians detected")
    tracker_logger.info("Active tracks: 5, Confirmed: 3, Tentative: 2")
    
    print("\n4. Warning Conditions (WARNING level)")
    print("-" * 70)
    detector_logger.warning("Detection latency 25.3ms exceeds target of 20ms")
    estimator_3d_logger.warning("No calibration data for camera 3")
    tracker_logger.warning("Track 7 lost: time_since_update=31 > max_age=30")
    
    print("\n5. Error Conditions (ERROR level)")
    print("-" * 70)
    detector_2d_logger.error("Detection failed for camera 1: CUDA out of memory")
    detector_logger.error("Detection pipeline failed: YOLOv8 inference error")
    
    print("\n6. Error Recovery (WARNING + INFO)")
    print("-" * 70)
    detector_logger.warning("Max errors reached (3), attempting to reload detector")
    detector_logger.info("Error recovery successful")
    
    print("\n" + "=" * 70)
    print("Log files written to:")
    print("  - logs/perception.log (perception module logs)")
    print("  - logs/sentinel.log (system-wide logs)")
    print("  - logs/errors.log (ERROR level only)")
    print("=" * 70)
    
    # Show sample log file content
    print("\nSample from logs/perception.log:")
    print("-" * 70)
    try:
        with open('logs/perception.log', 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:  # Last 10 lines
                print(line.rstrip())
    except FileNotFoundError:
        print("Log file not yet created")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)


if __name__ == '__main__':
    demo_detection_logging()
