"""Verification script for Driver Monitoring System (DMS)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_imports():
    """Verify all DMS modules can be imported."""
    logger.info("Verifying DMS imports...")
    
    try:
        from src.dms.face import FaceDetector
        logger.info("✓ FaceDetector imported")
    except Exception as e:
        logger.error(f"✗ FaceDetector import failed: {e}")
        return False
    
    try:
        from src.dms.gaze import GazeEstimator
        logger.info("✓ GazeEstimator imported")
    except Exception as e:
        logger.error(f"✗ GazeEstimator import failed: {e}")
        return False
    
    try:
        from src.dms.pose import HeadPoseEstimator
        logger.info("✓ HeadPoseEstimator imported")
    except Exception as e:
        logger.error(f"✗ HeadPoseEstimator import failed: {e}")
        return False
    
    try:
        from src.dms.drowsiness import DrowsinessDetector
        logger.info("✓ DrowsinessDetector imported")
    except Exception as e:
        logger.error(f"✗ DrowsinessDetector import failed: {e}")
        return False
    
    try:
        from src.dms.distraction import DistractionClassifier
        logger.info("✓ DistractionClassifier imported")
    except Exception as e:
        logger.error(f"✗ DistractionClassifier import failed: {e}")
        return False
    
    try:
        from src.dms.readiness import ReadinessCalculator
        logger.info("✓ ReadinessCalculator imported")
    except Exception as e:
        logger.error(f"✗ ReadinessCalculator import failed: {e}")
        return False
    
    try:
        from src.dms.monitor import DriverMonitor
        logger.info("✓ DriverMonitor imported")
    except Exception as e:
        logger.error(f"✗ DriverMonitor import failed: {e}")
        return False
    
    try:
        from src.dms import (
            DriverMonitor, FaceDetector, GazeEstimator,
            HeadPoseEstimator, DrowsinessDetector,
            DistractionClassifier, ReadinessCalculator
        )
        logger.info("✓ All DMS components imported from package")
    except Exception as e:
        logger.error(f"✗ Package import failed: {e}")
        return False
    
    return True


def verify_component_initialization():
    """Verify all components can be initialized."""
    logger.info("\nVerifying component initialization...")
    
    try:
        from src.dms.gaze import GazeEstimator
        estimator = GazeEstimator()
        logger.info("✓ GazeEstimator initialized")
    except Exception as e:
        logger.error(f"✗ GazeEstimator initialization failed: {e}")
        return False
    
    try:
        from src.dms.pose import HeadPoseEstimator
        estimator = HeadPoseEstimator()
        logger.info("✓ HeadPoseEstimator initialized")
    except Exception as e:
        logger.error(f"✗ HeadPoseEstimator initialization failed: {e}")
        return False
    
    try:
        from src.dms.drowsiness import DrowsinessDetector
        detector = DrowsinessDetector(fps=30)
        logger.info("✓ DrowsinessDetector initialized")
    except Exception as e:
        logger.error(f"✗ DrowsinessDetector initialization failed: {e}")
        return False
    
    try:
        from src.dms.distraction import DistractionClassifier
        classifier = DistractionClassifier()
        logger.info("✓ DistractionClassifier initialized")
    except Exception as e:
        logger.error(f"✗ DistractionClassifier initialization failed: {e}")
        return False
    
    try:
        from src.dms.readiness import ReadinessCalculator
        calculator = ReadinessCalculator()
        logger.info("✓ ReadinessCalculator initialized")
    except Exception as e:
        logger.error(f"✗ ReadinessCalculator initialization failed: {e}")
        return False
    
    return True


def verify_component_functionality():
    """Verify components work with test data."""
    logger.info("\nVerifying component functionality...")
    
    # Create test data
    test_landmarks = np.random.rand(68, 2) * np.array([640, 480])
    test_landmarks = test_landmarks.astype(np.float32)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test HeadPoseEstimator
    try:
        from src.dms.pose import HeadPoseEstimator
        estimator = HeadPoseEstimator()
        head_pose = estimator.estimate_head_pose(test_landmarks, (480, 640))
        assert 'roll' in head_pose
        assert 'pitch' in head_pose
        assert 'yaw' in head_pose
        logger.info(f"✓ HeadPoseEstimator: roll={head_pose['roll']:.1f}, pitch={head_pose['pitch']:.1f}, yaw={head_pose['yaw']:.1f}")
    except Exception as e:
        logger.error(f"✗ HeadPoseEstimator functionality failed: {e}")
        return False
    
    # Test GazeEstimator
    try:
        from src.dms.gaze import GazeEstimator
        estimator = GazeEstimator()
        gaze = estimator.estimate_gaze(test_frame, test_landmarks)
        assert 'pitch' in gaze
        assert 'yaw' in gaze
        assert 'attention_zone' in gaze
        logger.info(f"✓ GazeEstimator: pitch={gaze['pitch']:.1f}, yaw={gaze['yaw']:.1f}, zone={gaze['attention_zone']}")
    except Exception as e:
        logger.error(f"✗ GazeEstimator functionality failed: {e}")
        return False
    
    # Test DrowsinessDetector
    try:
        from src.dms.drowsiness import DrowsinessDetector
        detector = DrowsinessDetector(fps=30)
        drowsiness = detector.detect_drowsiness(test_landmarks, head_pose)
        assert 'score' in drowsiness
        assert 'perclos' in drowsiness
        assert 0.0 <= drowsiness['score'] <= 1.0
        logger.info(f"✓ DrowsinessDetector: score={drowsiness['score']:.2f}, perclos={drowsiness['perclos']:.2f}")
    except Exception as e:
        logger.error(f"✗ DrowsinessDetector functionality failed: {e}")
        return False
    
    # Test DistractionClassifier
    try:
        from src.dms.distraction import DistractionClassifier
        classifier = DistractionClassifier()
        distraction = classifier.classify_distraction(test_frame, gaze, head_pose)
        assert 'type' in distraction
        assert 'confidence' in distraction
        logger.info(f"✓ DistractionClassifier: type={distraction['type']}, confidence={distraction['confidence']:.2f}")
    except Exception as e:
        logger.error(f"✗ DistractionClassifier functionality failed: {e}")
        return False
    
    # Test ReadinessCalculator
    try:
        from src.dms.readiness import ReadinessCalculator
        calculator = ReadinessCalculator()
        readiness = calculator.calculate_readiness(drowsiness, gaze, distraction)
        assert 0.0 <= readiness <= 100.0
        logger.info(f"✓ ReadinessCalculator: readiness={readiness:.1f}/100")
    except Exception as e:
        logger.error(f"✗ ReadinessCalculator functionality failed: {e}")
        return False
    
    return True


def verify_dms_integration():
    """Verify DriverMonitor integration."""
    logger.info("\nVerifying DriverMonitor integration...")
    
    try:
        from src.core.config import ConfigManager
        from src.dms import DriverMonitor
        
        # Load config
        config = ConfigManager('configs/default.yaml').config
        
        # Initialize DMS
        dms = DriverMonitor(config)
        logger.info("✓ DriverMonitor initialized with config")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Analyze frame
        driver_state = dms.analyze(test_frame)
        
        # Verify output
        assert hasattr(driver_state, 'face_detected')
        assert hasattr(driver_state, 'landmarks')
        assert hasattr(driver_state, 'head_pose')
        assert hasattr(driver_state, 'gaze')
        assert hasattr(driver_state, 'eye_state')
        assert hasattr(driver_state, 'drowsiness')
        assert hasattr(driver_state, 'distraction')
        assert hasattr(driver_state, 'readiness_score')
        
        logger.info(f"✓ DriverMonitor.analyze() returned valid DriverState")
        logger.info(f"  - Face detected: {driver_state.face_detected}")
        logger.info(f"  - Readiness score: {driver_state.readiness_score:.1f}/100")
        logger.info(f"  - Attention zone: {driver_state.gaze['attention_zone']}")
        logger.info(f"  - Drowsiness: {driver_state.drowsiness['score']:.2f}")
        logger.info(f"  - Distraction: {driver_state.distraction['type']}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ DriverMonitor integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    logger.info("=" * 60)
    logger.info("DMS Verification Script")
    logger.info("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", verify_imports()))
    
    # Test initialization
    results.append(("Initialization", verify_component_initialization()))
    
    # Test functionality
    results.append(("Functionality", verify_component_functionality()))
    
    # Test integration
    results.append(("Integration", verify_dms_integration()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Verification Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("=" * 60)
    
    if all_passed:
        logger.info("✓ All DMS verification tests PASSED")
        return 0
    else:
        logger.error("✗ Some DMS verification tests FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
