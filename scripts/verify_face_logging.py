"""Verify FaceDetector logging functionality."""

import sys
from pathlib import Path
import numpy as np
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.logging import LoggerSetup
from dms.face import FaceDetector


def main():
    """Test FaceDetector logging."""
    print("=" * 60)
    print("FaceDetector Logging Verification")
    print("=" * 60)
    
    # Setup logging
    LoggerSetup.setup(log_level='DEBUG', log_dir='logs')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize FaceDetector
        print("\n1. Initializing FaceDetector...")
        detector = FaceDetector()
        print("   ✓ FaceDetector initialized")
        
        # Test with valid frame
        print("\n2. Testing face detection with valid frame...")
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        face_detected, landmarks = detector.detect_and_extract_landmarks(test_frame)
        print(f"   Face detected: {face_detected}")
        if landmarks is not None:
            print(f"   Landmarks shape: {landmarks.shape}")
        
        # Test with invalid frame
        print("\n3. Testing face detection with invalid frame...")
        face_detected, landmarks = detector.detect_and_extract_landmarks(None)
        print(f"   Face detected: {face_detected}")
        
        # Run multiple detections to trigger statistics logging
        print("\n4. Running 100 detections to trigger statistics...")
        for i in range(100):
            detector.detect_and_extract_landmarks(test_frame)
        print("   ✓ Statistics logged")
        
        # Get performance stats
        print("\n5. Performance Statistics:")
        stats = detector.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n" + "=" * 60)
        print("Verification Complete!")
        print("=" * 60)
        print("\nCheck logs/dms.log for detailed logging output")
        
    except ImportError as e:
        print(f"\n✗ Import Error: {e}")
        print("  MediaPipe may not be installed. Install with: pip install mediapipe")
        return 1
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
