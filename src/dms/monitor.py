"""Driver Monitoring System (DMS) main class."""

import logging
from typing import Optional
import numpy as np
import time

from ..core.interfaces import IDMS
from ..core.data_structures import DriverState
from .face import FaceDetector
from .gaze import GazeEstimator
from .pose import HeadPoseEstimator
from .drowsiness import DrowsinessDetector
from .distraction import DistractionClassifier
from .readiness import ReadinessCalculator

logger = logging.getLogger(__name__)


class DriverMonitor(IDMS):
    """
    Driver Monitoring System implementation.
    
    Integrates face detection, gaze estimation, head pose, drowsiness detection,
    distraction classification, and readiness scoring.
    """
    
    def __init__(self, config: dict):
        """
        Initialize DMS with all components.
        
        Args:
            config: Configuration dictionary with DMS settings
        """
        self.config = config
        dms_config = config.get('models', {}).get('dms', {})
        
        # Initialize components
        try:
            self.face_detector = FaceDetector()
        except Exception as e:
            logger.error(f"Failed to initialize FaceDetector: {e}")
            self.face_detector = None
        
        gaze_model_path = dms_config.get('gaze_weights')
        self.gaze_estimator = GazeEstimator(
            model_path=gaze_model_path,
            device=dms_config.get('device', 'cuda')
        )
        
        self.pose_estimator = HeadPoseEstimator()
        
        fps = config.get('cameras', {}).get('interior', {}).get('fps', 30)
        self.drowsiness_detector = DrowsinessDetector(fps=fps)
        
        distraction_model_path = dms_config.get('distraction_weights')
        self.distraction_classifier = DistractionClassifier(
            model_path=distraction_model_path,
            device=dms_config.get('device', 'cuda')
        )
        
        self.readiness_calculator = ReadinessCalculator()
        
        # Performance tracking
        self.last_valid_state = None
        self.error_count = 0
        self.max_errors = 5
        
        logger.info("DriverMonitor initialized successfully")
    
    def analyze(self, frame: np.ndarray) -> DriverState:
        """
        Analyze driver state from interior camera frame.
        
        Args:
            frame: Interior camera frame (H, W, 3) in BGR format
            
        Returns:
            DriverState dataclass with complete driver analysis
        """
        start_time = time.time()
        
        try:
            # Step 1: Face detection and landmark extraction
            face_detected, landmarks = self._detect_face(frame)
            
            if not face_detected or landmarks is None:
                # Return default state if no face detected
                return self._create_default_state()
            
            # Step 2: Head pose estimation
            head_pose = self.pose_estimator.estimate_head_pose(
                landmarks, frame.shape
            )
            
            # Step 3: Gaze estimation
            gaze = self.gaze_estimator.estimate_gaze(frame, landmarks)
            
            # Step 4: Drowsiness detection
            drowsiness = self.drowsiness_detector.detect_drowsiness(
                landmarks, head_pose
            )
            
            # Step 5: Distraction classification
            distraction = self.distraction_classifier.classify_distraction(
                frame, gaze, head_pose
            )
            
            # Step 6: Calculate readiness score
            readiness_score = self.readiness_calculator.calculate_readiness(
                drowsiness, gaze, distraction
            )
            
            # Create driver state
            driver_state = DriverState(
                face_detected=True,
                landmarks=landmarks,
                head_pose=head_pose,
                gaze=gaze,
                eye_state={
                    'left_ear': drowsiness.get('left_ear', 0.3),
                    'right_ear': drowsiness.get('right_ear', 0.3),
                    'perclos': drowsiness.get('perclos', 0.0)
                },
                drowsiness=drowsiness,
                distraction=distraction,
                readiness_score=readiness_score
            )
            
            # Cache valid state for error recovery
            self.last_valid_state = driver_state
            self.error_count = 0
            
            # Log performance
            processing_time = (time.time() - start_time) * 1000
            if processing_time > 25:
                logger.warning(f"DMS processing time: {processing_time:.1f}ms (target: 25ms)")
            
            return driver_state
        
        except Exception as e:
            logger.error(f"DMS analysis failed: {e}")
            return self._handle_error()
    
    def _detect_face(self, frame: np.ndarray) -> tuple:
        """
        Detect face and extract landmarks with error handling.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (face_detected, landmarks)
        """
        if self.face_detector is None:
            logger.warning("FaceDetector not available")
            return False, None
        
        try:
            return self.face_detector.detect_and_extract_landmarks(frame)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return False, None
    
    def _handle_error(self) -> DriverState:
        """
        Handle errors with automatic recovery.
        
        Returns:
            DriverState (either cached or default)
        """
        self.error_count += 1
        
        if self.error_count > self.max_errors:
            logger.error(f"DMS error count exceeded {self.max_errors}, resetting components")
            self._reset_components()
            self.error_count = 0
        
        # Return last valid state if available
        if self.last_valid_state is not None:
            logger.info("Returning cached driver state")
            return self.last_valid_state
        
        return self._create_default_state()
    
    def _reset_components(self):
        """Reset DMS components for error recovery."""
        try:
            # Reinitialize face detector
            if self.face_detector is not None:
                del self.face_detector
                self.face_detector = FaceDetector()
            
            logger.info("DMS components reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset DMS components: {e}")
    
    def _create_default_state(self) -> DriverState:
        """
        Create default driver state when analysis fails.
        
        Returns:
            DriverState with default values
        """
        return DriverState(
            face_detected=False,
            landmarks=np.zeros((68, 2), dtype=np.float32),
            head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
            eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.0},
            drowsiness={
                'score': 0.0,
                'yawn_detected': False,
                'micro_sleep': False,
                'head_nod': False,
                'perclos': 0.0
            },
            distraction={
                'type': 'safe_driving',
                'confidence': 0.5,
                'duration': 0.0,
                'eyes_off_road': False
            },
            readiness_score=50.0
        )
