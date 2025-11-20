"""Face detection and landmark extraction using MediaPipe."""

import logging
import time
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger("FaceDetector")


class FaceDetector:
    """Face detection and landmark extraction using MediaPipe."""
    
    def __init__(self):
        """Initialize MediaPipe face detection and face mesh."""
        logger.debug("FaceDetector initialization started")
        
        if mp is None:
            logger.error("MediaPipe initialization failed: package not installed")
            raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")
        
        try:
            # Initialize MediaPipe Face Detection
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for short-range (< 2m), 1 for full-range
                min_detection_confidence=0.5
            )
            logger.debug("MediaPipe Face Detection initialized: model_selection=0, min_confidence=0.5")
            
            # Initialize MediaPipe Face Mesh for landmarks
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.debug("MediaPipe Face Mesh initialized: max_faces=1, refine_landmarks=True")
            
            # Performance tracking
            self.processing_times = []
            self.detection_count = 0
            self.no_face_count = 0
            
            logger.info("FaceDetector initialized successfully with MediaPipe")
            
        except Exception as e:
            logger.error(f"FaceDetector initialization failed: {e}", exc_info=True)
            raise
    
    def detect_and_extract_landmarks(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect face and extract facial landmarks.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            
        Returns:
            Tuple of (face_detected, landmarks)
            - face_detected: Boolean indicating if face was found
            - landmarks: Array of shape (68, 2) with (x, y) coordinates, or None if no face
        """
        start_time = time.time()
        
        if frame is None or frame.size == 0:
            logger.debug("Face detection skipped: invalid frame")
            return False, None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with face mesh
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.no_face_count += 1
                processing_time = (time.time() - start_time) * 1000
                logger.debug(f"No face detected: processing_time={processing_time:.2f}ms")
                return False, None
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            h, w = frame.shape[:2]
            landmarks = []
            
            # MediaPipe Face Mesh has 468 landmarks, we need to select key 68 landmarks
            # that correspond to standard facial landmark indices
            # Key landmark indices for 68-point model approximation
            key_indices = self._get_68_landmark_indices()
            
            for idx in key_indices:
                if idx < len(face_landmarks.landmark):
                    lm = face_landmarks.landmark[idx]
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    landmarks.append([x, y])
            
            landmarks_array = np.array(landmarks, dtype=np.float32)
            
            # Ensure we have 68 landmarks
            if len(landmarks_array) < 68:
                # Pad with zeros if needed
                padding = np.zeros((68 - len(landmarks_array), 2), dtype=np.float32)
                landmarks_array = np.vstack([landmarks_array, padding])
                logger.debug(f"Landmark array padded: original_count={len(landmarks_array)}, padded_to=68")
            elif len(landmarks_array) > 68:
                # Truncate if we have more
                landmarks_array = landmarks_array[:68]
                logger.debug(f"Landmark array truncated: original_count={len(landmarks_array)}, truncated_to=68")
            
            # Track performance
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            self.detection_count += 1
            
            # Keep only last 100 measurements
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # Log performance warning if exceeding target (part of 25ms DMS budget)
            if processing_time > 10.0:
                logger.warning(f"Face detection slow: processing_time={processing_time:.2f}ms, target=10ms")
            else:
                logger.debug(f"Face detected successfully: landmarks=68, processing_time={processing_time:.2f}ms")
            
            # Periodic statistics logging
            if self.detection_count % 100 == 0:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                detection_rate = self.detection_count / (self.detection_count + self.no_face_count) * 100
                logger.info(
                    f"Face detection statistics: total_detections={self.detection_count}, "
                    f"no_face_count={self.no_face_count}, detection_rate={detection_rate:.1f}%, "
                    f"avg_processing_time={avg_time:.2f}ms"
                )
            
            return True, landmarks_array
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                f"Face detection failed: error={e}, processing_time={processing_time:.2f}ms",
                exc_info=True
            )
            return False, None
    
    def _get_68_landmark_indices(self) -> list:
        """
        Get MediaPipe landmark indices that approximate the 68-point facial landmark model.
        
        MediaPipe Face Mesh has 468 landmarks. We map key points to approximate
        the standard 68-point model used in dlib and similar libraries.
        
        Returns:
            List of 68 landmark indices
        """
        # Mapping based on MediaPipe Face Mesh topology
        # This is an approximation to get 68 key points
        indices = [
            # Jaw line (0-16): 17 points
            234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397,
            # Right eyebrow (17-21): 5 points
            70, 63, 105, 66, 107,
            # Left eyebrow (22-26): 5 points
            336, 296, 334, 293, 300,
            # Nose bridge (27-30): 4 points
            168, 6, 197, 195,
            # Nose base (31-35): 5 points
            5, 4, 1, 19, 94,
            # Right eye (36-41): 6 points
            33, 160, 158, 133, 153, 144,
            # Left eye (42-47): 6 points
            362, 385, 387, 263, 373, 380,
            # Outer lip (48-59): 12 points
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
            # Inner lip (60-67): 8 points
            78, 95, 88, 178, 87, 14, 317, 402
        ]
        
        return indices
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {
                'avg_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'detection_count': self.detection_count,
                'no_face_count': self.no_face_count
            }
        
        times = sorted(self.processing_times)
        return {
            'avg_ms': sum(times) / len(times),
            'min_ms': times[0],
            'max_ms': times[-1],
            'detection_count': self.detection_count,
            'no_face_count': self.no_face_count,
            'detection_rate': self.detection_count / (self.detection_count + self.no_face_count) * 100 if (self.detection_count + self.no_face_count) > 0 else 0.0
        }
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
                logger.debug("MediaPipe Face Detection closed")
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
                logger.debug("MediaPipe Face Mesh closed")
            
            # Log final statistics
            if hasattr(self, 'detection_count') and self.detection_count > 0:
                stats = self.get_performance_stats()
                logger.info(
                    f"FaceDetector cleanup: total_detections={stats['detection_count']}, "
                    f"avg_time={stats['avg_ms']:.2f}ms, detection_rate={stats['detection_rate']:.1f}%"
                )
        except Exception as e:
            logger.error(f"Error during FaceDetector cleanup: {e}")
