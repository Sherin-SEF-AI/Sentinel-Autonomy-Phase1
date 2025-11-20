"""Drowsiness detection using eye aspect ratio and behavioral indicators."""

import logging
from typing import Dict, Any
from collections import deque
import numpy as np
import time

logger = logging.getLogger(__name__)


class DrowsinessDetector:
    """Drowsiness detection using PERCLOS, yawning, and head nodding."""
    
    def __init__(self, fps: int = 30):
        """
        Initialize drowsiness detector.
        
        Args:
            fps: Frame rate for temporal calculations
        """
        self.fps = fps
        
        # PERCLOS calculation (60 frames = 2 seconds at 30fps)
        self.perclos_window = 60
        self.ear_history = deque(maxlen=self.perclos_window)
        
        # Eye closure threshold
        self.ear_threshold = 0.2
        
        # Yawn detection
        self.yawn_history = deque(maxlen=60 * fps)  # 1 minute history
        self.yawn_threshold = 0.6  # Mouth aspect ratio threshold
        self.yawning = False
        
        # Micro-sleep detection
        self.eyes_closed_start = None
        self.micro_sleep_threshold = 2.0  # seconds
        
        # Head nodding detection
        self.head_nod_history = deque(maxlen=60 * fps)  # 1 minute history
        self.previous_pitch = None
        self.nod_threshold = 15.0  # degrees
        
        # Drowsiness state
        self.drowsiness_start = None
        self.drowsiness_threshold = 3.0  # seconds
        
        logger.info(f"DrowsinessDetector initialized (fps={fps})")
    
    def detect_drowsiness(self, landmarks: np.ndarray, head_pose: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect drowsiness indicators.
        
        Args:
            landmarks: Facial landmarks (68, 2)
            head_pose: Head pose dictionary with roll, pitch, yaw
            
        Returns:
            Dictionary with:
                - score: Drowsiness score (0-1)
                - yawn_detected: Boolean
                - micro_sleep: Boolean
                - head_nod: Boolean
                - perclos: PERCLOS value (0-1)
        """
        if landmarks is None or len(landmarks) < 68:
            return self._default_drowsiness()
        
        try:
            # Calculate Eye Aspect Ratio (EAR)
            left_ear = self._calculate_ear(landmarks, eye='left')
            right_ear = self._calculate_ear(landmarks, eye='right')
            avg_ear = (left_ear + right_ear) / 2
            
            # Update EAR history
            self.ear_history.append(avg_ear)
            
            # Calculate PERCLOS
            perclos = self._calculate_perclos()
            
            # Detect micro-sleep
            micro_sleep = self._detect_micro_sleep(avg_ear)
            
            # Detect yawning
            yawn_detected = self._detect_yawn(landmarks)
            
            # Detect head nodding
            head_nod = self._detect_head_nod(head_pose)
            
            # Calculate drowsiness score
            drowsiness_score = self._calculate_drowsiness_score(
                perclos, micro_sleep, yawn_detected, head_nod
            )
            
            return {
                'score': float(drowsiness_score),
                'yawn_detected': yawn_detected,
                'micro_sleep': micro_sleep,
                'head_nod': head_nod,
                'perclos': float(perclos),
                'left_ear': float(left_ear),
                'right_ear': float(right_ear)
            }
        
        except Exception as e:
            logger.error(f"Drowsiness detection failed: {e}")
            return self._default_drowsiness()
    
    def _calculate_ear(self, landmarks: np.ndarray, eye: str) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Args:
            landmarks: Facial landmarks (68, 2)
            eye: 'left' or 'right'
            
        Returns:
            Eye aspect ratio
        """
        if eye == 'left':
            # Left eye landmarks (42-47)
            eye_points = landmarks[42:48]
        else:
            # Right eye landmarks (36-41)
            eye_points = landmarks[36:42]
        
        # Vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Avoid division by zero
        if h < 1e-6:
            return 0.3  # Default open eye value
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _calculate_perclos(self) -> float:
        """
        Calculate PERCLOS (Percentage of Eye Closure).
        
        Returns:
            PERCLOS value (0-1)
        """
        if len(self.ear_history) < self.perclos_window:
            return 0.0
        
        # Count frames where eyes are closed
        closed_frames = sum(1 for ear in self.ear_history if ear < self.ear_threshold)
        
        perclos = closed_frames / len(self.ear_history)
        return perclos
    
    def _detect_micro_sleep(self, ear: float) -> bool:
        """
        Detect micro-sleep events (eyes closed > 2 seconds).
        
        Args:
            ear: Current eye aspect ratio
            
        Returns:
            True if micro-sleep detected
        """
        current_time = time.time()
        
        if ear < self.ear_threshold:
            # Eyes are closed
            if self.eyes_closed_start is None:
                self.eyes_closed_start = current_time
            else:
                duration = current_time - self.eyes_closed_start
                if duration > self.micro_sleep_threshold:
                    return True
        else:
            # Eyes are open
            self.eyes_closed_start = None
        
        return False
    
    def _detect_yawn(self, landmarks: np.ndarray) -> bool:
        """
        Detect yawning based on mouth aspect ratio.
        
        Args:
            landmarks: Facial landmarks (68, 2)
            
        Returns:
            True if yawn detected
        """
        # Mouth landmarks (48-67)
        mouth_top = landmarks[51]
        mouth_bottom = landmarks[57]
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        
        # Calculate mouth aspect ratio (MAR)
        vertical = np.linalg.norm(mouth_top - mouth_bottom)
        horizontal = np.linalg.norm(mouth_left - mouth_right)
        
        if horizontal < 1e-6:
            mar = 0.0
        else:
            mar = vertical / horizontal
        
        # Detect yawn
        current_time = time.time()
        yawn_detected = False
        
        if mar > self.yawn_threshold:
            if not self.yawning:
                # New yawn started
                self.yawning = True
                self.yawn_history.append(current_time)
                yawn_detected = True
        else:
            self.yawning = False
        
        return yawn_detected
    
    def _detect_head_nod(self, head_pose: Dict[str, float]) -> bool:
        """
        Detect head nodding events.
        
        Args:
            head_pose: Head pose dictionary with pitch
            
        Returns:
            True if head nod detected
        """
        current_pitch = head_pose.get('pitch', 0.0)
        current_time = time.time()
        
        nod_detected = False
        
        if self.previous_pitch is not None:
            # Check for significant downward head movement
            pitch_change = current_pitch - self.previous_pitch
            
            if pitch_change < -self.nod_threshold:
                # Head moved down significantly
                self.head_nod_history.append(current_time)
                nod_detected = True
        
        self.previous_pitch = current_pitch
        
        return nod_detected
    
    def _calculate_drowsiness_score(self, perclos: float, micro_sleep: bool, 
                                   yawn_detected: bool, head_nod: bool) -> float:
        """
        Calculate overall drowsiness score.
        
        Args:
            perclos: PERCLOS value
            micro_sleep: Micro-sleep detected
            yawn_detected: Yawn detected
            head_nod: Head nod detected
            
        Returns:
            Drowsiness score (0-1)
        """
        score = 0.0
        
        # PERCLOS contribution (weight: 0.5)
        if perclos > 0.8:
            # Check if sustained for 3+ seconds
            current_time = time.time()
            if self.drowsiness_start is None:
                self.drowsiness_start = current_time
            else:
                duration = current_time - self.drowsiness_start
                if duration >= self.drowsiness_threshold:
                    score += 0.5
                else:
                    score += 0.3
        else:
            self.drowsiness_start = None
            score += perclos * 0.3
        
        # Micro-sleep contribution (weight: 0.3)
        if micro_sleep:
            score += 0.3
        
        # Yawn frequency contribution (weight: 0.1)
        yawn_count = self._count_recent_events(self.yawn_history, 60)
        if yawn_count > 3:
            score += 0.1
        elif yawn_count > 0:
            score += 0.05
        
        # Head nod frequency contribution (weight: 0.1)
        nod_count = self._count_recent_events(self.head_nod_history, 60)
        if nod_count > 2:
            score += 0.1
        elif nod_count > 0:
            score += 0.05
        
        return min(score, 1.0)
    
    def _count_recent_events(self, event_history: deque, window_seconds: int) -> int:
        """Count events in recent time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        count = sum(1 for event_time in event_history if event_time > cutoff_time)
        return count
    
    def _default_drowsiness(self) -> Dict[str, Any]:
        """Return default drowsiness state."""
        return {
            'score': 0.0,
            'yawn_detected': False,
            'micro_sleep': False,
            'head_nod': False,
            'perclos': 0.0,
            'left_ear': 0.3,
            'right_ear': 0.3
        }
