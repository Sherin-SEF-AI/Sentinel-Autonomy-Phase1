"""Traffic sign detection and recognition."""

import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.core.data_structures import TrafficSign


class TrafficSignDetector:
    """
    Detect and classify traffic signs using YOLO-based detection.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize traffic sign detector.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Sign categories
        self.sign_classes = {
            0: 'speed_limit_20', 1: 'speed_limit_30', 2: 'speed_limit_40',
            3: 'speed_limit_50', 4: 'speed_limit_60', 5: 'speed_limit_70',
            6: 'speed_limit_80', 7: 'stop', 8: 'yield',
            9: 'no_entry', 10: 'warning_pedestrian', 11: 'warning_children',
            12: 'warning_curve', 13: 'no_parking', 14: 'one_way'
        }

        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.nms_threshold = config.get('nms_threshold', 0.4)

        # Model (using existing YOLO detector - extend to traffic signs)
        self.enabled = config.get('enabled', True)

        # Speed limit tracking
        self.current_speed_limit = None
        self.speed_limit_confidence = 0.0

        self.logger.info(f"Traffic sign detector initialized (enabled={self.enabled})")

    def detect(
        self,
        frame: np.ndarray,
        camera_id: int
    ) -> List[TrafficSign]:
        """
        Detect traffic signs in camera frame.

        Args:
            frame: Input camera frame
            camera_id: Camera identifier

        Returns:
            List of detected traffic signs
        """
        if not self.enabled or frame is None:
            return []

        # Simplified placeholder - would use trained YOLO model for signs
        # For now, use color-based detection for demonstration
        detected_signs = self._detect_by_color(frame, camera_id)

        return detected_signs

    def _detect_by_color(
        self,
        frame: np.ndarray,
        camera_id: int
    ) -> List[TrafficSign]:
        """
        Detect signs using color-based heuristics (placeholder).

        Args:
            frame: Input frame
            camera_id: Camera ID

        Returns:
            List of detected signs
        """
        signs = []

        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect red signs (stop, speed limits, warnings)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500 or area > 50000:  # Filter by size
                continue

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0

            # Traffic signs are roughly circular/square
            if 0.7 < aspect_ratio < 1.3:
                # Placeholder classification
                sign = TrafficSign(
                    sign_type='speed_limit',
                    sign_class='speed_limit_50',
                    value=50,
                    position=(0.0, 0.0, 0.0),  # Would need 3D estimation
                    bbox_2d=(float(x), float(y), float(x + w), float(y + h)),
                    confidence=0.7,
                    camera_id=camera_id
                )
                signs.append(sign)

        return signs

    def get_current_speed_limit(self) -> Optional[int]:
        """Get the currently detected speed limit."""
        return self.current_speed_limit
