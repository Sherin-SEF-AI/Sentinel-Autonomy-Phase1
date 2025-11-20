"""Parking space detection and assistance."""

import logging
import cv2
import numpy as np
from typing import List, Dict, Optional

from src.core.data_structures import ParkingSpace, SegmentationOutput


class ParkingAssistant:
    """
    Detect parking spaces from BEV and assist with parking maneuvers.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize parking assistant.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Parking space dimensions
        self.min_width = config.get('min_width', 2.0)  # meters
        self.min_length = config.get('min_length', 4.5)  # meters
        self.vehicle_width = config.get('vehicle_width', 1.8)  # meters
        self.vehicle_length = config.get('vehicle_length', 4.5)  # meters
        self.parking_clearance = config.get('parking_clearance', 0.3)  # meters

        # Conversion (simplified - should use camera calibration)
        self.pixels_per_meter = config.get('pixels_per_meter', 30.0)

        self.enabled = config.get('enabled', True)

        self.logger.info("Parking assistant initialized")

    def detect_spaces(
        self,
        bev_seg: SegmentationOutput,
        timestamp: float
    ) -> List[ParkingSpace]:
        """
        Detect available parking spaces from BEV.

        Args:
            bev_seg: BEV segmentation output
            timestamp: Current timestamp

        Returns:
            List of detected parking spaces
        """
        if not self.enabled or bev_seg is None:
            return []

        spaces = []

        # Use segmentation to find parking areas
        # Assuming class 6 is 'parking_space' in segmentation
        if hasattr(bev_seg, 'class_map') and bev_seg.class_map is not None:
            parking_mask = (bev_seg.class_map == 6).astype(np.uint8)

            # Find contours of parking areas
            contours, _ = cv2.findContours(
                parking_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for idx, contour in enumerate(contours):
                area = cv2.contourArea(contour)

                # Calculate bounding rectangle
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Get dimensions
                width_px, height_px = rect[1]
                width_m = width_px / self.pixels_per_meter
                length_m = height_px / self.pixels_per_meter

                # Ensure width < length
                if width_m > length_m:
                    width_m, length_m = length_m, width_m

                # Check if space is large enough
                if (width_m >= self.min_width and
                    length_m >= self.min_length):

                    # Check if vehicle can fit
                    can_fit = (
                        width_m >= self.vehicle_width + self.parking_clearance and
                        length_m >= self.vehicle_length + self.parking_clearance
                    )

                    space = ParkingSpace(
                        space_id=idx,
                        corners=box.astype(np.float32),
                        width=width_m,
                        length=length_m,
                        occupied=not can_fit,  # Simplified occupancy check
                        confidence=0.7,
                        space_type='perpendicular'  # Would need orientation analysis
                    )
                    spaces.append(space)

        return spaces

    def calculate_parking_guidance(
        self,
        target_space: ParkingSpace,
        current_position: np.ndarray
    ) -> Dict:
        """
        Calculate steering guidance for parking maneuver.

        Args:
            target_space: Target parking space
            current_position: Current vehicle position in BEV (x, y, theta)

        Returns:
            Dictionary with guidance information
        """
        # Simplified parking guidance
        space_center = np.mean(target_space.corners, axis=0)

        # Calculate distance and angle to space
        dx = space_center[0] - current_position[0]
        dy = space_center[1] - current_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        angle_to_space = np.arctan2(dy, dx)

        # Current heading
        current_heading = current_position[2]
        heading_error = angle_to_space - current_heading

        # Normalize angle
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        guidance = {
            'distance_to_space': distance / self.pixels_per_meter,  # meters
            'heading_error': np.degrees(heading_error),  # degrees
            'recommended_action': 'forward' if distance > 50 else 'align'
        }

        return guidance
