"""Road surface condition analysis."""

import logging
import cv2
import numpy as np
from typing import Dict, List

from src.core.data_structures import RoadCondition, SegmentationOutput


class RoadSurfaceAnalyzer:
    """
    Analyze road surface conditions from camera and BEV segmentation.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize road surface analyzer.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        self.enabled = config.get('enabled', True)

        # Detection thresholds
        self.wet_threshold = config.get('wet_threshold', 0.3)  # Brightness threshold
        self.puddle_min_area = config.get('puddle_min_area', 100)  # pixels

        self.logger.info("Road surface analyzer initialized")

    def analyze(
        self,
        front_camera: np.ndarray,
        bev_seg: SegmentationOutput,
        timestamp: float
    ) -> RoadCondition:
        """
        Analyze road surface conditions.

        Args:
            front_camera: Front camera frame
            bev_seg: BEV segmentation output
            timestamp: Current timestamp

        Returns:
            RoadCondition analysis
        """
        # Default condition
        condition = RoadCondition(
            timestamp=timestamp,
            surface_type='dry',
            confidence=0.8,
            visibility='clear',
            hazards=[],
            friction_estimate=1.0
        )

        if not self.enabled or front_camera is None:
            return condition

        # Analyze road brightness (wet roads are darker/more reflective)
        road_roi = front_camera[int(front_camera.shape[0] * 0.5):, :]  # Lower half
        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        std_brightness = np.std(gray)

        # Detect reflections (high std with low avg = wet/reflective)
        if avg_brightness < 80 and std_brightness > 40:
            condition.surface_type = 'wet'
            condition.friction_estimate = 0.7
            condition.confidence = 0.6

        # Detect puddles from BEV segmentation
        if bev_seg and bev_seg.class_map is not None:
            # Look for low-confidence or anomalous regions in road class
            road_mask = (bev_seg.class_map == 0)  # Assuming class 0 is road
            low_conf_mask = (bev_seg.confidence < 0.5) & road_mask

            # Find contiguous low-confidence regions (potential puddles)
            contours, _ = cv2.findContours(
                low_conf_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.puddle_min_area:
                    condition.hazards.append('puddle')
                    break  # Just flag once

        # Analyze visibility from camera
        # High blur/low contrast = fog/rain
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Low variance = poor visibility
            condition.visibility = 'fog'
            condition.confidence = 0.5

        return condition
