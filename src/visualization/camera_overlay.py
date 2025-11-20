"""Camera overlay renderer for advanced features visualization."""

import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from src.core.data_structures import (
    Detection3D, DetectedLane, BlindSpotWarning, CollisionWarning,
    TrafficSign, PredictedInteraction
)


class CameraOverlayRenderer:
    """
    Renders overlays on camera frames for visualization of:
    - Detected lanes
    - Blind spot zones
    - Collision warning zones
    - Object detections with trajectories
    - Traffic signs
    - Predicted interactions
    """

    def __init__(self, config: Dict = None):
        """
        Initialize overlay renderer.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Rendering settings
        self.line_thickness = config.get('line_thickness', 3)
        self.font_scale = config.get('font_scale', 0.6)
        self.alpha = config.get('alpha', 0.3)  # Transparency for zones

        # Colors (BGR format)
        self.colors = {
            'lane_ego': (0, 255, 0),  # Green
            'lane_adjacent': (255, 0, 0),  # Blue
            'blind_spot_clear': (0, 255, 0),  # Green
            'blind_spot_warning': (0, 0, 255),  # Red
            'collision_zone': (0, 165, 255),  # Orange
            'detection_box': (255, 255, 0),  # Cyan
            'trajectory': (255, 0, 255),  # Magenta
            'interaction': (0, 255, 255),  # Yellow
        }

        self.logger.info("Camera overlay renderer initialized")

    def render_all_overlays(
        self,
        frame: np.ndarray,
        lanes: Optional[List[DetectedLane]] = None,
        detections: Optional[List[Detection3D]] = None,
        blind_spot: Optional[BlindSpotWarning] = None,
        collision: Optional[CollisionWarning] = None,
        signs: Optional[List[TrafficSign]] = None,
        interactions: Optional[List[PredictedInteraction]] = None
    ) -> np.ndarray:
        """
        Render all overlays on camera frame.

        Args:
            frame: Input camera frame
            lanes: Detected lanes
            detections: 3D detections
            blind_spot: Blind spot warning
            collision: Collision warning
            signs: Traffic signs
            interactions: Predicted interactions

        Returns:
            Frame with overlays rendered
        """
        if frame is None:
            return frame

        output = frame.copy()

        # Render lanes (bottom layer)
        if lanes:
            output = self.render_lanes(output, lanes)

        # Render collision zones
        if collision and collision.warning_level != 'none':
            output = self.render_collision_zone(output, collision)

        # Render blind spot zones
        if blind_spot:
            output = self.render_blind_spot_zones(output, blind_spot)

        # Render detections (top layer)
        if detections:
            output = self.render_detections(output, detections)

        # Render traffic signs
        if signs:
            output = self.render_traffic_signs(output, signs)

        # Render interactions
        if interactions:
            output = self.render_interactions(output, interactions)

        return output

    def render_lanes(
        self,
        frame: np.ndarray,
        lanes: List[DetectedLane]
    ) -> np.ndarray:
        """
        Render detected lanes on frame.

        Args:
            frame: Input frame
            lanes: List of detected lanes

        Returns:
            Frame with lanes rendered
        """
        output = frame.copy()

        for lane in lanes:
            # Choose color based on lane type
            if lane.lane_id in [1, 2]:  # Ego left/right
                color = self.colors['lane_ego']
                thickness = self.line_thickness
            else:
                color = self.colors['lane_adjacent']
                thickness = self.line_thickness - 1

            # Draw lane polyline
            points = lane.points.astype(np.int32)
            cv2.polylines(output, [points], isClosed=False, color=color, thickness=thickness)

            # Draw lane type indicator (dashed line visualization)
            if lane.lane_type == 'dashed':
                # Draw dashes
                for i in range(0, len(points) - 10, 20):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[min(i + 10, len(points) - 1)])
                    cv2.line(output, pt1, pt2, color, thickness + 1)

        return output

    def render_blind_spot_zones(
        self,
        frame: np.ndarray,
        warning: BlindSpotWarning
    ) -> np.ndarray:
        """
        Render blind spot zones as semi-transparent overlays.

        Args:
            frame: Input frame
            warning: Blind spot warning status

        Returns:
            Frame with blind spot zones rendered
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Define blind spot zones (approximate in image coordinates)
        left_zone = np.array([
            [0, height // 2],
            [width // 4, height // 2],
            [width // 4, height],
            [0, height]
        ], dtype=np.int32)

        right_zone = np.array([
            [3 * width // 4, height // 2],
            [width, height // 2],
            [width, height],
            [3 * width // 4, height]
        ], dtype=np.int32)

        # Render left blind spot
        if warning.left_blind_spot:
            color = self.colors['blind_spot_warning']
            cv2.fillPoly(overlay, [left_zone], color)
            # Add text
            cv2.putText(overlay, "BLIND SPOT", (20, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            color = self.colors['blind_spot_clear']
            # Just draw border
            cv2.polylines(overlay, [left_zone], isClosed=True, color=color, thickness=2)

        # Render right blind spot
        if warning.right_blind_spot:
            color = self.colors['blind_spot_warning']
            cv2.fillPoly(overlay, [right_zone], color)
            # Add text
            cv2.putText(overlay, "BLIND SPOT", (width - 180, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            color = self.colors['blind_spot_clear']
            # Just draw border
            cv2.polylines(overlay, [right_zone], isClosed=True, color=color, thickness=2)

        # Blend with alpha
        output = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)

        return output

    def render_collision_zone(
        self,
        frame: np.ndarray,
        warning: CollisionWarning
    ) -> np.ndarray:
        """
        Render forward collision warning zone.

        Args:
            frame: Input frame
            warning: Collision warning

        Returns:
            Frame with collision zone rendered
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Define forward collision zone (trapezoid)
        zone = np.array([
            [width // 3, height],
            [2 * width // 3, height],
            [width // 2 + 50, height // 2],
            [width // 2 - 50, height // 2]
        ], dtype=np.int32)

        # Color based on warning level
        if warning.warning_level == 'critical':
            color = (0, 0, 255)  # Red
        elif warning.warning_level == 'warning':
            color = (0, 165, 255)  # Orange
        elif warning.warning_level == 'caution':
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green

        # Fill zone with transparency
        cv2.fillPoly(overlay, [zone], color)

        # Draw border
        cv2.polylines(overlay, [zone], isClosed=True, color=color, thickness=3)

        # Add warning text
        if warning.warning_level != 'none':
            text = warning.warning_level.upper()
            if warning.time_to_collision:
                text += f" - TTC: {warning.time_to_collision:.1f}s"

            # Calculate text size and position
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

            # Center text in zone
            text_x = width // 2 - text_width // 2
            text_y = height // 2 + 50

            # Draw text background
            cv2.rectangle(overlay,
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + 10),
                         (0, 0, 0), -1)

            # Draw text
            cv2.putText(overlay, text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness)

        # Blend
        output = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)

        return output

    def render_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection3D]
    ) -> np.ndarray:
        """
        Render 3D detections (simplified 2D projection for now).

        Args:
            frame: Input frame
            detections: List of 3D detections

        Returns:
            Frame with detections rendered
        """
        output = frame.copy()

        # Note: This is simplified. Full implementation would require
        # camera calibration for proper 3D to 2D projection.

        # For now, just indicate object class and confidence
        y_offset = 60
        for i, det in enumerate(detections[:5]):  # Show top 5
            text = f"{det.class_name}: {det.confidence:.2f}"
            color = self.colors['detection_box']

            cv2.putText(output, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       color, 2)
            y_offset += 25

        return output

    def render_traffic_signs(
        self,
        frame: np.ndarray,
        signs: List[TrafficSign]
    ) -> np.ndarray:
        """
        Render detected traffic signs.

        Args:
            frame: Input frame
            signs: List of traffic signs

        Returns:
            Frame with signs rendered
        """
        output = frame.copy()

        for sign in signs:
            x1, y1, x2, y2 = [int(v) for v in sign.bbox_2d]

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Draw label
            label = sign.sign_class.replace('_', ' ').title()
            if sign.value:
                label += f": {sign.value}"

            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        return output

    def render_interactions(
        self,
        frame: np.ndarray,
        interactions: List[PredictedInteraction]
    ) -> np.ndarray:
        """
        Render predicted interactions.

        Args:
            frame: Input frame
            interactions: List of predicted interactions

        Returns:
            Frame with interactions rendered
        """
        output = frame.copy()
        height, width = frame.shape[:2]

        # Show critical interactions at top of frame
        critical = [i for i in interactions if i.risk_level in ['high', 'critical']]

        y_offset = 30
        for interaction in critical[:3]:  # Show top 3
            # Color based on risk
            if interaction.risk_level == 'critical':
                color = (0, 0, 255)  # Red
            elif interaction.risk_level == 'high':
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow

            text = f"⚠️ {interaction.description}"

            # Draw background
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(output,
                         (10, y_offset - text_height - 5),
                         (10 + text_width + 10, y_offset + 5),
                         (0, 0, 0), -1)

            # Draw text
            cv2.putText(output, text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            y_offset += 35

        return output
