"""Lane detection using computer vision."""

import logging
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict

from src.core.data_structures import DetectedLane


class LaneDetector:
    """
    Detect lanes from front camera using classical computer vision.

    Uses Canny edge detection, Hough transform, and polynomial fitting
    for robust lane detection under various conditions.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize lane detector.

        Args:
            config: Configuration dictionary with detection parameters
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Detection parameters
        self.roi_vertices = config.get('roi_vertices', None)  # Region of interest
        self.canny_low = config.get('canny_low', 50)
        self.canny_high = config.get('canny_high', 150)
        self.hough_threshold = config.get('hough_threshold', 30)
        self.min_line_length = config.get('min_line_length', 40)
        self.max_line_gap = config.get('max_line_gap', 100)

        # Lane filtering
        self.min_lane_confidence = config.get('min_lane_confidence', 0.5)
        self.lane_width_pixels = config.get('lane_width_pixels', 400)  # Approximate lane width in pixels

        # Temporal smoothing
        self.use_smoothing = config.get('use_smoothing', True)
        self.smoothing_alpha = config.get('smoothing_alpha', 0.8)
        self.previous_lanes: Optional[List[DetectedLane]] = None

        self.logger.info("Lane detector initialized")

    def detect(self, frame: np.ndarray) -> List[DetectedLane]:
        """
        Detect lanes in camera frame.

        Args:
            frame: Input camera frame (H, W, 3) BGR

        Returns:
            List of detected lanes
        """
        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        # Define default ROI if not set (focus on lower half, remove hood)
        if self.roi_vertices is None:
            self.roi_vertices = np.array([[
                (int(width * 0.1), height),  # Bottom left
                (int(width * 0.4), int(height * 0.6)),  # Top left
                (int(width * 0.6), int(height * 0.6)),  # Top right
                (int(width * 0.9), height)  # Bottom right
            ]], dtype=np.int32)

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Apply ROI mask
        roi_mask = np.zeros_like(edges)
        cv2.fillPoly(roi_mask, self.roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi_mask)

        # Hough line detection
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        if lines is None or len(lines) == 0:
            # No lines detected, return previous lanes if available
            if self.previous_lanes and self.use_smoothing:
                return self.previous_lanes
            return []

        # Classify lines into left and right lanes
        left_lines, right_lines = self._classify_lines(lines, width)

        # Fit polynomial to lane lines
        detected_lanes = []

        # Process left lane
        if left_lines:
            left_lane = self._fit_lane(left_lines, lane_id=1, width=width, height=height)
            if left_lane:
                detected_lanes.append(left_lane)

        # Process right lane
        if right_lines:
            right_lane = self._fit_lane(right_lines, lane_id=2, width=width, height=height)
            if right_lane:
                detected_lanes.append(right_lane)

        # Apply temporal smoothing if enabled
        if self.use_smoothing and self.previous_lanes:
            detected_lanes = self._smooth_lanes(detected_lanes, self.previous_lanes)

        self.previous_lanes = detected_lanes

        return detected_lanes

    def _classify_lines(
        self,
        lines: np.ndarray,
        image_width: int
    ) -> Tuple[List, List]:
        """
        Classify detected lines into left and right lanes based on slope and position.

        Args:
            lines: Detected lines from Hough transform
            image_width: Width of the image

        Returns:
            Tuple of (left_lines, right_lines)
        """
        left_lines = []
        right_lines = []
        center_x = image_width / 2

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate slope
            if x2 - x1 == 0:
                continue  # Skip vertical lines

            slope = (y2 - y1) / (x2 - x1)

            # Filter by slope (lanes should not be horizontal)
            if abs(slope) < 0.5:
                continue

            # Classify based on slope and position
            line_center_x = (x1 + x2) / 2

            if slope < 0 and line_center_x < center_x:
                # Left lane (negative slope, left side)
                left_lines.append(line[0])
            elif slope > 0 and line_center_x > center_x:
                # Right lane (positive slope, right side)
                right_lines.append(line[0])

        return left_lines, right_lines

    def _fit_lane(
        self,
        lines: List,
        lane_id: int,
        width: int,
        height: int
    ) -> Optional[DetectedLane]:
        """
        Fit a polynomial to detected lane lines.

        Args:
            lines: List of line segments
            lane_id: Lane identifier
            width: Image width
            height: Image height

        Returns:
            DetectedLane object or None
        """
        if not lines:
            return None

        # Collect all points from line segments
        x_points = []
        y_points = []

        for line in lines:
            x1, y1, x2, y2 = line
            x_points.extend([x1, x2])
            y_points.extend([y1, y2])

        if len(x_points) < 2:
            return None

        x_points = np.array(x_points)
        y_points = np.array(y_points)

        # Fit 2nd order polynomial (x = ay^2 + by + c)
        # We fit x as a function of y because lanes are nearly vertical in image
        try:
            coefficients = np.polyfit(y_points, x_points, 2)
        except np.linalg.LinAlgError:
            return None

        # Generate smooth lane points
        y_range = np.linspace(min(y_points), height, num=50)
        x_fitted = np.polyval(coefficients, y_range)

        # Filter out-of-bounds points
        valid_mask = (x_fitted >= 0) & (x_fitted < width)
        y_range = y_range[valid_mask]
        x_fitted = x_fitted[valid_mask]

        if len(x_fitted) < 10:
            return None

        # Create lane points array
        lane_points = np.column_stack((x_fitted, y_range)).astype(np.float32)

        # Calculate confidence based on number of supporting line segments
        confidence = min(len(lines) / 10.0, 1.0)

        # Determine lane type and color (simplified - would need more sophisticated detection)
        lane_type = 'dashed'  # Default
        lane_color = 'white'  # Default

        return DetectedLane(
            lane_id=lane_id,
            points=lane_points,
            coefficients=coefficients,
            confidence=confidence,
            lane_type=lane_type,
            color=lane_color
        )

    def _smooth_lanes(
        self,
        current_lanes: List[DetectedLane],
        previous_lanes: List[DetectedLane]
    ) -> List[DetectedLane]:
        """
        Apply temporal smoothing to lane detections.

        Args:
            current_lanes: Currently detected lanes
            previous_lanes: Previously detected lanes

        Returns:
            Smoothed lanes
        """
        if not previous_lanes:
            return current_lanes

        # Match lanes by ID
        smoothed_lanes = []

        for curr_lane in current_lanes:
            # Find matching previous lane
            prev_lane = None
            for pl in previous_lanes:
                if pl.lane_id == curr_lane.lane_id:
                    prev_lane = pl
                    break

            if prev_lane is None:
                # New lane, no smoothing
                smoothed_lanes.append(curr_lane)
                continue

            # Smooth coefficients
            smoothed_coeffs = (
                self.smoothing_alpha * curr_lane.coefficients +
                (1 - self.smoothing_alpha) * prev_lane.coefficients
            )

            # Regenerate points from smoothed coefficients
            y_range = curr_lane.points[:, 1]
            x_fitted = np.polyval(smoothed_coeffs, y_range)
            smoothed_points = np.column_stack((x_fitted, y_range)).astype(np.float32)

            # Create smoothed lane
            smoothed_lane = DetectedLane(
                lane_id=curr_lane.lane_id,
                points=smoothed_points,
                coefficients=smoothed_coeffs,
                confidence=curr_lane.confidence,
                lane_type=curr_lane.lane_type,
                color=curr_lane.color
            )
            smoothed_lanes.append(smoothed_lane)

        return smoothed_lanes

    def draw_lanes(self, frame: np.ndarray, lanes: List[DetectedLane]) -> np.ndarray:
        """
        Draw detected lanes on frame for visualization.

        Args:
            frame: Input frame
            lanes: Detected lanes

        Returns:
            Frame with lanes drawn
        """
        output = frame.copy()

        for lane in lanes:
            # Determine color based on lane ID
            if lane.lane_id == 1:  # Left ego lane
                color = (0, 255, 0)  # Green
            elif lane.lane_id == 2:  # Right ego lane
                color = (0, 0, 255)  # Red
            else:
                color = (255, 0, 0)  # Blue

            # Draw lane points
            points = lane.points.astype(np.int32)
            cv2.polylines(output, [points], isClosed=False, color=color, thickness=3)

        return output
