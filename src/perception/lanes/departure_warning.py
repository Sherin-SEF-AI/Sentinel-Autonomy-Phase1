"""Lane departure warning system."""

import logging
import numpy as np
from typing import List, Optional, Tuple, Dict

from src.core.data_structures import DetectedLane, LaneState, VehicleTelemetry


class LaneDepartureWarning:
    """
    Lane departure warning system that monitors lateral position and predicts lane crossings.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize lane departure warning system.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Warning thresholds
        self.lateral_offset_threshold = config.get('lateral_offset_threshold', 0.3)  # meters
        self.time_to_crossing_threshold = config.get('time_to_crossing_threshold', 2.0)  # seconds
        self.min_speed_for_warning = config.get('min_speed_for_warning', 8.0)  # m/s (~30 km/h)

        # Camera calibration (simplified - ideally from camera calibration)
        self.pixels_per_meter = config.get('pixels_per_meter', 30.0)  # Approximate at lane level
        self.vehicle_width = config.get('vehicle_width', 1.8)  # meters

        # State tracking
        self.previous_offset = 0.0
        self.departure_cooldown = 0  # Frames to wait before next warning

        self.logger.info("Lane departure warning system initialized")

    def assess(
        self,
        lanes: List[DetectedLane],
        vehicle_telemetry: Optional[VehicleTelemetry],
        image_width: int,
        timestamp: float
    ) -> LaneState:
        """
        Assess lane state and generate departure warnings.

        Args:
            lanes: Detected lanes
            vehicle_telemetry: Current vehicle state
            image_width: Width of the camera image
            timestamp: Current timestamp

        Returns:
            LaneState with departure warning information
        """
        # Initialize default state
        lane_state = LaneState(
            timestamp=timestamp,
            lanes_detected=lanes,
            ego_lane_center=None,
            lateral_offset=0.0,
            heading_angle=0.0,
            departure_warning=False,
            departure_side='none',
            time_to_lane_crossing=None
        )

        if not lanes or len(lanes) < 2:
            # Need at least 2 lanes (left and right ego lanes) for warning
            self.logger.debug("Insufficient lanes detected for departure warning")
            return lane_state

        # Find ego left and right lanes
        ego_left = None
        ego_right = None

        for lane in lanes:
            if lane.lane_id == 1:  # Left ego lane
                ego_left = lane
            elif lane.lane_id == 2:  # Right ego lane
                ego_right = lane

        if ego_left is None or ego_right is None:
            return lane_state

        # Calculate lane center at a reference point (e.g., bottom of image)
        # Use the last points (closest to vehicle)
        left_x = ego_left.points[-1, 0]
        right_x = ego_right.points[-1, 0]
        lane_center_x = (left_x + right_x) / 2
        vehicle_center_x = image_width / 2

        # Calculate lateral offset in pixels, then convert to meters
        lateral_offset_pixels = vehicle_center_x - lane_center_x
        lateral_offset_meters = lateral_offset_pixels / self.pixels_per_meter

        # Calculate heading angle (simplified - using lane slope at bottom)
        # Average the slopes of both lanes
        left_slope = self._calculate_slope_at_bottom(ego_left)
        right_slope = self._calculate_slope_at_bottom(ego_right)
        avg_slope = (left_slope + right_slope) / 2

        # Convert slope to heading angle (radians)
        # Note: In image coordinates, positive slope means heading right
        heading_angle = np.arctan(avg_slope) if avg_slope != 0 else 0.0

        # Update lane state
        lane_state.ego_lane_center = (lane_center_x, ego_left.points[-1, 1])
        lane_state.lateral_offset = lateral_offset_meters
        lane_state.heading_angle = heading_angle

        # Check if vehicle is moving fast enough for warning
        if vehicle_telemetry and vehicle_telemetry.speed < self.min_speed_for_warning:
            return lane_state

        # Calculate time to lane crossing
        ttc_left, ttc_right = self._calculate_time_to_crossing(
            lateral_offset_meters,
            heading_angle,
            vehicle_telemetry,
            left_x,
            right_x,
            vehicle_center_x
        )

        # Determine if warning should be issued
        warning_active = False
        warning_side = 'none'

        # Check if turn signal is active (suppress warning if intentional lane change)
        turn_signal_active = False
        if vehicle_telemetry:
            turn_signal_active = vehicle_telemetry.turn_signal != 'none'

        # Cooldown check
        if self.departure_cooldown > 0:
            self.departure_cooldown -= 1

        # Issue warning if drifting and no turn signal active
        if not turn_signal_active and self.departure_cooldown == 0:
            # Check left lane departure
            if ttc_left is not None and ttc_left < self.time_to_crossing_threshold:
                if abs(lateral_offset_meters) > self.lateral_offset_threshold:
                    warning_active = True
                    warning_side = 'left'
                    lane_state.time_to_lane_crossing = ttc_left
                    self.departure_cooldown = 30  # ~1 second cooldown at 30 FPS

            # Check right lane departure
            elif ttc_right is not None and ttc_right < self.time_to_crossing_threshold:
                if abs(lateral_offset_meters) > self.lateral_offset_threshold:
                    warning_active = True
                    warning_side = 'right'
                    lane_state.time_to_lane_crossing = ttc_right
                    self.departure_cooldown = 30

        lane_state.departure_warning = warning_active
        lane_state.departure_side = warning_side

        # Store for next iteration
        self.previous_offset = lateral_offset_meters

        if warning_active:
            self.logger.warning(
                f"Lane departure warning: {warning_side} side, "
                f"offset={lateral_offset_meters:.2f}m, TTC={lane_state.time_to_lane_crossing:.1f}s"
            )

        return lane_state

    def _calculate_slope_at_bottom(self, lane: DetectedLane) -> float:
        """
        Calculate lane slope at the bottom of the image (closest to vehicle).

        Args:
            lane: Detected lane

        Returns:
            Slope (dx/dy)
        """
        if len(lane.points) < 2:
            return 0.0

        # Use last few points for more stable slope estimate
        n_points = min(10, len(lane.points))
        points = lane.points[-n_points:]

        # Calculate average slope
        if len(points) > 1:
            x = points[:, 0]
            y = points[:, 1]
            if y[-1] - y[0] != 0:
                slope = (x[-1] - x[0]) / (y[-1] - y[0])
                return slope

        return 0.0

    def _calculate_time_to_crossing(
        self,
        lateral_offset: float,
        heading_angle: float,
        vehicle_telemetry: Optional[VehicleTelemetry],
        left_x: float,
        right_x: float,
        vehicle_x: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate time to cross left and right lane boundaries.

        Args:
            lateral_offset: Current lateral offset in meters
            heading_angle: Current heading angle relative to lane
            vehicle_telemetry: Vehicle state
            left_x: Left lane position (pixels)
            right_x: Right lane position (pixels)
            vehicle_x: Vehicle center position (pixels)

        Returns:
            Tuple of (time_to_left_crossing, time_to_right_crossing) in seconds or None
        """
        if vehicle_telemetry is None:
            return None, None

        # Calculate lateral velocity component
        # lateral_velocity = speed * sin(heading_angle) + steering contribution
        lateral_velocity = vehicle_telemetry.speed * np.sin(heading_angle)

        # Add steering contribution (simplified)
        if hasattr(vehicle_telemetry, 'steering_angle'):
            # Approximate lateral acceleration from steering
            lateral_accel = vehicle_telemetry.steering_angle * vehicle_telemetry.speed * 0.5
            lateral_velocity += lateral_accel

        # Calculate distance to each lane boundary (in pixels, then convert)
        dist_to_left_pixels = vehicle_x - left_x - (self.vehicle_width / 2 * self.pixels_per_meter)
        dist_to_right_pixels = right_x - vehicle_x - (self.vehicle_width / 2 * self.pixels_per_meter)

        dist_to_left_meters = dist_to_left_pixels / self.pixels_per_meter
        dist_to_right_meters = dist_to_right_pixels / self.pixels_per_meter

        # Calculate time to crossing
        ttc_left = None
        ttc_right = None

        if lateral_velocity < -0.01:  # Moving left
            if dist_to_left_meters > 0:
                ttc_left = abs(dist_to_left_meters / lateral_velocity)
        elif lateral_velocity > 0.01:  # Moving right
            if dist_to_right_meters > 0:
                ttc_right = dist_to_right_meters / lateral_velocity

        return ttc_left, ttc_right

    def reset(self):
        """Reset internal state."""
        self.previous_offset = 0.0
        self.departure_cooldown = 0
        self.logger.debug("Lane departure warning system reset")
