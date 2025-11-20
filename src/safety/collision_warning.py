"""Forward collision warning system."""

import logging
import numpy as np
from typing import List, Dict, Optional

from src.core.data_structures import Detection3D, CollisionWarning, VehicleTelemetry


class ForwardCollisionWarning:
    """
    Forward collision warning system with multi-stage alerts.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize forward collision warning system.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Warning level thresholds (Time-to-Collision in seconds)
        self.caution_ttc = config.get('caution_ttc', 3.5)
        self.warning_ttc = config.get('warning_ttc', 2.5)
        self.critical_ttc = config.get('critical_ttc', 1.5)
        self.emergency_braking_ttc = config.get('emergency_braking_ttc', 0.8)

        # Detection zone (in vehicle frame)
        self.zone_width = config.get('zone_width', 3.0)  # meters (±1.5m from center)
        self.zone_length_min = config.get('zone_length_min', 0.0)
        self.zone_length_max = config.get('zone_length_max', 100.0)

        # Filtering
        self.min_confidence = config.get('min_confidence', 0.5)
        self.min_speed_for_warning = config.get('min_speed_for_warning', 5.0)  # m/s

        # State tracking
        self.warning_hysteresis = config.get('warning_hysteresis', 3)
        self.warning_counter = 0
        self.last_warning_level = 'none'

        self.logger.info("Forward collision warning system initialized")

    def assess(
        self,
        detections: List[Detection3D],
        vehicle_telemetry: Optional[VehicleTelemetry],
        timestamp: float
    ) -> CollisionWarning:
        """
        Assess forward collision risk and generate warnings.

        Args:
            detections: List of 3D object detections
            vehicle_telemetry: Current vehicle state
            timestamp: Current timestamp

        Returns:
            CollisionWarning with warning level and recommendations
        """
        # Default state
        warning = CollisionWarning(
            timestamp=timestamp,
            warning_level='none',
            time_to_collision=None,
            target_object=None,
            recommended_deceleration=0.0,
            automatic_braking_required=False
        )

        # Check if vehicle is moving
        if vehicle_telemetry is None or vehicle_telemetry.speed < self.min_speed_for_warning:
            return warning

        ego_speed = vehicle_telemetry.speed

        # Find closest object in forward path
        closest_object = None
        min_ttc = float('inf')

        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue

            x, y, z, w, h, l, theta = detection.bbox_3d
            vx, vy, vz = detection.velocity

            # Check if object is in forward zone
            if not (self.zone_length_min <= x <= self.zone_length_max and
                    abs(y) <= self.zone_width / 2):
                continue

            # Calculate relative velocity
            relative_vx = vx - ego_speed  # Negative if ego is faster

            # Calculate TTC
            if x > 0:  # Object ahead
                if relative_vx < -0.1:  # Approaching (ego faster)
                    ttc = -x / relative_vx
                    if 0 < ttc < min_ttc:
                        min_ttc = ttc
                        closest_object = detection

        # Determine warning level
        if closest_object is not None and min_ttc < self.caution_ttc:
            warning.time_to_collision = min_ttc
            warning.target_object = closest_object

            # Determine warning level
            if min_ttc <= self.emergency_braking_ttc:
                warning.warning_level = 'critical'
                warning.automatic_braking_required = True
                # Recommend emergency braking
                warning.recommended_deceleration = 9.8  # m/s^2 (1g)
                self.warning_counter = self.warning_hysteresis
            elif min_ttc <= self.critical_ttc:
                warning.warning_level = 'critical'
                # Recommend hard braking
                x_target, _, _, _, _, _, _ = closest_object.bbox_3d
                warning.recommended_deceleration = (ego_speed ** 2) / (2 * max(x_target - 2.0, 0.1))
                self.warning_counter = self.warning_hysteresis
            elif min_ttc <= self.warning_ttc:
                warning.warning_level = 'warning'
                # Recommend moderate braking
                x_target, _, _, _, _, _, _ = closest_object.bbox_3d
                warning.recommended_deceleration = (ego_speed ** 2) / (2 * max(x_target - 5.0, 0.1))
                self.warning_counter = min(self.warning_counter + 1, self.warning_hysteresis)
            elif min_ttc <= self.caution_ttc:
                warning.warning_level = 'caution'
                # Suggest being ready to brake
                warning.recommended_deceleration = 2.0  # m/s^2
                self.warning_counter = max(self.warning_counter - 1, 0)
        else:
            # No threat detected
            self.warning_counter = max(self.warning_counter - 1, 0)
            if self.warning_counter == 0:
                warning.warning_level = 'none'

        # Apply hysteresis to prevent flickering
        if self.warning_counter < (self.warning_hysteresis // 2):
            if self.last_warning_level in ['critical', 'warning']:
                warning.warning_level = 'caution'
            elif self.last_warning_level == 'caution':
                warning.warning_level = 'none'

        # Log warnings
        if warning.warning_level != 'none' and warning.warning_level != self.last_warning_level:
            self.logger.warning(
                f"Forward collision warning: {warning.warning_level.upper()} - "
                f"TTC={min_ttc:.1f}s, decel={warning.recommended_deceleration:.1f}m/s²"
            )

        self.last_warning_level = warning.warning_level

        return warning
