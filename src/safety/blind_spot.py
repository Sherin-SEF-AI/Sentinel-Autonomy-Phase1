"""Blind spot monitoring system."""

import logging
import numpy as np
from typing import List, Dict, Optional

from src.core.data_structures import Detection3D, BlindSpotWarning, VehicleTelemetry


class BlindSpotMonitor:
    """
    Monitor blind spots using 3D object detections and issue warnings.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize blind spot monitor.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Blind spot zone definition (in vehicle frame, meters)
        self.left_zone_y_min = config.get('left_zone_y_min', 0.5)  # Left side
        self.left_zone_y_max = config.get('left_zone_y_max', 2.5)
        self.right_zone_y_min = config.get('right_zone_y_min', -2.5)  # Right side
        self.right_zone_y_max = config.get('right_zone_y_max', -0.5)
        self.zone_x_min = config.get('zone_x_min', -3.0)  # Behind vehicle
        self.zone_x_max = config.get('zone_x_max', 1.0)  # Slightly ahead

        # Warning thresholds
        self.min_object_confidence = config.get('min_object_confidence', 0.5)
        self.warning_hysteresis_frames = config.get('warning_hysteresis_frames', 5)

        # State tracking
        self.left_warning_counter = 0
        self.right_warning_counter = 0

        self.logger.info("Blind spot monitor initialized")

    def assess(
        self,
        detections: List[Detection3D],
        vehicle_telemetry: Optional[VehicleTelemetry],
        timestamp: float
    ) -> BlindSpotWarning:
        """
        Assess blind spot status and generate warnings.

        Args:
            detections: List of 3D object detections
            vehicle_telemetry: Current vehicle state
            timestamp: Current timestamp

        Returns:
            BlindSpotWarning with detection status
        """
        left_objects = []
        right_objects = []

        # Filter objects in blind spot zones
        for detection in detections:
            if detection.confidence < self.min_object_confidence:
                continue

            x, y, z, w, h, l, theta = detection.bbox_3d

            # Check if object is vehicle (most relevant for blind spot)
            if detection.class_name not in ['vehicle', 'cyclist', 'motorcyclist']:
                continue

            # Check left blind spot
            if (self.left_zone_y_min <= y <= self.left_zone_y_max and
                self.zone_x_min <= x <= self.zone_x_max):
                left_objects.append(detection)

            # Check right blind spot
            elif (self.right_zone_y_min <= y <= self.right_zone_y_max and
                  self.zone_x_min <= x <= self.zone_x_max):
                right_objects.append(detection)

        # Determine blind spot status with hysteresis
        left_blind_spot = len(left_objects) > 0
        right_blind_spot = len(right_objects) > 0

        # Apply hysteresis to reduce flickering
        if left_blind_spot:
            self.left_warning_counter = min(
                self.left_warning_counter + 1,
                self.warning_hysteresis_frames
            )
        else:
            self.left_warning_counter = max(self.left_warning_counter - 1, 0)

        if right_blind_spot:
            self.right_warning_counter = min(
                self.right_warning_counter + 1,
                self.warning_hysteresis_frames
            )
        else:
            self.right_warning_counter = max(self.right_warning_counter - 1, 0)

        # Activate warning only if counter exceeds threshold
        left_warning_active = self.left_warning_counter >= (self.warning_hysteresis_frames // 2)
        right_warning_active = self.right_warning_counter >= (self.warning_hysteresis_frames // 2)

        # Determine warning side and if turn signal is active
        warning_active = False
        warning_side = 'none'

        if vehicle_telemetry:
            turn_signal = vehicle_telemetry.turn_signal

            # Enhanced warning when turn signal is active and object in blind spot
            if turn_signal == 'left' and left_warning_active:
                warning_active = True
                warning_side = 'left'
                self.logger.warning("Blind spot warning: LEFT - vehicle in blind spot with turn signal active!")
            elif turn_signal == 'right' and right_warning_active:
                warning_active = True
                warning_side = 'right'
                self.logger.warning("Blind spot warning: RIGHT - vehicle in blind spot with turn signal active!")
            elif left_warning_active and right_warning_active:
                warning_active = True
                warning_side = 'both'
            elif left_warning_active:
                warning_side = 'left'
            elif right_warning_active:
                warning_side = 'right'

        return BlindSpotWarning(
            timestamp=timestamp,
            left_blind_spot=left_warning_active,
            right_blind_spot=right_warning_active,
            left_objects=left_objects,
            right_objects=right_objects,
            warning_active=warning_active,
            warning_side=warning_side
        )
