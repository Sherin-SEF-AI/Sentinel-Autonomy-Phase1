"""Trip statistics tracking and analytics."""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Optional, List

from src.core.data_structures import TripStats, DriverState, VehicleTelemetry


class TripTracker:
    """
    Track trip statistics and generate analytics.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize trip tracker.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        self.save_dir = Path(config.get('save_dir', 'data/trips'))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Current trip state
        self.current_trip: Optional[TripStats] = None
        self.is_active = False

        # Thresholds for event detection
        self.hard_brake_threshold = config.get('hard_brake_threshold', 4.0)  # m/s²
        self.rapid_accel_threshold = config.get('rapid_accel_threshold', 3.0)  # m/s²

        # Running statistics
        self.last_speed = 0.0
        self.last_timestamp = 0.0
        self.attention_scores = []

        self.logger.info("Trip tracker initialized")

    def start_trip(self):
        """Start a new trip."""
        if self.is_active:
            self.logger.warning("Trip already active, stopping previous trip")
            self.end_trip()

        self.current_trip = TripStats(
            start_time=time.time(),
            end_time=None,
            distance=0.0,
            duration=0.0,
            average_speed=0.0,
            max_speed=0.0,
            num_hard_brakes=0,
            num_rapid_accelerations=0,
            num_lane_departures=0,
            num_collision_warnings=0,
            num_blind_spot_warnings=0,
            average_attention_score=0.0,
            safety_score=100.0
        )
        self.is_active = True
        self.attention_scores = []
        self.last_timestamp = time.time()

        self.logger.info("Trip started")

    def update(
        self,
        vehicle_telemetry: Optional[VehicleTelemetry],
        driver_state: Optional[DriverState],
        events: Dict = None
    ):
        """
        Update trip statistics.

        Args:
            vehicle_telemetry: Current vehicle state
            driver_state: Current driver state
            events: Dictionary of events this frame
        """
        if not self.is_active or self.current_trip is None:
            return

        current_time = time.time()
        dt = current_time - self.last_timestamp

        if vehicle_telemetry:
            speed = vehicle_telemetry.speed

            # Update distance
            self.current_trip.distance += speed * dt

            # Update max speed
            self.current_trip.max_speed = max(self.current_trip.max_speed, speed)

            # Detect hard braking and rapid acceleration
            if dt > 0 and self.last_speed > 0:
                acceleration = (speed - self.last_speed) / dt

                if acceleration < -self.hard_brake_threshold:
                    self.current_trip.num_hard_brakes += 1
                    self.logger.info(f"Hard brake detected: {acceleration:.1f} m/s²")
                elif acceleration > self.rapid_accel_threshold:
                    self.current_trip.num_rapid_accelerations += 1
                    self.logger.info(f"Rapid acceleration detected: {acceleration:.1f} m/s²")

            self.last_speed = speed

        # Track driver attention
        if driver_state:
            self.attention_scores.append(driver_state.readiness_score)

        # Track events
        if events:
            if events.get('lane_departure'):
                self.current_trip.num_lane_departures += 1
            if events.get('collision_warning'):
                self.current_trip.num_collision_warnings += 1
            if events.get('blind_spot_warning'):
                self.current_trip.num_blind_spot_warnings += 1

        self.last_timestamp = current_time

    def end_trip(self) -> Optional[TripStats]:
        """
        End the current trip and calculate final statistics.

        Returns:
            Completed TripStats or None
        """
        if not self.is_active or self.current_trip is None:
            return None

        # Finalize statistics
        self.current_trip.end_time = time.time()
        self.current_trip.duration = self.current_trip.end_time - self.current_trip.start_time

        if self.current_trip.duration > 0:
            self.current_trip.average_speed = self.current_trip.distance / self.current_trip.duration

        if self.attention_scores:
            self.current_trip.average_attention_score = sum(self.attention_scores) / len(self.attention_scores)

        # Calculate safety score (0-100)
        safety_score = 100.0
        safety_score -= self.current_trip.num_hard_brakes * 5
        safety_score -= self.current_trip.num_rapid_accelerations * 3
        safety_score -= self.current_trip.num_lane_departures * 10
        safety_score -= self.current_trip.num_collision_warnings * 15
        safety_score -= self.current_trip.num_blind_spot_warnings * 5
        safety_score = max(0.0, min(100.0, safety_score))

        self.current_trip.safety_score = safety_score

        # Save trip data
        self._save_trip(self.current_trip)

        trip = self.current_trip
        self.current_trip = None
        self.is_active = False

        self.logger.info(
            f"Trip ended: duration={trip.duration:.0f}s, "
            f"distance={trip.distance:.1f}m, safety_score={trip.safety_score:.0f}"
        )

        return trip

    def _save_trip(self, trip: TripStats):
        """Save trip data to file."""
        try:
            filename = f"trip_{int(trip.start_time)}.json"
            filepath = self.save_dir / filename

            trip_data = {
                'start_time': trip.start_time,
                'end_time': trip.end_time,
                'duration': trip.duration,
                'distance': trip.distance,
                'average_speed': trip.average_speed,
                'max_speed': trip.max_speed,
                'num_hard_brakes': trip.num_hard_brakes,
                'num_rapid_accelerations': trip.num_rapid_accelerations,
                'num_lane_departures': trip.num_lane_departures,
                'num_collision_warnings': trip.num_collision_warnings,
                'num_blind_spot_warnings': trip.num_blind_spot_warnings,
                'average_attention_score': trip.average_attention_score,
                'safety_score': trip.safety_score
            }

            with open(filepath, 'w') as f:
                json.dump(trip_data, f, indent=2)

            self.logger.info(f"Trip data saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save trip data: {e}")

    def get_current_stats(self) -> Optional[TripStats]:
        """Get current trip statistics."""
        return self.current_trip
