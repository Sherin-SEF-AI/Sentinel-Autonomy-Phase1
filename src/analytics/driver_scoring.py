"""Real-time driver behavior scoring system."""

import logging
import time
from collections import deque
from typing import Dict, List, Optional

from src.core.data_structures import DriverScore, DriverState, VehicleTelemetry, Risk


class DriverScoringSystem:
    """
    Real-time driver behavior scoring and evaluation.
    """

    def __init__(self, config: Dict = None):
        """
        Initialize driver scoring system.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        config = config or {}

        # Scoring weights
        self.attention_weight = config.get('attention_weight', 0.4)
        self.smoothness_weight = config.get('smoothness_weight', 0.2)
        self.safety_weight = config.get('safety_weight', 0.3)
        self.hazard_response_weight = config.get('hazard_response_weight', 0.1)

        # Event tracking
        self.recent_events = deque(maxlen=100)
        self.event_window = config.get('event_window', 60.0)  # seconds

        # Score history for smoothing
        self.score_history = deque(maxlen=30)  # Last 30 scores (~1 second at 30 FPS)

        # Smoothness tracking
        self.speed_history = deque(maxlen=10)
        self.steering_history = deque(maxlen=10)

        self.logger.info("Driver scoring system initialized")

    def calculate_score(
        self,
        driver_state: DriverState,
        vehicle_telemetry: Optional[VehicleTelemetry],
        top_risks: List[Risk],
        timestamp: float
    ) -> DriverScore:
        """
        Calculate real-time driver score.

        Args:
            driver_state: Current driver state
            vehicle_telemetry: Vehicle telemetry
            top_risks: Current top risks
            timestamp: Current timestamp

        Returns:
            DriverScore with breakdown
        """
        # Calculate component scores
        attention_score = self._calculate_attention_score(driver_state)
        smoothness_score = self._calculate_smoothness_score(vehicle_telemetry)
        safety_score = self._calculate_safety_score(top_risks)
        hazard_response_score = self._calculate_hazard_response_score(
            top_risks, driver_state
        )

        # Calculate weighted overall score
        overall_score = (
            attention_score * self.attention_weight +
            smoothness_score * self.smoothness_weight +
            safety_score * self.safety_weight +
            hazard_response_score * self.hazard_response_weight
        )

        # Apply temporal smoothing
        self.score_history.append(overall_score)
        if len(self.score_history) > 5:
            overall_score = sum(self.score_history) / len(self.score_history)

        # Get recent events (within time window)
        recent_events_list = [
            event for event in self.recent_events
            if timestamp - event['timestamp'] < self.event_window
        ]

        return DriverScore(
            timestamp=timestamp,
            overall_score=overall_score,
            attention_score=attention_score,
            smoothness_score=smoothness_score,
            safety_score=safety_score,
            hazard_response_score=hazard_response_score,
            recent_events=recent_events_list[-10:]  # Last 10 events
        )

    def _calculate_attention_score(self, driver_state: DriverState) -> float:
        """Calculate attention component score."""
        if not driver_state.face_detected:
            return 0.0

        # Use readiness score as base
        score = driver_state.readiness_score

        # Penalties
        if driver_state.drowsiness.get('yawn_detected', False):
            score -= 20
        if driver_state.drowsiness.get('micro_sleep', False):
            score -= 30
        if driver_state.distraction.get('type') not in ['looking_ahead', None]:
            score -= 15

        return max(0.0, min(100.0, score))

    def _calculate_smoothness_score(
        self,
        vehicle_telemetry: Optional[VehicleTelemetry]
    ) -> float:
        """Calculate driving smoothness score."""
        if vehicle_telemetry is None:
            return 100.0

        score = 100.0

        # Track speed changes
        self.speed_history.append(vehicle_telemetry.speed)
        if len(self.speed_history) >= 3:
            # Calculate jerk (rate of acceleration change)
            speeds = list(self.speed_history)
            accelerations = [speeds[i+1] - speeds[i] for i in range(len(speeds)-1)]
            if len(accelerations) >= 2:
                jerks = [accelerations[i+1] - accelerations[i] for i in range(len(accelerations)-1)]
                avg_jerk = sum(abs(j) for j in jerks) / len(jerks) if jerks else 0
                # Penalize high jerk
                score -= min(avg_jerk * 10, 30)

        # Track steering smoothness
        self.steering_history.append(vehicle_telemetry.steering_angle)
        if len(self.steering_history) >= 3:
            steering_changes = [
                abs(self.steering_history[i+1] - self.steering_history[i])
                for i in range(len(self.steering_history)-1)
            ]
            avg_steering_change = sum(steering_changes) / len(steering_changes) if steering_changes else 0
            # Penalize erratic steering
            score -= min(avg_steering_change * 100, 20)

        return max(0.0, min(100.0, score))

    def _calculate_safety_score(self, top_risks: List[Risk]) -> float:
        """Calculate safety awareness score."""
        score = 100.0

        # Penalize based on number and severity of unacknowledged risks
        for risk in top_risks:
            if risk.urgency == 'critical':
                score -= 30 if not risk.driver_aware else 10
            elif risk.urgency == 'high':
                score -= 20 if not risk.driver_aware else 5
            elif risk.urgency == 'medium':
                score -= 10 if not risk.driver_aware else 2

        return max(0.0, min(100.0, score))

    def _calculate_hazard_response_score(
        self,
        top_risks: List[Risk],
        driver_state: DriverState
    ) -> float:
        """Calculate hazard response quality score."""
        if not top_risks:
            return 100.0

        score = 100.0

        # Check if driver is attentive to high-priority risks
        critical_risks = [r for r in top_risks if r.urgency in ['critical', 'high']]

        if critical_risks:
            aware_count = sum(1 for r in critical_risks if r.driver_aware)
            awareness_ratio = aware_count / len(critical_risks)
            score = awareness_ratio * 100

        return score

    def log_event(self, event_type: str, severity: str, details: Dict = None):
        """
        Log a driving event.

        Args:
            event_type: Type of event ('lane_departure', 'hard_brake', etc.)
            severity: Event severity ('low', 'medium', 'high', 'critical')
            details: Additional event details
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'severity': severity,
            'details': details or {}
        }
        self.recent_events.append(event)
        self.logger.debug(f"Logged event: {event_type} ({severity})")
