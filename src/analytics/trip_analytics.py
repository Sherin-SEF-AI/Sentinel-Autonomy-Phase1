"""
Trip Analytics

Tracks trip metrics including duration, distance, speed, safety scores,
and identifies high-risk segments.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TripSegment:
    """Represents a segment of a trip."""
    start_time: float
    end_time: float
    start_position: Tuple[float, float]  # (x, y)
    end_position: Tuple[float, float]
    distance: float  # meters
    avg_speed: float  # m/s
    max_risk: float  # 0-1
    alert_count: int
    critical_alert_count: int


@dataclass
class TripSummary:
    """Complete trip summary."""
    trip_id: str
    start_time: datetime
    end_time: datetime
    duration: float  # seconds
    distance: float  # meters
    avg_speed: float  # m/s
    max_speed: float  # m/s
    safety_score: float  # 0-100
    alert_counts: Dict[str, int]  # by urgency level
    high_risk_segments: List[TripSegment]
    driver_id: Optional[str] = None
    route_summary: Optional[str] = None


class TripAnalytics:
    """
    Tracks and analyzes trip metrics.
    
    Capabilities:
    - Track trip duration, distance, average speed
    - Calculate trip safety score
    - Count alerts by type
    - Identify high-risk segments
    """
    
    def __init__(self, config: dict):
        """
        Initialize trip analytics.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Current trip state
        self.trip_active = False
        self.trip_start_time: Optional[datetime] = None
        self.trip_id: Optional[str] = None
        self.driver_id: Optional[str] = None
        
        # Trip metrics
        self.total_distance = 0.0  # meters
        self.positions: List[Tuple[float, float, float]] = []  # (x, y, timestamp)
        self.speeds: List[float] = []  # m/s
        self.risk_scores: List[Tuple[float, float]] = []  # (timestamp, risk)
        self.alerts: List[Dict] = []  # Alert records
        
        # Segment tracking
        self.segment_duration = config.get('analytics', {}).get('segment_duration', 60.0)  # seconds
        self.high_risk_threshold = config.get('analytics', {}).get('high_risk_threshold', 0.7)
        
        # Trip history
        self.trip_history: List[TripSummary] = []
        self.max_history = 100
        
        logger.info("TripAnalytics initialized")
    
    def start_trip(self, driver_id: Optional[str] = None) -> str:
        """
        Start a new trip.
        
        Args:
            driver_id: Optional driver identifier
        
        Returns:
            Trip ID
        """
        if self.trip_active:
            logger.warning("Trip already active, ending previous trip")
            self.end_trip()
        
        self.trip_active = True
        self.trip_start_time = datetime.now()
        self.trip_id = f"trip_{self.trip_start_time.strftime('%Y%m%d_%H%M%S')}"
        self.driver_id = driver_id
        
        # Reset metrics
        self.total_distance = 0.0
        self.positions = []
        self.speeds = []
        self.risk_scores = []
        self.alerts = []
        
        logger.info(f"Trip started: {self.trip_id}, driver: {driver_id}")
        return self.trip_id
    
    def update(self,
               timestamp: float,
               position: Tuple[float, float],
               speed: float,
               risk_score: float,
               alerts: Optional[List] = None):
        """
        Update trip metrics with current frame data.
        
        Args:
            timestamp: Current timestamp
            position: Current position (x, y) in meters
            speed: Current speed in m/s
            risk_score: Current risk score (0-1)
            alerts: List of alerts generated this frame
        """
        if not self.trip_active:
            return
        
        # Update position and calculate distance
        if self.positions:
            prev_pos = self.positions[-1][:2]
            dx = position[0] - prev_pos[0]
            dy = position[1] - prev_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            self.total_distance += distance
        
        self.positions.append((position[0], position[1], timestamp))
        self.speeds.append(speed)
        self.risk_scores.append((timestamp, risk_score))
        
        # Record alerts
        if alerts:
            for alert in alerts:
                alert_record = {
                    'timestamp': timestamp,
                    'urgency': alert.urgency if hasattr(alert, 'urgency') else 'unknown',
                    'message': alert.message if hasattr(alert, 'message') else '',
                    'hazard_id': alert.hazard_id if hasattr(alert, 'hazard_id') else -1
                }
                self.alerts.append(alert_record)
    
    def end_trip(self) -> Optional[TripSummary]:
        """
        End the current trip and generate summary.
        
        Returns:
            TripSummary object or None if no active trip
        """
        if not self.trip_active:
            logger.warning("No active trip to end")
            return None
        
        end_time = datetime.now()
        duration = (end_time - self.trip_start_time).total_seconds()
        
        # Calculate metrics
        avg_speed = np.mean(self.speeds) if self.speeds else 0.0
        max_speed = np.max(self.speeds) if self.speeds else 0.0
        
        # Count alerts by urgency
        alert_counts = {
            'info': 0,
            'warning': 0,
            'critical': 0
        }
        for alert in self.alerts:
            urgency = alert.get('urgency', 'unknown')
            if urgency in alert_counts:
                alert_counts[urgency] += 1
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(
            duration, self.total_distance, alert_counts, self.risk_scores
        )
        
        # Identify high-risk segments
        high_risk_segments = self._identify_high_risk_segments()
        
        # Create summary
        summary = TripSummary(
            trip_id=self.trip_id,
            start_time=self.trip_start_time,
            end_time=end_time,
            duration=duration,
            distance=self.total_distance,
            avg_speed=avg_speed,
            max_speed=max_speed,
            safety_score=safety_score,
            alert_counts=alert_counts,
            high_risk_segments=high_risk_segments,
            driver_id=self.driver_id
        )
        
        # Add to history
        self.trip_history.append(summary)
        if len(self.trip_history) > self.max_history:
            self.trip_history.pop(0)
        
        logger.info(f"Trip ended: {self.trip_id}, duration: {duration:.1f}s, "
                   f"distance: {self.total_distance:.1f}m, safety: {safety_score:.1f}")
        
        # Reset state
        self.trip_active = False
        self.trip_id = None
        
        return summary
    
    def _calculate_safety_score(self,
                                duration: float,
                                distance: float,
                                alert_counts: Dict[str, int],
                                risk_scores: List[Tuple[float, float]]) -> float:
        """
        Calculate trip safety score (0-100).
        
        Based on:
        - Alert frequency (fewer is better)
        - Critical alert count (fewer is better)
        - Average risk score (lower is better)
        - High-risk time percentage (lower is better)
        
        Args:
            duration: Trip duration in seconds
            distance: Trip distance in meters
            alert_counts: Alert counts by urgency
            risk_scores: List of (timestamp, risk) tuples
        
        Returns:
            Safety score (0-100)
        """
        score = 100.0
        
        if duration < 60:  # Less than 1 minute
            return score  # Not enough data
        
        # Penalize critical alerts heavily
        critical_count = alert_counts.get('critical', 0)
        score -= critical_count * 15  # -15 points per critical alert
        
        # Penalize warnings moderately
        warning_count = alert_counts.get('warning', 0)
        score -= warning_count * 5  # -5 points per warning
        
        # Penalize info alerts slightly
        info_count = alert_counts.get('info', 0)
        score -= info_count * 1  # -1 point per info alert
        
        # Calculate average risk
        if risk_scores:
            risks = [r[1] for r in risk_scores]
            avg_risk = np.mean(risks)
            score -= avg_risk * 20  # Up to -20 points for high average risk
            
            # Penalize time spent in high-risk situations
            high_risk_count = sum(1 for r in risks if r > self.high_risk_threshold)
            high_risk_percentage = high_risk_count / len(risks)
            score -= high_risk_percentage * 30  # Up to -30 points
        
        return float(np.clip(score, 0, 100))
    
    def _identify_high_risk_segments(self) -> List[TripSegment]:
        """
        Identify high-risk segments of the trip.
        
        A segment is considered high-risk if:
        - Average risk score > threshold
        - Contains critical alerts
        
        Returns:
            List of high-risk TripSegment objects
        """
        if not self.positions or not self.risk_scores:
            return []
        
        segments = []
        segment_start_idx = 0
        
        # Divide trip into segments
        start_time = self.positions[0][2]
        
        for i, pos in enumerate(self.positions):
            timestamp = pos[2]
            
            # Check if segment duration reached
            if timestamp - start_time >= self.segment_duration:
                # Analyze segment
                segment = self._analyze_segment(segment_start_idx, i)
                
                # Add if high-risk
                if segment and segment.max_risk > self.high_risk_threshold:
                    segments.append(segment)
                
                # Start new segment
                segment_start_idx = i
                start_time = timestamp
        
        # Analyze final segment
        if segment_start_idx < len(self.positions) - 1:
            segment = self._analyze_segment(segment_start_idx, len(self.positions) - 1)
            if segment and segment.max_risk > self.high_risk_threshold:
                segments.append(segment)
        
        logger.info(f"Identified {len(segments)} high-risk segments")
        return segments
    
    def _analyze_segment(self, start_idx: int, end_idx: int) -> Optional[TripSegment]:
        """
        Analyze a trip segment.
        
        Args:
            start_idx: Start index in positions list
            end_idx: End index in positions list
        
        Returns:
            TripSegment object or None if invalid
        """
        if start_idx >= end_idx or end_idx >= len(self.positions):
            return None
        
        start_pos = self.positions[start_idx]
        end_pos = self.positions[end_idx]
        
        start_time = start_pos[2]
        end_time = end_pos[2]
        duration = end_time - start_time
        
        if duration <= 0:
            return None
        
        # Calculate distance
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate average speed
        avg_speed = distance / duration if duration > 0 else 0.0
        
        # Find max risk in segment
        segment_risks = [
            r[1] for r in self.risk_scores
            if start_time <= r[0] <= end_time
        ]
        max_risk = np.max(segment_risks) if segment_risks else 0.0
        
        # Count alerts in segment
        segment_alerts = [
            a for a in self.alerts
            if start_time <= a['timestamp'] <= end_time
        ]
        alert_count = len(segment_alerts)
        critical_alert_count = sum(
            1 for a in segment_alerts if a.get('urgency') == 'critical'
        )
        
        return TripSegment(
            start_time=start_time,
            end_time=end_time,
            start_position=(start_pos[0], start_pos[1]),
            end_position=(end_pos[0], end_pos[1]),
            distance=distance,
            avg_speed=avg_speed,
            max_risk=max_risk,
            alert_count=alert_count,
            critical_alert_count=critical_alert_count
        )
    
    def get_trip_summary(self, trip_id: str) -> Optional[TripSummary]:
        """
        Get summary for a specific trip.
        
        Args:
            trip_id: Trip identifier
        
        Returns:
            TripSummary or None if not found
        """
        for summary in self.trip_history:
            if summary.trip_id == trip_id:
                return summary
        return None
    
    def get_recent_trips(self, limit: int = 10) -> List[TripSummary]:
        """
        Get recent trip summaries.
        
        Args:
            limit: Maximum number of trips to return
        
        Returns:
            List of TripSummary objects
        """
        return self.trip_history[-limit:]
    
    def get_driver_trips(self, driver_id: str, limit: int = 10) -> List[TripSummary]:
        """
        Get trips for a specific driver.
        
        Args:
            driver_id: Driver identifier
            limit: Maximum number of trips to return
        
        Returns:
            List of TripSummary objects
        """
        driver_trips = [
            t for t in self.trip_history
            if t.driver_id == driver_id
        ]
        return driver_trips[-limit:]
    
    def get_statistics(self) -> Dict:
        """
        Get overall statistics across all trips.
        
        Returns:
            Dictionary with aggregate statistics
        """
        if not self.trip_history:
            return {
                'total_trips': 0,
                'total_distance': 0.0,
                'total_duration': 0.0,
                'avg_safety_score': 0.0,
                'total_alerts': 0
            }
        
        total_distance = sum(t.distance for t in self.trip_history)
        total_duration = sum(t.duration for t in self.trip_history)
        avg_safety = np.mean([t.safety_score for t in self.trip_history])
        
        total_alerts = sum(
            sum(t.alert_counts.values()) for t in self.trip_history
        )
        
        return {
            'total_trips': len(self.trip_history),
            'total_distance': total_distance,
            'total_duration': total_duration,
            'avg_distance': total_distance / len(self.trip_history),
            'avg_duration': total_duration / len(self.trip_history),
            'avg_safety_score': avg_safety,
            'total_alerts': total_alerts
        }
