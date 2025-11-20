"""
Metrics Tracker for Driver Behavior

Tracks reaction time, following distance, lane changes, speed profile, and risk tolerance.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Single snapshot of driver metrics."""
    timestamp: float
    reaction_time: Optional[float] = None  # seconds
    following_distance: Optional[float] = None  # meters
    speed: Optional[float] = None  # m/s
    lane_change: bool = False
    near_miss: bool = False
    risk_score: float = 0.0


class MetricsTracker:
    """
    Tracks driver behavior metrics over time.
    
    Metrics tracked:
    - Reaction time: Time from alert to driver action
    - Following distance: Distance to vehicle ahead
    - Lane change frequency: Number of lane changes per hour
    - Speed profile: Average, max, variance
    - Risk tolerance: Behavior during high-risk situations
    """
    
    def __init__(self, config: dict):
        """
        Initialize metrics tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Metrics storage
        self.reaction_times: List[float] = []
        self.following_distances: List[float] = []
        self.lane_changes: List[float] = []  # timestamps
        self.speeds: List[float] = []
        self.near_miss_events: List[Dict] = []
        self.risk_scores: List[float] = []
        
        # Alert tracking for reaction time
        self.pending_alerts: Dict[int, float] = {}  # alert_id -> timestamp
        
        # Session tracking
        self.session_start: Optional[float] = None
        self.session_duration: float = 0.0
        self.total_distance: float = 0.0
        
        # Previous state for change detection
        self.prev_lane_id: Optional[int] = None
        self.prev_speed: float = 0.0
        self.prev_timestamp: float = 0.0
        
        logger.info("MetricsTracker initialized")
    
    def start_session(self, timestamp: float):
        """Start a new tracking session."""
        self.session_start = timestamp
        self.prev_timestamp = timestamp
        logger.info(f"Metrics tracking session started at {timestamp}")
    
    def end_session(self, timestamp: float):
        """End the current tracking session."""
        if self.session_start is not None:
            self.session_duration = timestamp - self.session_start
            logger.info(f"Metrics tracking session ended. Duration: {self.session_duration:.1f}s")
    
    def update(self, 
               timestamp: float,
               speed: Optional[float] = None,
               following_distance: Optional[float] = None,
               lane_id: Optional[int] = None,
               risk_score: float = 0.0):
        """
        Update metrics with current state.
        
        Args:
            timestamp: Current timestamp
            speed: Vehicle speed in m/s
            following_distance: Distance to vehicle ahead in meters
            lane_id: Current lane ID
            risk_score: Current risk score (0-1)
        """
        # Update speed metrics
        if speed is not None:
            self.speeds.append(speed)
            
            # Update distance traveled
            if self.prev_timestamp > 0:
                dt = timestamp - self.prev_timestamp
                self.total_distance += speed * dt
        
        # Update following distance
        if following_distance is not None and following_distance > 0:
            self.following_distances.append(following_distance)
        
        # Detect lane changes
        if lane_id is not None and self.prev_lane_id is not None:
            if lane_id != self.prev_lane_id:
                self.lane_changes.append(timestamp)
                logger.debug(f"Lane change detected at {timestamp}")
        
        # Track risk scores
        self.risk_scores.append(risk_score)
        
        # Update previous state
        self.prev_lane_id = lane_id
        self.prev_speed = speed if speed is not None else self.prev_speed
        self.prev_timestamp = timestamp
    
    def record_alert(self, alert_id: int, timestamp: float):
        """
        Record an alert for reaction time tracking.
        
        Args:
            alert_id: Unique alert identifier
            timestamp: Alert timestamp
        """
        self.pending_alerts[alert_id] = timestamp
        logger.debug(f"Alert {alert_id} recorded at {timestamp}")
    
    def record_driver_action(self, alert_id: int, timestamp: float, action_type: str):
        """
        Record driver action in response to alert.
        
        Args:
            alert_id: Alert identifier
            timestamp: Action timestamp
            action_type: Type of action (brake, steer, etc.)
        """
        if alert_id in self.pending_alerts:
            alert_time = self.pending_alerts[alert_id]
            reaction_time = timestamp - alert_time
            
            if reaction_time > 0 and reaction_time < 10.0:  # Sanity check
                self.reaction_times.append(reaction_time)
                logger.info(f"Reaction time recorded: {reaction_time:.3f}s for action '{action_type}'")
            
            del self.pending_alerts[alert_id]
    
    def record_near_miss(self, timestamp: float, ttc: float, risk_score: float):
        """
        Record a near-miss event for risk tolerance analysis.
        
        Args:
            timestamp: Event timestamp
            ttc: Time to collision at event
            risk_score: Risk score at event
        """
        event = {
            'timestamp': timestamp,
            'ttc': ttc,
            'risk_score': risk_score
        }
        self.near_miss_events.append(event)
        logger.info(f"Near-miss event recorded: TTC={ttc:.2f}s, risk={risk_score:.3f}")
    
    def get_reaction_time_stats(self) -> Dict[str, float]:
        """
        Get reaction time statistics.
        
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not self.reaction_times:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        times = np.array(self.reaction_times)
        return {
            'mean': float(np.mean(times)),
            'median': float(np.median(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'count': len(times)
        }
    
    def get_following_distance_stats(self) -> Dict[str, float]:
        """
        Get following distance statistics.
        
        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not self.following_distances:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        distances = np.array(self.following_distances)
        return {
            'mean': float(np.mean(distances)),
            'median': float(np.median(distances)),
            'std': float(np.std(distances)),
            'min': float(np.min(distances)),
            'max': float(np.max(distances)),
            'count': len(distances)
        }
    
    def get_lane_change_frequency(self) -> float:
        """
        Get lane change frequency in changes per hour.
        
        Returns:
            Lane changes per hour
        """
        if self.session_duration <= 0:
            return 0.0
        
        hours = self.session_duration / 3600.0
        if hours > 0:
            return len(self.lane_changes) / hours
        return 0.0
    
    def get_speed_profile(self) -> Dict[str, float]:
        """
        Get speed profile statistics.
        
        Returns:
            Dictionary with mean, max, std speed
        """
        if not self.speeds:
            return {
                'mean': 0.0,
                'max': 0.0,
                'std': 0.0,
                'count': 0
            }
        
        speeds = np.array(self.speeds)
        return {
            'mean': float(np.mean(speeds)),
            'max': float(np.max(speeds)),
            'std': float(np.std(speeds)),
            'count': len(speeds)
        }
    
    def get_risk_tolerance(self) -> float:
        """
        Calculate risk tolerance from near-miss events and risk scores.
        
        Returns:
            Risk tolerance score (0-1), higher means more risk-tolerant
        """
        if not self.risk_scores:
            return 0.5  # Neutral
        
        # Calculate average risk score driver operates at
        avg_risk = np.mean(self.risk_scores)
        
        # Factor in near-miss events
        near_miss_factor = len(self.near_miss_events) / max(1, self.session_duration / 3600.0)
        
        # Combine factors
        risk_tolerance = min(1.0, avg_risk + near_miss_factor * 0.1)
        
        return float(risk_tolerance)
    
    def get_summary(self) -> Dict:
        """
        Get complete metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            'session_duration': self.session_duration,
            'total_distance': self.total_distance,
            'reaction_time': self.get_reaction_time_stats(),
            'following_distance': self.get_following_distance_stats(),
            'lane_change_frequency': self.get_lane_change_frequency(),
            'speed_profile': self.get_speed_profile(),
            'risk_tolerance': self.get_risk_tolerance(),
            'near_miss_count': len(self.near_miss_events)
        }
    
    def reset(self):
        """Reset all metrics."""
        self.reaction_times.clear()
        self.following_distances.clear()
        self.lane_changes.clear()
        self.speeds.clear()
        self.near_miss_events.clear()
        self.risk_scores.clear()
        self.pending_alerts.clear()
        
        self.session_start = None
        self.session_duration = 0.0
        self.total_distance = 0.0
        self.prev_lane_id = None
        self.prev_speed = 0.0
        self.prev_timestamp = 0.0
        
        logger.info("Metrics tracker reset")
