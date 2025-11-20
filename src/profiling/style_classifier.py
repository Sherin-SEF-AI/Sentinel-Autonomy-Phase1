"""
Driving Style Classifier

Classifies driving style as aggressive, normal, or cautious based on behavior metrics.
"""

import numpy as np
from typing import Dict, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DrivingStyle(Enum):
    """Driving style categories."""
    AGGRESSIVE = "aggressive"
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    UNKNOWN = "unknown"


class DrivingStyleClassifier:
    """
    Classifies driving style based on behavioral metrics.
    
    Classification is based on:
    - Reaction time (faster = more aggressive)
    - Following distance (shorter = more aggressive)
    - Lane change frequency (higher = more aggressive)
    - Speed profile (higher speeds = more aggressive)
    - Risk tolerance (higher = more aggressive)
    """
    
    def __init__(self, config: dict):
        """
        Initialize driving style classifier.
        
        Args:
            config: Configuration dictionary with thresholds
        """
        self.config = config
        
        # Classification thresholds
        self.thresholds = {
            'reaction_time': {
                'aggressive': 0.8,  # < 0.8s
                'cautious': 1.5     # > 1.5s
            },
            'following_distance': {
                'aggressive': 15.0,  # < 15m
                'cautious': 35.0     # > 35m
            },
            'lane_change_freq': {
                'aggressive': 8.0,   # > 8 per hour
                'cautious': 2.0      # < 2 per hour
            },
            'speed_variance': {
                'aggressive': 5.0,   # > 5 m/s std
                'cautious': 2.0      # < 2 m/s std
            },
            'risk_tolerance': {
                'aggressive': 0.6,   # > 0.6
                'cautious': 0.3      # < 0.3
            }
        }
        
        # Feature weights for classification
        self.feature_weights = {
            'reaction_time': 0.2,
            'following_distance': 0.25,
            'lane_change_freq': 0.15,
            'speed_variance': 0.15,
            'risk_tolerance': 0.25
        }
        
        # History for temporal smoothing
        self.classification_history = []
        self.max_history = 10
        
        logger.info("DrivingStyleClassifier initialized")
    
    def classify(self, metrics: Dict) -> DrivingStyle:
        """
        Classify driving style from metrics.
        
        Args:
            metrics: Dictionary with driver metrics from MetricsTracker
        
        Returns:
            DrivingStyle enum value
        """
        # Extract relevant metrics
        reaction_time = metrics.get('reaction_time', {}).get('mean', None)
        following_distance = metrics.get('following_distance', {}).get('mean', None)
        lane_change_freq = metrics.get('lane_change_frequency', None)
        speed_variance = metrics.get('speed_profile', {}).get('std', None)
        risk_tolerance = metrics.get('risk_tolerance', None)
        
        # Check if we have enough data
        if not self._has_sufficient_data(metrics):
            logger.warning("Insufficient data for style classification")
            return DrivingStyle.UNKNOWN
        
        # Calculate feature scores (-1: cautious, 0: normal, +1: aggressive)
        scores = {}
        
        if reaction_time is not None:
            scores['reaction_time'] = self._score_reaction_time(reaction_time)
        
        if following_distance is not None:
            scores['following_distance'] = self._score_following_distance(following_distance)
        
        if lane_change_freq is not None:
            scores['lane_change_freq'] = self._score_lane_change_freq(lane_change_freq)
        
        if speed_variance is not None:
            scores['speed_variance'] = self._score_speed_variance(speed_variance)
        
        if risk_tolerance is not None:
            scores['risk_tolerance'] = self._score_risk_tolerance(risk_tolerance)
        
        # Weighted average of scores
        total_weight = sum(self.feature_weights[k] for k in scores.keys())
        if total_weight == 0:
            return DrivingStyle.UNKNOWN
        
        weighted_score = sum(
            scores[k] * self.feature_weights[k] 
            for k in scores.keys()
        ) / total_weight
        
        # Classify based on weighted score
        if weighted_score > 0.3:
            style = DrivingStyle.AGGRESSIVE
        elif weighted_score < -0.3:
            style = DrivingStyle.CAUTIOUS
        else:
            style = DrivingStyle.NORMAL
        
        # Add to history for temporal smoothing
        self.classification_history.append(style)
        if len(self.classification_history) > self.max_history:
            self.classification_history.pop(0)
        
        # Apply temporal smoothing
        smoothed_style = self._smooth_classification()
        
        logger.info(f"Driving style classified: {smoothed_style.value} (score={weighted_score:.3f})")
        
        return smoothed_style
    
    def _has_sufficient_data(self, metrics: Dict) -> bool:
        """Check if we have sufficient data for classification."""
        reaction_count = metrics.get('reaction_time', {}).get('count', 0)
        distance_count = metrics.get('following_distance', {}).get('count', 0)
        
        # Need at least some reaction time and following distance data
        return reaction_count >= 3 and distance_count >= 10
    
    def _score_reaction_time(self, reaction_time: float) -> float:
        """Score reaction time feature."""
        if reaction_time < self.thresholds['reaction_time']['aggressive']:
            return 1.0  # Aggressive (very fast reactions)
        elif reaction_time > self.thresholds['reaction_time']['cautious']:
            return -1.0  # Cautious (slow reactions)
        else:
            # Linear interpolation
            mid = (self.thresholds['reaction_time']['aggressive'] + 
                   self.thresholds['reaction_time']['cautious']) / 2
            if reaction_time < mid:
                return (mid - reaction_time) / (mid - self.thresholds['reaction_time']['aggressive'])
            else:
                return -(reaction_time - mid) / (self.thresholds['reaction_time']['cautious'] - mid)
    
    def _score_following_distance(self, distance: float) -> float:
        """Score following distance feature."""
        if distance < self.thresholds['following_distance']['aggressive']:
            return 1.0  # Aggressive (short following distance)
        elif distance > self.thresholds['following_distance']['cautious']:
            return -1.0  # Cautious (long following distance)
        else:
            # Linear interpolation
            mid = (self.thresholds['following_distance']['aggressive'] + 
                   self.thresholds['following_distance']['cautious']) / 2
            if distance < mid:
                return (mid - distance) / (mid - self.thresholds['following_distance']['aggressive'])
            else:
                return -(distance - mid) / (self.thresholds['following_distance']['cautious'] - mid)
    
    def _score_lane_change_freq(self, freq: float) -> float:
        """Score lane change frequency feature."""
        if freq > self.thresholds['lane_change_freq']['aggressive']:
            return 1.0  # Aggressive (frequent lane changes)
        elif freq < self.thresholds['lane_change_freq']['cautious']:
            return -1.0  # Cautious (rare lane changes)
        else:
            # Linear interpolation
            mid = (self.thresholds['lane_change_freq']['aggressive'] + 
                   self.thresholds['lane_change_freq']['cautious']) / 2
            if freq > mid:
                return (freq - mid) / (self.thresholds['lane_change_freq']['aggressive'] - mid)
            else:
                return -(mid - freq) / (mid - self.thresholds['lane_change_freq']['cautious'])
    
    def _score_speed_variance(self, variance: float) -> float:
        """Score speed variance feature."""
        if variance > self.thresholds['speed_variance']['aggressive']:
            return 1.0  # Aggressive (high speed variance)
        elif variance < self.thresholds['speed_variance']['cautious']:
            return -1.0  # Cautious (low speed variance)
        else:
            # Linear interpolation
            mid = (self.thresholds['speed_variance']['aggressive'] + 
                   self.thresholds['speed_variance']['cautious']) / 2
            if variance > mid:
                return (variance - mid) / (self.thresholds['speed_variance']['aggressive'] - mid)
            else:
                return -(mid - variance) / (mid - self.thresholds['speed_variance']['cautious'])
    
    def _score_risk_tolerance(self, tolerance: float) -> float:
        """Score risk tolerance feature."""
        if tolerance > self.thresholds['risk_tolerance']['aggressive']:
            return 1.0  # Aggressive (high risk tolerance)
        elif tolerance < self.thresholds['risk_tolerance']['cautious']:
            return -1.0  # Cautious (low risk tolerance)
        else:
            # Linear interpolation
            mid = (self.thresholds['risk_tolerance']['aggressive'] + 
                   self.thresholds['risk_tolerance']['cautious']) / 2
            if tolerance > mid:
                return (tolerance - mid) / (self.thresholds['risk_tolerance']['aggressive'] - mid)
            else:
                return -(mid - tolerance) / (mid - self.thresholds['risk_tolerance']['cautious'])
    
    def _smooth_classification(self) -> DrivingStyle:
        """
        Apply temporal smoothing to classification.
        
        Returns most common style in recent history.
        """
        if not self.classification_history:
            return DrivingStyle.UNKNOWN
        
        # Count occurrences
        counts = {}
        for style in self.classification_history:
            counts[style] = counts.get(style, 0) + 1
        
        # Return most common
        return max(counts, key=counts.get)
    
    def get_style_description(self, style: DrivingStyle) -> str:
        """
        Get human-readable description of driving style.
        
        Args:
            style: DrivingStyle enum
        
        Returns:
            Description string
        """
        descriptions = {
            DrivingStyle.AGGRESSIVE: "Aggressive driving style with fast reactions, short following distances, and higher risk tolerance.",
            DrivingStyle.NORMAL: "Normal driving style with balanced behavior and moderate risk tolerance.",
            DrivingStyle.CAUTIOUS: "Cautious driving style with longer following distances, fewer lane changes, and lower risk tolerance.",
            DrivingStyle.UNKNOWN: "Insufficient data to determine driving style."
        }
        return descriptions.get(style, "Unknown driving style")
    
    def reset(self):
        """Reset classification history."""
        self.classification_history.clear()
        logger.info("Driving style classifier reset")
