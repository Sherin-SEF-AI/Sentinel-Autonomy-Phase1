"""
Threshold Adapter for Personalized Safety Settings

Adapts safety thresholds based on driver behavior and driving style.
"""

import numpy as np
from typing import Dict
from .style_classifier import DrivingStyle
import logging

logger = logging.getLogger(__name__)


class ThresholdAdapter:
    """
    Adapts safety thresholds based on driver profile.
    
    Personalizes:
    - TTC (Time-To-Collision) threshold based on reaction time
    - Following distance threshold based on driving style
    - Alert sensitivity based on risk tolerance
    
    Always applies 1.5x safety margin to ensure safety.
    """
    
    def __init__(self, config: dict):
        """
        Initialize threshold adapter.
        
        Args:
            config: Configuration dictionary with base thresholds
        """
        self.config = config
        
        # Base thresholds (from system config)
        self.base_ttc_threshold = config.get('base_ttc_threshold', 2.0)  # seconds
        self.base_following_distance = config.get('base_following_distance', 25.0)  # meters
        self.base_alert_sensitivity = config.get('base_alert_sensitivity', 0.7)  # 0-1
        
        # Safety margin multiplier
        self.safety_margin = 1.5
        
        # Current adapted thresholds
        self.adapted_ttc_threshold = self.base_ttc_threshold
        self.adapted_following_distance = self.base_following_distance
        self.adapted_alert_sensitivity = self.base_alert_sensitivity
        
        logger.info(f"ThresholdAdapter initialized with base TTC={self.base_ttc_threshold}s, "
                   f"following_distance={self.base_following_distance}m, "
                   f"alert_sensitivity={self.base_alert_sensitivity}")
    
    def adapt_thresholds(self, 
                        metrics: Dict,
                        driving_style: DrivingStyle) -> Dict[str, float]:
        """
        Adapt thresholds based on driver metrics and style.
        
        Args:
            metrics: Driver metrics from MetricsTracker
            driving_style: Classified driving style
        
        Returns:
            Dictionary with adapted thresholds
        """
        # Adapt TTC threshold based on reaction time
        reaction_time_mean = metrics.get('reaction_time', {}).get('mean', None)
        if reaction_time_mean is not None:
            self.adapted_ttc_threshold = self._adapt_ttc_threshold(reaction_time_mean)
        
        # Adapt following distance based on driving style
        self.adapted_following_distance = self._adapt_following_distance(driving_style)
        
        # Adapt alert sensitivity based on risk tolerance
        risk_tolerance = metrics.get('risk_tolerance', 0.5)
        self.adapted_alert_sensitivity = self._adapt_alert_sensitivity(risk_tolerance)
        
        adapted = {
            'ttc_threshold': self.adapted_ttc_threshold,
            'following_distance': self.adapted_following_distance,
            'alert_sensitivity': self.adapted_alert_sensitivity
        }
        
        logger.info(f"Thresholds adapted: TTC={adapted['ttc_threshold']:.2f}s, "
                   f"following_distance={adapted['following_distance']:.1f}m, "
                   f"alert_sensitivity={adapted['alert_sensitivity']:.2f}")
        
        return adapted
    
    def _adapt_ttc_threshold(self, reaction_time: float) -> float:
        """
        Adapt TTC threshold based on driver reaction time.
        
        Formula: TTC_threshold = reaction_time * safety_margin
        
        Args:
            reaction_time: Mean reaction time in seconds
        
        Returns:
            Adapted TTC threshold in seconds
        """
        # Apply safety margin
        adapted_ttc = reaction_time * self.safety_margin
        
        # Clamp to reasonable range [1.5, 4.0] seconds
        adapted_ttc = np.clip(adapted_ttc, 1.5, 4.0)
        
        logger.debug(f"TTC threshold adapted from {self.base_ttc_threshold:.2f}s to {adapted_ttc:.2f}s "
                    f"based on reaction time {reaction_time:.2f}s")
        
        return float(adapted_ttc)
    
    def _adapt_following_distance(self, driving_style: DrivingStyle) -> float:
        """
        Adapt following distance threshold based on driving style.
        
        Args:
            driving_style: Classified driving style
        
        Returns:
            Adapted following distance in meters
        """
        # Style-based multipliers
        style_multipliers = {
            DrivingStyle.AGGRESSIVE: 0.8,   # Shorter distance for aggressive drivers
            DrivingStyle.NORMAL: 1.0,       # Base distance
            DrivingStyle.CAUTIOUS: 1.3,     # Longer distance for cautious drivers
            DrivingStyle.UNKNOWN: 1.0       # Default to base
        }
        
        multiplier = style_multipliers.get(driving_style, 1.0)
        adapted_distance = self.base_following_distance * multiplier
        
        # Clamp to reasonable range [15, 40] meters
        adapted_distance = np.clip(adapted_distance, 15.0, 40.0)
        
        logger.debug(f"Following distance adapted from {self.base_following_distance:.1f}m to {adapted_distance:.1f}m "
                    f"for {driving_style.value} style")
        
        return float(adapted_distance)
    
    def _adapt_alert_sensitivity(self, risk_tolerance: float) -> float:
        """
        Adapt alert sensitivity based on risk tolerance.
        
        Higher risk tolerance -> lower sensitivity (fewer alerts)
        Lower risk tolerance -> higher sensitivity (more alerts)
        
        Args:
            risk_tolerance: Risk tolerance score (0-1)
        
        Returns:
            Adapted alert sensitivity (0-1)
        """
        # Inverse relationship: high tolerance = low sensitivity
        # Formula: sensitivity = base * (1.5 - risk_tolerance)
        adapted_sensitivity = self.base_alert_sensitivity * (1.5 - risk_tolerance)
        
        # Clamp to range [0.5, 0.9]
        adapted_sensitivity = np.clip(adapted_sensitivity, 0.5, 0.9)
        
        logger.debug(f"Alert sensitivity adapted from {self.base_alert_sensitivity:.2f} to {adapted_sensitivity:.2f} "
                    f"based on risk tolerance {risk_tolerance:.2f}")
        
        return float(adapted_sensitivity)
    
    def get_adapted_thresholds(self) -> Dict[str, float]:
        """
        Get current adapted thresholds.
        
        Returns:
            Dictionary with current thresholds
        """
        return {
            'ttc_threshold': self.adapted_ttc_threshold,
            'following_distance': self.adapted_following_distance,
            'alert_sensitivity': self.adapted_alert_sensitivity
        }
    
    def reset_to_defaults(self):
        """Reset thresholds to base values."""
        self.adapted_ttc_threshold = self.base_ttc_threshold
        self.adapted_following_distance = self.base_following_distance
        self.adapted_alert_sensitivity = self.base_alert_sensitivity
        logger.info("Thresholds reset to defaults")
    
    def get_safety_margin_info(self) -> Dict[str, float]:
        """
        Get information about safety margins applied.
        
        Returns:
            Dictionary with safety margin details
        """
        return {
            'safety_margin_multiplier': self.safety_margin,
            'ttc_margin': self.adapted_ttc_threshold - (self.adapted_ttc_threshold / self.safety_margin),
            'base_ttc': self.base_ttc_threshold,
            'adapted_ttc': self.adapted_ttc_threshold
        }
