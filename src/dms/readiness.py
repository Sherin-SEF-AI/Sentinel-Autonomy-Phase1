"""Driver readiness score calculation."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ReadinessCalculator:
    """Calculate driver readiness score from alertness, attention, and distraction."""
    
    # Weights for readiness score calculation
    ALERTNESS_WEIGHT = 0.4
    ATTENTION_WEIGHT = 0.3
    DISTRACTION_WEIGHT = 0.3
    
    def __init__(self):
        """Initialize readiness calculator."""
        logger.info("ReadinessCalculator initialized")
    
    def calculate_readiness(self, drowsiness: Dict[str, Any], 
                          gaze: Dict[str, Any],
                          distraction: Dict[str, Any]) -> float:
        """
        Calculate driver readiness score.
        
        Score = alertness (0.4) + attention (0.3) + distraction (0.3)
        
        Args:
            drowsiness: Drowsiness detection results
            gaze: Gaze estimation results
            distraction: Distraction classification results
            
        Returns:
            Readiness score (0-100)
        """
        try:
            # Calculate alertness component (inverse of drowsiness)
            drowsiness_score = drowsiness.get('score', 0.0)
            alertness = (1.0 - drowsiness_score) * 100
            
            # Calculate attention component (based on gaze)
            attention = self._calculate_attention_score(gaze)
            
            # Calculate distraction component (inverse of distraction)
            distraction_penalty = self._calculate_distraction_penalty(distraction)
            
            # Weighted sum
            readiness = (
                self.ALERTNESS_WEIGHT * alertness +
                self.ATTENTION_WEIGHT * attention +
                self.DISTRACTION_WEIGHT * distraction_penalty
            )
            
            # Clamp to [0, 100]
            readiness = max(0.0, min(100.0, readiness))
            
            return float(readiness)
        
        except Exception as e:
            logger.error(f"Readiness calculation failed: {e}")
            return 50.0  # Default neutral score
    
    def _calculate_attention_score(self, gaze: Dict[str, Any]) -> float:
        """
        Calculate attention score from gaze.
        
        Args:
            gaze: Gaze dictionary with attention_zone
            
        Returns:
            Attention score (0-100)
        """
        attention_zone = gaze.get('attention_zone', 'front')
        
        # Score based on attention zone
        zone_scores = {
            'front': 100.0,           # Full attention
            'front_left': 90.0,       # Mostly attentive
            'front_right': 90.0,      # Mostly attentive
            'left': 60.0,             # Partially attentive
            'right': 60.0,            # Partially attentive
            'rear_left': 30.0,        # Low attention
            'rear_right': 30.0,       # Low attention
            'rear': 20.0              # Very low attention
        }
        
        return zone_scores.get(attention_zone, 50.0)
    
    def _calculate_distraction_penalty(self, distraction: Dict[str, Any]) -> float:
        """
        Calculate distraction penalty score.
        
        Args:
            distraction: Distraction dictionary with type and duration
            
        Returns:
            Distraction penalty score (0-100, higher is better)
        """
        distraction_type = distraction.get('type', 'safe_driving')
        duration = distraction.get('duration', 0.0)
        eyes_off_road = distraction.get('eyes_off_road', False)
        
        # Base scores for distraction types
        type_scores = {
            'safe_driving': 100.0,
            'adjusting_controls': 80.0,
            'looking_at_passenger': 60.0,
            'eyes_off_road': 40.0,
            'phone_usage': 20.0,
            'hands_off_wheel': 10.0
        }
        
        base_score = type_scores.get(distraction_type, 50.0)
        
        # Apply duration penalty
        if duration > 0:
            # Reduce score based on duration (max 50% reduction)
            duration_penalty = min(duration / 10.0, 0.5)
            base_score *= (1.0 - duration_penalty)
        
        # Apply eyes off road penalty
        if eyes_off_road:
            base_score *= 0.5
        
        return base_score
