"""Risk calculation and assessment."""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np

from src.core.data_structures import Detection3D, DriverState, Hazard, Risk


class RiskCalculator:
    """Calculates base and contextual risk scores."""
    
    # Vulnerability scores by object type
    VULNERABILITY_SCORES = {
        'pedestrian': 1.0,
        'cyclist': 0.8,
        'vehicle': 0.4,
        'traffic_sign': 0.1,
        'traffic_light': 0.1
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk calculator.
        
        Args:
            config: Configuration dictionary with risk assessment settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Base risk weights
        weights = config.get('base_risk_weights', {})
        self.weight_ttc = weights.get('ttc', 0.4)
        self.weight_trajectory = weights.get('trajectory_conflict', 0.3)
        self.weight_vulnerability = weights.get('vulnerability', 0.2)
        self.weight_speed = weights.get('relative_speed', 0.1)
        
        # Thresholds
        thresholds = config.get('thresholds', {})
        self.hazard_threshold = thresholds.get('hazard_detection', 0.3)
        self.intervention_threshold = thresholds.get('intervention', 0.7)
        self.critical_threshold = thresholds.get('critical', 0.9)
        
        self.logger.info(
            f"Risk Calculator initialized: weights(ttc={self.weight_ttc}, "
            f"trajectory={self.weight_trajectory}, vulnerability={self.weight_vulnerability}, "
            f"speed={self.weight_speed})"
        )
    
    def calculate_base_risk(
        self,
        detection: Detection3D,
        ttc: float,
        trajectory_conflict: float,
        zone: str
    ) -> float:
        """
        Calculate base risk score for a detected object.
        
        Args:
            detection: 3D detection
            ttc: Time-to-collision in seconds
            trajectory_conflict: Trajectory conflict score (0-1)
            zone: Spatial zone of the object
            
        Returns:
            Base risk score (0-1)
        """
        # TTC component (inverse relationship)
        if ttc == float('inf'):
            ttc_score = 0.0
        else:
            # Normalize TTC: high risk for TTC < 3s, low risk for TTC > 10s
            ttc_score = np.clip(1.0 - (ttc / 10.0), 0.0, 1.0)
        
        # Trajectory conflict component (already 0-1)
        trajectory_score = trajectory_conflict
        
        # Vulnerability component
        vulnerability_score = self._get_vulnerability_score(detection.class_name)
        
        # Relative speed component
        speed_score = self._calculate_speed_score(detection.velocity)
        
        # Weighted sum
        base_risk = (
            self.weight_ttc * ttc_score +
            self.weight_trajectory * trajectory_score +
            self.weight_vulnerability * vulnerability_score +
            self.weight_speed * speed_score
        )
        
        return float(np.clip(base_risk, 0.0, 1.0))
    
    def _get_vulnerability_score(self, object_type: str) -> float:
        """
        Get vulnerability score for object type.
        
        Args:
            object_type: Object class name
            
        Returns:
            Vulnerability score (0-1)
        """
        return self.VULNERABILITY_SCORES.get(object_type.lower(), 0.5)
    
    def _calculate_speed_score(self, velocity: Tuple[float, float, float]) -> float:
        """
        Calculate risk score based on relative speed.
        
        Args:
            velocity: Velocity vector (vx, vy, vz)
            
        Returns:
            Speed score (0-1)
        """
        speed = np.linalg.norm(velocity)
        
        # Normalize speed: high risk for speed > 10 m/s, low risk for speed < 2 m/s
        speed_score = np.clip((speed - 2.0) / 8.0, 0.0, 1.0)
        
        return float(speed_score)
    
    def calculate_contextual_risk(
        self,
        base_risk: float,
        driver_aware: bool,
        driver_readiness: float
    ) -> float:
        """
        Calculate contextual risk score considering driver state.
        
        Args:
            base_risk: Base risk score (0-1)
            driver_aware: Whether driver is aware of the hazard
            driver_readiness: Driver readiness score (0-100)
            
        Returns:
            Contextual risk score (0-1, capped)
        """
        # Awareness penalty
        awareness_penalty = 1.0 if driver_aware else 2.0
        
        # Capacity factor (based on driver readiness)
        # Lower readiness = higher capacity factor
        capacity_factor = 2.0 - (driver_readiness / 100.0)
        
        # Contextual risk (can exceed 1.0 before clipping)
        contextual_risk = base_risk * awareness_penalty * capacity_factor
        
        # Clip to [0, 1] range for final score
        return float(np.clip(contextual_risk, 0.0, 1.0))
    
    def categorize_urgency(self, contextual_risk: float) -> str:
        """
        Categorize risk urgency level.
        
        Args:
            contextual_risk: Contextual risk score (0-1)
            
        Returns:
            Urgency level: 'low', 'medium', 'high', or 'critical'
        """
        if contextual_risk >= self.critical_threshold:
            return 'critical'
        elif contextual_risk >= self.intervention_threshold:
            return 'high'
        elif contextual_risk >= self.hazard_threshold:
            return 'medium'
        else:
            return 'low'
    
    def create_hazard(
        self,
        detection: Detection3D,
        ttc: float,
        trajectory: List[Tuple[float, float, float]],
        zone: str,
        base_risk: float
    ) -> Hazard:
        """
        Create Hazard object from detection and risk assessment.
        
        Args:
            detection: 3D detection
            ttc: Time-to-collision
            trajectory: Predicted trajectory waypoints
            zone: Spatial zone
            base_risk: Base risk score
            
        Returns:
            Hazard object
        """
        x, y, z, w, h, l, theta = detection.bbox_3d
        
        hazard = Hazard(
            object_id=detection.track_id,
            type=detection.class_name,
            position=(x, y, z),
            velocity=detection.velocity,
            trajectory=trajectory,
            ttc=ttc,
            zone=zone,
            base_risk=base_risk
        )
        
        return hazard
    
    def create_risk(
        self,
        hazard: Hazard,
        contextual_score: float,
        driver_aware: bool
    ) -> Risk:
        """
        Create Risk object from hazard and contextual assessment.
        
        Args:
            hazard: Hazard object
            contextual_score: Contextual risk score
            driver_aware: Whether driver is aware
            
        Returns:
            Risk object
        """
        urgency = self.categorize_urgency(contextual_score)
        intervention_needed = contextual_score >= self.intervention_threshold
        
        risk = Risk(
            hazard=hazard,
            contextual_score=contextual_score,
            driver_aware=driver_aware,
            urgency=urgency,
            intervention_needed=intervention_needed
        )
        
        return risk
