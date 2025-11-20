"""Advanced risk assessment with trajectory prediction integration."""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from src.core.data_structures import Detection3D, DriverState, Hazard, Risk
from src.intelligence.advanced_trajectory import (
    AdvancedTrajectoryPredictor,
    CollisionProbabilityCalculator,
    Trajectory
)
from src.intelligence.risk import RiskCalculator


class AdvancedRiskAssessor:
    """Advanced risk assessment with trajectory prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize advanced risk assessor.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.trajectory_predictor = AdvancedTrajectoryPredictor(
            config.get('trajectory_prediction', {})
        )
        
        self.collision_calculator = CollisionProbabilityCalculator()
        
        self.risk_calculator = RiskCalculator(config)
        
        # Configuration
        self.use_collision_probability = config.get('trajectory_prediction', {}).get(
            'use_collision_probability', True
        )
        self.collision_weight = config.get('trajectory_prediction', {}).get(
            'collision_probability_weight', 0.4
        )
        
        self.logger.info(
            f"Advanced Risk Assessor initialized: "
            f"use_collision_prob={self.use_collision_probability}"
        )
    
    def assess_hazards_with_trajectories(
        self,
        detections: List[Detection3D],
        ego_trajectory: Optional[Trajectory] = None,
        driver_state: Optional[DriverState] = None
    ) -> Tuple[List[Hazard], Dict[int, List[Trajectory]], Dict[int, float]]:
        """
        Assess hazards with advanced trajectory prediction.
        
        Args:
            detections: List of detected objects
            ego_trajectory: Optional ego vehicle trajectory
            driver_state: Optional driver state
            
        Returns:
            Tuple of (hazards, object_trajectories, collision_probabilities)
        """
        # Predict trajectories for all objects
        object_trajectories = self.trajectory_predictor.predict_all(detections)
        
        # Calculate collision probabilities if ego trajectory provided
        collision_probabilities = {}
        if ego_trajectory and self.use_collision_probability:
            # Get object sizes
            object_sizes = {}
            for detection in detections:
                _, _, _, w, h, l, _ = detection.bbox_3d
                # Use maximum dimension as size
                object_sizes[detection.track_id] = max(w, l)
            
            collision_probabilities = self.collision_calculator.calculate_all_collision_probabilities(
                ego_trajectory,
                object_trajectories,
                object_sizes
            )
        
        # Create hazards with enhanced risk scores
        hazards = []
        
        for detection in detections:
            track_id = detection.track_id
            
            # Get best trajectory for this object
            trajectories = object_trajectories.get(track_id, [])
            if not trajectories:
                continue
            
            best_trajectory = trajectories[0]  # Highest confidence
            
            # Get collision probability
            collision_prob = 0.0
            if track_id in collision_probabilities:
                collision_prob, _, _ = collision_probabilities[track_id]
            
            # Calculate enhanced base risk
            base_risk = self._calculate_enhanced_base_risk(
                detection,
                best_trajectory,
                collision_prob
            )
            
            # Create hazard
            hazard = Hazard(
                object_id=track_id,
                type=detection.class_name,
                position=tuple(detection.bbox_3d[:3]),
                velocity=detection.velocity,
                trajectory=best_trajectory.points,
                ttc=self._calculate_ttc_from_trajectory(best_trajectory, ego_trajectory),
                zone=self._determine_zone(detection),
                base_risk=base_risk
            )
            
            hazards.append(hazard)
        
        return hazards, object_trajectories, collision_probabilities
    
    def _calculate_enhanced_base_risk(
        self,
        detection: Detection3D,
        trajectory: Trajectory,
        collision_prob: float
    ) -> float:
        """
        Calculate enhanced base risk using trajectory and collision probability.
        
        Args:
            detection: Detection object
            trajectory: Predicted trajectory
            collision_prob: Collision probability
            
        Returns:
            Enhanced base risk score (0-1)
        """
        # Calculate traditional base risk components
        ttc = self._estimate_ttc(detection)
        trajectory_conflict = self._estimate_trajectory_conflict(trajectory)
        zone = self._determine_zone(detection)
        
        # Get traditional base risk
        base_risk = self.risk_calculator.calculate_base_risk(
            detection,
            ttc,
            trajectory_conflict,
            zone
        )
        
        # Enhance with collision probability if available
        if self.use_collision_probability and collision_prob > 0:
            # Blend traditional risk with collision probability
            enhanced_risk = (
                (1.0 - self.collision_weight) * base_risk +
                self.collision_weight * collision_prob
            )
            return float(np.clip(enhanced_risk, 0.0, 1.0))
        
        return base_risk
    
    def _estimate_ttc(self, detection: Detection3D) -> float:
        """Estimate time-to-collision for detection."""
        x, y, z, _, _, _, _ = detection.bbox_3d
        vx, vy, vz = detection.velocity
        
        # Distance to object
        distance = np.linalg.norm([x, y, z])
        
        # Relative velocity (assuming ego is stationary for simplicity)
        relative_velocity = np.linalg.norm([vx, vy, vz])
        
        if relative_velocity < 0.1:
            return float('inf')
        
        # Simple TTC calculation
        ttc = distance / relative_velocity
        
        return float(ttc)
    
    def _estimate_trajectory_conflict(self, trajectory: Trajectory) -> float:
        """Estimate trajectory conflict score."""
        # Check if trajectory crosses ego path
        # Simplified: check if any point is close to origin
        min_distance = float('inf')
        
        for point in trajectory.points:
            distance = np.linalg.norm(point)
            min_distance = min(min_distance, distance)
        
        # Convert distance to conflict score
        conflict_score = np.exp(-min_distance / 5.0)
        
        return float(np.clip(conflict_score, 0.0, 1.0))
    
    def _determine_zone(self, detection: Detection3D) -> str:
        """Determine spatial zone for detection."""
        x, y, z, _, _, _, _ = detection.bbox_3d
        
        # Calculate angle from vehicle forward direction
        angle = np.arctan2(y, x)  # radians
        angle_deg = np.degrees(angle)
        
        # Normalize to [0, 360)
        if angle_deg < 0:
            angle_deg += 360
        
        # Determine zone (8 zones, 45 degrees each)
        if angle_deg < 22.5 or angle_deg >= 337.5:
            return 'front'
        elif angle_deg < 67.5:
            return 'front-left'
        elif angle_deg < 112.5:
            return 'left'
        elif angle_deg < 157.5:
            return 'rear-left'
        elif angle_deg < 202.5:
            return 'rear'
        elif angle_deg < 247.5:
            return 'rear-right'
        elif angle_deg < 292.5:
            return 'right'
        else:
            return 'front-right'
    
    def _calculate_ttc_from_trajectory(
        self,
        object_trajectory: Trajectory,
        ego_trajectory: Optional[Trajectory]
    ) -> float:
        """Calculate TTC from trajectories."""
        if not ego_trajectory:
            # Fallback to simple distance-based TTC
            if object_trajectory.points:
                first_point = np.array(object_trajectory.points[0])
                distance = np.linalg.norm(first_point)
                
                # Estimate velocity from trajectory
                if len(object_trajectory.points) > 1:
                    second_point = np.array(object_trajectory.points[1])
                    dt = object_trajectory.timestamps[1] - object_trajectory.timestamps[0]
                    velocity = np.linalg.norm(second_point - first_point) / dt
                    
                    if velocity > 0.1:
                        return distance / velocity
            
            return float('inf')
        
        # Find minimum distance between trajectories
        min_distance = float('inf')
        min_time = float('inf')
        
        num_steps = min(len(object_trajectory.points), len(ego_trajectory.points))
        
        for i in range(num_steps):
            ego_pos = np.array(ego_trajectory.points[i])
            obj_pos = np.array(object_trajectory.points[i])
            
            distance = np.linalg.norm(ego_pos - obj_pos)
            
            if distance < min_distance:
                min_distance = distance
                min_time = object_trajectory.timestamps[i]
        
        # If minimum distance is within collision threshold, return time
        collision_threshold = 3.0  # meters
        if min_distance < collision_threshold:
            return float(min_time)
        
        return float('inf')
    
    def create_risks_with_trajectories(
        self,
        hazards: List[Hazard],
        driver_state: DriverState,
        attention_map: Dict[str, Any]
    ) -> List[Risk]:
        """
        Create Risk objects with trajectory-enhanced assessment.
        
        Args:
            hazards: List of hazards
            driver_state: Driver state
            attention_map: Attention mapping
            
        Returns:
            List of Risk objects
        """
        risks = []
        
        for hazard in hazards:
            # Check if driver is aware of this hazard
            driver_aware = self._is_driver_aware(hazard, attention_map)
            
            # Calculate contextual risk
            contextual_score = self.risk_calculator.calculate_contextual_risk(
                hazard.base_risk,
                driver_aware,
                driver_state.readiness_score
            )
            
            # Categorize urgency
            urgency = self.risk_calculator.categorize_urgency(contextual_score)
            
            # Determine if intervention needed
            intervention_needed = contextual_score >= self.risk_calculator.intervention_threshold
            
            risk = Risk(
                hazard=hazard,
                contextual_score=contextual_score,
                driver_aware=driver_aware,
                urgency=urgency,
                intervention_needed=intervention_needed
            )
            
            risks.append(risk)
        
        return risks
    
    def _is_driver_aware(
        self,
        hazard: Hazard,
        attention_map: Dict[str, Any]
    ) -> bool:
        """Check if driver is aware of hazard."""
        # Get driver's attention zone
        driver_zone = attention_map.get('current_zone', 'front')
        
        # Check if hazard zone matches driver attention
        return hazard.zone == driver_zone
