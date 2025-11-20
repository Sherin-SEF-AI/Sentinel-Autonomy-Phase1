"""Time-to-collision (TTC) calculator."""

import logging
from typing import Dict, Any
import numpy as np

from src.core.data_structures import Detection3D


class TTCCalculator:
    """Calculates time-to-collision for detected objects."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TTC calculator.
        
        Args:
            config: Configuration dictionary with ttc_calculation settings
        """
        self.logger = logging.getLogger(__name__)
        self.method = config.get('method', 'constant_velocity')
        self.safety_margin = config.get('safety_margin', 1.5)  # meters
        
        self.logger.info(f"TTC Calculator initialized: method={self.method}, safety_margin={self.safety_margin}m")
    
    def calculate_ttc(self, detection: Detection3D, ego_velocity: float = 0.0) -> float:
        """
        Calculate time-to-collision for a detected object.
        
        Args:
            detection: 3D detection with position and velocity
            ego_velocity: Ego vehicle velocity in m/s (from CAN bus)
            
        Returns:
            TTC in seconds (inf if no collision predicted)
        """
        if self.method == 'constant_velocity':
            return self._calculate_constant_velocity_ttc(detection, ego_velocity)
        else:
            self.logger.warning(f"Unknown TTC method: {self.method}, using constant_velocity")
            return self._calculate_constant_velocity_ttc(detection, ego_velocity)
    
    def _calculate_constant_velocity_ttc(self, detection: Detection3D, ego_velocity: float = 0.0) -> float:
        """
        Calculate TTC using constant velocity model.
        
        Args:
            detection: 3D detection with position and velocity
            ego_velocity: Ego vehicle velocity in m/s (from CAN bus)
            
        Returns:
            TTC in seconds (inf if no collision predicted)
        """
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity
        
        # Position relative to vehicle (vehicle is at origin)
        position = np.array([x, y, z])
        velocity = np.array([vx, vy, vz])
        
        # Add ego vehicle velocity to get relative velocity
        # Ego vehicle moves forward (positive x direction)
        ego_velocity_vector = np.array([ego_velocity, 0, 0])
        relative_velocity = velocity - ego_velocity_vector
        
        # Calculate distance to vehicle (considering safety margin)
        distance = np.linalg.norm(position[:2])  # Use x, y only (ignore z)
        distance_with_margin = max(0, distance - self.safety_margin)
        
        # Calculate relative velocity magnitude
        velocity_magnitude = np.linalg.norm(relative_velocity[:2])
        
        if velocity_magnitude < 0.1:  # Nearly stationary relative to ego
            return float('inf')
        
        # Calculate if object is approaching
        # Dot product of position and relative velocity vectors
        # Negative means approaching
        closing_rate = -np.dot(position[:2], relative_velocity[:2]) / distance if distance > 0 else 0
        
        if closing_rate <= 0:  # Not approaching
            return float('inf')
        
        # TTC = distance / closing_rate
        ttc = distance_with_margin / closing_rate
        
        return float(ttc)
    
    def is_collision_imminent(self, ttc: float, threshold: float = 3.0) -> bool:
        """
        Check if collision is imminent based on TTC threshold.
        
        Args:
            ttc: Time-to-collision in seconds
            threshold: TTC threshold in seconds
            
        Returns:
            True if collision is imminent
        """
        return ttc < threshold and ttc != float('inf')
    
    def calculate_ttc_for_all(self, detections: list) -> Dict[int, float]:
        """
        Calculate TTC for all detections.
        
        Args:
            detections: List of Detection3D objects
            
        Returns:
            Dictionary mapping track_id to TTC
        """
        ttc_map = {}
        
        for detection in detections:
            ttc = self.calculate_ttc(detection)
            ttc_map[detection.track_id] = ttc
        
        return ttc_map
