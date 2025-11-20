"""Trajectory prediction for detected objects."""

import logging
from typing import List, Tuple, Dict, Any
import numpy as np

from src.core.data_structures import Detection3D


class TrajectoryPredictor:
    """Predicts future trajectories of detected objects."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trajectory predictor.
        
        Args:
            config: Configuration dictionary with trajectory_prediction settings
        """
        self.logger = logging.getLogger(__name__)
        self.horizon = config.get('horizon', 3.0)  # seconds
        self.dt = config.get('dt', 0.1)  # time step in seconds
        self.method = config.get('method', 'linear')
        
        self.num_steps = int(self.horizon / self.dt)
        
        self.logger.info(
            f"Trajectory Predictor initialized: horizon={self.horizon}s, "
            f"dt={self.dt}s, steps={self.num_steps}, method={self.method}"
        )
    
    def predict(self, detection: Detection3D) -> List[Tuple[float, float, float]]:
        """
        Predict future trajectory for a detected object.
        
        Args:
            detection: 3D detection with position and velocity
            
        Returns:
            List of (x, y, z) waypoints representing predicted trajectory
        """
        if self.method == 'linear':
            return self._predict_linear(detection)
        else:
            self.logger.warning(f"Unknown prediction method: {self.method}, using linear")
            return self._predict_linear(detection)
    
    def _predict_linear(self, detection: Detection3D) -> List[Tuple[float, float, float]]:
        """
        Predict trajectory using linear (constant velocity) model.
        
        Args:
            detection: 3D detection with position and velocity
            
        Returns:
            List of trajectory waypoints
        """
        x, y, z, w, h, l, theta = detection.bbox_3d
        vx, vy, vz = detection.velocity
        
        # Current position
        position = np.array([x, y, z])
        velocity = np.array([vx, vy, vz])
        
        # Generate waypoints
        trajectory = []
        for step in range(self.num_steps + 1):
            t = step * self.dt
            # Linear prediction: position + velocity * time
            future_pos = position + velocity * t
            trajectory.append(tuple(future_pos.tolist()))
        
        return trajectory
    
    def predict_all(self, detections: List[Detection3D]) -> Dict[int, List[Tuple[float, float, float]]]:
        """
        Predict trajectories for all detections.
        
        Args:
            detections: List of Detection3D objects
            
        Returns:
            Dictionary mapping track_id to trajectory waypoints
        """
        trajectories = {}
        
        for detection in detections:
            trajectory = self.predict(detection)
            trajectories[detection.track_id] = trajectory
        
        return trajectories
    
    def check_trajectory_conflict(
        self, 
        trajectory1: List[Tuple[float, float, float]], 
        trajectory2: List[Tuple[float, float, float]],
        threshold: float = 2.0
    ) -> bool:
        """
        Check if two trajectories have potential conflict.
        
        Args:
            trajectory1: First trajectory waypoints
            trajectory2: Second trajectory waypoints
            threshold: Distance threshold for conflict (meters)
            
        Returns:
            True if trajectories conflict
        """
        # Check each time step
        min_steps = min(len(trajectory1), len(trajectory2))
        
        for i in range(min_steps):
            pos1 = np.array(trajectory1[i])
            pos2 = np.array(trajectory2[i])
            
            # Calculate distance at this time step
            distance = np.linalg.norm(pos1 - pos2)
            
            if distance < threshold:
                return True
        
        return False
    
    def calculate_trajectory_conflict_score(
        self,
        trajectory1: List[Tuple[float, float, float]],
        trajectory2: List[Tuple[float, float, float]]
    ) -> float:
        """
        Calculate conflict score between two trajectories.
        
        Args:
            trajectory1: First trajectory waypoints
            trajectory2: Second trajectory waypoints
            
        Returns:
            Conflict score (0-1, higher means more conflict)
        """
        if not trajectory1 or not trajectory2:
            return 0.0
        
        min_steps = min(len(trajectory1), len(trajectory2))
        
        # Calculate minimum distance across all time steps
        min_distance = float('inf')
        
        for i in range(min_steps):
            pos1 = np.array(trajectory1[i])
            pos2 = np.array(trajectory2[i])
            distance = np.linalg.norm(pos1 - pos2)
            min_distance = min(min_distance, distance)
        
        # Convert distance to conflict score
        # Close distances = high conflict
        # Use exponential decay: score = exp(-distance / scale)
        scale = 5.0  # meters
        conflict_score = np.exp(-min_distance / scale)
        
        return float(np.clip(conflict_score, 0.0, 1.0))
