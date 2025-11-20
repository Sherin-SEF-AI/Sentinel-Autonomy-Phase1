"""Trajectory visualization helpers for GUI integration."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.intelligence.advanced_trajectory import Trajectory


logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Convert trajectory predictions to visualization format."""
    
    def __init__(self):
        """Initialize trajectory visualizer."""
        self.logger = logging.getLogger(__name__)
    
    def prepare_trajectories_for_display(
        self,
        object_trajectories: Dict[int, List[Trajectory]],
        collision_probabilities: Optional[Dict[int, tuple]] = None,
        show_all_hypotheses: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Prepare trajectories for BEV canvas display.
        
        Args:
            object_trajectories: Dictionary mapping track_id to list of trajectory hypotheses
            collision_probabilities: Optional collision probabilities (track_id -> (prob, step, hyp_idx))
            show_all_hypotheses: If True, show all hypotheses; if False, show only best
            
        Returns:
            List of trajectory dictionaries for BEV canvas
        """
        display_trajectories = []
        
        for track_id, trajectories in object_trajectories.items():
            if not trajectories:
                continue
            
            # Get collision probability for this object
            collision_prob = 0.0
            best_hyp_idx = 0
            if collision_probabilities and track_id in collision_probabilities:
                collision_prob, _, best_hyp_idx = collision_probabilities[track_id]
            
            if show_all_hypotheses:
                # Show all trajectory hypotheses
                for hyp_idx, trajectory in enumerate(trajectories):
                    # Use collision prob only for the best hypothesis
                    hyp_collision_prob = collision_prob if hyp_idx == best_hyp_idx else 0.0
                    
                    display_traj = self._convert_trajectory_to_display(
                        track_id,
                        trajectory,
                        hyp_collision_prob,
                        hypothesis_index=hyp_idx
                    )
                    display_trajectories.append(display_traj)
            else:
                # Show only best trajectory (highest confidence or collision risk)
                if best_hyp_idx < len(trajectories):
                    best_trajectory = trajectories[best_hyp_idx]
                else:
                    best_trajectory = trajectories[0]
                
                display_traj = self._convert_trajectory_to_display(
                    track_id,
                    best_trajectory,
                    collision_prob
                )
                display_trajectories.append(display_traj)
        
        self.logger.debug(f"Prepared {len(display_trajectories)} trajectories for display")
        return display_trajectories
    
    def _convert_trajectory_to_display(
        self,
        track_id: int,
        trajectory: Trajectory,
        collision_prob: float,
        hypothesis_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Convert Trajectory object to display format.
        
        Args:
            track_id: Object track ID
            trajectory: Trajectory object
            collision_prob: Collision probability
            hypothesis_index: Optional hypothesis index for multi-hypothesis display
            
        Returns:
            Dictionary for BEV canvas
        """
        # Extract 2D points (x, y) from 3D trajectory
        points_2d = [(p[0], p[1]) for p in trajectory.points]
        
        # Extract uncertainty (use trace of covariance as scalar uncertainty)
        uncertainty_values = []
        for cov in trajectory.uncertainty:
            # Calculate standard deviation from covariance trace
            std = np.sqrt(np.trace(cov) / 3.0)  # Average over 3 dimensions
            uncertainty_values.append(float(std))
        
        # Build display dictionary
        display_dict = {
            'object_id': track_id,
            'points': points_2d,
            'uncertainty': uncertainty_values,
            'collision_probability': float(collision_prob),
            'confidence': float(trajectory.confidence),
            'model': trajectory.model,
            'timestamps': trajectory.timestamps
        }
        
        # Add hypothesis index if provided
        if hypothesis_index is not None:
            display_dict['hypothesis_index'] = hypothesis_index
        
        return display_dict
    
    def prepare_ego_trajectory_for_display(
        self,
        ego_trajectory: Trajectory
    ) -> Dict[str, Any]:
        """
        Prepare ego vehicle trajectory for display.
        
        Args:
            ego_trajectory: Ego vehicle trajectory
            
        Returns:
            Dictionary for BEV canvas
        """
        return self._convert_trajectory_to_display(
            track_id=-1,  # Special ID for ego vehicle
            trajectory=ego_trajectory,
            collision_prob=0.0
        )
    
    def create_trajectory_legend(
        self,
        object_trajectories: Dict[int, List[Trajectory]],
        collision_probabilities: Optional[Dict[int, tuple]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create legend information for trajectory display.
        
        Args:
            object_trajectories: Object trajectories
            collision_probabilities: Collision probabilities
            
        Returns:
            List of legend entries
        """
        legend_entries = []
        
        # Model type legend
        model_types = set()
        for trajectories in object_trajectories.values():
            for traj in trajectories:
                model_types.add(traj.model)
        
        for model in sorted(model_types):
            legend_entries.append({
                'type': 'model',
                'label': f'{model.upper()} Model',
                'description': self._get_model_description(model)
            })
        
        # Collision probability legend
        if collision_probabilities:
            legend_entries.append({
                'type': 'collision',
                'label': 'Collision Risk',
                'description': 'Color indicates collision probability',
                'color_scale': [
                    ('Green', 'Low risk (<30%)'),
                    ('Yellow', 'Medium risk (30-60%)'),
                    ('Orange', 'High risk (60-80%)'),
                    ('Red', 'Critical risk (>80%)')
                ]
            })
        
        return legend_entries
    
    def _get_model_description(self, model: str) -> str:
        """Get description for model type."""
        descriptions = {
            'lstm': 'Learning-based prediction using historical patterns',
            'cv': 'Constant velocity physics model',
            'ca': 'Constant acceleration physics model',
            'ct': 'Constant turn rate physics model',
            'merged': 'Ensemble of multiple models'
        }
        return descriptions.get(model, 'Unknown model')
    
    def filter_trajectories_by_distance(
        self,
        trajectories: List[Dict[str, Any]],
        max_distance: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Filter trajectories to only show those within distance threshold.
        
        Args:
            trajectories: List of trajectory dictionaries
            max_distance: Maximum distance in meters
            
        Returns:
            Filtered list of trajectories
        """
        filtered = []
        
        for traj in trajectories:
            points = traj.get('points', [])
            if not points:
                continue
            
            # Check if any point is within max distance
            within_range = False
            for x, y in points:
                distance = np.sqrt(x**2 + y**2)
                if distance <= max_distance:
                    within_range = True
                    break
            
            if within_range:
                filtered.append(traj)
        
        return filtered
    
    def filter_trajectories_by_collision_risk(
        self,
        trajectories: List[Dict[str, Any]],
        min_collision_prob: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Filter trajectories to only show those with significant collision risk.
        
        Args:
            trajectories: List of trajectory dictionaries
            min_collision_prob: Minimum collision probability threshold
            
        Returns:
            Filtered list of trajectories
        """
        return [
            traj for traj in trajectories
            if traj.get('collision_probability', 0.0) >= min_collision_prob
        ]
    
    def annotate_trajectory_with_time(
        self,
        trajectory: Dict[str, Any],
        time_intervals: List[float] = [1.0, 2.0, 3.0]
    ) -> Dict[str, Any]:
        """
        Add time annotations to trajectory.
        
        Args:
            trajectory: Trajectory dictionary
            time_intervals: Time points to annotate (seconds)
            
        Returns:
            Trajectory with time annotations
        """
        timestamps = trajectory.get('timestamps', [])
        points = trajectory.get('points', [])
        
        if not timestamps or not points:
            return trajectory
        
        # Find points closest to time intervals
        annotations = []
        for target_time in time_intervals:
            # Find closest timestamp
            closest_idx = min(
                range(len(timestamps)),
                key=lambda i: abs(timestamps[i] - target_time)
            )
            
            if abs(timestamps[closest_idx] - target_time) < 0.5:  # Within 0.5s
                annotations.append({
                    'time': target_time,
                    'point': points[closest_idx],
                    'label': f'{target_time:.1f}s'
                })
        
        trajectory['time_annotations'] = annotations
        return trajectory
