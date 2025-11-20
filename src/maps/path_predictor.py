"""Path prediction using lane graph and A* routing."""

import logging
from typing import Dict, List, Optional, Tuple
import heapq
import numpy as np

from src.core.data_structures import Lane

logger = logging.getLogger(__name__)


class PathPredictor:
    """Predicts driver intended path through road network."""
    
    def __init__(self, lanes: Dict[str, Lane]):
        """Initialize path predictor.
        
        Args:
            lanes: Dictionary of lane_id -> Lane
        """
        self.lanes = lanes
        self.logger = logging.getLogger(__name__ + '.PathPredictor')
        
        # Build lane graph
        self._build_lane_graph()
        
        self.logger.info(f"Initialized path predictor with {len(lanes)} lanes")
    
    def _build_lane_graph(self):
        """Build directed graph of lane connectivity."""
        self.lane_graph: Dict[str, List[Tuple[str, float]]] = {}
        
        for lane_id, lane in self.lanes.items():
            # Add successors with costs (lane length)
            successors = []
            
            for successor_id in lane.successors:
                if successor_id in self.lanes:
                    # Calculate lane length as cost
                    successor_lane = self.lanes[successor_id]
                    cost = self._calculate_lane_length(successor_lane)
                    successors.append((successor_id, cost))
            
            self.lane_graph[lane_id] = successors
        
        self.logger.debug(f"Built lane graph with {len(self.lane_graph)} nodes")
    
    def _calculate_lane_length(self, lane: Lane) -> float:
        """Calculate total length of lane centerline.
        
        Args:
            lane: Lane object
            
        Returns:
            Length in meters
        """
        if not lane.centerline or len(lane.centerline) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(lane.centerline) - 1):
            x1, y1, _ = lane.centerline[i]
            x2, y2, _ = lane.centerline[i + 1]
            length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return length
    
    def predict_path(self, current_lane_id: str,
                    turn_signal: str = 'none',
                    destination: Optional[Tuple[float, float]] = None,
                    horizon: int = 5) -> List[str]:
        """Predict intended path through lane network.
        
        Args:
            current_lane_id: Current lane ID
            turn_signal: Turn signal state ('left', 'right', 'none')
            destination: Optional destination position (x, y)
            horizon: Number of lanes to predict ahead
            
        Returns:
            List of predicted lane IDs
        """
        if current_lane_id not in self.lanes:
            self.logger.warning(f"Current lane {current_lane_id} not found")
            return []
        
        # If destination provided, use A* search
        if destination is not None:
            return self._predict_path_to_destination(
                current_lane_id, destination, horizon
            )
        
        # Otherwise, use turn signal heuristic
        return self._predict_path_from_signal(
            current_lane_id, turn_signal, horizon
        )
    
    def _predict_path_from_signal(self, current_lane_id: str,
                                  turn_signal: str, horizon: int) -> List[str]:
        """Predict path based on turn signal.
        
        Args:
            current_lane_id: Current lane ID
            turn_signal: Turn signal state
            horizon: Number of lanes ahead
            
        Returns:
            List of predicted lane IDs
        """
        path = [current_lane_id]
        current = current_lane_id
        
        for _ in range(horizon):
            if current not in self.lane_graph:
                break
            
            successors = self.lane_graph[current]
            
            if not successors:
                break
            
            # Choose successor based on turn signal
            if len(successors) == 1:
                # Only one option
                next_lane = successors[0][0]
            else:
                # Multiple options - use turn signal
                next_lane = self._choose_successor_by_signal(
                    current, successors, turn_signal
                )
            
            path.append(next_lane)
            current = next_lane
        
        return path
    
    def _choose_successor_by_signal(self, current_lane_id: str,
                                   successors: List[Tuple[str, float]],
                                   turn_signal: str) -> str:
        """Choose successor lane based on turn signal.
        
        Args:
            current_lane_id: Current lane ID
            successors: List of (successor_id, cost) tuples
            turn_signal: Turn signal state
            
        Returns:
            Chosen successor lane ID
        """
        if turn_signal == 'none' or len(successors) == 1:
            # Default to first successor (straight)
            return successors[0][0]
        
        current_lane = self.lanes[current_lane_id]
        
        # Get current lane heading
        if len(current_lane.centerline) >= 2:
            x1, y1, _ = current_lane.centerline[-2]
            x2, y2, _ = current_lane.centerline[-1]
            current_heading = np.arctan2(y2 - y1, x2 - x1)
        else:
            return successors[0][0]
        
        # Calculate heading change for each successor
        best_successor = successors[0][0]
        best_score = -float('inf')
        
        for successor_id, _ in successors:
            successor_lane = self.lanes[successor_id]
            
            if len(successor_lane.centerline) >= 2:
                x1, y1, _ = successor_lane.centerline[0]
                x2, y2, _ = successor_lane.centerline[1]
                successor_heading = np.arctan2(y2 - y1, x2 - x1)
                
                # Calculate heading change
                heading_change = self._normalize_angle(successor_heading - current_heading)
                
                # Score based on turn signal
                if turn_signal == 'left':
                    # Prefer left turns (positive heading change)
                    score = heading_change
                elif turn_signal == 'right':
                    # Prefer right turns (negative heading change)
                    score = -heading_change
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_successor = successor_id
        
        return best_successor
    
    def _predict_path_to_destination(self, start_lane_id: str,
                                    destination: Tuple[float, float],
                                    max_length: int) -> List[str]:
        """Predict path to destination using A* search.
        
        Args:
            start_lane_id: Starting lane ID
            destination: Destination position (x, y)
            max_length: Maximum path length
            
        Returns:
            List of lane IDs forming path
        """
        # A* search
        # Priority queue: (f_score, g_score, lane_id, path)
        open_set = [(0.0, 0.0, start_lane_id, [start_lane_id])]
        closed_set = set()
        
        dest_x, dest_y = destination
        
        while open_set:
            f_score, g_score, current_id, path = heapq.heappop(open_set)
            
            if current_id in closed_set:
                continue
            
            closed_set.add(current_id)
            
            # Check if we've reached destination or max length
            if len(path) >= max_length:
                return path
            
            current_lane = self.lanes[current_id]
            
            # Check if destination is on current lane
            if self._is_point_on_lane(destination, current_lane):
                return path
            
            # Expand successors
            if current_id not in self.lane_graph:
                continue
            
            for successor_id, edge_cost in self.lane_graph[current_id]:
                if successor_id in closed_set:
                    continue
                
                # Calculate g_score (cost so far)
                new_g_score = g_score + edge_cost
                
                # Calculate h_score (heuristic to destination)
                successor_lane = self.lanes[successor_id]
                h_score = self._heuristic_to_destination(successor_lane, destination)
                
                # Calculate f_score
                new_f_score = new_g_score + h_score
                
                # Add to open set
                new_path = path + [successor_id]
                heapq.heappush(open_set, (new_f_score, new_g_score, successor_id, new_path))
        
        # No path found, return current path
        return [start_lane_id]
    
    def _heuristic_to_destination(self, lane: Lane, 
                                  destination: Tuple[float, float]) -> float:
        """Calculate heuristic distance from lane to destination.
        
        Args:
            lane: Lane object
            destination: Destination position (x, y)
            
        Returns:
            Estimated distance in meters
        """
        if not lane.centerline:
            return float('inf')
        
        dest_x, dest_y = destination
        
        # Use distance from lane end to destination
        end_x, end_y, _ = lane.centerline[-1]
        
        return np.sqrt((dest_x - end_x)**2 + (dest_y - end_y)**2)
    
    def _is_point_on_lane(self, point: Tuple[float, float], lane: Lane,
                         threshold: float = 5.0) -> bool:
        """Check if point is on lane.
        
        Args:
            point: Point position (x, y)
            lane: Lane object
            threshold: Distance threshold in meters
            
        Returns:
            True if point is on lane
        """
        if not lane.centerline:
            return False
        
        px, py = point
        
        # Check distance to any point on centerline
        for cx, cy, _ in lane.centerline:
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            if dist < threshold:
                return True
        
        return False
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_lane_sequence(self, current_lane_id: str,
                         distance: float) -> List[Tuple[str, float]]:
        """Get sequence of lanes for a given distance ahead.
        
        Args:
            current_lane_id: Current lane ID
            distance: Distance to look ahead in meters
            
        Returns:
            List of (lane_id, distance_along_lane) tuples
        """
        if current_lane_id not in self.lanes:
            return []
        
        sequence = []
        remaining_distance = distance
        current = current_lane_id
        
        while remaining_distance > 0:
            if current not in self.lanes:
                break
            
            lane = self.lanes[current]
            lane_length = self._calculate_lane_length(lane)
            
            if lane_length >= remaining_distance:
                # Destination is on this lane
                sequence.append((current, remaining_distance))
                break
            else:
                # Add full lane and continue to next
                sequence.append((current, lane_length))
                remaining_distance -= lane_length
                
                # Get next lane (use first successor)
                if current in self.lane_graph and self.lane_graph[current]:
                    current = self.lane_graph[current][0][0]
                else:
                    break
        
        return sequence
