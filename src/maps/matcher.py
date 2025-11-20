"""Map matching for lane-level localization."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.data_structures import Lane

logger = logging.getLogger(__name__)


class MapMatcher:
    """Performs map matching to determine current lane and position."""
    
    def __init__(self, lanes: Dict[str, Lane]):
        """Initialize map matcher.
        
        Args:
            lanes: Dictionary of lane_id -> Lane
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug(f"MapMatcher initialization started: num_lanes={len(lanes)}")
        
        self.lanes = lanes
        self.current_lane_id: Optional[str] = None
        self.last_position: Optional[Tuple[float, float]] = None
        
        # Performance tracking
        self.match_count = 0
        self.successful_matches = 0
        
        self.logger.info(
            f"MapMatcher initialized: num_lanes={len(lanes)}, "
            f"lane_ids={list(lanes.keys())[:5]}{'...' if len(lanes) > 5 else ''}"
        )
    
    def match(self, position: Tuple[float, float], heading: Optional[float] = None,
              gps_accuracy: float = 5.0) -> Optional[Dict]:
        """Match vehicle position to lane.
        
        Args:
            position: Vehicle position (x, y) in map frame
            heading: Vehicle heading in radians (optional)
            gps_accuracy: GPS accuracy in meters for search radius
            
        Returns:
            Dictionary with:
                - lane_id: Matched lane ID
                - lateral_offset: Distance from lane center (m)
                - longitudinal_position: Position along lane (m)
                - confidence: Match confidence 0-1
        """
        import time
        match_start = time.time()
        self.match_count += 1
        
        heading_str = f"{heading:.3f}" if heading is not None else "None"
        self.logger.debug(
            f"Map matching started: position=({position[0]:.2f}, {position[1]:.2f}), "
            f"heading={heading_str}, "
            f"gps_accuracy={gps_accuracy:.1f}m"
        )
        
        if not self.lanes:
            self.logger.warning("Map matching failed: no lanes available")
            return None
        
        x, y = position
        
        # Find candidate lanes within search radius
        candidates = self._find_candidate_lanes(position, gps_accuracy * 2)
        
        if not candidates:
            self.logger.debug(
                f"Map matching failed: no candidate lanes found near position "
                f"({x:.2f}, {y:.2f}) within {gps_accuracy * 2:.1f}m radius"
            )
            return None
        
        self.logger.debug(f"Found {len(candidates)} candidate lanes: {candidates}")
        
        # Score each candidate
        best_match = None
        best_score = -1
        
        for lane_id in candidates:
            lane = self.lanes[lane_id]
            result = self._match_to_lane(position, heading, lane)
            
            if result is None:
                self.logger.debug(f"Lane {lane_id} match failed: no valid result")
                continue
            
            # Calculate match score
            score = self._calculate_match_score(
                result['lateral_offset'],
                result['heading_diff'] if heading is not None else 0,
                lane_id == self.current_lane_id
            )
            
            self.logger.debug(
                f"Lane {lane_id} score: {score:.3f} "
                f"(lateral_offset={result['lateral_offset']:.2f}m, "
                f"heading_diff={result['heading_diff']:.3f}rad)"
            )
            
            if score > best_score:
                best_score = score
                best_match = result
                best_match['lane_id'] = lane_id
                best_match['confidence'] = score
        
        match_duration = (time.time() - match_start) * 1000  # Convert to ms
        
        if best_match:
            self.successful_matches += 1
            prev_lane = self.current_lane_id
            self.current_lane_id = best_match['lane_id']
            self.last_position = position
            
            # Log lane change
            if prev_lane and prev_lane != self.current_lane_id:
                self.logger.info(
                    f"Lane change detected: {prev_lane} -> {self.current_lane_id}, "
                    f"confidence={best_match['confidence']:.2f}"
                )
            
            self.logger.debug(
                f"Map matching completed: lane_id={best_match['lane_id']}, "
                f"lateral_offset={best_match['lateral_offset']:.2f}m, "
                f"longitudinal_position={best_match['longitudinal_position']:.1f}m, "
                f"confidence={best_match['confidence']:.2f}, "
                f"duration={match_duration:.2f}ms"
            )
            
            # Warn if matching is slow
            if match_duration > 5.0:
                self.logger.warning(
                    f"Slow map matching detected: duration={match_duration:.2f}ms "
                    f"(target: <5ms), num_candidates={len(candidates)}"
                )
        else:
            self.logger.warning(
                f"Map matching failed: no valid match found among {len(candidates)} candidates, "
                f"position=({x:.2f}, {y:.2f}), duration={match_duration:.2f}ms"
            )
        
        # Periodic statistics logging
        if self.match_count % 100 == 0:
            success_rate = (self.successful_matches / self.match_count) * 100
            self.logger.info(
                f"Map matching statistics: attempts={self.match_count}, "
                f"successful={self.successful_matches}, "
                f"success_rate={success_rate:.1f}%, "
                f"current_lane={self.current_lane_id}"
            )
        
        return best_match
    
    def _find_candidate_lanes(self, position: Tuple[float, float], 
                             search_radius: float) -> List[str]:
        """Find lanes within search radius of position.
        
        Args:
            position: Vehicle position (x, y)
            search_radius: Search radius in meters
            
        Returns:
            List of candidate lane IDs
        """
        x, y = position
        candidates = []
        
        # Prioritize current lane if available
        if self.current_lane_id and self.current_lane_id in self.lanes:
            candidates.append(self.current_lane_id)
            self.logger.debug(
                f"Current lane {self.current_lane_id} added as priority candidate"
            )
        
        # Check all lanes
        lanes_checked = 0
        for lane_id, lane in self.lanes.items():
            if lane_id == self.current_lane_id:
                continue
            
            lanes_checked += 1
            
            # Check if any point on centerline is within search radius
            for cx, cy, _ in lane.centerline:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < search_radius:
                    candidates.append(lane_id)
                    self.logger.debug(
                        f"Lane {lane_id} added as candidate: min_dist={dist:.2f}m"
                    )
                    break
        
        self.logger.debug(
            f"Candidate search completed: checked={lanes_checked} lanes, "
            f"found={len(candidates)} candidates, search_radius={search_radius:.1f}m"
        )
        
        return candidates
    
    def _match_to_lane(self, position: Tuple[float, float], 
                       heading: Optional[float], lane: Lane) -> Optional[Dict]:
        """Match position to a specific lane.
        
        Args:
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            lane: Lane to match against
            
        Returns:
            Dictionary with match details or None
        """
        if not lane.centerline:
            return None
        
        x, y = position
        
        # Find closest point on centerline
        min_dist = float('inf')
        closest_idx = 0
        closest_point = None
        
        for i, (cx, cy, _) in enumerate(lane.centerline):
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                closest_point = (cx, cy)
        
        if closest_point is None:
            return None
        
        # Calculate lateral offset (perpendicular distance to centerline)
        lateral_offset = self._calculate_lateral_offset(
            position, closest_idx, lane.centerline
        )
        
        # Calculate longitudinal position along lane
        longitudinal_position = self._calculate_longitudinal_position(
            closest_idx, lane.centerline
        )
        
        # Calculate heading difference if heading provided
        heading_diff = 0.0
        if heading is not None:
            lane_heading = self._calculate_lane_heading(closest_idx, lane.centerline)
            heading_diff = abs(self._normalize_angle(heading - lane_heading))
        
        return {
            'lateral_offset': lateral_offset,
            'longitudinal_position': longitudinal_position,
            'heading_diff': heading_diff,
            'closest_idx': closest_idx,
        }
    
    def _calculate_lateral_offset(self, position: Tuple[float, float],
                                  idx: int, centerline: List[Tuple[float, float, float]]) -> float:
        """Calculate perpendicular distance from position to centerline.
        
        Args:
            position: Vehicle position (x, y)
            idx: Index of closest point on centerline
            centerline: Lane centerline points
            
        Returns:
            Lateral offset in meters (positive = left, negative = right)
        """
        x, y = position
        cx, cy, _ = centerline[idx]
        
        # Get lane direction at this point
        if idx < len(centerline) - 1:
            next_cx, next_cy, _ = centerline[idx + 1]
            dx = next_cx - cx
            dy = next_cy - cy
        elif idx > 0:
            prev_cx, prev_cy, _ = centerline[idx - 1]
            dx = cx - prev_cx
            dy = cy - prev_cy
        else:
            # Single point, can't determine direction
            return np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Normalize direction vector
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return np.sqrt((x - cx)**2 + (y - cy)**2)
        
        dx /= length
        dy /= length
        
        # Vector from centerline point to vehicle
        to_vehicle_x = x - cx
        to_vehicle_y = y - cy
        
        # Calculate perpendicular distance using cross product
        # Cross product gives signed area, which equals perpendicular distance for unit vector
        # Positive = left of centerline, negative = right
        lateral_offset = dx * to_vehicle_y - dy * to_vehicle_x
        
        return lateral_offset
    
    def _calculate_longitudinal_position(self, idx: int, 
                                        centerline: List[Tuple[float, float, float]]) -> float:
        """Calculate distance along centerline to closest point.
        
        Args:
            idx: Index of closest point
            centerline: Lane centerline points
            
        Returns:
            Distance in meters from start of lane
        """
        distance = 0.0
        
        for i in range(idx):
            x1, y1, _ = centerline[i]
            x2, y2, _ = centerline[i + 1]
            distance += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance
    
    def _calculate_lane_heading(self, idx: int, 
                               centerline: List[Tuple[float, float, float]]) -> float:
        """Calculate lane heading at given index.
        
        Args:
            idx: Index on centerline
            centerline: Lane centerline points
            
        Returns:
            Heading in radians
        """
        if idx < len(centerline) - 1:
            x1, y1, _ = centerline[idx]
            x2, y2, _ = centerline[idx + 1]
        elif idx > 0:
            x1, y1, _ = centerline[idx - 1]
            x2, y2, _ = centerline[idx]
        else:
            return 0.0
        
        return np.arctan2(y2 - y1, x2 - x1)
    
    def _calculate_match_score(self, lateral_offset: float, heading_diff: float,
                              is_current_lane: bool) -> float:
        """Calculate match confidence score.
        
        Args:
            lateral_offset: Lateral distance from lane center
            heading_diff: Heading difference in radians
            is_current_lane: Whether this is the current lane
            
        Returns:
            Score from 0 to 1
        """
        # Lateral offset score (exponential decay)
        lateral_score = np.exp(-abs(lateral_offset) / 2.0)
        
        # Heading score (exponential decay)
        heading_score = np.exp(-heading_diff / (np.pi / 4))
        
        # Combine scores
        score = 0.6 * lateral_score + 0.4 * heading_score
        
        # Bonus for current lane (hysteresis)
        if is_current_lane:
            score *= 1.2
        
        return min(score, 1.0)
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def get_current_lane(self) -> Optional[Lane]:
        """Get current matched lane.
        
        Returns:
            Current Lane object or None
        """
        if self.current_lane_id and self.current_lane_id in self.lanes:
            lane = self.lanes[self.current_lane_id]
            self.logger.debug(
                f"Current lane retrieved: lane_id={self.current_lane_id}, "
                f"centerline_points={len(lane.centerline)}"
            )
            return lane
        
        self.logger.debug("No current lane available")
        return None
    
    def get_statistics(self) -> Dict:
        """Get map matching statistics.
        
        Returns:
            Dictionary with matching statistics
        """
        success_rate = (self.successful_matches / self.match_count * 100) if self.match_count > 0 else 0
        
        stats = {
            'match_attempts': self.match_count,
            'successful_matches': self.successful_matches,
            'success_rate': success_rate,
            'current_lane_id': self.current_lane_id,
            'num_lanes': len(self.lanes)
        }
        
        self.logger.debug(f"Statistics retrieved: {stats}")
        return stats
