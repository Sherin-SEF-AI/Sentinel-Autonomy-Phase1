"""Feature query system for HD maps."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.core.data_structures import Lane, MapFeature

logger = logging.getLogger(__name__)


class FeatureQuery:
    """Query system for map features with spatial indexing."""
    
    def __init__(self, lanes: Dict[str, Lane], features: List[MapFeature]):
        """Initialize feature query system.
        
        Args:
            lanes: Dictionary of lane_id -> Lane
            features: List of map features
        """
        self.lanes = lanes
        self.features = features
        self.logger = logging.getLogger(__name__ + '.FeatureQuery')
        
        # Build spatial index for features
        self._build_spatial_index()
        
        self.logger.info(
            f"Initialized feature query with {len(lanes)} lanes "
            f"and {len(features)} features"
        )
    
    def _build_spatial_index(self):
        """Build simple grid-based spatial index for features."""
        # Grid cell size in meters
        self.grid_size = 50.0
        self.feature_grid: Dict[Tuple[int, int], List[int]] = {}
        
        for i, feature in enumerate(self.features):
            x, y, _ = feature.position
            grid_x = int(x / self.grid_size)
            grid_y = int(y / self.grid_size)
            
            key = (grid_x, grid_y)
            if key not in self.feature_grid:
                self.feature_grid[key] = []
            self.feature_grid[key].append(i)
        
        self.logger.debug(f"Built spatial index with {len(self.feature_grid)} grid cells")
    
    def query_nearby_features(self, position: Tuple[float, float], 
                             radius: float = 100.0,
                             feature_types: Optional[List[str]] = None) -> List[MapFeature]:
        """Query features within radius of position.
        
        Args:
            position: Query position (x, y)
            radius: Search radius in meters
            feature_types: Optional list of feature types to filter
            
        Returns:
            List of MapFeature objects within radius
        """
        x, y = position
        
        # Determine grid cells to search
        grid_radius = int(np.ceil(radius / self.grid_size))
        center_grid_x = int(x / self.grid_size)
        center_grid_y = int(y / self.grid_size)
        
        nearby_features = []
        
        # Search grid cells
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_key = (center_grid_x + dx, center_grid_y + dy)
                
                if grid_key not in self.feature_grid:
                    continue
                
                # Check features in this cell
                for feature_idx in self.feature_grid[grid_key]:
                    feature = self.features[feature_idx]
                    
                    # Filter by type if specified
                    if feature_types and feature.type not in feature_types:
                        continue
                    
                    # Check distance
                    fx, fy, _ = feature.position
                    dist = np.sqrt((x - fx)**2 + (y - fy)**2)
                    
                    if dist <= radius:
                        nearby_features.append(feature)
        
        return nearby_features
    
    def query_upcoming_features(self, position: Tuple[float, float],
                               heading: float, lookahead: float = 100.0,
                               feature_types: Optional[List[str]] = None) -> List[Dict]:
        """Query features ahead of vehicle along heading direction.
        
        Args:
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            lookahead: Lookahead distance in meters
            feature_types: Optional list of feature types to filter
            
        Returns:
            List of dictionaries with:
                - feature: MapFeature object
                - distance: Distance ahead in meters
                - lateral_offset: Lateral offset from path (m)
        """
        # Get nearby features
        nearby = self.query_nearby_features(position, lookahead, feature_types)
        
        x, y = position
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        upcoming = []
        
        for feature in nearby:
            fx, fy, _ = feature.position
            
            # Transform to vehicle frame
            dx = fx - x
            dy = fy - y
            
            # Rotate to vehicle heading
            forward = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h
            
            # Only include features ahead
            if forward > 0 and forward <= lookahead:
                upcoming.append({
                    'feature': feature,
                    'distance': forward,
                    'lateral_offset': lateral,
                })
        
        # Sort by distance
        upcoming.sort(key=lambda x: x['distance'])
        
        return upcoming
    
    def query_lane_features(self, lane_id: str, 
                           longitudinal_position: float,
                           lookahead: float = 100.0,
                           feature_types: Optional[List[str]] = None) -> List[Dict]:
        """Query features along a specific lane.
        
        Args:
            lane_id: Lane ID to query
            longitudinal_position: Current position along lane (m)
            lookahead: Lookahead distance in meters
            feature_types: Optional list of feature types to filter
            
        Returns:
            List of dictionaries with feature and distance along lane
        """
        if lane_id not in self.lanes:
            self.logger.warning(f"Lane {lane_id} not found")
            return []
        
        lane = self.lanes[lane_id]
        
        # Get features near lane
        lane_features = []
        
        # Sample points along lane ahead
        current_dist = 0.0
        for i in range(len(lane.centerline) - 1):
            x1, y1, _ = lane.centerline[i]
            x2, y2, _ = lane.centerline[i + 1]
            
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if current_dist >= longitudinal_position and current_dist <= longitudinal_position + lookahead:
                # Query features near this segment
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                nearby = self.query_nearby_features((mid_x, mid_y), 10.0, feature_types)
                
                for feature in nearby:
                    # Calculate distance along lane
                    fx, fy, _ = feature.position
                    
                    # Project onto lane segment
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        dx /= length
                        dy /= length
                        
                        to_feature_x = fx - x1
                        to_feature_y = fy - y1
                        
                        projection = to_feature_x * dx + to_feature_y * dy
                        feature_dist = current_dist + projection
                        
                        if feature_dist >= longitudinal_position:
                            lane_features.append({
                                'feature': feature,
                                'distance': feature_dist - longitudinal_position,
                            })
            
            current_dist += segment_length
            
            if current_dist > longitudinal_position + lookahead:
                break
        
        # Remove duplicates and sort
        seen = set()
        unique_features = []
        for item in lane_features:
            if item['feature'].feature_id not in seen:
                seen.add(item['feature'].feature_id)
                unique_features.append(item)
        
        unique_features.sort(key=lambda x: x['distance'])
        
        return unique_features
    
    def get_traffic_signs_ahead(self, position: Tuple[float, float],
                               heading: float, lookahead: float = 100.0) -> List[Dict]:
        """Get traffic signs ahead of vehicle.
        
        Args:
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            lookahead: Lookahead distance in meters
            
        Returns:
            List of traffic signs with distance and attributes
        """
        return self.query_upcoming_features(
            position, heading, lookahead, feature_types=['sign']
        )
    
    def get_traffic_lights_ahead(self, position: Tuple[float, float],
                                heading: float, lookahead: float = 100.0) -> List[Dict]:
        """Get traffic lights ahead of vehicle.
        
        Args:
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            lookahead: Lookahead distance in meters
            
        Returns:
            List of traffic lights with distance and attributes
        """
        return self.query_upcoming_features(
            position, heading, lookahead, feature_types=['light']
        )
    
    def get_crosswalks_ahead(self, position: Tuple[float, float],
                            heading: float, lookahead: float = 100.0) -> List[Dict]:
        """Get crosswalks ahead of vehicle.
        
        Args:
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            lookahead: Lookahead distance in meters
            
        Returns:
            List of crosswalks with distance
        """
        return self.query_upcoming_features(
            position, heading, lookahead, feature_types=['crosswalk']
        )
    
    def get_intersections_ahead(self, lane_id: str,
                               longitudinal_position: float,
                               lookahead: float = 100.0) -> List[Dict]:
        """Get intersections ahead on current lane.
        
        Args:
            lane_id: Current lane ID
            longitudinal_position: Position along lane (m)
            lookahead: Lookahead distance in meters
            
        Returns:
            List of intersections with distance
        """
        if lane_id not in self.lanes:
            return []
        
        lane = self.lanes[lane_id]
        intersections = []
        
        # Check for lane splits/merges (multiple successors/predecessors)
        if len(lane.successors) > 1:
            # Calculate distance to end of lane
            total_length = 0.0
            for i in range(len(lane.centerline) - 1):
                x1, y1, _ = lane.centerline[i]
                x2, y2, _ = lane.centerline[i + 1]
                total_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            distance_to_end = total_length - longitudinal_position
            
            if 0 < distance_to_end <= lookahead:
                # Get position of intersection
                if lane.centerline:
                    x, y, _ = lane.centerline[-1]
                    intersections.append({
                        'type': 'lane_split',
                        'position': (x, y, 0.0),
                        'distance': distance_to_end,
                        'options': lane.successors,
                    })
        
        return intersections
