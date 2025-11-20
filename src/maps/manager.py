"""HD Map Manager - Main interface for HD map integration."""

import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from src.core.data_structures import Lane, MapFeature
from src.maps.parser import HDMapParser, OpenDRIVEParser, Lanelet2Parser
from src.maps.matcher import MapMatcher
from src.maps.query import FeatureQuery
from src.maps.path_predictor import PathPredictor

logger = logging.getLogger(__name__)


class HDMapManager:
    """
    HD Map Manager.
    
    Provides unified interface for HD map functionality:
    - Map loading and parsing
    - Map matching and localization
    - Feature queries
    - Path prediction
    
    Optimized for <5ms query performance.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize HD map manager.
        
        Args:
            config: Configuration dictionary with map settings
        """
        self.logger = logging.getLogger(__name__ + '.HDMapManager')
        self.config = config or {}
        
        # Map data
        self.lanes: Dict[str, Lane] = {}
        self.features: List[MapFeature] = []
        self.metadata: Dict = {}
        self.map_format: Optional[str] = None
        
        # Components
        self.matcher: Optional[MapMatcher] = None
        self.query: Optional[FeatureQuery] = None
        self.path_predictor: Optional[PathPredictor] = None
        
        # Cache for performance
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.nearby_features_cache: Dict = {}
        self.cache_ttl = 0.5  # seconds
        self.last_cache_update = 0.0
        
        # Performance tracking
        self.query_times: List[float] = []
        self.max_query_time = 0.005  # 5ms target
        
        self.logger.info("HDMapManager initialized")
    
    def load_map(self, map_file: str) -> bool:
        """Load HD map from file.
        
        Args:
            map_file: Path to map file (.xodr or .osm)
            
        Returns:
            True if successful
        """
        self.logger.info(f"Loading map: {map_file}")
        
        try:
            # Determine format from file extension
            path = Path(map_file)
            
            if not path.exists():
                self.logger.error(f"Map file not found: {map_file}")
                return False
            
            # Select parser
            if path.suffix.lower() in ['.xodr', '.xml']:
                parser = OpenDRIVEParser()
                self.map_format = 'opendrive'
            elif path.suffix.lower() == '.osm':
                parser = Lanelet2Parser()
                self.map_format = 'lanelet2'
            else:
                self.logger.error(f"Unsupported map format: {path.suffix}")
                return False
            
            # Parse map
            start_time = time.time()
            map_data = parser.parse(str(map_file))
            parse_time = time.time() - start_time
            
            # Extract data
            self.lanes = map_data.get('lanes', {})
            self.features = map_data.get('features', [])
            self.metadata = map_data.get('metadata', {})
            
            self.logger.info(
                f"Map loaded in {parse_time:.3f}s: "
                f"{len(self.lanes)} lanes, {len(self.features)} features"
            )
            
            # Initialize components
            self._initialize_components()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")
            return False
    
    def _initialize_components(self):
        """Initialize map components after loading."""
        if not self.lanes:
            self.logger.warning("No lanes available, skipping component initialization")
            return
        
        # Initialize map matcher
        self.matcher = MapMatcher(self.lanes)
        
        # Initialize feature query with spatial indexing
        self.query = FeatureQuery(self.lanes, self.features)
        
        # Initialize path predictor
        self.path_predictor = PathPredictor(self.lanes)
        
        self.logger.info("Map components initialized")
    
    def match_position(self, position: Tuple[float, float],
                      heading: Optional[float] = None,
                      gps_accuracy: float = 5.0) -> Optional[Dict]:
        """Match vehicle position to lane.
        
        Args:
            position: Vehicle position (x, y) in map frame
            heading: Vehicle heading in radians (optional)
            gps_accuracy: GPS accuracy in meters
            
        Returns:
            Dictionary with match results or None
        """
        if not self.matcher:
            self.logger.warning("Map matcher not initialized")
            return None
        
        start_time = time.time()
        
        result = self.matcher.match(position, heading, gps_accuracy)
        
        query_time = time.time() - start_time
        self._track_query_time(query_time)
        
        return result
    
    def query_nearby_features(self, position: Tuple[float, float],
                             radius: float = 100.0,
                             feature_types: Optional[List[str]] = None,
                             use_cache: bool = True) -> List[MapFeature]:
        """Query features near position.
        
        Args:
            position: Query position (x, y)
            radius: Search radius in meters
            feature_types: Optional list of feature types to filter
            use_cache: Whether to use cached results
            
        Returns:
            List of MapFeature objects
        """
        if not self.query:
            self.logger.warning("Feature query not initialized")
            return []
        
        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = (position, radius, tuple(feature_types) if feature_types else None)
            current_time = time.time()
            
            if cache_key in self.nearby_features_cache:
                cached_result, cache_time = self.nearby_features_cache[cache_key]
                if current_time - cache_time < self.cache_ttl:
                    return cached_result
        
        start_time = time.time()
        
        result = self.query.query_nearby_features(position, radius, feature_types)
        
        query_time = time.time() - start_time
        self._track_query_time(query_time)
        
        # Update cache
        if use_cache and self.cache_enabled:
            self.nearby_features_cache[cache_key] = (result, time.time())
        
        return result
    
    def query_upcoming_features(self, position: Tuple[float, float],
                               heading: float, lookahead: float = 100.0,
                               feature_types: Optional[List[str]] = None) -> List[Dict]:
        """Query features ahead of vehicle.
        
        Args:
            position: Vehicle position (x, y)
            heading: Vehicle heading in radians
            lookahead: Lookahead distance in meters
            feature_types: Optional list of feature types to filter
            
        Returns:
            List of dictionaries with feature and distance
        """
        if not self.query:
            self.logger.warning("Feature query not initialized")
            return []
        
        start_time = time.time()
        
        result = self.query.query_upcoming_features(
            position, heading, lookahead, feature_types
        )
        
        query_time = time.time() - start_time
        self._track_query_time(query_time)
        
        return result
    
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
        if not self.path_predictor:
            self.logger.warning("Path predictor not initialized")
            return []
        
        start_time = time.time()
        
        result = self.path_predictor.predict_path(
            current_lane_id, turn_signal, destination, horizon
        )
        
        query_time = time.time() - start_time
        self._track_query_time(query_time)
        
        return result
    
    def get_current_lane(self) -> Optional[Lane]:
        """Get current matched lane.
        
        Returns:
            Current Lane object or None
        """
        if not self.matcher:
            return None
        
        return self.matcher.get_current_lane()
    
    def get_lane(self, lane_id: str) -> Optional[Lane]:
        """Get lane by ID.
        
        Args:
            lane_id: Lane ID
            
        Returns:
            Lane object or None
        """
        return self.lanes.get(lane_id)
    
    def get_all_lanes(self) -> Dict[str, Lane]:
        """Get all lanes.
        
        Returns:
            Dictionary of lane_id -> Lane
        """
        return self.lanes
    
    def get_all_features(self) -> List[MapFeature]:
        """Get all features.
        
        Returns:
            List of MapFeature objects
        """
        return self.features
    
    def get_map_metadata(self) -> Dict:
        """Get map metadata.
        
        Returns:
            Dictionary with map metadata
        """
        return self.metadata
    
    def is_loaded(self) -> bool:
        """Check if map is loaded.
        
        Returns:
            True if map is loaded
        """
        return len(self.lanes) > 0
    
    def clear_cache(self):
        """Clear query cache."""
        self.nearby_features_cache.clear()
        self.logger.debug("Query cache cleared")
    
    def _track_query_time(self, query_time: float):
        """Track query performance.
        
        Args:
            query_time: Query time in seconds
        """
        self.query_times.append(query_time)
        
        # Keep only last 100 queries
        if len(self.query_times) > 100:
            self.query_times.pop(0)
        
        # Log warning if exceeding target
        if query_time > self.max_query_time:
            self.logger.warning(
                f"Query time {query_time*1000:.2f}ms exceeds target "
                f"{self.max_query_time*1000:.0f}ms"
            )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.query_times:
            return {
                'avg_query_time_ms': 0.0,
                'max_query_time_ms': 0.0,
                'p95_query_time_ms': 0.0,
                'num_queries': 0,
            }
        
        import numpy as np
        
        times_ms = [t * 1000 for t in self.query_times]
        
        return {
            'avg_query_time_ms': np.mean(times_ms),
            'max_query_time_ms': np.max(times_ms),
            'p95_query_time_ms': np.percentile(times_ms, 95),
            'num_queries': len(self.query_times),
            'cache_size': len(self.nearby_features_cache),
        }
