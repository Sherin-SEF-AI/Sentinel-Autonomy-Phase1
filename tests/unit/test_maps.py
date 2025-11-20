"""Unit tests for HD map integration module."""

import pytest
import numpy as np
from src.core.data_structures import Lane, MapFeature
from src.maps.matcher import MapMatcher
from src.maps.query import FeatureQuery
from src.maps.path_predictor import PathPredictor
from src.maps.manager import HDMapManager


@pytest.fixture
def sample_lanes():
    """Create sample lanes for testing."""
    lanes = {}
    
    # Straight lane
    lane1 = Lane(
        lane_id='lane_1',
        centerline=[
            (0.0, 0.0, 0.0),
            (50.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
        ],
        left_boundary=[
            (0.0, 1.75, 0.0),
            (50.0, 1.75, 0.0),
            (100.0, 1.75, 0.0),
        ],
        right_boundary=[
            (0.0, -1.75, 0.0),
            (50.0, -1.75, 0.0),
            (100.0, -1.75, 0.0),
        ],
        width=3.5,
        speed_limit=50.0,
        lane_type='driving',
        predecessors=[],
        successors=['lane_2']
    )
    lanes['lane_1'] = lane1
    
    # Successor lane
    lane2 = Lane(
        lane_id='lane_2',
        centerline=[
            (100.0, 0.0, 0.0),
            (150.0, 0.0, 0.0),
        ],
        left_boundary=[
            (100.0, 1.75, 0.0),
            (150.0, 1.75, 0.0),
        ],
        right_boundary=[
            (100.0, -1.75, 0.0),
            (150.0, -1.75, 0.0),
        ],
        width=3.5,
        speed_limit=50.0,
        lane_type='driving',
        predecessors=['lane_1'],
        successors=[]
    )
    lanes['lane_2'] = lane2
    
    return lanes


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    return [
        MapFeature(
            feature_id='sign_1',
            type='sign',
            position=(50.0, 3.0, 0.0),
            attributes={'sign_type': 'speed_limit'},
            geometry=[(50.0, 3.0)]
        ),
        MapFeature(
            feature_id='light_1',
            type='light',
            position=(90.0, 3.0, 0.0),
            attributes={'light_type': 'traffic_light'},
            geometry=[(90.0, 3.0)]
        ),
    ]


class TestMapMatcher:
    """Tests for MapMatcher."""
    
    def test_match_center_of_lane(self, sample_lanes):
        """Test matching to center of lane."""
        matcher = MapMatcher(sample_lanes)
        
        result = matcher.match((50.0, 0.0), heading=0.0)
        
        assert result is not None
        assert result['lane_id'] == 'lane_1'
        assert abs(result['lateral_offset']) < 0.1
        assert result['confidence'] > 0.8
    
    def test_match_left_side(self, sample_lanes):
        """Test matching to left side of lane."""
        matcher = MapMatcher(sample_lanes)
        
        result = matcher.match((50.0, 1.0), heading=0.0)
        
        assert result is not None
        assert result['lane_id'] == 'lane_1'
        assert 0.9 < result['lateral_offset'] < 1.1
    
    def test_match_right_side(self, sample_lanes):
        """Test matching to right side of lane."""
        matcher = MapMatcher(sample_lanes)
        
        result = matcher.match((50.0, -1.0), heading=0.0)
        
        assert result is not None
        assert result['lane_id'] == 'lane_1'
        assert -1.1 < result['lateral_offset'] < -0.9
    
    def test_no_match_far_away(self, sample_lanes):
        """Test no match when far from lanes."""
        matcher = MapMatcher(sample_lanes)
        
        result = matcher.match((50.0, 100.0), heading=0.0)
        
        assert result is None


class TestFeatureQuery:
    """Tests for FeatureQuery."""
    
    def test_query_nearby_features(self, sample_lanes, sample_features):
        """Test querying nearby features."""
        query = FeatureQuery(sample_lanes, sample_features)
        
        nearby = query.query_nearby_features((50.0, 0.0), radius=10.0)
        
        assert len(nearby) == 1
        assert nearby[0].feature_id == 'sign_1'
    
    def test_query_upcoming_features(self, sample_lanes, sample_features):
        """Test querying upcoming features."""
        query = FeatureQuery(sample_lanes, sample_features)
        
        upcoming = query.query_upcoming_features(
            (30.0, 0.0), heading=0.0, lookahead=50.0
        )
        
        assert len(upcoming) >= 1
        assert upcoming[0]['feature'].feature_id == 'sign_1'
        assert 15.0 < upcoming[0]['distance'] < 25.0
    
    def test_filter_by_type(self, sample_lanes, sample_features):
        """Test filtering features by type."""
        query = FeatureQuery(sample_lanes, sample_features)
        
        signs = query.query_nearby_features(
            (70.0, 0.0), radius=50.0, feature_types=['sign']
        )
        
        assert len(signs) == 1
        assert all(f.type == 'sign' for f in signs)


class TestPathPredictor:
    """Tests for PathPredictor."""
    
    def test_predict_straight_path(self, sample_lanes):
        """Test predicting straight path."""
        predictor = PathPredictor(sample_lanes)
        
        path = predictor.predict_path('lane_1', turn_signal='none', horizon=2)
        
        assert len(path) == 2
        assert path[0] == 'lane_1'
        assert path[1] == 'lane_2'
    
    def test_predict_with_horizon(self, sample_lanes):
        """Test prediction with limited horizon."""
        predictor = PathPredictor(sample_lanes)
        
        path = predictor.predict_path('lane_1', turn_signal='none', horizon=1)
        
        assert len(path) == 1
        assert path[0] == 'lane_1'


class TestHDMapManager:
    """Tests for HDMapManager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = HDMapManager()
        
        assert manager is not None
        assert not manager.is_loaded()
    
    def test_manual_map_loading(self, sample_lanes, sample_features):
        """Test manual map data loading."""
        manager = HDMapManager()
        
        manager.lanes = sample_lanes
        manager.features = sample_features
        manager._initialize_components()
        
        assert manager.is_loaded()
        assert len(manager.get_all_lanes()) == 2
        assert len(manager.get_all_features()) == 2
    
    def test_match_position(self, sample_lanes, sample_features):
        """Test position matching through manager."""
        manager = HDMapManager()
        manager.lanes = sample_lanes
        manager.features = sample_features
        manager._initialize_components()
        
        result = manager.match_position((50.0, 0.0), heading=0.0)
        
        assert result is not None
        assert result['lane_id'] == 'lane_1'
    
    def test_query_features(self, sample_lanes, sample_features):
        """Test feature queries through manager."""
        manager = HDMapManager()
        manager.lanes = sample_lanes
        manager.features = sample_features
        manager._initialize_components()
        
        nearby = manager.query_nearby_features((50.0, 0.0), radius=10.0)
        
        assert len(nearby) >= 1
    
    def test_performance_tracking(self, sample_lanes, sample_features):
        """Test performance tracking."""
        manager = HDMapManager()
        manager.lanes = sample_lanes
        manager.features = sample_features
        manager._initialize_components()
        
        # Run some queries
        for _ in range(5):
            manager.match_position((50.0, 0.0), heading=0.0)
        
        stats = manager.get_performance_stats()
        
        assert stats['num_queries'] >= 5
        assert stats['avg_query_time_ms'] >= 0
        assert stats['p95_query_time_ms'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
