"""Test suite for maps matcher module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.maps.matcher import MapMatcher
from src.core.data_structures import Lane


@pytest.fixture
def sample_lane():
    """Fixture providing a sample lane for testing."""
    # Create a straight lane along x-axis
    centerline = [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
        (40.0, 0.0, 0.0),
    ]
    left_boundary = [
        (0.0, 1.75, 0.0),
        (10.0, 1.75, 0.0),
        (20.0, 1.75, 0.0),
        (30.0, 1.75, 0.0),
        (40.0, 1.75, 0.0),
    ]
    right_boundary = [
        (0.0, -1.75, 0.0),
        (10.0, -1.75, 0.0),
        (20.0, -1.75, 0.0),
        (30.0, -1.75, 0.0),
        (40.0, -1.75, 0.0),
    ]
    
    return Lane(
        lane_id="lane_001",
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        width=3.5,
        speed_limit=50.0,
        lane_type="driving",
        predecessors=[],
        successors=["lane_002"]
    )


@pytest.fixture
def curved_lane():
    """Fixture providing a curved lane for testing."""
    # Create a curved lane (quarter circle)
    centerline = []
    for i in range(10):
        angle = i * np.pi / 18  # 10 degrees per segment
        x = 20.0 * np.sin(angle)
        y = 20.0 * (1 - np.cos(angle))
        centerline.append((x, y, 0.0))
    
    left_boundary = []
    right_boundary = []
    for x, y, z in centerline:
        # Simplified boundaries (not geometrically perfect)
        left_boundary.append((x - 1.75, y, z))
        right_boundary.append((x + 1.75, y, z))
    
    return Lane(
        lane_id="lane_002",
        centerline=centerline,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        width=3.5,
        speed_limit=40.0,
        lane_type="driving",
        predecessors=["lane_001"],
        successors=[]
    )


@pytest.fixture
def lanes_dict(sample_lane, curved_lane):
    """Fixture providing a dictionary of lanes."""
    return {
        "lane_001": sample_lane,
        "lane_002": curved_lane,
    }


@pytest.fixture
def map_matcher(lanes_dict):
    """Fixture creating a MapMatcher instance for testing."""
    return MapMatcher(lanes_dict)


class TestMapMatcher:
    """Test suite for MapMatcher class."""
    
    def test_initialization(self, map_matcher, lanes_dict):
        """Test that MapMatcher initializes correctly with valid lanes."""
        assert map_matcher is not None
        assert map_matcher.lanes == lanes_dict
        assert map_matcher.current_lane_id is None
        assert map_matcher.last_position is None
        assert map_matcher.match_count == 0
        assert map_matcher.successful_matches == 0
    
    def test_initialization_empty_lanes(self):
        """Test initialization with empty lanes dictionary."""
        matcher = MapMatcher({})
        assert matcher.lanes == {}
        assert len(matcher.lanes) == 0
    
    def test_match_on_centerline(self, map_matcher):
        """Test matching when vehicle is exactly on lane centerline."""
        # Position on centerline of lane_001
        position = (15.0, 0.0)
        heading = 0.0  # Heading along x-axis
        
        result = map_matcher.match(position, heading)
        
        assert result is not None
        assert result['lane_id'] == 'lane_001'
        assert abs(result['lateral_offset']) < 0.1  # Should be very close to 0
        assert result['confidence'] > 0.9  # High confidence
        assert 10.0 < result['longitudinal_position'] < 20.0
    
    def test_match_left_of_centerline(self, map_matcher):
        """Test matching when vehicle is left of centerline."""
        # Position 1 meter left of centerline
        position = (15.0, 1.0)
        heading = 0.0
        
        result = map_matcher.match(position, heading)
        
        assert result is not None
        assert result['lane_id'] == 'lane_001'
        assert 0.9 < result['lateral_offset'] < 1.1  # Approximately 1m left
        assert result['confidence'] > 0.5
    
    def test_match_right_of_centerline(self, map_matcher):
        """Test matching when vehicle is right of centerline."""
        # Position 1 meter right of centerline
        position = (15.0, -1.0)
        heading = 0.0
        
        result = map_matcher.match(position, heading)
        
        assert result is not None
        assert result['lane_id'] == 'lane_001'
        assert -1.1 < result['lateral_offset'] < -0.9  # Approximately 1m right
        assert result['confidence'] > 0.5
    
    def test_match_with_heading(self, map_matcher):
        """Test matching with vehicle heading information."""
        position = (15.0, 0.0)
        heading = 0.0  # Aligned with lane
        
        result = map_matcher.match(position, heading)
        
        assert result is not None
        assert result['lane_id'] == 'lane_001'
        assert result['confidence'] > 0.9
    
    def test_match_with_misaligned_heading(self, map_matcher):
        """Test matching with misaligned heading reduces confidence."""
        position = (15.0, 0.0)
        heading = np.pi / 2  # 90 degrees off
        
        result = map_matcher.match(position, heading)
        
        assert result is not None
        # Confidence should be lower due to heading mismatch
        assert result['confidence'] < 0.7
    
    def test_match_without_heading(self, map_matcher):
        """Test matching without heading information."""
        position = (15.0, 0.0)
        
        result = map_matcher.match(position, heading=None)
        
        assert result is not None
        assert result['lane_id'] == 'lane_001'
        assert 'confidence' in result
    
    def test_match_far_from_lanes(self, map_matcher):
        """Test matching when vehicle is far from all lanes."""
        # Position 100 meters away from any lane
        position = (15.0, 100.0)
        
        result = map_matcher.match(position)
        
        # Should fail to match
        assert result is None
    
    def test_match_updates_current_lane(self, map_matcher):
        """Test that successful match updates current_lane_id."""
        position = (15.0, 0.0)
        
        assert map_matcher.current_lane_id is None
        
        result = map_matcher.match(position)
        
        assert result is not None
        assert map_matcher.current_lane_id == result['lane_id']
    
    def test_match_lane_change_detection(self, map_matcher):
        """Test detection of lane changes."""
        # First match on lane_001
        position1 = (15.0, 0.0)
        result1 = map_matcher.match(position1)
        assert result1['lane_id'] == 'lane_001'
        
        # Move to lane_002 (curved lane)
        position2 = (5.0, 3.0)
        result2 = map_matcher.match(position2)
        
        # Should detect lane change
        assert result2 is not None
        if result2['lane_id'] != result1['lane_id']:
            assert map_matcher.current_lane_id == result2['lane_id']
    
    def test_match_hysteresis_current_lane(self, map_matcher):
        """Test that current lane gets preference (hysteresis)."""
        # Establish current lane
        position1 = (15.0, 0.0)
        map_matcher.match(position1)
        current_lane = map_matcher.current_lane_id
        
        # Move slightly but stay in same lane
        position2 = (16.0, 0.1)
        result = map_matcher.match(position2)
        
        # Should stay in same lane due to hysteresis
        assert result['lane_id'] == current_lane
    
    def test_match_statistics_tracking(self, map_matcher):
        """Test that match statistics are tracked correctly."""
        initial_count = map_matcher.match_count
        initial_success = map_matcher.successful_matches
        
        # Successful match
        map_matcher.match((15.0, 0.0))
        assert map_matcher.match_count == initial_count + 1
        assert map_matcher.successful_matches == initial_success + 1
        
        # Failed match
        map_matcher.match((1000.0, 1000.0))
        assert map_matcher.match_count == initial_count + 2
        assert map_matcher.successful_matches == initial_success + 1
    
    def test_match_with_custom_gps_accuracy(self, map_matcher):
        """Test matching with different GPS accuracy values."""
        position = (15.0, 0.0)
        
        # High accuracy (small search radius)
        result1 = map_matcher.match(position, gps_accuracy=1.0)
        assert result1 is not None
        
        # Low accuracy (large search radius)
        result2 = map_matcher.match(position, gps_accuracy=20.0)
        assert result2 is not None
    
    def test_get_current_lane(self, map_matcher, sample_lane):
        """Test retrieving current lane object."""
        # No current lane initially
        assert map_matcher.get_current_lane() is None
        
        # Match to establish current lane
        map_matcher.match((15.0, 0.0))
        
        current_lane = map_matcher.get_current_lane()
        assert current_lane is not None
        assert current_lane.lane_id == map_matcher.current_lane_id
    
    def test_get_statistics(self, map_matcher):
        """Test retrieving matching statistics."""
        # Initial statistics
        stats = map_matcher.get_statistics()
        assert stats['match_attempts'] == 0
        assert stats['successful_matches'] == 0
        assert stats['success_rate'] == 0
        assert stats['current_lane_id'] is None
        assert stats['num_lanes'] == 2
        
        # After some matches
        map_matcher.match((15.0, 0.0))
        map_matcher.match((16.0, 0.0))
        
        stats = map_matcher.get_statistics()
        assert stats['match_attempts'] == 2
        assert stats['successful_matches'] == 2
        assert stats['success_rate'] == 100.0
        assert stats['current_lane_id'] is not None
    
    def test_find_candidate_lanes(self, map_matcher):
        """Test finding candidate lanes within search radius."""
        position = (15.0, 0.0)
        search_radius = 10.0
        
        candidates = map_matcher._find_candidate_lanes(position, search_radius)
        
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert 'lane_001' in candidates
    
    def test_find_candidate_lanes_small_radius(self, map_matcher):
        """Test finding candidates with very small search radius."""
        position = (15.0, 0.0)
        search_radius = 0.1
        
        candidates = map_matcher._find_candidate_lanes(position, search_radius)
        
        # Should still find lane_001 since position is on it
        assert 'lane_001' in candidates
    
    def test_find_candidate_lanes_prioritizes_current(self, map_matcher):
        """Test that current lane is prioritized in candidate search."""
        # Establish current lane
        map_matcher.match((15.0, 0.0))
        current_lane = map_matcher.current_lane_id
        
        # Search for candidates
        candidates = map_matcher._find_candidate_lanes((16.0, 0.0), 10.0)
        
        # Current lane should be first
        assert candidates[0] == current_lane
    
    def test_match_to_lane(self, map_matcher, sample_lane):
        """Test matching to a specific lane."""
        position = (15.0, 0.5)
        heading = 0.0
        
        result = map_matcher._match_to_lane(position, heading, sample_lane)
        
        assert result is not None
        assert 'lateral_offset' in result
        assert 'longitudinal_position' in result
        assert 'heading_diff' in result
        assert 'closest_idx' in result
    
    def test_match_to_lane_empty_centerline(self, map_matcher):
        """Test matching to lane with empty centerline."""
        empty_lane = Lane(
            lane_id="empty",
            centerline=[],
            left_boundary=[],
            right_boundary=[],
            width=3.5,
            speed_limit=50.0,
            lane_type="driving",
            predecessors=[],
            successors=[]
        )
        
        result = map_matcher._match_to_lane((15.0, 0.0), 0.0, empty_lane)
        
        assert result is None
    
    def test_calculate_lateral_offset(self, map_matcher, sample_lane):
        """Test lateral offset calculation."""
        # Position 1m left of centerline
        position = (15.0, 1.0)
        idx = 1  # Second point on centerline
        
        offset = map_matcher._calculate_lateral_offset(
            position, idx, sample_lane.centerline
        )
        
        # Should be approximately 1.0 (positive = left)
        assert 0.9 < offset < 1.1
    
    def test_calculate_lateral_offset_right(self, map_matcher, sample_lane):
        """Test lateral offset calculation for right side."""
        # Position 1m right of centerline
        position = (15.0, -1.0)
        idx = 1
        
        offset = map_matcher._calculate_lateral_offset(
            position, idx, sample_lane.centerline
        )
        
        # Should be approximately -1.0 (negative = right)
        assert -1.1 < offset < -0.9
    
    def test_calculate_longitudinal_position(self, map_matcher, sample_lane):
        """Test longitudinal position calculation."""
        # At second point (index 1)
        idx = 1
        
        position = map_matcher._calculate_longitudinal_position(
            idx, sample_lane.centerline
        )
        
        # Should be approximately 10m (distance to first segment)
        assert 9.5 < position < 10.5
    
    def test_calculate_longitudinal_position_start(self, map_matcher, sample_lane):
        """Test longitudinal position at start of lane."""
        idx = 0
        
        position = map_matcher._calculate_longitudinal_position(
            idx, sample_lane.centerline
        )
        
        # Should be 0 at start
        assert position == 0.0
    
    def test_calculate_lane_heading(self, map_matcher, sample_lane):
        """Test lane heading calculation."""
        idx = 1
        
        heading = map_matcher._calculate_lane_heading(idx, sample_lane.centerline)
        
        # Lane is along x-axis, so heading should be 0
        assert abs(heading) < 0.1
    
    def test_calculate_lane_heading_single_point(self, map_matcher):
        """Test lane heading with single point centerline."""
        centerline = [(0.0, 0.0, 0.0)]
        
        heading = map_matcher._calculate_lane_heading(0, centerline)
        
        # Should return 0 for single point
        assert heading == 0.0
    
    def test_calculate_match_score(self, map_matcher):
        """Test match score calculation."""
        # Perfect match
        score1 = map_matcher._calculate_match_score(0.0, 0.0, False)
        assert score1 > 0.9
        
        # Large lateral offset
        score2 = map_matcher._calculate_match_score(5.0, 0.0, False)
        assert score2 < 0.5
        
        # Large heading difference
        score3 = map_matcher._calculate_match_score(0.0, np.pi / 2, False)
        assert score3 < 0.7
    
    def test_calculate_match_score_current_lane_bonus(self, map_matcher):
        """Test that current lane gets bonus in match score."""
        lateral_offset = 0.5
        heading_diff = 0.1
        
        score_not_current = map_matcher._calculate_match_score(
            lateral_offset, heading_diff, False
        )
        score_current = map_matcher._calculate_match_score(
            lateral_offset, heading_diff, True
        )
        
        # Current lane should have higher score
        assert score_current > score_not_current
    
    def test_normalize_angle(self, map_matcher):
        """Test angle normalization to [-pi, pi]."""
        # Already normalized
        assert abs(map_matcher._normalize_angle(0.0)) < 1e-6
        assert abs(map_matcher._normalize_angle(np.pi) - np.pi) < 1e-6
        
        # Needs normalization
        angle1 = map_matcher._normalize_angle(3 * np.pi)
        assert -np.pi <= angle1 <= np.pi
        
        angle2 = map_matcher._normalize_angle(-3 * np.pi)
        assert -np.pi <= angle2 <= np.pi
    
    def test_match_edge_case_none_position(self, map_matcher):
        """Test matching with None position (should handle gracefully)."""
        # This should raise an error or be handled
        with pytest.raises((TypeError, AttributeError)):
            map_matcher.match(None)
    
    def test_match_edge_case_invalid_position(self, map_matcher):
        """Test matching with invalid position format."""
        # Single value instead of tuple
        with pytest.raises((TypeError, ValueError)):
            map_matcher.match(15.0)
    
    @pytest.mark.performance
    def test_performance_single_match(self, map_matcher):
        """Test that single match completes within performance requirements."""
        import time
        
        position = (15.0, 0.0)
        heading = 0.0
        
        start_time = time.perf_counter()
        result = map_matcher.match(position, heading)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        assert result is not None
        assert execution_time_ms < 5.0, f"Match took {execution_time_ms:.2f}ms, expected < 5ms"
    
    @pytest.mark.performance
    def test_performance_multiple_matches(self, map_matcher):
        """Test performance with multiple consecutive matches."""
        import time
        
        positions = [
            (10.0, 0.0),
            (15.0, 0.0),
            (20.0, 0.0),
            (25.0, 0.0),
            (30.0, 0.0),
        ]
        
        times = []
        for position in positions:
            start_time = time.perf_counter()
            map_matcher.match(position, heading=0.0)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        
        assert avg_time < 5.0, f"Average match time {avg_time:.2f}ms exceeds 5ms"
        assert max_time < 10.0, f"Max match time {max_time:.2f}ms exceeds 10ms"
    
    def test_match_with_many_lanes(self):
        """Test matching performance with many lanes."""
        # Create 100 lanes
        lanes = {}
        for i in range(100):
            centerline = [
                (float(j), float(i * 5), 0.0) for j in range(0, 50, 10)
            ]
            lanes[f"lane_{i:03d}"] = Lane(
                lane_id=f"lane_{i:03d}",
                centerline=centerline,
                left_boundary=[],
                right_boundary=[],
                width=3.5,
                speed_limit=50.0,
                lane_type="driving",
                predecessors=[],
                successors=[]
            )
        
        matcher = MapMatcher(lanes)
        
        # Match should still be fast
        import time
        start_time = time.perf_counter()
        result = matcher.match((25.0, 10.0))
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        assert result is not None
        assert execution_time_ms < 50.0, f"Match with 100 lanes took {execution_time_ms:.2f}ms"
    
    def test_curved_lane_matching(self, map_matcher, curved_lane):
        """Test matching on curved lane."""
        # Position on curved lane
        position = (5.0, 3.0)
        
        result = map_matcher.match(position)
        
        assert result is not None
        # Should match to one of the lanes
        assert result['lane_id'] in ['lane_001', 'lane_002']
    
    def test_match_logging(self, map_matcher, caplog):
        """Test that matching produces appropriate log messages."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        position = (15.0, 0.0)
        map_matcher.match(position)
        
        # Check that debug logs were produced
        assert any("Map matching started" in record.message for record in caplog.records)
        assert any("Map matching completed" in record.message for record in caplog.records)
    
    def test_match_statistics_logging(self, map_matcher, caplog):
        """Test that statistics are logged periodically."""
        import logging
        caplog.set_level(logging.INFO)
        
        # Perform 100 matches to trigger statistics logging
        for i in range(100):
            map_matcher.match((10.0 + i * 0.1, 0.0))
        
        # Should have statistics log
        assert any("Map matching statistics" in record.message for record in caplog.records)
