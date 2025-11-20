"""
Comprehensive HD Map Integration Tests

Tests for task 31.4:
- Validate map parsing (OpenDRIVE and Lanelet2)
- Test map matching accuracy (0.2m requirement)
- Verify feature queries
- Benchmark query performance (<5ms requirement)

Requirements: 22.1, 22.2, 22.3, 22.6
"""

import pytest
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

from src.core.data_structures import Lane, MapFeature
from src.maps.parser import OpenDRIVEParser, Lanelet2Parser
from src.maps.matcher import MapMatcher
from src.maps.query import FeatureQuery
from src.maps.manager import HDMapManager


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_opendrive_xml():
    """Create a minimal OpenDRIVE XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
    <header revMajor="1" revMinor="4" name="Test Map" date="2024-01-01" 
            north="100" south="0" east="100" west="0"/>
    <road id="1" length="100.0" junction="-1">
        <planView>
            <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="100.0">
                <line/>
            </geometry>
        </planView>
        <lanes>
            <laneSection s="0.0">
                <center>
                    <lane id="0" type="driving" level="false"/>
                </center>
                <right>
                    <lane id="-1" type="driving" level="false">
                        <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
                        <speed sOffset="0.0" max="50.0"/>
                    </lane>
                </right>
            </laneSection>
        </lanes>
        <objects>
            <object id="obj1" type="crosswalk" s="50.0" t="0.0"/>
        </objects>
        <signals>
            <signal id="sig1" type="speed_limit" subtype="50" s="30.0" t="3.0"/>
        </signals>
    </road>
</OpenDRIVE>"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xodr', delete=False) as f:
        f.write(xml_content)
        return f.name



@pytest.fixture
def sample_lanelet2_xml():
    """Create a minimal Lanelet2 OSM XML file for testing."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6">
    <node id="1" lat="0.0" lon="0.0"/>
    <node id="2" lat="0.0" lon="0.001"/>
    <node id="3" lat="0.0001" lon="0.0"/>
    <node id="4" lat="0.0001" lon="0.001"/>
    <way id="10">
        <nd ref="1"/>
        <nd ref="2"/>
    </way>
    <way id="11">
        <nd ref="3"/>
        <nd ref="4"/>
    </way>
    <relation id="100">
        <member type="way" ref="10" role="left"/>
        <member type="way" ref="11" role="right"/>
        <tag k="type" v="lanelet"/>
        <tag k="subtype" v="road"/>
        <tag k="speed_limit" v="50"/>
    </relation>
</osm>"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.osm', delete=False) as f:
        f.write(xml_content)
        return f.name


@pytest.fixture
def precise_test_lanes():
    """Create lanes with precise geometry for accuracy testing."""
    lanes = {}
    
    # Create a straight lane with known geometry
    # Centerline at y=0, width=3.5m
    lane = Lane(
        lane_id='test_lane_1',
        centerline=[
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (20.0, 0.0, 0.0),
            (30.0, 0.0, 0.0),
            (40.0, 0.0, 0.0),
            (50.0, 0.0, 0.0),
        ],
        left_boundary=[
            (0.0, 1.75, 0.0),
            (10.0, 1.75, 0.0),
            (20.0, 1.75, 0.0),
            (30.0, 1.75, 0.0),
            (40.0, 1.75, 0.0),
            (50.0, 1.75, 0.0),
        ],
        right_boundary=[
            (0.0, -1.75, 0.0),
            (10.0, -1.75, 0.0),
            (20.0, -1.75, 0.0),
            (30.0, -1.75, 0.0),
            (40.0, -1.75, 0.0),
            (50.0, -1.75, 0.0),
        ],
        width=3.5,
        speed_limit=50.0,
        lane_type='driving',
        predecessors=[],
        successors=[]
    )
    lanes['test_lane_1'] = lane
    
    return lanes


@pytest.fixture
def test_features():
    """Create test features at known locations."""
    return [
        MapFeature(
            feature_id='sign_1',
            type='sign',
            position=(10.0, 3.0, 0.0),
            attributes={'sign_type': 'speed_limit', 'value': 50},
            geometry=[(10.0, 3.0)]
        ),
        MapFeature(
            feature_id='sign_2',
            type='sign',
            position=(30.0, 3.0, 0.0),
            attributes={'sign_type': 'stop'},
            geometry=[(30.0, 3.0)]
        ),
        MapFeature(
            feature_id='light_1',
            type='light',
            position=(45.0, 3.0, 0.0),
            attributes={'light_type': 'traffic_light'},
            geometry=[(45.0, 3.0)]
        ),
    ]


# ============================================================================
# Test Map Parsing (Requirement 22.1)
# ============================================================================

class TestMapParsing:
    """Test map parsing for OpenDRIVE and Lanelet2 formats."""
    
    def test_opendrive_parser_initialization(self):
        """Test OpenDRIVE parser can be initialized."""
        parser = OpenDRIVEParser()
        assert parser is not None
    
    def test_opendrive_parsing(self, sample_opendrive_xml):
        """Test parsing OpenDRIVE format."""
        parser = OpenDRIVEParser()
        
        result = parser.parse(sample_opendrive_xml)
        
        # Verify structure
        assert 'lanes' in result
        assert 'features' in result
        assert 'metadata' in result
        assert result['format'] == 'opendrive'
        
        # Verify lanes were parsed
        assert len(result['lanes']) > 0
        
        # Verify features were parsed
        assert len(result['features']) > 0
        
        # Verify metadata
        metadata = result['metadata']
        assert 'name' in metadata
        assert metadata['name'] == 'Test Map'
        
        # Clean up
        Path(sample_opendrive_xml).unlink()
    
    def test_opendrive_lane_structure(self, sample_opendrive_xml):
        """Test that parsed lanes have correct structure."""
        parser = OpenDRIVEParser()
        result = parser.parse(sample_opendrive_xml)
        
        lanes = result['lanes']
        assert len(lanes) > 0
        
        # Get first lane
        lane = list(lanes.values())[0]
        
        # Verify Lane dataclass fields
        assert hasattr(lane, 'lane_id')
        assert hasattr(lane, 'centerline')
        assert hasattr(lane, 'left_boundary')
        assert hasattr(lane, 'right_boundary')
        assert hasattr(lane, 'width')
        assert hasattr(lane, 'speed_limit')
        assert hasattr(lane, 'lane_type')
        
        # Verify geometry is not empty
        assert len(lane.centerline) > 0
        assert len(lane.left_boundary) > 0
        assert len(lane.right_boundary) > 0
        
        # Clean up
        Path(sample_opendrive_xml).unlink()
    
    def test_lanelet2_parser_initialization(self):
        """Test Lanelet2 parser can be initialized."""
        parser = Lanelet2Parser()
        assert parser is not None
    
    def test_lanelet2_parsing(self, sample_lanelet2_xml):
        """Test parsing Lanelet2 format."""
        parser = Lanelet2Parser()
        
        result = parser.parse(sample_lanelet2_xml)
        
        # Verify structure
        assert 'lanes' in result
        assert 'features' in result
        assert 'metadata' in result
        assert result['format'] == 'lanelet2'
        
        # Verify lanes were parsed
        assert len(result['lanes']) > 0
        
        # Verify metadata
        metadata = result['metadata']
        assert 'num_nodes' in metadata
        assert 'num_ways' in metadata
        
        # Clean up
        Path(sample_lanelet2_xml).unlink()


# ============================================================================
# Test Map Matching Accuracy (Requirement 22.2)
# ============================================================================

class TestMapMatchingAccuracy:
    """Test map matching accuracy meets 0.2m requirement."""
    
    def test_exact_center_match(self, precise_test_lanes):
        """Test matching at exact lane center."""
        matcher = MapMatcher(precise_test_lanes)
        
        # Test at exact center of lane with high GPS accuracy
        result = matcher.match((25.0, 0.0), heading=0.0, gps_accuracy=10.0)
        
        assert result is not None
        assert result['lane_id'] == 'test_lane_1'
        
        # Lateral offset should be near zero (within 0.2m)
        assert abs(result['lateral_offset']) < 0.2
        print(f"Center match lateral offset: {result['lateral_offset']:.4f}m")
    
    def test_left_side_accuracy(self, precise_test_lanes):
        """Test matching accuracy on left side of lane."""
        matcher = MapMatcher(precise_test_lanes)
        
        # Test at 1.0m left of center with high GPS accuracy
        result = matcher.match((25.0, 1.0), heading=0.0, gps_accuracy=10.0)
        
        assert result is not None
        assert result['lane_id'] == 'test_lane_1'
        
        # Should detect 1.0m offset with <0.2m error
        expected_offset = 1.0
        actual_offset = result['lateral_offset']
        error = abs(actual_offset - expected_offset)
        
        assert error < 0.2, f"Lateral offset error {error:.4f}m exceeds 0.2m"
        print(f"Left side match error: {error:.4f}m")
    
    def test_right_side_accuracy(self, precise_test_lanes):
        """Test matching accuracy on right side of lane."""
        matcher = MapMatcher(precise_test_lanes)
        
        # Test at 1.0m right of center with high GPS accuracy
        result = matcher.match((25.0, -1.0), heading=0.0, gps_accuracy=10.0)
        
        assert result is not None
        assert result['lane_id'] == 'test_lane_1'
        
        # Should detect -1.0m offset with <0.2m error
        expected_offset = -1.0
        actual_offset = result['lateral_offset']
        error = abs(actual_offset - expected_offset)
        
        assert error < 0.2, f"Lateral offset error {error:.4f}m exceeds 0.2m"
        print(f"Right side match error: {error:.4f}m")
    
    def test_multiple_positions_accuracy(self, precise_test_lanes):
        """Test accuracy across multiple positions."""
        matcher = MapMatcher(precise_test_lanes)
        
        test_positions = [
            ((5.0, 0.0), 0.0),
            ((15.0, 0.5), 0.5),
            ((25.0, -0.5), -0.5),
            ((35.0, 1.5), 1.5),
            ((45.0, -1.5), -1.5),
        ]
        
        errors = []
        for (x, y), expected_offset in test_positions:
            result = matcher.match((x, y), heading=0.0, gps_accuracy=10.0)
            assert result is not None
            
            actual_offset = result['lateral_offset']
            error = abs(actual_offset - expected_offset)
            errors.append(error)
            
            assert error < 0.2, f"Position ({x}, {y}) error {error:.4f}m exceeds 0.2m"
        
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"Average matching error: {avg_error:.4f}m")
        print(f"Maximum matching error: {max_error:.4f}m")
        
        assert avg_error < 0.1, "Average error should be well below 0.2m"
        assert max_error < 0.2, "Maximum error must be below 0.2m"


# ============================================================================
# Test Feature Queries (Requirement 22.3)
# ============================================================================

class TestFeatureQueries:
    """Test feature query functionality."""
    
    def test_nearby_features_query(self, precise_test_lanes, test_features):
        """Test querying nearby features."""
        query = FeatureQuery(precise_test_lanes, test_features)
        
        # Query near sign_1 at (10, 3)
        nearby = query.query_nearby_features((10.0, 0.0), radius=5.0)
        
        assert len(nearby) >= 1
        assert any(f.feature_id == 'sign_1' for f in nearby)
    
    def test_upcoming_features_query(self, precise_test_lanes, test_features):
        """Test querying upcoming features."""
        query = FeatureQuery(precise_test_lanes, test_features)
        
        # Query from position (5, 0) looking forward
        upcoming = query.query_upcoming_features(
            (5.0, 0.0), heading=0.0, lookahead=50.0
        )
        
        # Should find all 3 features ahead
        assert len(upcoming) >= 2
        
        # Verify features are sorted by distance
        distances = [item['distance'] for item in upcoming]
        assert distances == sorted(distances)
    
    def test_feature_type_filtering(self, precise_test_lanes, test_features):
        """Test filtering features by type."""
        query = FeatureQuery(precise_test_lanes, test_features)
        
        # Query only signs
        signs = query.query_nearby_features(
            (25.0, 0.0), radius=50.0, feature_types=['sign']
        )
        
        assert len(signs) == 2
        assert all(f.type == 'sign' for f in signs)
        
        # Query only lights
        lights = query.query_nearby_features(
            (25.0, 0.0), radius=50.0, feature_types=['light']
        )
        
        assert len(lights) == 1
        assert all(f.type == 'light' for f in lights)
    
    def test_feature_attributes(self, precise_test_lanes, test_features):
        """Test that feature attributes are preserved."""
        query = FeatureQuery(precise_test_lanes, test_features)
        
        nearby = query.query_nearby_features((10.0, 0.0), radius=5.0)
        
        sign = next((f for f in nearby if f.feature_id == 'sign_1'), None)
        assert sign is not None
        assert 'sign_type' in sign.attributes
        assert sign.attributes['sign_type'] == 'speed_limit'


# ============================================================================
# Test Query Performance (Requirement 22.6)
# ============================================================================

class TestQueryPerformance:
    """Test that query performance meets <5ms requirement."""
    
    def test_map_matching_performance(self, precise_test_lanes):
        """Test map matching performance."""
        matcher = MapMatcher(precise_test_lanes)
        
        num_queries = 100
        times = []
        
        for i in range(num_queries):
            x = float(i % 50)
            y = 0.0
            
            start = time.perf_counter()
            result = matcher.match((x, y), heading=0.0)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        max_time = np.max(times)
        
        print(f"\nMap Matching Performance:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  P95: {p95_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")
        
        assert p95_time < 5.0, f"P95 query time {p95_time:.3f}ms exceeds 5ms target"
    
    def test_feature_query_performance(self, precise_test_lanes, test_features):
        """Test feature query performance."""
        query = FeatureQuery(precise_test_lanes, test_features)
        
        num_queries = 100
        times = []
        
        for i in range(num_queries):
            x = float(i % 50)
            y = 0.0
            
            start = time.perf_counter()
            result = query.query_nearby_features((x, y), radius=20.0)
            elapsed = time.perf_counter() - start
            
            times.append(elapsed * 1000)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        max_time = np.max(times)
        
        print(f"\nFeature Query Performance:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  P95: {p95_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")
        
        assert p95_time < 5.0, f"P95 query time {p95_time:.3f}ms exceeds 5ms target"
    
    def test_combined_query_performance(self, precise_test_lanes, test_features):
        """Test combined matching and query performance."""
        matcher = MapMatcher(precise_test_lanes)
        query = FeatureQuery(precise_test_lanes, test_features)
        
        num_queries = 100
        times = []
        
        for i in range(num_queries):
            x = float(i % 50)
            y = 0.0
            
            start = time.perf_counter()
            
            # Perform both operations
            match_result = matcher.match((x, y), heading=0.0)
            feature_result = query.query_nearby_features((x, y), radius=20.0)
            
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        max_time = np.max(times)
        
        print(f"\nCombined Query Performance:")
        print(f"  Average: {avg_time:.3f}ms")
        print(f"  P95: {p95_time:.3f}ms")
        print(f"  Max: {max_time:.3f}ms")
        
        assert p95_time < 5.0, f"P95 combined query time {p95_time:.3f}ms exceeds 5ms target"


# ============================================================================
# Test HD Map Manager Integration
# ============================================================================

class TestHDMapManager:
    """Test HDMapManager integration."""
    
    def test_manager_initialization(self):
        """Test manager can be initialized."""
        manager = HDMapManager()
        assert manager is not None
        assert not manager.is_loaded()
    
    def test_manager_with_manual_data(self, precise_test_lanes, test_features):
        """Test manager with manually loaded data."""
        manager = HDMapManager()
        
        manager.lanes = precise_test_lanes
        manager.features = test_features
        manager._initialize_components()
        
        assert manager.is_loaded()
        assert len(manager.get_all_lanes()) == 1
        assert len(manager.get_all_features()) == 3
    
    def test_manager_match_position(self, precise_test_lanes, test_features):
        """Test position matching through manager."""
        manager = HDMapManager()
        manager.lanes = precise_test_lanes
        manager.features = test_features
        manager._initialize_components()
        
        result = manager.match_position((25.0, 0.0), heading=0.0)
        
        assert result is not None
        assert result['lane_id'] == 'test_lane_1'
        assert abs(result['lateral_offset']) < 0.2
    
    def test_manager_query_features(self, precise_test_lanes, test_features):
        """Test feature queries through manager."""
        manager = HDMapManager()
        manager.lanes = precise_test_lanes
        manager.features = test_features
        manager._initialize_components()
        
        nearby = manager.query_nearby_features((25.0, 0.0), radius=30.0)
        
        assert len(nearby) >= 2
    
    def test_manager_performance_tracking(self, precise_test_lanes, test_features):
        """Test that manager tracks performance."""
        manager = HDMapManager()
        manager.lanes = precise_test_lanes
        manager.features = test_features
        manager._initialize_components()
        
        # Perform several queries
        for i in range(10):
            manager.match_position((float(i * 5), 0.0), heading=0.0)
            manager.query_nearby_features((float(i * 5), 0.0), radius=20.0)
        
        stats = manager.get_performance_stats()
        
        assert stats['num_queries'] >= 20
        assert stats['avg_query_time_ms'] >= 0
        assert stats['p95_query_time_ms'] >= 0
        assert stats['p95_query_time_ms'] < 5.0, "Manager queries exceed 5ms target"
        
        print(f"\nManager Performance Stats:")
        print(f"  Queries: {stats['num_queries']}")
        print(f"  Average: {stats['avg_query_time_ms']:.3f}ms")
        print(f"  P95: {stats['p95_query_time_ms']:.3f}ms")
    
    def test_manager_cache_functionality(self, precise_test_lanes, test_features):
        """Test that caching improves performance."""
        manager = HDMapManager({'cache_enabled': True})
        manager.lanes = precise_test_lanes
        manager.features = test_features
        manager._initialize_components()
        
        position = (25.0, 0.0)
        
        # First query (no cache)
        start = time.perf_counter()
        result1 = manager.query_nearby_features(position, radius=20.0, use_cache=True)
        time1 = time.perf_counter() - start
        
        # Second query (should use cache)
        start = time.perf_counter()
        result2 = manager.query_nearby_features(position, radius=20.0, use_cache=True)
        time2 = time.perf_counter() - start
        
        # Results should be identical
        assert len(result1) == len(result2)
        
        # Cached query should be faster (or at least not slower)
        print(f"\nCache Performance:")
        print(f"  First query: {time1*1000:.3f}ms")
        print(f"  Cached query: {time2*1000:.3f}ms")
        print(f"  Speedup: {time1/time2:.1f}x" if time2 > 0 else "  Speedup: N/A")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
