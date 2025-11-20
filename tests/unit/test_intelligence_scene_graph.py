"""Test suite for scene graph builder module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.intelligence.scene_graph import SceneGraphBuilder
from src.core.data_structures import Detection3D


@pytest.fixture
def scene_graph_builder():
    """Fixture creating an instance of SceneGraphBuilder for testing."""
    return SceneGraphBuilder()


@pytest.fixture
def sample_detection():
    """Fixture providing a sample 3D detection."""
    return Detection3D(
        bbox_3d=(5.0, 2.0, 0.0, 2.0, 1.5, 4.5, 0.0),  # x, y, z, w, h, l, theta
        class_name='car',
        confidence=0.95,
        velocity=(10.0, 0.0, 0.0),
        track_id=1
    )


@pytest.fixture
def multiple_detections():
    """Fixture providing multiple 3D detections for testing."""
    return [
        Detection3D(
            bbox_3d=(5.0, 2.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='car',
            confidence=0.95,
            velocity=(10.0, 0.0, 0.0),
            track_id=1
        ),
        Detection3D(
            bbox_3d=(10.0, -3.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='car',
            confidence=0.92,
            velocity=(8.0, 0.0, 0.0),
            track_id=2
        ),
        Detection3D(
            bbox_3d=(3.0, 0.5, 0.0, 0.5, 1.8, 0.5, 0.0),
            class_name='pedestrian',
            confidence=0.88,
            velocity=(1.0, 0.5, 0.0),
            track_id=3
        )
    ]


class TestSceneGraphBuilder:
    """Test suite for SceneGraphBuilder class."""
    
    def test_initialization(self, scene_graph_builder):
        """Test that SceneGraphBuilder initializes correctly."""
        assert scene_graph_builder is not None
        assert scene_graph_builder.logger is not None
        assert scene_graph_builder.logger.name == 'src.intelligence.scene_graph'
    
    def test_build_empty_detections(self, scene_graph_builder):
        """Test building scene graph with no detections."""
        scene_graph = scene_graph_builder.build([])
        
        assert scene_graph is not None
        assert scene_graph['objects'] == []
        assert scene_graph['relationships'] == []
        assert scene_graph['spatial_map'] == {}
    
    def test_build_single_detection(self, scene_graph_builder, sample_detection):
        """Test building scene graph with a single detection."""
        scene_graph = scene_graph_builder.build([sample_detection])
        
        assert scene_graph is not None
        assert len(scene_graph['objects']) == 1
        assert scene_graph['num_objects'] == 1
        
        # Verify object properties
        obj = scene_graph['objects'][0]
        assert obj['id'] == 1
        assert obj['type'] == 'car'
        assert obj['position'] == (5.0, 2.0, 0.0)
        assert obj['dimensions'] == (2.0, 1.5, 4.5)
        assert obj['orientation'] == 0.0
        assert obj['velocity'] == (10.0, 0.0, 0.0)
        assert obj['confidence'] == 0.95
        
        # Single object should have no relationships
        assert len(scene_graph['relationships']) == 0
    
    def test_build_multiple_detections(self, scene_graph_builder, multiple_detections):
        """Test building scene graph with multiple detections."""
        scene_graph = scene_graph_builder.build(multiple_detections)
        
        assert scene_graph is not None
        assert len(scene_graph['objects']) == 3
        assert scene_graph['num_objects'] == 3
        
        # Verify all objects are present
        object_ids = [obj['id'] for obj in scene_graph['objects']]
        assert 1 in object_ids
        assert 2 in object_ids
        assert 3 in object_ids
        
        # Verify relationships (3 objects = 3 relationships: 1-2, 1-3, 2-3)
        assert len(scene_graph['relationships']) == 3
    
    def test_relationships_calculation(self, scene_graph_builder, multiple_detections):
        """Test that spatial relationships are calculated correctly."""
        scene_graph = scene_graph_builder.build(multiple_detections)
        relationships = scene_graph['relationships']
        
        # Find relationship between objects 1 and 2
        rel_1_2 = next((r for r in relationships if 
                       (r['object1_id'] == 1 and r['object2_id'] == 2)), None)
        
        assert rel_1_2 is not None
        
        # Calculate expected distance: sqrt((10-5)^2 + (-3-2)^2 + 0^2) = sqrt(25 + 25) = 7.07
        expected_distance = np.sqrt((10.0 - 5.0)**2 + (-3.0 - 2.0)**2)
        assert abs(rel_1_2['distance'] - expected_distance) < 0.01
        
        # Verify relative position
        assert len(rel_1_2['relative_position']) == 3
        assert rel_1_2['relative_position'][0] == 5.0  # x difference
        assert rel_1_2['relative_position'][1] == -5.0  # y difference
        
        # Verify proximity categorization
        assert rel_1_2['proximity'] in ['very_close', 'close', 'near', 'far']
    
    def test_proximity_categorization(self, scene_graph_builder):
        """Test proximity categorization based on distance."""
        assert scene_graph_builder._categorize_proximity(1.0) == 'very_close'
        assert scene_graph_builder._categorize_proximity(1.99) == 'very_close'
        assert scene_graph_builder._categorize_proximity(2.0) == 'close'
        assert scene_graph_builder._categorize_proximity(4.99) == 'close'
        assert scene_graph_builder._categorize_proximity(5.0) == 'near'
        assert scene_graph_builder._categorize_proximity(9.99) == 'near'
        assert scene_graph_builder._categorize_proximity(10.0) == 'far'
        assert scene_graph_builder._categorize_proximity(100.0) == 'far'
    
    def test_spatial_map_creation(self, scene_graph_builder, multiple_detections):
        """Test that spatial map is created correctly."""
        scene_graph = scene_graph_builder.build(multiple_detections)
        spatial_map = scene_graph['spatial_map']
        
        assert 'grid' in spatial_map
        assert 'grid_size' in spatial_map
        assert 'grid_resolution' in spatial_map
        assert 'grid_cells' in spatial_map
        
        # Verify grid parameters
        assert spatial_map['grid_size'] == 64
        assert spatial_map['grid_resolution'] == 0.5
        assert spatial_map['grid_cells'] == 128
        
        # Verify grid contains object IDs
        grid = spatial_map['grid']
        assert isinstance(grid, dict)
        
        # At least some cells should be occupied
        assert len(grid) > 0
        
        # Each cell should contain a list of object IDs
        for cell_key, object_ids in grid.items():
            assert isinstance(object_ids, list)
            assert all(isinstance(obj_id, int) for obj_id in object_ids)
    
    def test_spatial_map_grid_coordinates(self, scene_graph_builder):
        """Test that objects are placed in correct grid cells."""
        # Create detection at known position
        detection = Detection3D(
            bbox_3d=(0.0, 0.0, 0.0, 2.0, 1.5, 4.5, 0.0),  # At origin
            class_name='car',
            confidence=0.95,
            velocity=(0.0, 0.0, 0.0),
            track_id=100
        )
        
        scene_graph = scene_graph_builder.build([detection])
        spatial_map = scene_graph['spatial_map']
        grid = spatial_map['grid']
        
        # Object at (0, 0) should be in center cell
        # Grid coordinates: (0 + 32) / 0.5 = 64, (0 + 32) / 0.5 = 64
        expected_cell = "64,64"
        
        # Check if object is in the expected cell or nearby (due to rounding)
        found = False
        for cell_key, object_ids in grid.items():
            if 100 in object_ids:
                found = True
                # Verify it's near the center
                x, y = map(int, cell_key.split(','))
                assert 60 <= x <= 68  # Allow some tolerance
                assert 60 <= y <= 68
                break
        
        assert found, "Object not found in spatial grid"
    
    def test_spatial_map_out_of_bounds(self, scene_graph_builder):
        """Test that objects outside grid bounds are handled correctly."""
        # Create detection far outside grid (grid is 64m x 64m)
        detection = Detection3D(
            bbox_3d=(100.0, 100.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='car',
            confidence=0.95,
            velocity=(0.0, 0.0, 0.0),
            track_id=999
        )
        
        scene_graph = scene_graph_builder.build([detection])
        spatial_map = scene_graph['spatial_map']
        grid = spatial_map['grid']
        
        # Object should not appear in grid (out of bounds)
        for cell_key, object_ids in grid.items():
            assert 999 not in object_ids
    
    def test_build_with_close_objects(self, scene_graph_builder):
        """Test scene graph with very close objects."""
        close_detections = [
            Detection3D(
                bbox_3d=(5.0, 0.0, 0.0, 2.0, 1.5, 4.5, 0.0),
                class_name='car',
                confidence=0.95,
                velocity=(10.0, 0.0, 0.0),
                track_id=1
            ),
            Detection3D(
                bbox_3d=(5.5, 0.5, 0.0, 2.0, 1.5, 4.5, 0.0),
                class_name='car',
                confidence=0.92,
                velocity=(10.0, 0.0, 0.0),
                track_id=2
            )
        ]
        
        scene_graph = scene_graph_builder.build(close_detections)
        relationships = scene_graph['relationships']
        
        assert len(relationships) == 1
        rel = relationships[0]
        
        # Distance should be small
        assert rel['distance'] < 2.0
        assert rel['proximity'] == 'very_close'
    
    def test_build_preserves_all_detection_data(self, scene_graph_builder, sample_detection):
        """Test that all detection data is preserved in scene graph."""
        scene_graph = scene_graph_builder.build([sample_detection])
        obj = scene_graph['objects'][0]
        
        # Verify all fields are present and correct
        assert obj['id'] == sample_detection.track_id
        assert obj['type'] == sample_detection.class_name
        assert obj['confidence'] == sample_detection.confidence
        assert obj['velocity'] == sample_detection.velocity
        
        # Verify bbox_3d is correctly unpacked
        x, y, z, w, h, l, theta = sample_detection.bbox_3d
        assert obj['position'] == (x, y, z)
        assert obj['dimensions'] == (w, h, l)
        assert obj['orientation'] == theta
    
    @pytest.mark.performance
    def test_performance_small_scene(self, scene_graph_builder, multiple_detections):
        """Test that scene graph building completes within performance requirements."""
        import time
        
        start_time = time.perf_counter()
        scene_graph = scene_graph_builder.build(multiple_detections)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Scene graph building should be very fast (< 5ms for small scenes)
        assert execution_time_ms < 5.0, f"Execution took {execution_time_ms:.2f}ms, expected < 5ms"
        assert scene_graph is not None
    
    @pytest.mark.performance
    def test_performance_large_scene(self, scene_graph_builder):
        """Test performance with a large number of objects."""
        import time
        
        # Create 50 detections (realistic worst case)
        large_detections = []
        for i in range(50):
            detection = Detection3D(
                bbox_3d=(float(i), float(i % 10), 0.0, 2.0, 1.5, 4.5, 0.0),
                class_name='car',
                confidence=0.9,
                velocity=(10.0, 0.0, 0.0),
                track_id=i
            )
            large_detections.append(detection)
        
        start_time = time.perf_counter()
        scene_graph = scene_graph_builder.build(large_detections)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should still complete quickly even with many objects (< 10ms target)
        assert execution_time_ms < 10.0, f"Execution took {execution_time_ms:.2f}ms, expected < 10ms"
        assert scene_graph['num_objects'] == 50
        # 50 objects = 50*49/2 = 1225 relationships
        assert len(scene_graph['relationships']) == 1225
    
    def test_relationship_symmetry(self, scene_graph_builder, multiple_detections):
        """Test that relationships are not duplicated (only one direction)."""
        scene_graph = scene_graph_builder.build(multiple_detections)
        relationships = scene_graph['relationships']
        
        # Check that we don't have both (A, B) and (B, A)
        pairs = set()
        for rel in relationships:
            id1, id2 = rel['object1_id'], rel['object2_id']
            # Ensure id1 < id2 (relationships should be ordered)
            assert id1 < id2, "Relationships should have object1_id < object2_id"
            
            pair = (id1, id2)
            assert pair not in pairs, f"Duplicate relationship found: {pair}"
            pairs.add(pair)
    
    def test_scene_graph_structure(self, scene_graph_builder, multiple_detections):
        """Test that scene graph has the expected structure."""
        scene_graph = scene_graph_builder.build(multiple_detections)
        
        # Verify top-level keys
        assert 'objects' in scene_graph
        assert 'relationships' in scene_graph
        assert 'spatial_map' in scene_graph
        assert 'num_objects' in scene_graph
        
        # Verify objects structure
        for obj in scene_graph['objects']:
            assert 'id' in obj
            assert 'type' in obj
            assert 'position' in obj
            assert 'dimensions' in obj
            assert 'orientation' in obj
            assert 'velocity' in obj
            assert 'confidence' in obj
        
        # Verify relationships structure
        for rel in scene_graph['relationships']:
            assert 'object1_id' in rel
            assert 'object2_id' in rel
            assert 'distance' in rel
            assert 'relative_position' in rel
            assert 'proximity' in rel
        
        # Verify spatial_map structure
        spatial_map = scene_graph['spatial_map']
        assert 'grid' in spatial_map
        assert 'grid_size' in spatial_map
        assert 'grid_resolution' in spatial_map
        assert 'grid_cells' in spatial_map
