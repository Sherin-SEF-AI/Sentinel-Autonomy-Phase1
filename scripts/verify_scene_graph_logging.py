"""Verify SceneGraphBuilder logging implementation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import logging.config
import yaml
from src.intelligence.scene_graph import SceneGraphBuilder
from src.core.data_structures import Detection3D


def setup_logging():
    """Setup logging from config file."""
    with open('configs/logging.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)


def create_test_detections():
    """Create test detections for scene graph."""
    detections = [
        Detection3D(
            bbox_3d=(5.0, 2.0, 0.0, 1.8, 1.5, 4.5, 0.0),
            class_name='car',
            confidence=0.95,
            velocity=(10.0, 0.0, 0.0),
            track_id=1
        ),
        Detection3D(
            bbox_3d=(8.0, -1.0, 0.0, 1.8, 1.5, 4.5, 0.0),
            class_name='car',
            confidence=0.92,
            velocity=(12.0, 0.0, 0.0),
            track_id=2
        ),
        Detection3D(
            bbox_3d=(3.0, 1.5, 0.0, 0.6, 1.8, 0.6, 0.0),
            class_name='pedestrian',
            confidence=0.88,
            velocity=(1.5, 0.0, 0.0),
            track_id=3
        ),
    ]
    return detections


def main():
    """Test SceneGraphBuilder logging."""
    print("=" * 60)
    print("SceneGraphBuilder Logging Verification")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting SceneGraphBuilder logging verification")
    
    # Initialize scene graph builder
    print("\n1. Initializing SceneGraphBuilder...")
    builder = SceneGraphBuilder()
    
    # Test with empty detections
    print("\n2. Testing with empty detections...")
    empty_graph = builder.build([])
    print(f"   Empty graph: {empty_graph['num_objects']} objects")
    
    # Test with sample detections
    print("\n3. Testing with sample detections...")
    detections = create_test_detections()
    scene_graph = builder.build(detections)
    
    print(f"\n   Scene Graph Results:")
    print(f"   - Objects: {scene_graph['num_objects']}")
    print(f"   - Relationships: {len(scene_graph['relationships'])}")
    print(f"   - Occupied cells: {len(scene_graph['spatial_map']['grid'])}")
    
    # Display relationships
    print(f"\n   Spatial Relationships:")
    for rel in scene_graph['relationships']:
        print(f"   - Object {rel['object1_id']} <-> Object {rel['object2_id']}: "
              f"{rel['distance']:.2f}m ({rel['proximity']})")
    
    # Test multiple times for performance
    print("\n4. Testing performance (10 iterations)...")
    import time
    times = []
    for i in range(10):
        start = time.time()
        builder.build(detections)
        duration = (time.time() - start) * 1000
        times.append(duration)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    print(f"   Average time: {avg_time:.2f}ms")
    print(f"   Max time: {max_time:.2f}ms")
    print(f"   Target: 2.0ms")
    
    if avg_time < 2.0:
        print(f"   ✓ Performance target met!")
    else:
        print(f"   ⚠ Performance target exceeded by {avg_time - 2.0:.2f}ms")
    
    # Check log files
    print("\n5. Checking log files...")
    log_files = [
        'logs/intelligence.log',
        'logs/sentinel.log'
    ]
    
    for log_file in log_files:
        if Path(log_file).exists():
            print(f"   ✓ {log_file} exists")
            # Count SceneGraphBuilder entries
            with open(log_file, 'r') as f:
                lines = f.readlines()
                scene_graph_lines = [l for l in lines if 'SceneGraphBuilder' in l]
                print(f"     - SceneGraphBuilder entries: {len(scene_graph_lines)}")
        else:
            print(f"   ✗ {log_file} not found")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)
    
    logger.info("SceneGraphBuilder logging verification completed")


if __name__ == '__main__':
    main()
