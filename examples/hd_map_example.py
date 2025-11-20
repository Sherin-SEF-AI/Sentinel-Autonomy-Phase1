"""
HD Map Integration Example

Demonstrates HD map loading, map matching, feature queries, and path prediction.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.maps.manager import HDMapManager
from src.core.data_structures import Lane, MapFeature

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_map():
    """Create a simple sample map for demonstration."""
    logger.info("Creating sample map data...")
    
    # Create sample lanes
    lanes = {}
    
    # Lane 1: Straight road
    lane1 = Lane(
        lane_id='road_1_lane_1',
        centerline=[
            (0.0, 0.0, 0.0),
            (50.0, 0.0, 0.0),
            (100.0, 0.0, 0.0),
            (150.0, 0.0, 0.0),
        ],
        left_boundary=[
            (0.0, 1.75, 0.0),
            (50.0, 1.75, 0.0),
            (100.0, 1.75, 0.0),
            (150.0, 1.75, 0.0),
        ],
        right_boundary=[
            (0.0, -1.75, 0.0),
            (50.0, -1.75, 0.0),
            (100.0, -1.75, 0.0),
            (150.0, -1.75, 0.0),
        ],
        width=3.5,
        speed_limit=50.0,
        lane_type='driving',
        predecessors=[],
        successors=['road_2_lane_1', 'road_2_lane_2']
    )
    lanes['road_1_lane_1'] = lane1
    
    # Lane 2: Left turn
    lane2 = Lane(
        lane_id='road_2_lane_1',
        centerline=[
            (150.0, 0.0, 0.0),
            (160.0, 5.0, 0.0),
            (165.0, 15.0, 0.0),
            (165.0, 25.0, 0.0),
        ],
        left_boundary=[
            (150.0, 1.75, 0.0),
            (161.75, 5.0, 0.0),
            (166.75, 15.0, 0.0),
            (166.75, 25.0, 0.0),
        ],
        right_boundary=[
            (150.0, -1.75, 0.0),
            (158.25, 5.0, 0.0),
            (163.25, 15.0, 0.0),
            (163.25, 25.0, 0.0),
        ],
        width=3.5,
        speed_limit=40.0,
        lane_type='driving',
        predecessors=['road_1_lane_1'],
        successors=[]
    )
    lanes['road_2_lane_1'] = lane2
    
    # Lane 3: Straight continuation
    lane3 = Lane(
        lane_id='road_2_lane_2',
        centerline=[
            (150.0, 0.0, 0.0),
            (200.0, 0.0, 0.0),
            (250.0, 0.0, 0.0),
        ],
        left_boundary=[
            (150.0, 1.75, 0.0),
            (200.0, 1.75, 0.0),
            (250.0, 1.75, 0.0),
        ],
        right_boundary=[
            (150.0, -1.75, 0.0),
            (200.0, -1.75, 0.0),
            (250.0, -1.75, 0.0),
        ],
        width=3.5,
        speed_limit=50.0,
        lane_type='driving',
        predecessors=['road_1_lane_1'],
        successors=[]
    )
    lanes['road_2_lane_2'] = lane3
    
    # Create sample features
    features = [
        MapFeature(
            feature_id='sign_1',
            type='sign',
            position=(80.0, 3.0, 0.0),
            attributes={'sign_type': 'speed_limit', 'value': 50},
            geometry=[(80.0, 3.0)]
        ),
        MapFeature(
            feature_id='light_1',
            type='light',
            position=(145.0, 3.0, 0.0),
            attributes={'light_type': 'traffic_light', 'state': 'green'},
            geometry=[(145.0, 3.0)]
        ),
        MapFeature(
            feature_id='crosswalk_1',
            type='crosswalk',
            position=(120.0, 0.0, 0.0),
            attributes={},
            geometry=[
                (120.0, -2.0),
                (120.0, 2.0)
            ]
        ),
    ]
    
    return lanes, features


def demonstrate_map_matching(manager):
    """Demonstrate map matching functionality."""
    logger.info("\n=== Map Matching Demo ===")
    
    # Test positions
    test_positions = [
        ((50.0, 0.0), 0.0, "Center of lane 1"),
        ((50.0, 1.0), 0.0, "Left side of lane 1"),
        ((50.0, -1.0), 0.0, "Right side of lane 1"),
        ((155.0, 2.0), 0.5, "On left turn"),
    ]
    
    for position, heading, description in test_positions:
        logger.info(f"\nTesting: {description}")
        logger.info(f"Position: {position}, Heading: {heading:.2f} rad")
        
        result = manager.match_position(position, heading)
        
        if result:
            logger.info(f"  ✓ Matched to lane: {result['lane_id']}")
            logger.info(f"  ✓ Lateral offset: {result['lateral_offset']:.2f}m")
            logger.info(f"  ✓ Longitudinal position: {result['longitudinal_position']:.2f}m")
            logger.info(f"  ✓ Confidence: {result['confidence']:.2f}")
        else:
            logger.warning("  ✗ No match found")


def demonstrate_feature_queries(manager):
    """Demonstrate feature query functionality."""
    logger.info("\n=== Feature Query Demo ===")
    
    position = (100.0, 0.0)
    heading = 0.0
    
    # Query nearby features
    logger.info(f"\nQuerying features near {position}...")
    nearby = manager.query_nearby_features(position, radius=50.0)
    logger.info(f"Found {len(nearby)} nearby features:")
    for feature in nearby:
        logger.info(f"  - {feature.type}: {feature.feature_id} at {feature.position}")
    
    # Query upcoming features
    logger.info(f"\nQuerying upcoming features (lookahead: 100m)...")
    upcoming = manager.query_upcoming_features(position, heading, lookahead=100.0)
    logger.info(f"Found {len(upcoming)} upcoming features:")
    for item in upcoming:
        feature = item['feature']
        distance = item['distance']
        logger.info(f"  - {feature.type}: {feature.feature_id} in {distance:.1f}m")
    
    # Query specific feature types
    logger.info(f"\nQuerying traffic signs ahead...")
    signs = manager.query_upcoming_features(
        position, heading, lookahead=100.0, feature_types=['sign']
    )
    logger.info(f"Found {len(signs)} traffic signs:")
    for item in signs:
        feature = item['feature']
        distance = item['distance']
        attrs = feature.attributes
        logger.info(f"  - Sign in {distance:.1f}m: {attrs}")


def demonstrate_path_prediction(manager):
    """Demonstrate path prediction functionality."""
    logger.info("\n=== Path Prediction Demo ===")
    
    current_lane = 'road_1_lane_1'
    
    # Predict with no turn signal
    logger.info(f"\nPredicting path from {current_lane} (no turn signal)...")
    path = manager.predict_path(current_lane, turn_signal='none', horizon=3)
    logger.info(f"Predicted path: {' -> '.join(path)}")
    
    # Predict with left turn signal
    logger.info(f"\nPredicting path from {current_lane} (left turn signal)...")
    path = manager.predict_path(current_lane, turn_signal='left', horizon=3)
    logger.info(f"Predicted path: {' -> '.join(path)}")
    
    # Predict with right turn signal
    logger.info(f"\nPredicting path from {current_lane} (right turn signal)...")
    path = manager.predict_path(current_lane, turn_signal='right', horizon=3)
    logger.info(f"Predicted path: {' -> '.join(path)}")


def demonstrate_performance(manager):
    """Demonstrate performance monitoring."""
    logger.info("\n=== Performance Monitoring ===")
    
    # Run multiple queries
    position = (100.0, 0.0)
    heading = 0.0
    
    for i in range(10):
        manager.match_position(position, heading)
        manager.query_nearby_features(position, radius=50.0)
        manager.query_upcoming_features(position, heading, lookahead=100.0)
    
    # Get performance stats
    stats = manager.get_performance_stats()
    
    logger.info(f"\nPerformance Statistics:")
    logger.info(f"  Average query time: {stats['avg_query_time_ms']:.3f}ms")
    logger.info(f"  Max query time: {stats['max_query_time_ms']:.3f}ms")
    logger.info(f"  P95 query time: {stats['p95_query_time_ms']:.3f}ms")
    logger.info(f"  Total queries: {stats['num_queries']}")
    logger.info(f"  Cache size: {stats['cache_size']}")
    
    # Check if meeting target
    target_ms = 5.0
    if stats['p95_query_time_ms'] < target_ms:
        logger.info(f"  ✓ Meeting <{target_ms}ms target!")
    else:
        logger.warning(f"  ✗ Exceeding {target_ms}ms target")


def main():
    """Main demonstration function."""
    logger.info("HD Map Integration Example")
    logger.info("=" * 50)
    
    # Create manager
    config = {
        'cache_enabled': True,
        'cache_ttl': 0.5,
    }
    
    manager = HDMapManager(config)
    
    # Create sample map data
    lanes, features = create_sample_map()
    
    # Manually set map data (normally would use load_map)
    manager.lanes = lanes
    manager.features = features
    manager.metadata = {'format': 'sample', 'num_lanes': len(lanes)}
    
    # Initialize components
    manager._initialize_components()
    
    logger.info(f"\nMap loaded: {len(lanes)} lanes, {len(features)} features")
    
    # Run demonstrations
    try:
        demonstrate_map_matching(manager)
        demonstrate_feature_queries(manager)
        demonstrate_path_prediction(manager)
        demonstrate_performance(manager)
        
        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
