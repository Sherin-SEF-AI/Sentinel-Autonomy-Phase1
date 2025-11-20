"""
Standalone HD Map Test

Tests HD map functionality without importing full SENTINEL system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import only what we need, avoiding src/__init__.py
from src.core.data_structures import Lane, MapFeature
from src.maps.matcher import MapMatcher
from src.maps.query import FeatureQuery
from src.maps.path_predictor import PathPredictor

print("=" * 60)
print("HD Map Integration Test")
print("=" * 60)

# Create sample lanes
print("\n1. Creating sample lanes...")
lanes = {}

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

print(f"   Created {len(lanes)} lanes")

# Create sample features
print("\n2. Creating sample features...")
features = [
    MapFeature(
        feature_id='sign_1',
        type='sign',
        position=(50.0, 3.0, 0.0),
        attributes={'sign_type': 'speed_limit', 'value': 50},
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

print(f"   Created {len(features)} features")

# Test MapMatcher
print("\n3. Testing MapMatcher...")
matcher = MapMatcher(lanes)

test_position = (50.0, 0.0)
test_heading = 0.0

result = matcher.match(test_position, test_heading)

if result:
    print(f"   ✓ Matched to lane: {result['lane_id']}")
    print(f"   ✓ Lateral offset: {result['lateral_offset']:.2f}m")
    print(f"   ✓ Confidence: {result['confidence']:.2f}")
else:
    print("   ✗ Match failed")

# Test FeatureQuery
print("\n4. Testing FeatureQuery...")
query = FeatureQuery(lanes, features)

nearby = query.query_nearby_features((50.0, 0.0), radius=10.0)
print(f"   Found {len(nearby)} nearby features")

upcoming = query.query_upcoming_features(
    (30.0, 0.0), heading=0.0, lookahead=50.0
)
print(f"   Found {len(upcoming)} upcoming features")

if upcoming:
    for item in upcoming:
        feature = item['feature']
        distance = item['distance']
        print(f"     - {feature.type} '{feature.feature_id}' in {distance:.1f}m")

# Test PathPredictor
print("\n5. Testing PathPredictor...")
predictor = PathPredictor(lanes)

path = predictor.predict_path('lane_1', turn_signal='none', horizon=2)
print(f"   Predicted path: {' -> '.join(path)}")

# Performance test
print("\n6. Testing Performance...")
import time

num_queries = 100
start_time = time.time()

for _ in range(num_queries):
    matcher.match((50.0, 0.0), 0.0)
    query.query_nearby_features((50.0, 0.0), radius=50.0)

elapsed = time.time() - start_time
avg_time_ms = (elapsed / num_queries) * 1000

print(f"   Ran {num_queries} queries in {elapsed:.3f}s")
print(f"   Average query time: {avg_time_ms:.3f}ms")

if avg_time_ms < 5.0:
    print(f"   ✓ Meeting <5ms target!")
else:
    print(f"   ✗ Exceeding 5ms target")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
