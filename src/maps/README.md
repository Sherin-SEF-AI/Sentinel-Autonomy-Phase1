# HD Map Integration Module

This module provides HD map integration for the SENTINEL system, enabling lane-level localization and anticipation of upcoming road features.

## Features

- **Map Parsing**: Support for OpenDRIVE and Lanelet2 formats
- **Map Matching**: Lane-level localization with 0.2m accuracy
- **Feature Queries**: Fast spatial queries for signs, lights, crosswalks
- **Path Prediction**: A* routing with turn signal integration
- **BEV Overlay**: Render map features on bird's eye view
- **Performance**: <5ms query time with spatial indexing and caching

## Components

### HDMapManager
Main interface for all HD map functionality. Coordinates parsing, matching, queries, and path prediction.

```python
from src.maps import HDMapManager

# Initialize manager
manager = HDMapManager(config)

# Load map
manager.load_map('maps/city_map.xodr')

# Match position to lane
match = manager.match_position((100.0, 50.0), heading=0.5)
print(f"Lane: {match['lane_id']}, Offset: {match['lateral_offset']:.2f}m")

# Query upcoming features
features = manager.query_upcoming_features(
    position=(100.0, 50.0),
    heading=0.5,
    lookahead=100.0,
    feature_types=['sign', 'light']
)

# Predict path
path = manager.predict_path(
    current_lane_id='road_1_lane_2',
    turn_signal='left',
    horizon=5
)
```

### Map Parsers

#### OpenDRIVEParser
Parses OpenDRIVE (.xodr) format maps.

```python
from src.maps.parser import OpenDRIVEParser

parser = OpenDRIVEParser()
map_data = parser.parse('map.xodr')

lanes = map_data['lanes']
features = map_data['features']
```

#### Lanelet2Parser
Parses Lanelet2 (.osm) format maps.

```python
from src.maps.parser import Lanelet2Parser

parser = Lanelet2Parser()
map_data = parser.parse('map.osm')
```

### MapMatcher
Performs map matching to determine current lane and lateral offset.

```python
from src.maps.matcher import MapMatcher

matcher = MapMatcher(lanes)

# Match position
result = matcher.match(
    position=(100.0, 50.0),
    heading=0.5,
    gps_accuracy=5.0
)

if result:
    print(f"Lane: {result['lane_id']}")
    print(f"Lateral offset: {result['lateral_offset']:.2f}m")
    print(f"Confidence: {result['confidence']:.2f}")
```

### FeatureQuery
Spatial query system for map features with grid-based indexing.

```python
from src.maps.query import FeatureQuery

query = FeatureQuery(lanes, features)

# Query nearby features
nearby = query.query_nearby_features(
    position=(100.0, 50.0),
    radius=50.0,
    feature_types=['sign', 'light']
)

# Query upcoming features
upcoming = query.query_upcoming_features(
    position=(100.0, 50.0),
    heading=0.5,
    lookahead=100.0
)

# Get traffic signs ahead
signs = query.get_traffic_signs_ahead(
    position=(100.0, 50.0),
    heading=0.5,
    lookahead=100.0
)
```

### PathPredictor
Predicts intended path through lane network using A* search.

```python
from src.maps.path_predictor import PathPredictor

predictor = PathPredictor(lanes)

# Predict path with turn signal
path = predictor.predict_path(
    current_lane_id='road_1_lane_2',
    turn_signal='left',
    horizon=5
)

# Predict path to destination
path = predictor.predict_path(
    current_lane_id='road_1_lane_2',
    destination=(500.0, 200.0),
    max_length=10
)
```

## Data Structures

### Lane
Represents a lane from the HD map.

```python
@dataclass
class Lane:
    lane_id: str
    centerline: List[Tuple[float, float, float]]  # (x, y, z) points
    left_boundary: List[Tuple[float, float, float]]
    right_boundary: List[Tuple[float, float, float]]
    width: float
    speed_limit: Optional[float]
    lane_type: str  # 'driving', 'parking', 'shoulder'
    predecessors: List[str]  # Lane IDs
    successors: List[str]  # Lane IDs
```

### MapFeature
Represents a map feature (sign, light, crosswalk, etc.).

```python
@dataclass
class MapFeature:
    feature_id: str
    type: str  # 'lane', 'sign', 'light', 'crosswalk', 'boundary'
    position: Tuple[float, float, float]  # (x, y, z)
    attributes: Dict[str, Any]  # Type-specific attributes
    geometry: List[Tuple[float, float]]  # Polyline points
```

## Configuration

```yaml
hd_map:
  enabled: true
  map_file: "maps/city_map.xodr"
  format: "opendrive"  # 'opendrive' or 'lanelet2'
  
  # Map matching
  gps_accuracy: 5.0  # meters
  match_threshold: 0.2  # meters
  
  # Feature queries
  lookahead_distance: 100.0  # meters
  cache_enabled: true
  cache_ttl: 0.5  # seconds
  
  # Performance
  max_query_time: 0.005  # 5ms target
```

## GUI Integration

### BEV Map Overlay
The BEV canvas widget supports map overlay rendering.

```python
# Enable map overlay
bev_canvas.set_show_map(True)

# Update map overlay
bev_canvas.update_map_overlay(
    lanes=manager.get_all_lanes(),
    features=manager.get_all_features(),
    vehicle_position=(100.0, 50.0)
)
```

### Map View Dock Widget
Dedicated dock widget for map visualization.

```python
from src.gui.widgets.map_view_dock import MapViewDock

# Create dock widget
map_dock = MapViewDock()

# Update map
map_dock.update_map(
    lanes=manager.get_all_lanes(),
    features=manager.get_all_features()
)

# Update vehicle position
map_dock.update_vehicle_position(
    position=(100.0, 50.0),
    heading=0.5,
    current_lane_id='road_1_lane_2'
)

# Update upcoming features
map_dock.update_upcoming_features(upcoming_features)
```

## Performance Optimization

### Spatial Indexing
Features are indexed using a grid-based spatial index for fast queries.

- Grid cell size: 50m x 50m
- Query complexity: O(k) where k is features in nearby cells
- Target query time: <5ms

### Caching
Frequently accessed queries are cached with TTL.

- Cache TTL: 0.5 seconds
- Cache key: (position, radius, feature_types)
- Automatic cache invalidation

### Performance Monitoring

```python
# Get performance statistics
stats = manager.get_performance_stats()

print(f"Average query time: {stats['avg_query_time_ms']:.2f}ms")
print(f"P95 query time: {stats['p95_query_time_ms']:.2f}ms")
print(f"Cache size: {stats['cache_size']}")
```

## Map Formats

### OpenDRIVE
Industry standard format for HD maps.

- File extension: `.xodr`
- XML-based format
- Supports roads, lanes, signals, objects
- Reference: https://www.asam.net/standards/detail/opendrive/

### Lanelet2
Open-source HD map format.

- File extension: `.osm`
- OSM XML format
- Supports lanelets, regulatory elements
- Reference: https://github.com/fzi-forschungszentrum-informatik/Lanelet2

## Requirements

Requirement 22: HD Map Integration

- 22.1: Load HD maps in OpenDRIVE or Lanelet2 format ✓
- 22.2: Map matching with 0.2m accuracy ✓
- 22.3: Query upcoming features within 100m lookahead ✓
- 22.4: Predict driver intended path ✓
- 22.5: Overlay HD map features on BEV display ✓
- 22.6: Complete queries within 5ms ✓

## Example Usage

```python
from src.maps import HDMapManager

# Initialize
config = {
    'map_file': 'maps/city_map.xodr',
    'cache_enabled': True,
}

manager = HDMapManager(config)

# Load map
if manager.load_map(config['map_file']):
    print("Map loaded successfully")
    
    # Main loop
    while True:
        # Get vehicle state
        position = get_vehicle_position()  # (x, y)
        heading = get_vehicle_heading()  # radians
        turn_signal = get_turn_signal()  # 'left', 'right', 'none'
        
        # Match to lane
        match = manager.match_position(position, heading)
        
        if match:
            lane_id = match['lane_id']
            lateral_offset = match['lateral_offset']
            
            # Query upcoming features
            upcoming = manager.query_upcoming_features(
                position, heading, lookahead=100.0
            )
            
            # Predict path
            path = manager.predict_path(
                lane_id, turn_signal, horizon=5
            )
            
            # Use results for risk assessment
            process_map_data(match, upcoming, path)
```

## Testing

```bash
# Run map integration tests
pytest tests/test_maps.py

# Run performance benchmarks
pytest tests/test_maps_performance.py
```

## Logging

The module uses Python's logging framework:

```python
import logging

# Enable debug logging
logging.getLogger('src.maps').setLevel(logging.DEBUG)
```

Log levels:
- DEBUG: Detailed query information
- INFO: Map loading, component initialization
- WARNING: Performance warnings, missing data
- ERROR: Parsing errors, query failures
