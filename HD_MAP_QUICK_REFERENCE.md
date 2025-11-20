# HD Map Integration - Quick Reference

## Overview
HD map integration provides lane-level localization and anticipation of upcoming road features for the SENTINEL system.

## Quick Start

### 1. Enable HD Map
Edit `configs/default.yaml`:
```yaml
hd_map:
  enabled: true
  map_file: "maps/your_map.xodr"  # or .osm for Lanelet2
  format: "opendrive"  # or "lanelet2"
```

### 2. Load Map
```python
from src.maps.manager import HDMapManager

manager = HDMapManager(config['hd_map'])
if manager.load_map(config['hd_map']['map_file']):
    print("Map loaded successfully")
```

### 3. Use Map Features
```python
# Match position to lane
match = manager.match_position(
    position=(x, y),
    heading=heading_rad,
    gps_accuracy=5.0
)

# Query upcoming features
features = manager.query_upcoming_features(
    position=(x, y),
    heading=heading_rad,
    lookahead=100.0
)

# Predict path
path = manager.predict_path(
    current_lane_id=match['lane_id'],
    turn_signal='left',  # 'left', 'right', 'none'
    horizon=5
)
```

## Key Classes

### HDMapManager
Main interface for all HD map functionality.

**Methods:**
- `load_map(map_file)` - Load map from file
- `match_position(position, heading)` - Match to lane
- `query_nearby_features(position, radius)` - Get nearby features
- `query_upcoming_features(position, heading, lookahead)` - Get features ahead
- `predict_path(lane_id, turn_signal, horizon)` - Predict intended path
- `get_performance_stats()` - Get query performance metrics

### MapMatcher
Lane-level localization.

**Returns:**
```python
{
    'lane_id': str,
    'lateral_offset': float,  # meters (+ left, - right)
    'longitudinal_position': float,  # meters along lane
    'confidence': float,  # 0-1
}
```

### FeatureQuery
Spatial queries for map features.

**Feature Types:**
- `'sign'` - Traffic signs
- `'light'` - Traffic lights
- `'crosswalk'` - Pedestrian crossings
- `'boundary'` - Road boundaries

### PathPredictor
Path prediction through lane network.

**Turn Signals:**
- `'left'` - Prefer left turns
- `'right'` - Prefer right turns
- `'none'` - Continue straight

## GUI Integration

### BEV Map Overlay
```python
# Enable map overlay
bev_canvas.set_show_map(True)

# Update overlay
bev_canvas.update_map_overlay(
    lanes=manager.get_all_lanes(),
    features=manager.get_all_features(),
    vehicle_position=(x, y)
)
```

### Map View Dock
```python
from src.gui.widgets.map_view_dock import MapViewDock

# Create widget
map_dock = MapViewDock()

# Update map
map_dock.update_map(
    lanes=manager.get_all_lanes(),
    features=manager.get_all_features()
)

# Update vehicle
map_dock.update_vehicle_position(
    position=(x, y),
    heading=heading_rad,
    current_lane_id=match['lane_id']
)
```

## Map Formats

### OpenDRIVE (.xodr)
Industry standard HD map format.
- XML-based
- Supports roads, lanes, signals, objects
- Reference: https://www.asam.net/standards/detail/opendrive/

### Lanelet2 (.osm)
Open-source HD map format.
- OSM XML format
- Supports lanelets, regulatory elements
- Reference: https://github.com/fzi-forschungszentrum-informatik/Lanelet2

## Performance

### Targets
- Query time: <5ms (p95)
- Map matching accuracy: 0.2m lateral
- Lookahead distance: 100m

### Optimization
- Spatial indexing (50m grid cells)
- Query result caching (0.5s TTL)
- Performance monitoring and logging

### Check Performance
```python
stats = manager.get_performance_stats()
print(f"P95 query time: {stats['p95_query_time_ms']:.2f}ms")
print(f"Cache size: {stats['cache_size']}")
```

## Common Use Cases

### 1. Lane-Level Localization
```python
match = manager.match_position((x, y), heading)
if match:
    lane = manager.get_lane(match['lane_id'])
    print(f"On {lane.lane_type} lane")
    print(f"Speed limit: {lane.speed_limit} km/h")
    print(f"Offset from center: {match['lateral_offset']:.2f}m")
```

### 2. Upcoming Traffic Signs
```python
signs = manager.query_upcoming_features(
    position=(x, y),
    heading=heading,
    lookahead=100.0,
    feature_types=['sign']
)

for item in signs:
    sign = item['feature']
    distance = item['distance']
    print(f"Sign in {distance:.0f}m: {sign.attributes}")
```

### 3. Traffic Light Detection
```python
lights = manager.query_upcoming_features(
    position=(x, y),
    heading=heading,
    lookahead=100.0,
    feature_types=['light']
)

if lights and lights[0]['distance'] < 50:
    print(f"Traffic light ahead in {lights[0]['distance']:.0f}m")
```

### 4. Path Planning
```python
# Get current lane
match = manager.match_position((x, y), heading)

# Predict path based on turn signal
path = manager.predict_path(
    current_lane_id=match['lane_id'],
    turn_signal=vehicle_turn_signal,
    horizon=5
)

# Get features along predicted path
for lane_id in path:
    lane = manager.get_lane(lane_id)
    print(f"Lane {lane_id}: {lane.lane_type}, {lane.speed_limit} km/h")
```

### 5. Intersection Detection
```python
from src.maps.query import FeatureQuery

query = FeatureQuery(manager.get_all_lanes(), manager.get_all_features())

intersections = query.get_intersections_ahead(
    lane_id=match['lane_id'],
    longitudinal_position=match['longitudinal_position'],
    lookahead=100.0
)

if intersections:
    print(f"Intersection in {intersections[0]['distance']:.0f}m")
    print(f"Options: {intersections[0]['options']}")
```

## Troubleshooting

### Map Not Loading
- Check file path is correct
- Verify file format (.xodr or .osm)
- Check file is valid XML
- Enable debug logging: `logging.getLogger('src.maps').setLevel(logging.DEBUG)`

### No Match Found
- Increase `gps_accuracy` parameter
- Check vehicle position is within map bounds
- Verify map has lanes near position
- Check coordinate system matches map

### Slow Queries
- Check performance stats: `manager.get_performance_stats()`
- Reduce lookahead distance
- Enable caching if disabled
- Consider smaller map area

### Missing Features
- Verify map file includes feature data
- Check feature types are supported
- Increase search radius
- Check coordinate transformations

## Configuration Reference

```yaml
hd_map:
  enabled: false              # Enable/disable HD map
  map_file: "maps/map.xodr"  # Path to map file
  format: "opendrive"         # Map format
  
  matching:
    gps_accuracy: 5.0         # GPS accuracy (m)
    match_threshold: 0.2      # Match threshold (m)
  
  query:
    lookahead_distance: 100.0 # Lookahead (m)
    search_radius: 50.0       # Search radius (m)
  
  cache:
    enabled: true             # Enable caching
    ttl: 0.5                  # Cache TTL (s)
  
  performance:
    max_query_time: 0.005     # Target time (s)
    log_slow_queries: true    # Log slow queries
```

## API Reference

See `src/maps/README.md` for complete API documentation.

## Examples

- `examples/hd_map_example.py` - Complete demonstration
- `tests/unit/test_maps.py` - Unit tests with examples

## Support

For issues or questions:
1. Check `src/maps/README.md` for detailed documentation
2. Review example code in `examples/hd_map_example.py`
3. Enable debug logging for troubleshooting
4. Check performance stats for optimization
