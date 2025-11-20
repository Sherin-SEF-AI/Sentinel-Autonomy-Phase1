# Data Structures Logging - Quick Reference

## Module Information

**Module**: `src/core/data_structures.py`  
**Logger Name**: `src.core.data_structures`  
**Log Level**: INFO  
**Log File**: `logs/sentinel.log`

## New Data Structures (Task 27)

### 1. MapFeature
HD map feature representation for lanes, signs, lights, crosswalks, and boundaries.

```python
feature = MapFeature(
    feature_id="sign_001",
    type="sign",  # 'lane', 'sign', 'light', 'crosswalk', 'boundary'
    position=(50.0, 10.0, 2.0),  # x, y, z in vehicle frame
    attributes={"sign_type": "stop", "text": "STOP"},
    geometry=[(50.0, 10.0), (50.0, 11.0)]  # Polyline points
)
```

### 2. Lane
Lane geometry with centerline, boundaries, and connectivity.

```python
lane = Lane(
    lane_id="lane_001",
    centerline=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],  # (x, y, z) points
    left_boundary=[(0.0, 1.75, 0.0), (10.0, 1.75, 0.0)],
    right_boundary=[(0.0, -1.75, 0.0), (10.0, -1.75, 0.0)],
    width=3.5,  # meters
    speed_limit=50.0,  # km/h (optional)
    lane_type="driving",  # 'driving', 'parking', 'shoulder'
    predecessors=["lane_000"],  # Connected lane IDs
    successors=["lane_002"]
)
```

### 3. VehicleTelemetry
CAN bus vehicle data for trajectory prediction.

```python
telemetry = VehicleTelemetry(
    timestamp=1234567890.0,
    speed=15.0,  # m/s
    steering_angle=0.05,  # radians
    brake_pressure=0.0,  # bar
    throttle_position=0.3,  # 0-1
    gear=3,
    turn_signal="none"  # 'left', 'right', 'none'
)
```

## Logging Configuration

### In `configs/logging.yaml`:
```yaml
src.core.data_structures:
  level: INFO
  handlers: [file_all]
  propagate: false
```

### In Python code:
```python
import logging
logger = logging.getLogger(__name__)
```

## Usage Examples

### Creating Data Structures
```python
from src.core.data_structures import MapFeature, Lane, VehicleTelemetry

# Create map feature
stop_sign = MapFeature(
    feature_id="sign_123",
    type="sign",
    position=(25.0, 5.0, 2.0),
    attributes={"sign_type": "stop"},
    geometry=[(25.0, 5.0)]
)

# Create lane
current_lane = Lane(
    lane_id="lane_main_01",
    centerline=[(0, 0, 0), (50, 0, 0), (100, 0, 0)],
    left_boundary=[(0, 1.75, 0), (50, 1.75, 0), (100, 1.75, 0)],
    right_boundary=[(0, -1.75, 0), (50, -1.75, 0), (100, -1.75, 0)],
    width=3.5,
    speed_limit=60.0,
    lane_type="driving",
    predecessors=[],
    successors=["lane_main_02"]
)

# Create telemetry
vehicle_data = VehicleTelemetry(
    timestamp=time.time(),
    speed=20.0,
    steering_angle=0.0,
    brake_pressure=0.0,
    throttle_position=0.4,
    gear=4,
    turn_signal="none"
)
```

## Integration Points

### HD Map Manager
```python
from src.maps.manager import HDMapManager
from src.core.data_structures import Lane, MapFeature

# Query returns Lane and MapFeature objects
lanes = map_manager.query_lanes(position, radius=50.0)
features = map_manager.query_features(position, radius=50.0)
```

### Path Predictor
```python
from src.maps.path_predictor import PathPredictor
from src.core.data_structures import Lane

# Uses Lane connectivity for path prediction
predicted_path = predictor.predict_path(current_lane, turn_signal="left")
```

### Contextual Intelligence
```python
from src.intelligence.engine import ContextualIntelligence
from src.core.data_structures import MapFeature

# Uses map features for enhanced risk assessment
risk = intelligence.assess_with_map(detections, driver_state, map_features)
```

## Verification

```bash
# Run verification script
python3 scripts/verify_data_structures_simple.py

# Expected output:
# ✓ All checks passed!
# Summary:
#   - Logging setup: ✓ Complete
#   - New data structures: ✓ 3 added
#   - Configuration: ✓ logging.yaml updated
```

## Performance Notes

- **Zero logging overhead**: Dataclasses have no runtime logging
- **Efficient**: Simple structures optimized for real-time use
- **Type-safe**: Full type hints for validation
- **Memory efficient**: Uses numpy arrays where appropriate

## Future Enhancements

1. **Validation methods**: Add logging for data validation
2. **Factory methods**: Log complex initialization
3. **Serialization**: Log data export/import operations
4. **CAN bus integration**: Populate VehicleTelemetry from vehicle

## Status

✅ Logging setup complete and verified  
✅ Ready for HD Map integration (Task 27)  
✅ Ready for CAN bus integration (future)
