# Map Matcher Logging Implementation Summary

## Overview
Comprehensive logging has been added to `src/maps/matcher.py` to support real-time performance monitoring and debugging of the HD map matching functionality.

## Changes Made

### 1. Module-Level Logging Setup
- ✅ Logger initialized at module level: `logger = logging.getLogger(__name__)`
- ✅ Instance logger in MapMatcher class for detailed tracking

### 2. Logging Configuration (configs/logging.yaml)
Added HD Map module loggers:
```yaml
# HD Map Module
src.maps:
  level: INFO
  handlers: [file_all]
  propagate: false

src.maps.matcher:
  level: DEBUG  # Detailed logging for map matching operations
  handlers: [file_all]
  propagate: false

src.maps.parser:
  level: INFO
  handlers: [file_all]
  propagate: false

src.maps.query:
  level: INFO
  handlers: [file_all]
  propagate: false

src.maps.path_predictor:
  level: INFO
  handlers: [file_all]
  propagate: false

src.maps.manager:
  level: INFO
  handlers: [file_all]
  propagate: false
```

### 3. Logging Points Added

#### Initialization (`__init__`)
- **DEBUG**: Initialization started with lane count
- **INFO**: Initialization completed with lane IDs preview
- Tracks: `num_lanes`, `lane_ids` (first 5)

#### Map Matching (`match`)
- **DEBUG**: Match started with position, heading, GPS accuracy
- **DEBUG**: Candidate lanes found with count and IDs
- **DEBUG**: Per-lane scoring results
- **INFO**: Lane change detection
- **DEBUG**: Match completed with full details
- **WARNING**: Slow matching (>5ms target)
- **WARNING**: Match failure with diagnostics
- **INFO**: Periodic statistics (every 100 matches)

Key metrics logged:
- `position`: Vehicle position (x, y)
- `heading`: Vehicle heading in radians
- `gps_accuracy`: GPS accuracy in meters
- `num_candidates`: Number of candidate lanes
- `lane_id`: Matched lane ID
- `lateral_offset`: Distance from lane center
- `longitudinal_position`: Position along lane
- `confidence`: Match confidence score
- `duration`: Processing time in milliseconds
- `success_rate`: Percentage of successful matches

#### Candidate Search (`_find_candidate_lanes`)
- **DEBUG**: Current lane prioritization
- **DEBUG**: Each candidate lane added with distance
- **DEBUG**: Search completion summary

#### Lane Retrieval (`get_current_lane`)
- **DEBUG**: Current lane retrieved with details
- **DEBUG**: No current lane available

#### Statistics (`get_statistics`)
- **DEBUG**: Statistics retrieved
- Returns: match attempts, successful matches, success rate, current lane

### 4. Performance Monitoring

#### Timing Tracking
```python
match_start = time.time()
# ... matching logic ...
match_duration = (time.time() - match_start) * 1000  # ms
```

#### Performance Warnings
- Logs warning if matching exceeds 5ms target
- Includes candidate count for optimization insights

#### Statistics Tracking
- `match_count`: Total match attempts
- `successful_matches`: Successful matches
- `success_rate`: Calculated percentage
- Logged every 100 matches

### 5. State Transition Logging

#### Lane Changes
```python
if prev_lane and prev_lane != self.current_lane_id:
    self.logger.info(
        f"Lane change detected: {prev_lane} -> {self.current_lane_id}, "
        f"confidence={best_match['confidence']:.2f}"
    )
```

#### Match Failures
- Logs when no lanes available
- Logs when no candidates found
- Logs when no valid match among candidates
- Includes position and search parameters

## Log Message Patterns

All log messages follow SENTINEL conventions:

### Successful Operations
```
"Map matching completed: lane_id=lane_1, lateral_offset=0.45m, confidence=0.92, duration=2.34ms"
```

### Failures
```
"Map matching failed: no valid match found among 3 candidates, position=(125.34, 67.89), duration=3.21ms"
```

### State Changes
```
"Lane change detected: lane_1 -> lane_2, confidence=0.87"
```

### Performance Issues
```
"Slow map matching detected: duration=6.78ms (target: <5ms), num_candidates=8"
```

## Performance Considerations

### Target Metrics
- **Matching Time**: <5ms per match
- **Success Rate**: >90% in normal conditions
- **Log Overhead**: Minimal with DEBUG level in production

### Optimization Insights
Logging provides data for:
- Identifying slow matching scenarios
- Analyzing candidate search efficiency
- Tracking lane change frequency
- Monitoring match confidence trends

## Integration with SENTINEL System

### Real-Time Requirements
- Logging designed for 30+ FPS operation
- DEBUG level for development/testing
- INFO level for production (reduces overhead)
- Performance warnings help maintain <100ms total latency

### Diagnostic Capabilities
- Trace complete matching pipeline
- Identify GPS accuracy issues
- Debug lane geometry problems
- Monitor matching algorithm performance

## Testing Verification

To verify logging:

```bash
# Run map matching tests
pytest tests/unit/test_maps.py -v -s

# Check log output
tail -f logs/sentinel.log | grep "src.maps.matcher"

# Run HD map integration test
python test_hd_map_standalone.py
```

## Example Log Output

```
2024-11-19 10:30:45 - src.maps.matcher - INFO - MapMatcher initialized: num_lanes=15, lane_ids=['lane_1', 'lane_2', 'lane_3', 'lane_4', 'lane_5']...
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Map matching started: position=(125.34, 67.89), heading=0.785, gps_accuracy=5.0m
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Current lane lane_1 added as priority candidate
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Lane lane_2 added as candidate: min_dist=3.45m
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Candidate search completed: checked=14 lanes, found=2 candidates, search_radius=10.0m
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Lane lane_1 score: 0.923 (lateral_offset=0.45m, heading_diff=0.012rad)
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Lane lane_2 score: 0.654 (lateral_offset=1.23m, heading_diff=0.234rad)
2024-11-19 10:30:45 - src.maps.matcher - DEBUG - Map matching completed: lane_id=lane_1, lateral_offset=0.45m, longitudinal_position=45.6m, confidence=0.92, duration=2.34ms
```

## Status

✅ **COMPLETE** - Map matcher logging fully implemented and integrated with SENTINEL logging infrastructure.

## Related Files
- `src/maps/matcher.py` - Map matching implementation with logging
- `configs/logging.yaml` - Logging configuration
- `tests/unit/test_maps.py` - Unit tests
- `test_hd_map_standalone.py` - Integration test
- `HD_MAP_QUICK_REFERENCE.md` - HD map module reference
