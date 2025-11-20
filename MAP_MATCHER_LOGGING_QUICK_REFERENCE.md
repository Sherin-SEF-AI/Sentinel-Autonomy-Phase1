# Map Matcher Logging Quick Reference

## Log Levels by Operation

| Operation | Level | When Used |
|-----------|-------|-----------|
| Initialization | INFO | MapMatcher created |
| Match started | DEBUG | Each match attempt |
| Candidate search | DEBUG | Finding nearby lanes |
| Lane scoring | DEBUG | Evaluating each candidate |
| Match completed | DEBUG | Successful match |
| Lane change | INFO | Lane transition detected |
| Match failure | WARNING | No valid match found |
| Slow matching | WARNING | Duration >5ms |
| Statistics | INFO | Every 100 matches |

## Key Metrics Logged

### Match Operation
```python
{
    'position': (x, y),           # Vehicle position in map frame
    'heading': float,             # Vehicle heading (radians)
    'gps_accuracy': float,        # GPS accuracy (meters)
    'num_candidates': int,        # Candidate lanes found
    'lane_id': str,              # Matched lane ID
    'lateral_offset': float,      # Distance from center (m)
    'longitudinal_position': float, # Position along lane (m)
    'confidence': float,          # Match confidence (0-1)
    'duration': float            # Processing time (ms)
}
```

### Statistics
```python
{
    'match_attempts': int,        # Total match attempts
    'successful_matches': int,    # Successful matches
    'success_rate': float,        # Success percentage
    'current_lane_id': str,      # Current lane
    'num_lanes': int             # Total lanes in map
}
```

## Common Log Messages

### Successful Match
```
Map matching completed: lane_id=lane_1, lateral_offset=0.45m, 
longitudinal_position=45.6m, confidence=0.92, duration=2.34ms
```

### Lane Change
```
Lane change detected: lane_1 -> lane_2, confidence=0.87
```

### Match Failure
```
Map matching failed: no valid match found among 3 candidates, 
position=(125.34, 67.89), duration=3.21ms
```

### Performance Warning
```
Slow map matching detected: duration=6.78ms (target: <5ms), num_candidates=8
```

### Periodic Statistics
```
Map matching statistics: attempts=100, successful=94, 
success_rate=94.0%, current_lane=lane_1
```

## Performance Targets

| Metric | Target | Warning Threshold |
|--------|--------|-------------------|
| Match Duration | <5ms | >5ms |
| Success Rate | >90% | <80% |
| Candidates | 1-5 | >10 |

## Debugging Tips

### High Match Failures
1. Check GPS accuracy parameter
2. Verify lane geometry in map
3. Review position accuracy
4. Check search radius

### Slow Matching
1. Reduce number of lanes in map
2. Optimize candidate search radius
3. Check lane centerline density
4. Review scoring algorithm

### Frequent Lane Changes
1. Verify match confidence thresholds
2. Check lateral offset calculations
3. Review hysteresis settings
4. Validate heading data

## Log Filtering

### View all map matching
```bash
tail -f logs/sentinel.log | grep "src.maps.matcher"
```

### View only warnings/errors
```bash
tail -f logs/sentinel.log | grep "src.maps.matcher" | grep -E "WARNING|ERROR"
```

### View lane changes
```bash
tail -f logs/sentinel.log | grep "Lane change detected"
```

### View performance issues
```bash
tail -f logs/sentinel.log | grep "Slow map matching"
```

### View statistics
```bash
tail -f logs/sentinel.log | grep "Map matching statistics"
```

## Configuration

### Development (Verbose)
```yaml
src.maps.matcher:
  level: DEBUG
  handlers: [console, file_all]
  propagate: false
```

### Production (Minimal)
```yaml
src.maps.matcher:
  level: INFO
  handlers: [file_all]
  propagate: false
```

### Troubleshooting (Maximum Detail)
```yaml
src.maps.matcher:
  level: DEBUG
  handlers: [console, file_all, file_performance]
  propagate: false
```

## Integration Points

### Called By
- `HDMapManager.update_position()` - Main update loop
- `PathPredictor.predict_path()` - Path prediction
- GUI map view dock - Visualization

### Calls To
- Lane geometry calculations
- Distance computations
- Heading calculations

## Related Modules
- `src.maps.manager` - HD map management
- `src.maps.query` - Feature queries
- `src.maps.path_predictor` - Path prediction
- `src.maps.parser` - Map file parsing
