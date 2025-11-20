# Driver Profiling Module - Logging Quick Reference

## Quick Start

```bash
# Verify logging configuration
python3 scripts/verify_profiling_logging_simple.py

# Run full verification (requires dependencies)
python3 scripts/verify_profiling_logging.py

# Check logs
tail -f logs/sentinel.log | grep profiling
```

## Logger Names

```python
import logging

# Component loggers
face_rec_logger = logging.getLogger('src.profiling.face_recognition')
metrics_logger = logging.getLogger('src.profiling.metrics_tracker')
style_logger = logging.getLogger('src.profiling.style_classifier')
threshold_logger = logging.getLogger('src.profiling.threshold_adapter')
report_logger = logging.getLogger('src.profiling.report_generator')
profile_logger = logging.getLogger('src.profiling.profile_manager')
```

## Log Levels by Component

| Component | Level | Rationale |
|-----------|-------|-----------|
| face_recognition | INFO | Driver identification events |
| metrics_tracker | DEBUG | Detailed per-frame metrics |
| style_classifier | INFO | Classification results |
| threshold_adapter | INFO | Threshold adaptations |
| report_generator | INFO | Report generation |
| profile_manager | INFO | Profile operations |

## Common Log Messages

### Face Recognition
```
INFO - FaceRecognitionSystem initialized with threshold=0.6
INFO - Driver matched: driver_abc123 (similarity=0.847)
INFO - No driver match found (best similarity=0.523)
WARNING - Face detection model not found, using fallback
ERROR - Failed to extract face embedding: {error}
```

### Metrics Tracker
```
INFO - MetricsTracker initialized
INFO - Metrics tracking session started at 1234567890.123
DEBUG - Lane change detected at 1234567891.456
INFO - Reaction time recorded: 0.850s for action 'brake'
INFO - Near-miss event recorded: TTC=1.50s, risk=0.800
INFO - Metrics tracking session ended. Duration: 1800.5s
```

### Style Classifier
```
INFO - DrivingStyleClassifier initialized
INFO - Driving style classified: aggressive (score=0.650)
INFO - Driving style classified: cautious (score=-0.420)
WARNING - Insufficient data for style classification
```

### Threshold Adapter
```
INFO - ThresholdAdapter initialized with base TTC=2.0s, following_distance=25.0m
INFO - Thresholds adapted: TTC=1.80s, following_distance=20.0m, alert_sensitivity=0.65
DEBUG - TTC threshold adapted from 2.00s to 1.80s based on reaction time 1.20s
DEBUG - Following distance adapted from 25.0m to 20.0m for aggressive style
```

### Report Generator
```
INFO - DriverReportGenerator initialized
INFO - Report generated for driver_abc123: Safety=82.5, Attention=88.0, Eco=75.5
```

### Profile Manager
```
INFO - ProfileManager initialized with 5 profiles
INFO - Loaded profile: driver_abc123
INFO - Driver identified: driver_abc123 (similarity=0.847)
INFO - New driver profile created: driver_xyz789
INFO - Session started for driver driver_abc123
INFO - Session ended for driver driver_abc123
INFO - Profile updated for driver_abc123: sessions=12, safety=85.0, style=normal
INFO - Profile saved: driver_abc123
ERROR - Failed to load profile from profiles/corrupt.json: {error}
```

## Performance Targets

| Operation | Target Latency | Log Level |
|-----------|---------------|-----------|
| Face recognition | <20ms | INFO |
| Metrics update | <1ms | DEBUG |
| Style classification | <5ms | INFO |
| Threshold adaptation | <2ms | INFO |
| Report generation | <10ms | INFO |
| Profile save | <50ms | INFO |

## Configuration

### Enable/Disable Logging

```yaml
# configs/logging.yaml
loggers:
  src.profiling:
    level: INFO  # Change to DEBUG for detailed logs
    handlers: [file_all]
    propagate: false
```

### Adjust Log Levels

```python
# Runtime adjustment
import logging
logging.getLogger('src.profiling.metrics_tracker').setLevel(logging.INFO)
```

## Troubleshooting

### Issue: No logs appearing

**Check:**
1. Logger configuration in `configs/logging.yaml`
2. Log file permissions in `logs/` directory
3. Log level settings (DEBUG vs INFO)

**Solution:**
```bash
# Verify configuration
python3 scripts/verify_profiling_logging_simple.py

# Check log file
ls -la logs/sentinel.log
```

### Issue: Too many DEBUG logs

**Check:**
- `metrics_tracker` is set to DEBUG level
- Generates logs every frame (~30 per second)

**Solution:**
```yaml
# Change to INFO in configs/logging.yaml
src.profiling.metrics_tracker:
  level: INFO  # Was DEBUG
```

### Issue: Missing driver identification logs

**Check:**
1. Face detection working
2. Camera providing frames
3. Face recognition logger level

**Solution:**
```python
# Enable detailed face recognition logging
logging.getLogger('src.profiling.face_recognition').setLevel(logging.DEBUG)
```

## Integration Example

```python
from src.profiling import ProfileManager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize profile manager
config = {
    'profiles_dir': 'profiles/',
    'auto_save': True,
    'face_recognition': {'recognition_threshold': 0.6}
}
manager = ProfileManager(config)

# Identify driver (logs automatically)
driver_id = manager.identify_driver(camera_frame)
# LOG: INFO - Driver identified: driver_abc123 (similarity=0.847)

# Start session (logs automatically)
manager.start_session(driver_id, timestamp)
# LOG: INFO - Session started for driver driver_abc123

# Update metrics (logs at DEBUG level)
tracker = manager.get_metrics_tracker()
tracker.update(timestamp, speed=25.0, following_distance=28.0)
# LOG: DEBUG - Metrics updated

# End session (logs automatically)
manager.end_session(timestamp)
# LOG: INFO - Session ended for driver driver_abc123
# LOG: INFO - Driving style classified: normal (score=0.050)
# LOG: INFO - Report generated for driver_abc123: Safety=85.0, Attention=88.0, Eco=78.0
# LOG: INFO - Profile updated for driver_abc123: sessions=1, safety=85.0, style=normal
# LOG: INFO - Profile saved: driver_abc123
```

## Log Analysis

### Extract profiling events
```bash
grep "profiling" logs/sentinel.log
```

### Count driver identifications
```bash
grep "Driver identified" logs/sentinel.log | wc -l
```

### Find slow operations
```bash
grep "profiling" logs/sentinel.log | grep -E "[0-9]+\.[0-9]+ms" | awk '{if ($NF > 50) print}'
```

### Session statistics
```bash
grep "Session ended" logs/sentinel.log | tail -10
```

## Best Practices

1. **Use INFO for production** - Minimal overhead, captures key events
2. **Use DEBUG for development** - Detailed metrics, higher overhead
3. **Monitor performance** - Track latencies in logs
4. **Regular log rotation** - Prevent disk space issues
5. **Structured messages** - Include context (driver_id, timestamps, metrics)

## Summary

✓ **6 components** with dedicated loggers  
✓ **Hierarchical logging** from root to component level  
✓ **Performance-aware** with <1ms per-frame overhead  
✓ **Comprehensive coverage** of all profiling operations  
✓ **Easy troubleshooting** with structured log messages  

For detailed information, see `DRIVER_PROFILING_LOGGING_SUMMARY.md`
