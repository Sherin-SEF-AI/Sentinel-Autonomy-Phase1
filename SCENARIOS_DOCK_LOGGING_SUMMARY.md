# Scenarios Dock Widget - Logging Summary

## Overview
Comprehensive logging has been implemented for the Scenarios Dock Widget (`src/gui/widgets/scenarios_dock.py`), which provides scenario management interface with browsing, filtering, export, and deletion capabilities.

## Logging Configuration

### Logger Setup
```python
import logging
logger = logging.getLogger(__name__)
```

### Configuration in `configs/logging.yaml`
```yaml
src.gui.widgets.scenarios_dock:
  level: INFO
  handlers: [file_all]
  propagate: false
```

## Logging Points

### 1. Initialization
- **Level**: INFO
- **Message**: "ScenariosDockWidget initialized"
- **Context**: Widget creation complete

### 2. Scenario Refresh Operations
```python
# Start
logger.info("Scenario refresh started")

# Directory check
logger.warning(f"Scenarios directory not found: path={self.scenarios_path}")
logger.debug(f"Found {len(scenario_dirs)} scenario directories")

# Loading progress
logger.debug(f"Scenarios sorted by timestamp (newest first)")
logger.warning(f"Failed to load {failed_count} scenario(s)")

# Completion
logger.info(f"Scenario refresh completed: loaded={loaded_count}, failed={failed_count}, duration={refresh_duration:.3f}s")
```

### 3. Metadata Loading
```python
# Missing metadata
logger.warning(f"Metadata file not found: scenario={scenario_name}, path={metadata_path}")

# Success
logger.debug(f"Metadata loaded: scenario={scenario_name}, duration={duration:.1f}s, trigger={trigger_type}")

# Errors
logger.error(f"Invalid JSON in metadata: scenario={scenario_name}, error={e}")
logger.error(f"Failed to load metadata: scenario={scenario_name}, error={e}")
```

### 4. Thumbnail Loading
```python
# Debug info
logger.debug(f"No metadata available for thumbnail: scenario={scenario_name}")
logger.debug(f"No video file found for thumbnail: scenario={scenario_name}")

# Warnings
logger.warning(f"Video file not found for thumbnail: scenario={scenario_name}, path={video_path}")
logger.warning(f"Failed to read first frame: scenario={scenario_name}, video={bev_file}")

# Success
logger.debug(f"Thumbnail loaded: scenario={scenario_name}, size={width}x{height}")

# Errors
logger.error(f"Failed to load thumbnail: scenario={scenario_name}, error={e}")
```

### 5. Filtering Operations
```python
# Filter application
logger.debug(f"Applying filter: search='{search_text}', type={filter_type}")

# Results
logger.debug(f"Filter applied: showing={visible_count}, total={total_count}, filtered_out={total_count - visible_count}")
logger.debug(f"Filter applied: showing all {total_count} scenario(s)")
```

### 6. User Interactions
```python
# Selection
logger.debug(f"Scenario selected: {scenario_name}")
logger.debug("Scenario selection cleared")

# Double-click
logger.info(f"Scenario double-clicked: {scenario_name}")

# Export request
logger.info(f"Export requested for scenario: {scenario_name}")
```

### 7. Export Operations
```python
# Start
logger.info(f"Scenario export started: scenario={scenario_name}, dest={export_dir}")
logger.debug(f"Export size: {total_size / (1024*1024):.2f} MB")

# Success
logger.info(f"Scenario export completed: scenario={scenario_name}, dest={dest_dir}, size_mb={total_size/(1024*1024):.2f}, duration={export_duration:.3f}s")

# Failure
logger.error(f"Scenario export failed: scenario={scenario_name}, dest={export_dir}, error={e}")
```

### 8. Delete Operations
```python
# Start
logger.info(f"Scenario deletion started: scenario={scenario_name}")

# Success
logger.info(f"Scenario deleted: scenario={scenario_name}, size_mb={size_mb:.2f}")

# Failure
logger.error(f"Scenario deletion failed: scenario={scenario_name}, error={e}")
```

## Log Message Patterns

### Successful Operations
- **Pattern**: "Operation completed: key_metrics"
- **Example**: `"Scenario refresh completed: loaded=5, failed=0, duration=0.234s"`

### Failed Operations
- **Pattern**: "Operation failed: context, error=details"
- **Example**: `"Scenario export failed: scenario=20241115_103045, dest=/export, error=Permission denied"`

### State Changes
- **Pattern**: "State changed: details"
- **Example**: `"Scenario selected: 20241115_103045"`

### Performance Metrics
- **Pattern**: "Operation: metric=value"
- **Example**: `"Export size: 125.45 MB"`

## Performance Considerations

### Non-Blocking Operations
- Thumbnail loading is lazy (only when visible)
- Metadata loading is sequential but fast (JSON parsing)
- Export/delete operations show progress dialogs

### Logging Overhead
- DEBUG level for frequent operations (filtering, selection)
- INFO level for user-initiated actions (refresh, export, delete)
- ERROR level for failures only

### Typical Log Volume
- **Initialization**: 1 INFO message
- **Refresh (10 scenarios)**: 1 INFO + 10 DEBUG + 1 INFO = ~12 messages
- **Filter change**: 1 DEBUG message
- **Selection**: 1 DEBUG message per selection
- **Export**: 3 INFO + 1 DEBUG = 4 messages
- **Delete**: 2 INFO messages

## Integration with SENTINEL System

### GUI Module Context
- Part of PyQt6 GUI application
- Runs in main GUI thread
- Interacts with file system for scenario management

### Related Modules
- **Recording Module**: Creates scenarios that this widget displays
- **Playback Module**: Replays scenarios selected in this widget
- **Main Window**: Hosts this dock widget

### Performance Impact
- Minimal impact on real-time processing (GUI-only)
- File I/O operations are user-initiated
- No continuous polling or updates

## Debugging Guide

### Common Issues

1. **Scenarios not loading**
   - Check: `"Scenarios directory not found"` warning
   - Verify: scenarios_path configuration
   - Look for: `"Failed to load metadata"` errors

2. **Thumbnails not showing**
   - Check: `"Video file not found for thumbnail"` warnings
   - Verify: Video files exist in scenario directories
   - Look for: `"Failed to read first frame"` warnings

3. **Export failures**
   - Check: `"Scenario export failed"` errors
   - Verify: Destination directory permissions
   - Look for: File size in debug logs

4. **Delete failures**
   - Check: `"Scenario deletion failed"` errors
   - Verify: File permissions
   - Look for: Scenario directory path in logs

### Log Analysis Examples

**Successful refresh:**
```
INFO - Scenario refresh started
DEBUG - Found 5 scenario directories
DEBUG - Scenarios sorted by timestamp (newest first)
DEBUG - Metadata loaded: scenario=20241115_103045, duration=15.2s, trigger=critical
DEBUG - Metadata loaded: scenario=20241115_102030, duration=8.5s, trigger=high_risk
...
INFO - Scenario refresh completed: loaded=5, failed=0, duration=0.234s
```

**Export operation:**
```
INFO - Export requested for scenario: 20241115_103045
INFO - Scenario export started: scenario=20241115_103045, dest=/home/user/exports
DEBUG - Export size: 125.45 MB
INFO - Scenario export completed: scenario=20241115_103045, dest=/home/user/exports/20241115_103045, size_mb=125.45, duration=2.345s
```

**Failed metadata loading:**
```
WARNING - Metadata file not found: scenario=20241115_100000, path=scenarios/20241115_100000/metadata.json
WARNING - Failed to load 1 scenario(s)
```

## Verification

### Test Logging
Run the test suite to verify logging:
```bash
pytest tests/unit/test_scenarios_dock.py -v -s
```

### Manual Verification
1. Start GUI application
2. Open Scenarios dock widget
3. Check logs for initialization message
4. Refresh scenarios and verify log messages
5. Select, export, or delete scenarios
6. Verify appropriate log messages appear

### Log File Location
- Main log: `logs/sentinel.log`
- All GUI logs included in main log file

## Summary

The Scenarios Dock Widget logging implementation provides:
- ✅ Comprehensive coverage of all operations
- ✅ Appropriate log levels (DEBUG for details, INFO for actions, ERROR for failures)
- ✅ Rich context in all messages (scenario names, paths, sizes, durations)
- ✅ Performance metrics for operations
- ✅ Clear error messages for troubleshooting
- ✅ Minimal performance impact on GUI responsiveness

The logging follows SENTINEL's patterns and integrates seamlessly with the existing logging infrastructure.
