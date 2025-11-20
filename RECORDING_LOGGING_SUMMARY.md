# Recording Module Logging Setup - Summary

## Overview
Comprehensive logging has been configured for the SENTINEL Recording Module, which handles automatic scenario recording, export, and playback functionality.

## Changes Made

### 1. Logging Configuration (configs/logging.yaml)

Added detailed logger configurations for all recording module components:

```yaml
# Recording Module
src.recording.trigger:
  level: DEBUG
  handlers: [file_all]
  propagate: false

src.recording.recorder:
  level: DEBUG
  handlers: [file_all]
  propagate: false

src.recording.exporter:
  level: INFO
  handlers: [file_all]
  propagate: false

src.recording.playback:
  level: INFO
  handlers: [file_all]
  propagate: false

src.recording.scenario_recorder:
  level: INFO
  handlers: [file_all]
  propagate: false
```

**Rationale for Log Levels:**
- **DEBUG** for `trigger` and `recorder`: These components handle frame-by-frame decisions and need detailed logging for debugging trigger logic and buffer management
- **INFO** for `exporter`, `playback`, and `scenario_recorder`: These are higher-level operations that don't need frame-level detail

### 2. Module Implementation Status

All recording module files already have comprehensive logging implemented:

#### src/recording/trigger.py
- ✅ Logger initialized: `logging.getLogger(__name__)`
- ✅ Initialization logging with configuration parameters
- ✅ DEBUG logging for each trigger type activation
- ✅ INFO logging when recording is triggered
- ✅ Detailed metadata in trigger events

**Key Log Points:**
- Trigger initialization with thresholds
- High risk score triggers (> 0.7)
- Distraction during hazard triggers
- System intervention triggers
- Low TTC triggers (< 1.5s)

#### src/recording/recorder.py
- ✅ Logger initialized: `logging.getLogger(__name__)`
- ✅ Initialization logging with buffer size and max duration
- ✅ INFO logging for recording start/stop with frame counts
- ✅ WARNING logging for max duration exceeded
- ✅ DEBUG logging for recording cleared

**Key Log Points:**
- Frame recorder initialization
- Recording started with pre-trigger frame count
- Recording stopped with duration and frame count
- Max duration warnings
- Recording cleared events

#### src/recording/exporter.py
- ✅ Logger initialized: `logging.getLogger(__name__)`
- ✅ Initialization logging with storage path
- ✅ INFO logging for export operations
- ✅ DEBUG logging for individual video exports
- ✅ WARNING logging for empty frame lists

**Key Log Points:**
- Exporter initialization with storage path
- Scenario export start with directory path
- Individual camera video exports
- Successful export with frame count and duration
- Empty frame warnings

#### src/recording/playback.py
- ✅ Logger initialized: `logging.getLogger(__name__)`
- ✅ Initialization logging with storage path
- ✅ INFO logging for scenario loading
- ✅ ERROR logging for missing files/scenarios
- ✅ WARNING logging for video read failures
- ✅ DEBUG logging for scenario close

**Key Log Points:**
- Playback initialization
- Scenario loading with frame count and duration
- Missing file/scenario errors
- Video frame read failures
- Scenario close events

#### src/recording/scenario_recorder.py
- ✅ Logger initialized: `logging.getLogger(__name__)`
- ✅ Initialization logging with enabled status
- ✅ INFO logging for recording lifecycle events
- ✅ INFO logging for scenario export
- ✅ WARNING logging for disabled operations

**Key Log Points:**
- Scenario recorder initialization
- Recording started/stopped events
- Scenario export with path
- Disabled operation warnings
- Recording state changes

### 3. Logging Patterns Used

All recording module components follow consistent logging patterns:

**Initialization:**
```python
self.logger.info(f"ComponentName initialized - param1={value1}, param2={value2}")
```

**State Changes:**
```python
self.logger.info(f"Recording started at t={timestamp:.3f}, included {count} pre-trigger frames")
```

**Trigger Events:**
```python
self.logger.debug(f"High risk trigger: score={score:.2f}, hazard={type}")
```

**Export Operations:**
```python
self.logger.info(f"Scenario exported successfully - {len(frames)} frames, duration={duration:.2f}s")
```

**Error Conditions:**
```python
self.logger.error(f"Scenario not found: {path}")
self.logger.warning(f"No frames to export")
```

## Module Architecture

The recording module consists of 5 main components:

1. **RecordingTrigger**: Determines when to start recording based on risk thresholds
2. **FrameRecorder**: Maintains circular buffer and records frames during active sessions
3. **ScenarioExporter**: Exports recorded scenarios to MP4 videos and JSON annotations
4. **ScenarioPlayback**: Loads and plays back recorded scenarios
5. **ScenarioRecorder**: High-level API integrating all components

## Performance Considerations

The recording module is designed to have minimal impact on real-time performance:

- **Circular buffer**: Pre-allocates memory for 90 frames (3 seconds at 30 FPS)
- **Asynchronous export**: Recording continues while export happens in background
- **Efficient serialization**: Converts numpy arrays to lists only during export
- **Video compression**: Uses MP4 codec for efficient storage

## Log Output Examples

### Normal Operation (No Triggers)
```
INFO - src.recording.scenario_recorder - ScenarioRecorder initialized
INFO - src.recording.trigger - RecordingTrigger initialized - risk_threshold=0.7, ttc_threshold=1.5
INFO - src.recording.recorder - FrameRecorder initialized - buffer_size=90, max_duration=30.0s
INFO - src.recording.exporter - ScenarioExporter initialized - storage_path=scenarios/
INFO - src.recording.playback - ScenarioPlayback initialized - storage_path=scenarios/
```

### Triggered Recording
```
DEBUG - src.recording.trigger - High risk trigger: score=0.85, hazard=vehicle
INFO - src.recording.trigger - Recording triggered: 1 events at t=123.456
INFO - src.recording.scenario_recorder - Recording started at t=123.456
INFO - src.recording.recorder - Recording started at t=123.456, included 90 pre-trigger frames
```

### Recording Stop and Export
```
INFO - src.recording.recorder - Recording stopped - captured 450 frames, duration=15.00s
INFO - src.recording.scenario_recorder - Recording stopped
INFO - src.recording.exporter - Exporting scenario to scenarios/20251115_143022
DEBUG - src.recording.exporter - Exported interior video: interior.mp4
DEBUG - src.recording.exporter - Exported front_left video: front_left.mp4
DEBUG - src.recording.exporter - Exported front_right video: front_right.mp4
DEBUG - src.recording.exporter - Exported bev video: bev.mp4
INFO - src.recording.exporter - Scenario exported successfully - 450 frames, duration=15.00s
INFO - src.recording.scenario_recorder - Scenario exported to scenarios/20251115_143022
```

### Playback Operations
```
INFO - src.recording.playback - Loaded scenario: 20251115_143022, frames=450, duration=15.00s
WARNING - src.recording.playback - Failed to read frame 100 from interior
```

## Integration with System

The recording module integrates with the main SENTINEL pipeline:

1. **Main Loop**: Calls `scenario_recorder.process_frame()` for every frame
2. **Automatic Triggers**: Monitors risk assessment and driver state
3. **Pre-trigger Buffer**: Maintains 3 seconds of context before trigger
4. **Export on Demand**: Can export scenarios manually or automatically
5. **Playback**: Supports loading and reviewing recorded scenarios

## Testing

Logging can be verified using:

```bash
# Run recording tests
pytest tests/test_recording.py -v

# Run recording example
python examples/recording_example.py

# Verify logging output
python scripts/verify_recording.py
```

## Configuration

Recording behavior is controlled via `configs/default.yaml`:

```yaml
recording:
  enabled: true
  triggers:
    risk_threshold: 0.7
    ttc_threshold: 1.5
  storage_path: "scenarios/"
  max_duration: 30.0
```

## Summary

✅ All recording module components have comprehensive logging
✅ Logging configuration added to configs/logging.yaml
✅ Consistent logging patterns across all files
✅ Appropriate log levels for different operations
✅ Minimal performance impact on real-time processing
✅ Detailed context in all log messages

The recording module logging is production-ready and provides excellent visibility into:
- Trigger activation and reasoning
- Recording lifecycle (start/stop/export)
- Frame buffer management
- Video export progress
- Playback operations
- Error conditions and warnings
