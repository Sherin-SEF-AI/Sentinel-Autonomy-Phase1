# Data Recording Module

The Data Recording Module automatically captures critical scenarios for analysis and system improvement.

## Features

- **Automatic Trigger Detection**: Records when risk scores are high, driver is distracted, or TTC is low
- **Pre-trigger Buffer**: Includes 3 seconds of context before trigger event
- **Multi-modal Recording**: Captures all camera feeds, BEV output, detections, driver state, and risk scores
- **Scenario Export**: Exports as MP4 videos with JSON annotations
- **Playback Support**: Frame-by-frame navigation of recorded scenarios

## Components

### RecordingTrigger
Determines when to start recording based on:
- Risk score > 0.7
- Driver distraction during hazard
- System intervention (alert generated)
- TTC < 1.5 seconds

### FrameRecorder
Maintains circular buffer and records frames during active sessions.

### ScenarioExporter
Exports scenarios to disk with:
- MP4 videos for each camera
- metadata.json with scenario info
- annotations.json with frame-by-frame data

### ScenarioPlayback
Loads and navigates recorded scenarios.

### ScenarioRecorder
Main API integrating all components.

## Usage

```python
from src.recording import ScenarioRecorder

# Initialize
config = {
    'enabled': True,
    'triggers': {
        'risk_threshold': 0.7,
        'ttc_threshold': 1.5
    },
    'storage_path': 'scenarios/',
    'max_duration': 30.0
}
recorder = ScenarioRecorder(config)

# Process frames (in main loop)
recorder.process_frame(
    timestamp=time.time(),
    camera_bundle=camera_bundle,
    bev_output=bev_output,
    detections_3d=detections,
    driver_state=driver_state,
    risk_assessment=risks,
    alerts=alerts
)

# Manual recording control
recorder.start_recording(timestamp)
recorder.stop_recording()
recorder.export_scenario(location="37.7749,-122.4194")

# Playback
scenarios = recorder.list_scenarios()
recorder.load_scenario(scenarios[0])
frame = recorder.get_playback_frame(0)
next_frame = recorder.next_playback_frame()
```

## Scenario Directory Structure

```
scenarios/
└── 20241115_103045/
    ├── metadata.json
    ├── annotations.json
    ├── interior.mp4
    ├── front_left.mp4
    ├── front_right.mp4
    └── bev.mp4
```

## Configuration

```yaml
recording:
  enabled: true
  triggers:
    risk_threshold: 0.7
    ttc_threshold: 1.5
  storage_path: "scenarios/"
  max_duration: 30.0
```

## Requirements

- Requirement 8.1: Recording triggers
- Requirement 8.2: Frame recording
- Requirement 8.3: Playback support
- Requirement 8.4: Scenario export
