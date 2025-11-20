# Scenarios Dock Widget - Quick Reference

## Overview
The Scenarios Dock Widget provides a comprehensive interface for managing and replaying recorded safety scenarios in the SENTINEL system.

## Features

### Scenario List
- **Thumbnail Preview**: 120x90 pixel preview from first frame
- **Timestamp**: When the scenario was recorded
- **Duration**: Length of the scenario in seconds
- **Trigger Type**: Why the scenario was recorded (color-coded)
  - 🔴 Critical (red)
  - 🟠 High Risk (orange)
  - 🟡 Near Miss (yellow)
  - 🟡 Distracted (light yellow)
  - 🔵 Manual (blue)

### Search and Filter
- **Search Box**: Filter by scenario name or trigger reason
- **Filter Dropdown**: Filter by trigger type
  - All
  - Critical
  - High Risk
  - Near Miss
  - Distracted
  - Manual

### Actions
- **Refresh**: Reload scenarios from disk
- **Export**: Copy scenario to another location
- **Delete**: Remove scenario (with confirmation)
- **Double-Click**: Open replay dialog

## Scenario Replay Dialog

### Video Display
- 2x2 grid showing all camera views:
  - Interior Camera
  - Front Left Camera
  - Front Right Camera
  - Bird's Eye View

### Playback Controls
- **Play/Pause**: Start/stop playback
- **◄ Step Back**: Go to previous frame
- **Step Forward ►**: Go to next frame
- **Speed**: Adjust playback speed (0.25x to 2x)
- **Annotations**: Toggle annotations overlay

### Timeline
- **Slider**: Scrub through scenario
- **Display**: Shows "Frame: X / Y | Time: Z.ZZs"

## Usage in Code

### Basic Setup
```python
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtCore import Qt
from gui.widgets import ScenariosDockWidget

# Create dock widget
scenarios_dock = ScenariosDockWidget('scenarios/')

# Add to main window
main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, scenarios_dock)
```

### Connect Signals
```python
# Handle scenario selection
def on_scenario_selected(scenario_name):
    print(f"Selected: {scenario_name}")
    metadata = scenarios_dock.get_scenario_metadata(scenario_name)
    print(f"Duration: {metadata['duration']:.1f}s")

scenarios_dock.scenario_selected.connect(on_scenario_selected)

# Handle replay request
def on_replay_requested(scenario_name):
    from gui.widgets import ScenarioReplayDialog
    dialog = ScenarioReplayDialog(scenario_name, 'scenarios/', main_window)
    dialog.exec()

scenarios_dock.scenario_replay_requested.connect(on_replay_requested)
```

### Programmatic Control
```python
# Refresh scenarios list
scenarios_dock.refresh_scenarios()

# Get selected scenario
scenario_name = scenarios_dock.get_selected_scenario()

# Get scenario metadata
metadata = scenarios_dock.get_scenario_metadata(scenario_name)
```

## Scenario Directory Structure

Each scenario is stored in a directory with the following structure:

```
scenarios/
└── 20241116_103045/
    ├── metadata.json       # Scenario metadata
    ├── annotations.json    # Frame-by-frame annotations
    ├── interior.mp4        # Interior camera video
    ├── front_left.mp4      # Front left camera video
    ├── front_right.mp4     # Front right camera video
    └── bev.mp4            # Bird's eye view video
```

### metadata.json Format
```json
{
  "timestamp": "2024-11-16T10:30:45.123Z",
  "duration": 15.5,
  "num_frames": 465,
  "trigger": {
    "type": "critical",
    "reason": "High risk collision detected"
  },
  "files": {
    "interior": "interior.mp4",
    "front_left": "front_left.mp4",
    "front_right": "front_right.mp4",
    "bev": "bev.mp4"
  }
}
```

### annotations.json Format
```json
{
  "frames": [
    {
      "timestamp": 0.033,
      "detections_3d": [...],
      "driver_state": {...},
      "risk_assessment": {...},
      "alerts": [...]
    }
  ]
}
```

## Keyboard Shortcuts (in Replay Dialog)

- **Space**: Play/Pause
- **Left Arrow**: Step backward
- **Right Arrow**: Step forward
- **Esc**: Close dialog

## Tips

1. **Performance**: Thumbnails are generated on-demand. For large scenario collections, initial loading may take a moment.

2. **Search**: Use the search box to quickly find scenarios by date, time, or trigger reason.

3. **Filtering**: Combine search and filter for precise results (e.g., search "collision" + filter "Critical").

4. **Export**: Use export to backup important scenarios or share with team members.

5. **Playback Speed**: Use slower speeds (0.25x, 0.5x) to analyze critical moments frame-by-frame.

6. **Annotations**: Toggle annotations off to see raw camera footage without overlays.

## Troubleshooting

### No Scenarios Showing
- Check that scenarios directory exists
- Verify metadata.json files are present
- Click Refresh button to reload

### Replay Dialog Won't Open
- Ensure video files exist in scenario directory
- Check that video files are valid MP4 format
- Verify annotations.json is present

### Thumbnails Not Loading
- Check that at least one video file exists
- Verify video files are not corrupted
- Try refreshing the scenarios list

### Export/Delete Not Working
- Ensure you have write permissions
- Check available disk space
- Verify scenario directory is not in use

## Integration with Main Window

To add the scenarios dock to the main SENTINEL window:

```python
# In main_window.py
from .widgets import ScenariosDockWidget

class SENTINELMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Create scenarios dock
        self.scenarios_dock = ScenariosDockWidget('scenarios/')
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.scenarios_dock)
        
        # Connect to replay handler
        self.scenarios_dock.scenario_replay_requested.connect(self._on_replay_scenario)
    
    def _on_replay_scenario(self, scenario_name):
        from .widgets import ScenarioReplayDialog
        dialog = ScenarioReplayDialog(scenario_name, 'scenarios/', self)
        dialog.exec()
```

## Related Components

- **ScenarioRecorder**: Records scenarios (src/recording/scenario_recorder.py)
- **ScenarioPlayback**: Playback API (src/recording/playback.py)
- **ScenarioExporter**: Export scenarios (src/recording/exporter.py)

## See Also

- Task 10 Summary: Data Recording Module
- Task 22 Summary: Scenarios Dock Widget Implementation
- Recording Module README: src/recording/README.md
