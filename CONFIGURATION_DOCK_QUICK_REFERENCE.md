# Configuration Dock Widget - Quick Reference

## Overview
The Configuration Dock Widget provides a comprehensive interface for adjusting SENTINEL system parameters through a tabbed GUI with real-time updates, save/reset functionality, and profile management.

## Quick Start

### Basic Usage
```python
from gui.widgets import ConfigurationDockWidget

# Create dock
config_dock = ConfigurationDockWidget('configs/default.yaml')

# Add to main window
main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, config_dock)

# Connect signals
config_dock.config_changed.connect(on_config_changed)
config_dock.config_saved.connect(on_config_saved)
```

## Configuration Tabs

### 1. Cameras Tab
- **Interior Camera**: Device ID, FPS
- **Front Left Camera**: Device ID, FPS
- **Front Right Camera**: Device ID, FPS

### 2. Detection Tab
- **Detection Model**: Confidence threshold, NMS threshold
- **Multi-View Fusion**: 3D IoU threshold
- **Object Tracking**: Max age, min hits, IoU threshold

### 3. DMS Tab
- **Segmentation Model**: Temporal smoothing alpha

### 4. Risk Tab
- **Time-To-Collision**: Safety margin
- **Trajectory Prediction**: Horizon, time step
- **Base Risk Weights**: TTC, trajectory conflict, vulnerability, relative speed
- **Risk Thresholds**: Hazard detection, intervention, critical

### 5. Alerts Tab
- **Alert Suppression**: Duplicate window, max simultaneous
- **Alert Escalation**: Critical, high, medium thresholds
- **Alert Modalities**: Visual duration/flash rate, audio volume

## Key Features

### Parameter Controls
- **Slider**: Visual adjustment with mouse
- **SpinBox**: Precise numeric entry
- **Range Display**: Shows min/max values
- **Units**: Displays parameter units
- **Tooltips**: Hover for parameter descriptions

### Save & Reset
- **Save**: Persists changes to YAML file (creates backup)
- **Reset**: Restores original values
- **Unsaved Indicator**: Shows "⚠ Unsaved Changes" when modified

### Profile Management
- **Import**: Load configuration from file
- **Export**: Save configuration to file (with timestamp)
- **Sharing**: Easy configuration sharing between systems

## API Reference

### ConfigurationDockWidget

#### Constructor
```python
ConfigurationDockWidget(config_path: str = 'configs/default.yaml')
```

#### Methods
```python
get_config() -> Dict[str, Any]
# Returns current configuration dictionary

has_changes() -> bool
# Returns True if there are unsaved changes
```

#### Signals
```python
config_changed = pyqtSignal(dict)
# Emitted when any parameter changes
# Args: Updated configuration dictionary

config_saved = pyqtSignal(str)
# Emitted when configuration is saved
# Args: Path to saved configuration file
```

### LabeledSlider

#### Constructor
```python
LabeledSlider(
    label: str,
    min_value: float,
    max_value: float,
    current_value: float,
    step: float = 0.1,
    decimals: int = 2,
    units: str = "",
    tooltip: str = ""
)
```

#### Methods
```python
get_value() -> float
# Returns current value

set_value(value: float)
# Sets value programmatically
```

#### Signals
```python
value_changed = pyqtSignal(float)
# Emitted when value changes
# Args: New value
```

## Usage Examples

### Listen for Configuration Changes
```python
def on_config_changed(config: dict):
    # Update system with new parameters
    threshold = config['models']['detection']['confidence_threshold']
    detector.set_threshold(threshold)
    print(f"Detection threshold updated: {threshold}")

config_dock.config_changed.connect(on_config_changed)
```

### Check for Unsaved Changes
```python
if config_dock.has_changes():
    reply = QMessageBox.question(
        self,
        "Unsaved Changes",
        "Save configuration before closing?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    if reply == QMessageBox.StandardButton.Yes:
        # Trigger save
        config_dock._on_save_clicked()
```

### Get Current Configuration
```python
current_config = config_dock.get_config()

# Access specific parameters
fps = current_config['cameras']['interior']['fps']
confidence = current_config['models']['detection']['confidence_threshold']
safety_margin = current_config['risk_assessment']['ttc_calculation']['safety_margin']
```

### Create Custom Parameter Control
```python
# Create a labeled slider for custom parameter
custom_slider = LabeledSlider(
    label="Custom Parameter",
    min_value=0.0,
    max_value=10.0,
    current_value=5.0,
    step=0.5,
    decimals=1,
    units="units",
    tooltip="Description of custom parameter"
)

# Connect to handler
custom_slider.value_changed.connect(
    lambda value: print(f"Custom parameter: {value}")
)
```

## Configuration File Format

The widget works with YAML configuration files:

```yaml
cameras:
  interior:
    device: 0
    fps: 30

models:
  detection:
    confidence_threshold: 0.5
    nms_threshold: 0.4

risk_assessment:
  thresholds:
    critical: 0.9
    intervention: 0.7

alerts:
  suppression:
    duplicate_window: 5.0
```

## Tips & Best Practices

### 1. Real-Time Updates
- Most parameters can be updated in real-time
- Connect to `config_changed` signal for immediate application
- Some parameters may require system restart (document these)

### 2. Validation
- Sliders enforce min/max ranges automatically
- SpinBoxes validate numeric input
- Custom validation can be added in signal handlers

### 3. Profile Management
- Export configurations before major changes
- Use descriptive filenames for profiles
- Share profiles across team for consistency

### 4. Backup Strategy
- Save creates automatic backup (.backup extension)
- Keep multiple profile versions for rollback
- Test new configurations before deployment

### 5. Performance
- Configuration changes are lightweight
- No performance impact on main system
- Updates are asynchronous via signals

## Troubleshooting

### Configuration Not Loading
- Check file path is correct
- Verify YAML syntax is valid
- Check file permissions

### Changes Not Saving
- Ensure write permissions on config file
- Check disk space
- Verify backup creation succeeds

### Parameters Not Updating
- Verify signal connections
- Check parameter key paths match config structure
- Enable debug logging to trace updates

## Logging

Enable debug logging to see detailed operation:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Log output includes:
- Configuration loading/saving
- Parameter changes with values
- Import/export operations
- Error conditions

## Integration with Main Window

Add to main window menu:
```python
# In main window menu creation
config_menu = menubar.addMenu("&Configuration")

show_config_action = QAction("&Show Configuration", self)
show_config_action.triggered.connect(
    lambda: config_dock.setVisible(True)
)
config_menu.addAction(show_config_action)
```

Add to toolbar:
```python
toolbar.addAction("Config", lambda: config_dock.setVisible(True))
```

## Keyboard Shortcuts

Suggested shortcuts for main window:
- `Ctrl+,` - Show configuration dock
- `Ctrl+S` - Save configuration (when dock has focus)
- `Ctrl+R` - Reset configuration (when dock has focus)

## Related Components

- **Main Window**: Hosts the configuration dock
- **System Components**: Listen to config changes
- **Config Manager**: Loads/saves configuration files
- **Theme Manager**: Applies styling to dock

## Support

For issues or questions:
1. Check log output for errors
2. Verify configuration file format
3. Test with default configuration
4. Review TASK_23_SUMMARY.md for details

## Version History

- **v1.0** (2024-11-16): Initial implementation
  - 5 configuration tabs
  - LabeledSlider widget
  - Save/reset functionality
  - Profile import/export
  - Real-time updates
