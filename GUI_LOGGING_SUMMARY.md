# GUI Module Logging Summary

## Overview

Comprehensive logging has been implemented for the SENTINEL PyQt6 GUI module to track user interactions, system state changes, window management, and performance metrics.

## Logging Configuration

### Module Loggers Added to `configs/logging.yaml`

```yaml
# GUI Module
src.gui:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.main_window:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.widgets:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.widgets.live_monitor:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.widgets.video_display:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.widgets.bev_canvas:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.themes:
  level: INFO
  handlers: [file_all]
  propagate: false

src.gui.themes.theme_manager:
  level: INFO
  handlers: [file_all]
  propagate: false
```

## Logging Implementation

### 1. Module Initialization (`src/gui/__init__.py`)

**Added:**
```python
import logging
logger = logging.getLogger(__name__)
logger.debug("GUI module initialized")
```

**Logs:**
- Module initialization confirmation

### 2. Main Window (`src/gui/main_window.py`)

**Already Implemented - Comprehensive Logging:**

#### Initialization & Setup
- `INFO`: "SENTINEL Main Window initialized"
- `DEBUG`: "Main UI structure initialized"
- `DEBUG`: "Menu bar created"
- `DEBUG`: "Toolbar created"
- `DEBUG`: "Status bar created"
- `DEBUG`: "Keyboard shortcuts configured"
- `DEBUG`: "Window state restored"

#### Monitor Detection
- `INFO`: "Detected {n} monitor(s):"
- `INFO`: "  Monitor {i}: {name} - {width}x{height} at ({x}, {y})"

#### Window Management
- `WARNING`: "Window not on any valid screen, moving to primary screen"
- `DEBUG`: "Window state saved"

#### System Control
- `INFO`: "Starting SENTINEL system"
- `INFO`: "Stopping SENTINEL system"
- `INFO`: "Restarting SENTINEL system"

#### User Actions
- `INFO`: "Open configuration requested"
- `INFO`: "Save configuration requested"
- `INFO`: "Export scenario requested"
- `INFO`: "Camera calibration requested"
- `INFO`: "Entered fullscreen mode"
- `INFO`: "Exited fullscreen mode"
- `INFO`: "Changing theme to: {theme_name}"
- `INFO`: "Changing accent color to: {accent_color}"
- `INFO`: "Moved window to monitor {index}: {name}"
- `WARNING`: "Invalid monitor index: {index}"
- `INFO`: "Resetting layout"
- `INFO`: "Starting recording"
- `INFO`: "Stopping recording"
- `INFO`: "Taking screenshot"
- `INFO`: "Opening settings"
- `INFO`: "Generating trip report"
- `INFO`: "Generating driver report"
- `INFO`: "Exporting data"
- `INFO`: "Opening documentation"

### 3. Live Monitor Widget (`src/gui/widgets/live_monitor.py`)

**Already Implemented:**

#### Initialization
- `INFO`: "LiveMonitorWidget initialized with update rate: {rate} Hz"

#### Frame Updates
- `DEBUG`: "Frame update timer started at {rate} Hz"
- `DEBUG`: "Frame update timer stopped"
- `DEBUG`: "All frames cleared"

#### Frame Display
- `DEBUG`: "Updated interior camera frame: {width}x{height}"
- `DEBUG`: "Updated front left camera frame: {width}x{height}"
- `DEBUG`: "Updated front right camera frame: {width}x{height}"
- `DEBUG`: "Updated BEV frame: {width}x{height}"

#### Performance
- `DEBUG`: "Frame update completed: duration={duration:.2f}ms"
- `WARNING`: "Frame update slow: duration={duration:.2f}ms (target: {target}ms)"

### 4. Video Display Widget (`src/gui/widgets/video_display.py`)

**Already Implemented:**

#### Initialization
- `DEBUG`: "VideoDisplayWidget created: {title}"

#### Frame Updates
- `DEBUG`: "Frame updated: {width}x{height}, conversion={duration:.2f}ms"
- `WARNING`: "Frame conversion slow: {duration:.2f}ms"

#### Errors
- `ERROR`: "Failed to convert frame: {error}"

### 5. BEV Canvas Widget (`src/gui/widgets/bev_canvas.py`)

**Already Implemented:**

#### Initialization
- `INFO`: "BEVCanvas initialized: size={width}x{height}, scale={scale}m/px"

#### Rendering
- `DEBUG`: "BEV image updated: {width}x{height}"
- `DEBUG`: "Rendered {n} detections"
- `DEBUG`: "Rendered {n} trajectories"
- `DEBUG`: "Rendered {n} zones"

#### Performance
- `DEBUG`: "Render completed: duration={duration:.2f}ms"
- `WARNING`: "Render slow: duration={duration:.2f}ms (target: 33ms)"

#### Interactions
- `DEBUG`: "Mouse clicked at BEV position: ({x:.2f}, {y:.2f})"
- `DEBUG`: "Detection selected: id={id}, class={class_name}"

### 6. Theme Manager (`src/gui/themes/theme_manager.py`)

**Already Implemented:**

#### Initialization
- `INFO`: "ThemeManager initialized with theme: {theme}"

#### Theme Changes
- `INFO`: "Theme changed: {old_theme} -> {new_theme}"
- `INFO`: "Accent color changed: {old_color} -> {new_color}"
- `DEBUG`: "Theme applied successfully"

#### Errors
- `ERROR`: "Failed to load theme file: {path}"
- `WARNING`: "Unknown theme requested: {theme}, using default"

## Log File Organization

All GUI logs are written to:
- **Primary**: `logs/sentinel.log` (all logs)
- **Errors**: `logs/errors.log` (ERROR and CRITICAL only)
- **Console**: INFO level and above

## Performance Considerations

### Logging Overhead
- **INFO level** used for user actions and state changes (minimal overhead)
- **DEBUG level** used for frame updates and rendering (disabled in production)
- Frame-by-frame logging only at DEBUG level to avoid performance impact

### Target Performance
- GUI update rate: 30 Hz (33ms per frame)
- Logging overhead: < 1ms per frame at INFO level
- No logging in tight rendering loops (use periodic summaries instead)

## Usage Examples

### Starting the GUI Application

```python
from src.gui import SENTINELMainWindow
from src.gui.themes import ThemeManager
from PyQt6.QtWidgets import QApplication
import sys

# Setup logging
from src.core.logging import LoggerSetup
LoggerSetup.setup(log_level='INFO', log_dir='logs')

# Create application
app = QApplication(sys.argv)

# Create theme manager
theme_manager = ThemeManager(theme='dark')

# Create main window
window = SENTINELMainWindow(theme_manager)
window.show()

# Run application
sys.exit(app.exec())
```

**Expected Logs:**
```
INFO - src.gui - GUI module initialized
INFO - src.gui.themes.theme_manager - ThemeManager initialized with theme: dark
DEBUG - src.gui.main_window - Main UI structure initialized
DEBUG - src.gui.main_window - Menu bar created
DEBUG - src.gui.main_window - Toolbar created
DEBUG - src.gui.main_window - Status bar created
DEBUG - src.gui.main_window - Keyboard shortcuts configured
INFO - src.gui.main_window - Detected 2 monitor(s):
INFO - src.gui.main_window -   Monitor 0: Display-1 - 1920x1080 at (0, 0)
INFO - src.gui.main_window -   Monitor 1: Display-2 - 2560x1440 at (1920, 0)
DEBUG - src.gui.main_window - Window state restored
INFO - src.gui.main_window - SENTINEL Main Window initialized
INFO - src.gui.widgets.live_monitor - LiveMonitorWidget initialized with update rate: 30 Hz
```

### System Start/Stop

```python
# User clicks "Start System" button
```

**Expected Logs:**
```
INFO - src.gui.main_window - Starting SENTINEL system
DEBUG - src.gui.widgets.live_monitor - Frame update timer started at 30 Hz
```

```python
# User clicks "Stop System" button
```

**Expected Logs:**
```
INFO - src.gui.main_window - Stopping SENTINEL system
DEBUG - src.gui.widgets.live_monitor - Frame update timer stopped
DEBUG - src.gui.widgets.live_monitor - All frames cleared
```

### Theme Changes

```python
# User changes theme to light mode
```

**Expected Logs:**
```
INFO - src.gui.main_window - Changing theme to: light
INFO - src.gui.themes.theme_manager - Theme changed: dark -> light
DEBUG - src.gui.themes.theme_manager - Theme applied successfully
```

### Frame Updates (DEBUG level)

```python
# Frame updates at 30 Hz
```

**Expected Logs (every frame at DEBUG level):**
```
DEBUG - src.gui.widgets.video_display - Frame updated: 640x480, conversion=2.34ms
DEBUG - src.gui.widgets.video_display - Frame updated: 1280x720, conversion=3.12ms
DEBUG - src.gui.widgets.video_display - Frame updated: 1280x720, conversion=3.45ms
DEBUG - src.gui.widgets.bev_canvas - BEV image updated: 640x640
DEBUG - src.gui.widgets.bev_canvas - Rendered 5 detections
DEBUG - src.gui.widgets.bev_canvas - Rendered 3 trajectories
DEBUG - src.gui.widgets.bev_canvas - Render completed: duration=8.23ms
DEBUG - src.gui.widgets.live_monitor - Frame update completed: duration=18.45ms
```

## Monitoring and Debugging

### Key Metrics to Monitor

1. **Frame Update Rate**
   - Target: 30 Hz (33ms per frame)
   - Warning threshold: > 40ms
   - Log at DEBUG level

2. **Rendering Performance**
   - BEV canvas render time: < 10ms
   - Video frame conversion: < 5ms
   - Total update time: < 33ms

3. **User Interactions**
   - All user actions logged at INFO level
   - State transitions logged at INFO level
   - Errors logged at ERROR level

### Debugging Tips

1. **Enable DEBUG logging** for detailed frame-by-frame analysis:
   ```python
   LoggerSetup.set_level('DEBUG')
   ```

2. **Filter logs by module**:
   ```bash
   grep "src.gui.widgets.bev_canvas" logs/sentinel.log
   ```

3. **Monitor performance**:
   ```bash
   grep "duration=" logs/sentinel.log | grep "WARNING"
   ```

4. **Track user actions**:
   ```bash
   grep "INFO.*src.gui.main_window" logs/sentinel.log
   ```

## Integration with SENTINEL System

The GUI logging integrates seamlessly with the main SENTINEL system logging:

1. **Shared log files**: All logs go to `logs/sentinel.log`
2. **Consistent format**: Same timestamp and format across all modules
3. **Performance tracking**: GUI performance metrics included in system performance logs
4. **Error aggregation**: All errors collected in `logs/errors.log`

## Future Enhancements

1. **GUI-specific log file**: Consider adding `logs/gui.log` for GUI-only logs
2. **Performance metrics**: Add dedicated performance logger for GUI metrics
3. **User action analytics**: Track user interaction patterns for UX improvements
4. **Crash reporting**: Enhanced error logging with stack traces and system state

## Validation

### Test Logging Setup

```bash
# Run GUI tests with logging enabled
pytest tests/test_gui.py -v --log-cli-level=DEBUG

# Verify log files created
ls -lh logs/

# Check log content
tail -f logs/sentinel.log
```

### Expected Output

```
logs/sentinel.log - Contains all GUI logs
logs/errors.log - Contains GUI errors (if any)
```

## Status

✅ **Complete** - GUI module logging fully implemented and integrated with SENTINEL system logging infrastructure.

All GUI components have comprehensive logging at appropriate levels:
- User interactions: INFO
- State changes: INFO
- Performance metrics: DEBUG
- Errors: ERROR
- Initialization: INFO/DEBUG

The logging configuration supports real-time monitoring and debugging while maintaining minimal performance overhead at INFO level.
