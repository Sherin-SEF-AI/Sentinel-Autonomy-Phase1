# SENTINEL PyQt6 GUI

Professional desktop GUI application for monitoring and controlling the SENTINEL contextual safety intelligence platform.

## Overview

The SENTINEL GUI provides a comprehensive interface for:
- Real-time monitoring of camera feeds and BEV output
- Driver state visualization
- Risk assessment dashboard
- System configuration and control
- Scenario recording and playback
- Performance monitoring

## Architecture

```
src/gui/
├── __init__.py              # Module initialization
├── main_window.py           # Main application window
├── widgets/                 # Custom widgets
│   ├── __init__.py
│   ├── live_monitor.py      # 2x2 camera grid
│   └── video_display.py     # Video frame display
├── themes/                  # Theme system
│   ├── __init__.py
│   ├── theme_manager.py     # Theme management
│   ├── dark.qss            # Dark theme stylesheet
│   └── light.qss           # Light theme stylesheet
├── dock_widgets/           # Dockable panels (future)
├── dialogs/                # Modal dialogs (future)
└── resources/              # Icons, images, fonts (future)
```

## Features

### Main Window (Task 16.1) ✓

- **Menu Bar**: File, System, View, Tools, Analytics, Help menus
- **Toolbar**: Quick action buttons (Start, Stop, Record, Screenshot)
- **Status Bar**: System status indicators
- **Keyboard Shortcuts**:
  - `F5`: Start system
  - `F6`: Stop system
  - `F11`: Toggle fullscreen
  - `Ctrl+Q`: Quit application
  - `Ctrl+R`: Toggle recording
  - `Ctrl+S`: Take screenshot

### Central Monitoring Widget (Task 16.2) ✓

- **2x2 Camera Grid Layout**:
  ```
  ┌─────────────┬─────────────┐
  │  Interior   │ Front Left  │
  ├─────────────┼─────────────┤
  │ Front Right │     BEV     │
  └─────────────┴─────────────┘
  ```
- **Real-time Updates**: 30 FPS frame updates
- **Responsive Layout**: QSplitter for resizable panels
- **Efficient Rendering**: Numpy array to QPixmap conversion

### Theme System (Task 16.3) ✓

- **Dark Theme**: Modern dark theme with reduced eye strain
- **Light Theme**: Clean light theme for bright environments
- **Configurable Accent Colors**:
  - Blue (default)
  - Green
  - Red
  - Purple
  - Orange
  - Teal
- **Dynamic Switching**: Change themes without restart
- **Persistent Preferences**: Theme choice saved across sessions

### Window State Persistence (Task 16.4) ✓

- **Geometry Persistence**: Window size and position saved
- **Dock State**: Dock widget positions and visibility saved
- **User Preferences**: Theme, shortcuts, and settings saved
- **Automatic Restore**: State restored on application startup

### Multi-Monitor Support (Task 16.5) ✓

- **Monitor Detection**: Automatically detects all available monitors
- **Window Dragging**: Drag window across monitors
- **Floating Docks**: Dock widgets can float on secondary monitors
- **Monitor Selection**: Menu option to move window to specific monitor
- **Layout Persistence**: Monitor-specific layouts saved

## Usage

### Running the GUI

```bash
# Run the GUI application
python src/gui_main.py

# Or use the test script
python test_gui.py
```

### Basic Operations

1. **Start System**: Click "Start System" button or press `F5`
2. **Stop System**: Click "Stop System" button or press `F6`
3. **Change Theme**: View > Theme > Dark/Light Theme
4. **Change Accent**: View > Theme > Accent Color > [Color]
5. **Move to Monitor**: View > Move to Monitor > [Monitor]
6. **Fullscreen**: Press `F11` or View > Fullscreen

### Integrating with SENTINEL System

```python
from PyQt6.QtWidgets import QApplication
from gui.main_window import SENTINELMainWindow
from gui.themes import ThemeManager

# Create application
app = QApplication(sys.argv)

# Create theme manager
theme_manager = ThemeManager(app)

# Create main window
main_window = SENTINELMainWindow(theme_manager)

# Update camera frames
main_window.live_monitor.update_camera_frame('interior', interior_frame)
main_window.live_monitor.update_camera_frame('front_left', front_left_frame)
main_window.live_monitor.update_camera_frame('front_right', front_right_frame)
main_window.live_monitor.update_camera_frame('bev', bev_frame)

# Or update all at once
main_window.live_monitor.update_all_frames({
    'interior': interior_frame,
    'front_left': front_left_frame,
    'front_right': front_right_frame,
    'bev': bev_frame
})

# Show window
main_window.show()

# Run event loop
app.exec()
```

## Components

### VideoDisplayWidget

Displays individual camera feeds with:
- Automatic aspect ratio preservation
- Smooth scaling
- Camera name label
- No signal indicator

```python
from gui.widgets import VideoDisplayWidget

# Create widget
video_widget = VideoDisplayWidget("Camera Name")

# Update frame
video_widget.update_frame(numpy_array)

# Clear frame
video_widget.clear_frame()
```

### LiveMonitorWidget

Central monitoring widget with 2x2 grid:

```python
from gui.widgets import LiveMonitorWidget

# Create widget
monitor = LiveMonitorWidget()

# Start updates (30 FPS)
monitor.start_updates()

# Update frames
monitor.update_camera_frame('interior', frame)
monitor.update_all_frames(frames_dict)

# Stop updates
monitor.stop_updates()

# Change update rate
monitor.set_update_rate(60)  # 60 Hz
```

### ThemeManager

Manages application themes:

```python
from gui.themes import ThemeManager

# Create theme manager
theme_manager = ThemeManager(app)

# Apply theme
theme_manager.apply_theme()

# Change theme
theme_manager.set_theme('dark')
theme_manager.set_theme('light')

# Change accent color
theme_manager.set_accent_color('blue')
theme_manager.set_accent_color('#0078d4')

# Toggle theme
theme_manager.toggle_theme()

# Get current settings
current_theme = theme_manager.get_current_theme()
current_accent = theme_manager.get_current_accent()
```

## Customization

### Adding Custom Accent Colors

Edit `src/gui/themes/theme_manager.py`:

```python
DEFAULT_ACCENT_COLORS = {
    'blue': '#0078d4',
    'green': '#107c10',
    'custom': '#ff00ff',  # Add your color
}
```

### Modifying Themes

Edit theme stylesheets:
- `src/gui/themes/dark.qss` - Dark theme
- `src/gui/themes/light.qss` - Light theme

Use `{accent_color}` placeholder for accent color substitution.

## Future Enhancements

The following features will be implemented in future tasks:

- **Task 17**: Interactive BEV Canvas with object overlays
- **Task 18**: Driver State Panel with gauges and metrics
- **Task 19**: Risk Assessment Panel with hazard list
- **Task 20**: Alerts Panel with audio and visual alerts
- **Task 21**: Performance Monitoring Dock
- **Task 22**: Scenarios Dock with playback controls
- **Task 23**: Configuration Dock with live tuning
- **Task 24**: Worker Thread Integration for SENTINEL system

## Requirements

- Python 3.10+
- PyQt6 >= 6.5.0
- PyQtGraph >= 0.13.0 (for future tasks)
- NumPy >= 1.24.0

## Testing

```bash
# Test GUI launch
python test_gui.py

# Run unit tests (when available)
pytest tests/test_gui.py
```

## Troubleshooting

### GUI doesn't start

- Ensure PyQt6 is installed: `pip install PyQt6`
- Check display environment variable: `echo $DISPLAY`
- Try running with: `QT_DEBUG_PLUGINS=1 python src/gui_main.py`

### Theme not applying

- Check theme files exist in `src/gui/themes/`
- Verify QSettings permissions
- Reset settings: Delete `~/.config/SENTINEL/SentinelGUI.conf`

### Window not visible

- Check if window is on disconnected monitor
- Use View > Move to Monitor to move window
- Or reset layout: View > Reset Layout

## License

© 2024 SENTINEL Project
