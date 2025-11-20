# BEV Canvas Widget

Interactive Bird's Eye View canvas widget for the SENTINEL GUI application.

## Features

### Core Functionality
- **BEV Image Display**: Renders 640x640 pixel bird's eye view images
- **Zoom & Pan**: Mouse wheel zoom (0.5x to 5.0x) and drag-to-pan controls
- **Interactive Selection**: Click objects to select and view details

### Overlays

#### 1. Object Overlays
- 3D bounding boxes in top-down view
- Color-coded by object class
- Track IDs and confidence scores
- Click detection and highlighting

#### 2. Trajectory Visualization
- Predicted paths as dashed lines
- Uncertainty bounds (semi-transparent)
- Color-coded by collision probability
- Waypoint markers

#### 3. Attention Zones
- 8 directional zones around vehicle
- Highlight attended zones
- Color-coded by risk level
- Zone labels with risk scores

#### 4. Distance Grid
- Concentric circles at 5-meter intervals
- Radial lines every 45 degrees
- Distance labels
- Vehicle position marker

### Capture & Recording
- **Screenshot**: Save current view to PNG with timestamp
- **Video Recording**: Record to MP4 at configurable FPS

## Usage

### Basic Setup

```python
from src.gui.widgets import BEVCanvas

# Create canvas
canvas = BEVCanvas()

# Update BEV image
import numpy as np
bev_image = np.zeros((640, 640, 3), dtype=np.uint8)
canvas.update_bev_image(bev_image)
```

### Object Overlays

```python
detections = [
    {
        'object_id': 1,
        'bbox_3d': (x, y, z, w, h, l, theta),  # Vehicle frame coordinates
        'class_name': 'vehicle',
        'confidence': 0.95,
        'track_id': 101
    }
]
canvas.update_objects(detections)
```

### Trajectory Visualization

```python
trajectories = [
    {
        'object_id': 1,
        'points': [(x1, y1), (x2, y2), ...],  # Vehicle frame coordinates
        'uncertainty': [std1, std2, ...],  # Optional
        'collision_probability': 0.2
    }
]
canvas.update_trajectories(trajectories)
```

### Attention Zones

```python
zones_data = {
    'attended_zone': 'front',  # Zone driver is looking at
    'zone_risks': {
        'front': 0.3,
        'front_right': 0.5,
        'right': 0.7,
        # ... other zones
    }
}
canvas.update_attention_zones(zones_data)
```

### Visibility Controls

```python
canvas.set_show_objects(True)       # Toggle object overlays
canvas.set_show_trajectories(True)  # Toggle trajectories
canvas.set_show_zones(True)         # Toggle attention zones
canvas.set_show_grid(True)          # Toggle distance grid
```

### Screenshot & Recording

```python
# Take screenshot
canvas.capture_screenshot("screenshot.png")

# Start recording
canvas.start_recording("video.mp4", fps=30)

# Stop recording
canvas.stop_recording()

# Check if recording
if canvas.is_recording():
    print("Recording in progress")
```

### Object Selection

```python
# Connect to object click signal
canvas.object_clicked.connect(lambda obj_id: print(f"Selected: {obj_id}"))

# Programmatically highlight object
canvas.highlight_object(object_id)

# Clear highlights
canvas.unhighlight_all_objects()
```

## Coordinate Systems

### Vehicle Frame
- Origin: Center of rear axle
- X-axis: Forward (front of vehicle)
- Y-axis: Left
- Units: Meters

### BEV/Scene Frame
- Origin: Top-left corner of image
- X-axis: Right
- Y-axis: Down (forward in vehicle frame)
- Units: Pixels (0.1 meters per pixel)

### Conversion

```python
# Vehicle to scene
scene_x, scene_y = canvas.get_scene_coordinates(vehicle_x, vehicle_y)

# Scene to vehicle
vehicle_x, vehicle_y = canvas.get_vehicle_coordinates(scene_x, scene_y)
```

## Object Classes & Colors

| Class | Color |
|-------|-------|
| Vehicle | Blue (#00AAFF) |
| Pedestrian | Orange (#FFAA00) |
| Cyclist | Red-Orange (#FF5500) |
| Traffic Sign | Yellow (#FFFF00) |
| Traffic Light | Green (#00FF00) |
| Obstacle | Red (#FF0000) |

## Trajectory Colors

| Collision Probability | Color |
|----------------------|-------|
| < 0.3 (Low) | Green |
| 0.3 - 0.6 (Medium) | Yellow |
| 0.6 - 0.8 (High) | Orange |
| > 0.8 (Critical) | Red |

## Zone Colors

### Attended Zones
- Low risk: Cyan (#00C8FF)
- Medium risk: Light Blue (#0096FF)
- High risk: Blue (#6464FF)

### Unattended Zones
- Low risk: Gray (#646464)
- Medium risk: Yellow (#FFC800)
- High risk: Red (#FF3232)

## Signals

- `object_clicked(int)`: Emitted when an object is clicked with its object_id

## Mouse Controls

- **Left Click**: Select object
- **Middle Click + Drag**: Pan view
- **Mouse Wheel**: Zoom in/out

## Performance

- GPU-accelerated rendering
- Efficient item management with dictionaries
- Z-ordered layers for proper overlay stacking
- Incremental updates (only changed items redrawn)

## Dependencies

- PyQt6
- NumPy
- OpenCV (cv2) - Optional, for video recording only

## Testing

Run the test script to see all features in action:

```bash
python test_bev_canvas.py
```

## Integration

The BEV canvas is integrated into `LiveMonitorWidget` and replaces the standard video display for the BEV view. It provides the same interface for frame updates while adding interactive capabilities.

```python
from src.gui.widgets import LiveMonitorWidget

monitor = LiveMonitorWidget()

# Update BEV frame (automatically uses BEVCanvas)
monitor.update_camera_frame('bev', bev_image)

# Access BEV canvas directly
monitor.bev_canvas.update_objects(detections)
monitor.bev_canvas.update_trajectories(trajectories)
```
