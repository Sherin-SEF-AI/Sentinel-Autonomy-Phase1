# Multi-View Object Detection Module

This module implements multi-view 3D object detection with tracking for the SENTINEL system.

## Components

### Detector2D (`detector_2d.py`)
- Uses YOLOv8 for per-camera 2D object detection
- Detects automotive classes: vehicles, pedestrians, cyclists, traffic signs/lights
- Configurable confidence and NMS thresholds
- Target: 20ms per camera

### Estimator3D (`estimator_3d.py`)
- Estimates 3D bounding boxes from 2D detections
- Uses camera extrinsics to transform to vehicle coordinate frame
- Applies pinhole camera model for depth estimation
- Outputs (x, y, z, w, h, l, θ) in vehicle frame

### MultiViewFusion (`fusion.py`)
- Fuses detections from multiple camera views
- Uses IoU-based clustering to identify same objects
- Applies confidence weighting for robust fusion
- Outputs unified 3D detection list

### ObjectTracker (`tracker.py`)
- Tracks objects across frames using DeepSORT-inspired algorithm
- Maintains consistent track IDs (minimum 30 frames)
- Uses Kalman filter for state estimation and velocity calculation
- Configurable max_age, min_hits, iou_threshold

### ObjectDetector (`detector.py`)
- Main class integrating all components
- Implements IObjectDetector interface
- Includes error recovery mechanism
- Returns both 2D and 3D detections

## Pipeline Flow

```
Camera Frames (Dict[int, np.ndarray])
    ↓
[Detector2D] → 2D Detections per camera
    ↓
[Estimator3D] → 3D Bounding boxes
    ↓
[MultiViewFusion] → Fused 3D detections
    ↓
[ObjectTracker] → Tracked objects with IDs and velocities
    ↓
Detection2D (per camera) + Detection3D (tracked)
```

## Configuration

```yaml
models:
  detection:
    architecture: "YOLOv8"
    variant: "yolov8m"
    weights: "models/yolov8m_automotive.pt"
    confidence_threshold: 0.5
    nms_threshold: 0.4
    device: "cuda"

fusion:
  method: "hungarian"
  iou_threshold_3d: 0.3
  confidence_weighting: true

tracking:
  algorithm: "DeepSORT"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
```

## Usage

```python
from src.perception.detection import ObjectDetector
from src.core.config import ConfigManager

# Load configuration
config = ConfigManager('configs/default.yaml')
detection_config = {
    'detection': config.get('models.detection'),
    'fusion': config.get('fusion'),
    'tracking': config.get('tracking')
}

# Load calibration data
calibration_data = {
    1: config.load_calibration('configs/calibration/front_left.yaml'),
    2: config.load_calibration('configs/calibration/front_right.yaml')
}

# Initialize detector
detector = ObjectDetector(detection_config, calibration_data)

# Detect objects
frames = {
    1: front_left_frame,  # (720, 1280, 3)
    2: front_right_frame  # (720, 1280, 3)
}

detections_2d, detections_3d = detector.detect(frames)

# Access results
for det_3d in detections_3d:
    x, y, z, w, h, l, theta = det_3d.bbox_3d
    vx, vy, vz = det_3d.velocity
    print(f"Track {det_3d.track_id}: {det_3d.class_name} at ({x:.1f}, {y:.1f}, {z:.1f})")
    print(f"  Velocity: ({vx:.1f}, {vy:.1f}, {vz:.1f}) m/s")
```

## Performance

- **2D Detection**: ~20ms per camera (YOLOv8m on GPU)
- **3D Estimation**: <1ms per detection
- **Fusion**: <5ms for typical scene
- **Tracking**: <5ms for typical scene
- **Total**: ~50ms for 2 cameras with 10-20 objects

## Requirements

- `ultralytics`: YOLOv8 implementation
- `scipy`: Hungarian algorithm for matching
- `numpy`: Numerical operations

## Coordinate Systems

### Vehicle Frame
- Origin: Center of rear axle
- X-axis: Forward
- Y-axis: Left
- Z-axis: Up

### Camera Frame
- Origin: Camera optical center
- Z-axis: Along optical axis (forward)
- X-axis: Right
- Y-axis: Down

## Error Handling

The detector includes automatic error recovery:
- Tracks consecutive inference failures
- Reloads model after 3 consecutive errors
- Returns last valid output during recovery
- Logs all errors for debugging

## Testing

```bash
# Run detection tests
pytest tests/test_detection.py -v

# Run with specific camera setup
python examples/detection_example.py
```
