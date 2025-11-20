# Camera Management Module

The Camera Management Module provides multi-threaded camera capture, timestamp synchronization, and calibration loading for the SENTINEL system.

## Components

### 1. CameraCapture
Multi-threaded camera capture from USB devices with health monitoring.

**Features:**
- Thread-safe circular buffer for frames
- Automatic reconnection on camera disconnection
- Health monitoring and failure detection
- Configurable resolution and FPS

### 2. TimestampSync
Software-based timestamp synchronization for multi-camera frames.

**Features:**
- Configurable tolerance (default: ±5ms)
- Frame dropping for out-of-sync frames
- Synchronization statistics tracking

### 3. CalibrationLoader
Loads camera calibration data from YAML files.

**Features:**
- Intrinsic parameters (focal length, principal point, distortion)
- Extrinsic parameters (translation, rotation)
- Homography matrices for BEV transformation

### 4. CameraManager
Integrates all components and implements the ICameraManager interface.

**Features:**
- Manages multiple cameras simultaneously
- Synchronized frame bundle generation
- Graceful degradation on camera failure
- Automatic reconnection logic
- Health monitoring

## Usage

```python
from src.camera import CameraManager
from src.core.config import ConfigManager

# Load configuration
config_manager = ConfigManager('configs/default.yaml')
config = config_manager.config

# Create and start camera manager
camera_manager = CameraManager(config)
camera_manager.start()

# Get synchronized frame bundle
bundle = camera_manager.get_frame_bundle()
if bundle:
    print(f"Interior: {bundle.interior.shape}")
    print(f"Front Left: {bundle.front_left.shape}")
    print(f"Front Right: {bundle.front_right.shape}")
    print(f"Timestamp: {bundle.timestamp}")

# Check health
if camera_manager.is_healthy():
    print("All cameras operational")

# Get statistics
stats = camera_manager.get_statistics()
print(f"Frames captured: {stats['frame_count']}")

# Stop cameras
camera_manager.stop()
```

## Configuration

Camera configuration in `configs/default.yaml`:

```yaml
cameras:
  interior:
    device: 0
    resolution: [640, 480]
    fps: 30
    calibration: "configs/calibration/interior.yaml"
  
  front_left:
    device: 1
    resolution: [1280, 720]
    fps: 30
    calibration: "configs/calibration/front_left.yaml"
  
  front_right:
    device: 2
    resolution: [1280, 720]
    fps: 30
    calibration: "configs/calibration/front_right.yaml"
```

## Calibration Format

Calibration files in `configs/calibration/*.yaml`:

```yaml
intrinsics:
  fx: 800.0
  fy: 800.0
  cx: 640.0
  cy: 360.0
  distortion: [k1, k2, p1, p2, k3]

extrinsics:
  translation: [x, y, z]  # meters from vehicle origin
  rotation: [roll, pitch, yaw]  # radians

homography:
  matrix: [[h11, h12, h13],
           [h21, h22, h23],
           [h31, h32, h33]]
```

## Requirements Met

This implementation satisfies the following requirements:

- **Requirement 1.1**: Multi-camera initialization with configured resolution and frame rate
- **Requirement 1.2**: Timestamp synchronization within ±5ms tolerance
- **Requirement 1.3**: Camera disconnection detection within 1 second
- **Requirement 1.4**: Automatic camera reconnection
- **Requirement 1.5**: Calibration data loading from persistent storage
- **Requirement 11.2**: Graceful degradation on camera failure

## Error Handling

The module implements robust error handling:

1. **Camera Disconnection**: Detected within 1 second, automatic reconnection attempted
2. **Synchronization Failure**: Frames outside tolerance are dropped
3. **Graceful Degradation**: System continues with available cameras
4. **Health Monitoring**: Continuous monitoring of camera status

## Performance

- **Capture Rate**: 30 FPS per camera
- **Synchronization Tolerance**: ±5ms
- **Buffer Size**: 5 frames per camera (configurable)
- **Reconnection Delay**: 1 second

## Testing

Run tests with:

```bash
pytest tests/test_camera.py -v
```

## Example

See `examples/camera_example.py` for a complete usage example.
