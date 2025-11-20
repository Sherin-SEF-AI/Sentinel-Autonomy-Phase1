# Driver Monitoring System (DMS)

The Driver Monitoring System (DMS) module monitors driver state including attention, drowsiness, and distraction through interior camera analysis.

## Overview

The DMS integrates multiple computer vision and machine learning techniques to provide comprehensive driver state monitoring:

- **Face Detection & Landmarks**: MediaPipe-based facial landmark extraction (68 points)
- **Gaze Estimation**: L2CS-Net model for pitch/yaw estimation and attention zone mapping
- **Head Pose**: PnP algorithm for roll/pitch/yaw calculation
- **Drowsiness Detection**: PERCLOS, yawning, micro-sleep, and head nodding detection
- **Distraction Classification**: MobileNetV3-based classification of distraction types
- **Readiness Score**: Weighted combination of alertness, attention, and distraction (0-100)

## Architecture

```
DriverMonitor (IDMS)
├── FaceDetector (MediaPipe)
│   └── detect_and_extract_landmarks()
├── GazeEstimator (L2CS-Net)
│   └── estimate_gaze() → pitch, yaw, attention_zone
├── HeadPoseEstimator (PnP)
│   └── estimate_head_pose() → roll, pitch, yaw
├── DrowsinessDetector
│   └── detect_drowsiness() → score, PERCLOS, yawn, micro-sleep
├── DistractionClassifier (MobileNetV3)
│   └── classify_distraction() → type, confidence, duration
└── ReadinessCalculator
    └── calculate_readiness() → score (0-100)
```

## Components

### FaceDetector (`face.py`)

Detects face and extracts 68 facial landmarks using MediaPipe Face Mesh.

**Key Features:**
- Real-time face detection
- 68-point landmark extraction (approximated from MediaPipe's 468 landmarks)
- Robust to varying lighting conditions

**Usage:**
```python
from src.dms.face import FaceDetector

detector = FaceDetector()
face_detected, landmarks = detector.detect_and_extract_landmarks(frame)
```

### GazeEstimator (`gaze.py`)

Estimates gaze direction and maps to 8 attention zones around the vehicle.

**Attention Zones:**
- `front`: -30° to 30° (primary driving zone)
- `front_left`: 30° to 75°
- `left`: 75° to 105°
- `rear_left`: 105° to 150°
- `rear`: 150° to 180° / -180° to -150°
- `rear_right`: -150° to -105°
- `right`: -105° to -75°
- `front_right`: -75° to -30°

**Usage:**
```python
from src.dms.gaze import GazeEstimator

estimator = GazeEstimator(model_path='models/l2cs_gaze.pth')
gaze = estimator.estimate_gaze(frame, landmarks)
# gaze = {'pitch': 5.2, 'yaw': -12.3, 'attention_zone': 'front'}
```

### HeadPoseEstimator (`pose.py`)

Calculates head pose (roll, pitch, yaw) using PnP algorithm with 3D face model.

**Usage:**
```python
from src.dms.pose import HeadPoseEstimator

estimator = HeadPoseEstimator()
head_pose = estimator.estimate_head_pose(landmarks, frame.shape)
# head_pose = {'roll': 2.1, 'pitch': -5.3, 'yaw': 8.7}
```

### DrowsinessDetector (`drowsiness.py`)

Detects drowsiness using multiple indicators:

**Indicators:**
- **PERCLOS**: Percentage of eye closure over 60-frame window (2 seconds)
- **Eye Aspect Ratio (EAR)**: Geometric ratio of eye openness
- **Yawning**: Mouth aspect ratio detection with frequency tracking
- **Micro-sleep**: Eyes closed for >2 seconds
- **Head Nodding**: Significant downward head movements

**Thresholds:**
- PERCLOS > 80% for 3+ seconds → Drowsy
- Yawn frequency > 3 per minute → Drowsy
- Head nod frequency > 2 per minute → Drowsy

**Usage:**
```python
from src.dms.drowsiness import DrowsinessDetector

detector = DrowsinessDetector(fps=30)
drowsiness = detector.detect_drowsiness(landmarks, head_pose)
# drowsiness = {
#     'score': 0.65,
#     'yawn_detected': False,
#     'micro_sleep': False,
#     'head_nod': False,
#     'perclos': 0.45
# }
```

### DistractionClassifier (`distraction.py`)

Classifies driver distraction type using MobileNetV3 or rule-based approach.

**Distraction Types:**
- `safe_driving`: Driver attentive
- `phone_usage`: Using phone
- `looking_at_passenger`: Interacting with passenger
- `adjusting_controls`: Adjusting vehicle controls
- `eyes_off_road`: Eyes off road for >2 seconds
- `hands_off_wheel`: Hands not on steering wheel

**Usage:**
```python
from src.dms.distraction import DistractionClassifier

classifier = DistractionClassifier(model_path='models/distraction_clf.pth')
distraction = classifier.classify_distraction(frame, gaze, head_pose)
# distraction = {
#     'type': 'safe_driving',
#     'confidence': 0.92,
#     'duration': 0.0,
#     'eyes_off_road': False
# }
```

### ReadinessCalculator (`readiness.py`)

Calculates driver readiness score (0-100) from alertness, attention, and distraction.

**Formula:**
```
Readiness = 0.4 × Alertness + 0.3 × Attention + 0.3 × (1 - Distraction)
```

**Interpretation:**
- 80-100: Fully ready
- 60-79: Mostly ready
- 40-59: Moderately ready
- 20-39: Low readiness
- 0-19: Very low readiness

**Usage:**
```python
from src.dms.readiness import ReadinessCalculator

calculator = ReadinessCalculator()
readiness = calculator.calculate_readiness(drowsiness, gaze, distraction)
# readiness = 78.5
```

### DriverMonitor (`monitor.py`)

Main DMS class that integrates all components and implements the IDMS interface.

**Usage:**
```python
from src.core.config import ConfigManager
from src.dms import DriverMonitor

config = ConfigManager('configs/default.yaml').config
dms = DriverMonitor(config)

# Analyze frame
driver_state = dms.analyze(interior_frame)

# Access results
print(f"Readiness: {driver_state.readiness_score:.1f}")
print(f"Attention Zone: {driver_state.gaze['attention_zone']}")
print(f"Drowsiness: {driver_state.drowsiness['score']:.2f}")
print(f"Distraction: {driver_state.distraction['type']}")
```

## Output: DriverState

The `analyze()` method returns a `DriverState` dataclass:

```python
@dataclass
class DriverState:
    face_detected: bool                    # Face detection status
    landmarks: np.ndarray                  # (68, 2) facial landmarks
    head_pose: Dict[str, float]           # roll, pitch, yaw
    gaze: Dict[str, Any]                  # pitch, yaw, attention_zone
    eye_state: Dict[str, float]           # left_ear, right_ear, perclos
    drowsiness: Dict[str, Any]            # score, yawn, micro_sleep, head_nod
    distraction: Dict[str, Any]           # type, confidence, duration
    readiness_score: float                # 0-100
```

## Performance

**Target:** 25ms processing time per frame

**Breakdown:**
- Face detection: ~8ms
- Gaze estimation: ~5ms
- Head pose: ~2ms
- Drowsiness detection: ~3ms
- Distraction classification: ~5ms
- Readiness calculation: ~1ms
- **Total:** ~24ms

## Error Handling

The DMS includes robust error handling:

1. **Face Detection Failure**: Returns default state with `face_detected=False`
2. **Model Inference Errors**: Falls back to geometric/rule-based methods
3. **Component Failures**: Caches last valid state and attempts recovery
4. **Automatic Reset**: Reinitializes components after 5 consecutive errors

## Configuration

DMS configuration in `configs/default.yaml`:

```yaml
cameras:
  interior:
    device: 0
    resolution: [640, 480]
    fps: 30

models:
  dms:
    face_detection: "MediaPipe"
    gaze_model: "L2CS-Net"
    gaze_weights: "models/l2cs_gaze.pth"
    drowsiness_weights: "models/drowsiness_model.pth"
    distraction_weights: "models/distraction_clf.pth"
    device: "cuda"
```

## Example

Run the DMS example:

```bash
python examples/dms_example.py
```

This will:
1. Initialize the DMS with all components
2. Open the interior camera
3. Display real-time driver state analysis
4. Show facial landmarks, gaze direction, and metrics

## Requirements

**Dependencies:**
- OpenCV: Camera capture and image processing
- MediaPipe: Face detection and landmarks
- PyTorch: Deep learning models
- NumPy: Numerical operations

**Models:**
- L2CS-Net: Gaze estimation (optional, falls back to geometric)
- MobileNetV3: Distraction classification (optional, falls back to rules)

## Testing

Run DMS tests:

```bash
# Unit tests
pytest tests/test_dms.py -v

# Integration tests
pytest tests/test_integration.py::test_dms_pipeline -v
```

## Future Enhancements

- [ ] Emotion recognition
- [ ] Driver identification
- [ ] Fatigue prediction
- [ ] Cognitive load estimation
- [ ] Multi-driver support
