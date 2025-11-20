# Design Document

## Overview

SENTINEL is a real-time contextual safety intelligence platform that integrates multi-camera perception, driver monitoring, and risk assessment into a unified system. The architecture follows a modular pipeline design where data flows from camera capture through perception modules, contextual intelligence, and finally to alert generation and visualization.

The system operates at 30+ FPS with end-to-end latency under 100ms, processing synchronized camera feeds through parallel perception pipelines (BEV generation, object detection, driver monitoring) before fusing results in the contextual intelligence engine. This design enables the unique capability of correlating environmental hazards with driver awareness state to generate context-aware safety interventions.

**Key Design Principles:**
- Modular architecture with clear interfaces between components
- Parallel processing pipelines to maximize throughput
- Configuration-driven behavior (no hardcoded parameters)
- Graceful degradation on component failure
- Real-time performance constraints enforced at each stage

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENTINEL SYSTEM                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
         ┌──────────▼──────────┐   ┌─────────▼──────────┐
         │  Camera Management  │   │  Config Manager    │
         │  - Capture          │   │  - YAML Loading    │
         │  - Sync             │   │  - Hot Reload      │
         │  - Calibration      │   └────────────────────┘
         └──────────┬──────────┘
                    │
         ┌──────────┴──────────┐
         │  Camera Bundle      │
         │  (Synchronized)     │
         └──────────┬──────────┘
                    │
         ┌──────────┴────────────────────────┐
         │                                    │
    ┌────▼─────┐                    ┌────────▼────────┐
    │   DMS    │                    │   Perception    │
    │ Pipeline │                    │    Pipeline     │
    └────┬─────┘                    └────────┬────────┘
         │                                    │
         │                          ┌─────────┴─────────┐
         │                          │                   │
         │                    ┌─────▼──────┐   ┌───────▼────────┐
         │                    │ BEV Gen    │   │ Multi-View     │
         │                    │ + Stitch   │   │ Detection      │
         │                    └─────┬──────┘   └───────┬────────┘
         │                          │                   │
         │                    ┌─────▼──────┐           │
         │                    │ Semantic   │           │
         │                    │ Seg (BEV)  │           │
         │                    └─────┬──────┘           │
         │                          │                   │
         │                          └─────────┬─────────┘
         │                                    │
         └──────────┬─────────────────────────┘
                    │
         ┌──────────▼──────────────────┐
         │ Contextual Intelligence     │
         │ - Scene Graph               │
         │ - Attention Mapping         │
         │ - Risk Assessment           │
         │ - Trajectory Prediction     │
         └──────────┬──────────────────┘
                    │
         ┌──────────▼──────────────────┐
         │  Alert & Action System      │
         │  - Priority Queue           │
         │  - Suppression Logic        │
         │  - Multi-Modal Dispatch     │
         └──────────┬──────────────────┘
                    │
         ┌──────────┴──────────────────┐
         │                             │
    ┌────▼─────────┐          ┌───────▼──────────┐
    │ Visualization│          │ Data Recording   │
    │ Dashboard    │          │ & Playback       │
    └──────────────┘          └──────────────────┘
```

### Module Breakdown

**1. Camera Management Module**
- Manages 3 USB cameras (1 interior, 2 external)
- Multi-threaded capture with timestamp synchronization
- Loads and applies calibration parameters
- Handles camera disconnection/reconnection
- Outputs synchronized CameraBundle

**2. BEV Generation Module**
- Transforms external camera views to top-down perspective
- Applies inverse perspective mapping using homography matrices
- Stitches multiple BEV views with multi-band blending
- Generates valid region masks
- Target: 15ms processing time

**3. Semantic Segmentation Module**
- Classifies BEV pixels into 9 classes
- Uses deep learning model (BEVFormer-Tiny or similar)
- Applies temporal smoothing for stability
- Outputs class map and confidence scores
- Target: 15ms inference time

**4. Multi-View Object Detection Module**
- Detects objects in each camera view
- Estimates 3D bounding boxes from 2D detections
- Fuses multi-view detections using Hungarian algorithm
- Tracks objects with DeepSORT
- Estimates object velocities
- Target: 20ms per camera

**5. Driver Monitoring System (DMS)**
- Face detection and landmark extraction
- Gaze estimation (pitch, yaw)
- Head pose calculation
- Drowsiness detection (PERCLOS, yawning, micro-sleep)
- Distraction classification
- Readiness score calculation
- Target: 25ms processing time

**6. Contextual Intelligence Engine**
- Builds spatial scene graph
- Maps driver attention to spatial zones
- Calculates Time-To-Collision
- Predicts object trajectories
- Computes contextual risk scores
- Identifies attention-risk mismatches
- Target: 10ms processing time

**7. Alert & Action System**
- Prioritizes risks
- Generates alerts with urgency levels
- Suppresses redundant alerts
- Dispatches multi-modal alerts
- Adapts to driver cognitive load

**8. Data Recording Module**
- Triggers on high-risk scenarios
- Records all camera feeds and system outputs
- Exports scenarios in JSON + video format
- Supports playback and analysis

**9. Visualization Dashboard**
- Web-based real-time display
- Shows BEV, detections, driver state, risks
- Performance monitoring
- Scenario playback interface

## Components and Interfaces

### Core Data Structures

```python
@dataclass
class CameraBundle:
    """Synchronized frame bundle from all cameras"""
    timestamp: float
    interior: np.ndarray  # (480, 640, 3)
    front_left: np.ndarray  # (720, 1280, 3)
    front_right: np.ndarray  # (720, 1280, 3)

@dataclass
class BEVOutput:
    """Bird's eye view transformation result"""
    timestamp: float
    image: np.ndarray  # (640, 640, 3)
    mask: np.ndarray  # (640, 640) bool - valid regions

@dataclass
class SegmentationOutput:
    """Semantic segmentation result"""
    timestamp: float
    class_map: np.ndarray  # (640, 640) int8
    confidence: np.ndarray  # (640, 640) float32

@dataclass
class Detection2D:
    """2D detection in camera view"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    camera_id: int

@dataclass
class Detection3D:
    """3D detection in vehicle frame"""
    bbox_3d: Tuple[float, float, float, float, float, float, float]  # x, y, z, w, h, l, θ
    class_name: str
    confidence: float
    velocity: Tuple[float, float, float]  # vx, vy, vz
    track_id: int

@dataclass
class DriverState:
    """Complete driver state"""
    face_detected: bool
    landmarks: np.ndarray  # (68, 2)
    head_pose: Dict[str, float]  # roll, pitch, yaw
    gaze: Dict[str, Any]  # pitch, yaw, attention_zone
    eye_state: Dict[str, float]  # left_ear, right_ear, perclos
    drowsiness: Dict[str, Any]  # score, yawn_detected, micro_sleep, head_nod
    distraction: Dict[str, Any]  # type, confidence, duration
    readiness_score: float  # 0-100

@dataclass
class Hazard:
    """Identified hazard"""
    object_id: int
    type: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    trajectory: List[Tuple[float, float, float]]
    ttc: float
    zone: str
    base_risk: float  # 0-1

@dataclass
class Risk:
    """Contextual risk assessment"""
    hazard: Hazard
    contextual_score: float  # 0-1
    driver_aware: bool
    urgency: str  # 'low', 'medium', 'high', 'critical'
    intervention_needed: bool

@dataclass
class RiskAssessment:
    """Complete risk assessment output"""
    scene_graph: Dict[str, Any]
    hazards: List[Hazard]
    attention_map: Dict[str, Any]
    top_risks: List[Risk]

@dataclass
class Alert:
    """Generated alert"""
    timestamp: float
    urgency: str  # 'info', 'warning', 'critical'
    modalities: List[str]  # ['visual', 'audio', 'haptic']
    message: str
    hazard_id: int
    dismissed: bool
```

### Module Interfaces

```python
class ICameraManager(ABC):
    """Camera management interface"""
    @abstractmethod
    def start(self) -> None:
        """Start camera capture threads"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop camera capture"""
        pass
    
    @abstractmethod
    def get_frame_bundle(self) -> Optional[CameraBundle]:
        """Get synchronized frame bundle"""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if all cameras are operational"""
        pass

class IBEVGenerator(ABC):
    """BEV generation interface"""
    @abstractmethod
    def generate(self, frames: List[np.ndarray]) -> BEVOutput:
        """Transform camera views to BEV"""
        pass

class ISemanticSegmentor(ABC):
    """Semantic segmentation interface"""
    @abstractmethod
    def segment(self, bev_image: np.ndarray) -> SegmentationOutput:
        """Segment BEV image"""
        pass

class IObjectDetector(ABC):
    """Object detection interface"""
    @abstractmethod
    def detect(self, frames: Dict[int, np.ndarray]) -> Tuple[Dict[int, List[Detection2D]], List[Detection3D]]:
        """Detect and fuse objects from multiple views"""
        pass

class IDMS(ABC):
    """Driver monitoring interface"""
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> DriverState:
        """Analyze driver state"""
        pass

class IContextualIntelligence(ABC):
    """Contextual intelligence interface"""
    @abstractmethod
    def assess(self, detections: List[Detection3D], driver_state: DriverState, 
               bev_seg: SegmentationOutput) -> RiskAssessment:
        """Assess contextual risks"""
        pass

class IAlertSystem(ABC):
    """Alert system interface"""
    @abstractmethod
    def process(self, risks: RiskAssessment, driver: DriverState) -> List[Alert]:
        """Generate and dispatch alerts"""
        pass
```

### Configuration Structure

```yaml
# config/default.yaml
system:
  name: "SENTINEL"
  version: "1.0"
  log_level: "INFO"

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

bev:
  output_size: [640, 640]
  scale: 0.1  # meters per pixel
  vehicle_position: [320, 480]
  blending_method: "multiband"
  blend_width: 50

models:
  segmentation:
    architecture: "BEVFormer-Tiny"
    weights: "models/bev_segmentation.pth"
    device: "cuda"
    precision: "fp16"
    temporal_smoothing: true
    smoothing_alpha: 0.7
  
  detection:
    architecture: "YOLOv8"
    variant: "yolov8m"
    weights: "models/yolov8m_automotive.pt"
    confidence_threshold: 0.5
    nms_threshold: 0.4
  
  dms:
    face_detection: "MediaPipe"
    gaze_model: "L2CS-Net"
    gaze_weights: "models/l2cs_gaze.pth"
    drowsiness_weights: "models/drowsiness_model.pth"
    distraction_weights: "models/distraction_clf.pth"

fusion:
  method: "hungarian"
  iou_threshold_3d: 0.3
  confidence_weighting: true

tracking:
  algorithm: "DeepSORT"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

risk_assessment:
  ttc_calculation:
    method: "constant_velocity"
    safety_margin: 1.5
  
  trajectory_prediction:
    horizon: 3.0
    dt: 0.1
    method: "linear"
  
  zone_mapping:
    num_zones: 8
  
  base_risk_weights:
    ttc: 0.4
    trajectory_conflict: 0.3
    vulnerability: 0.2
    relative_speed: 0.1
  
  thresholds:
    hazard_detection: 0.3
    intervention: 0.7
    critical: 0.9

alerts:
  suppression:
    duplicate_window: 5.0
    max_simultaneous: 2
  
  escalation:
    critical_threshold: 0.9
    high_threshold: 0.7
    medium_threshold: 0.5
  
  modalities:
    visual:
      display_duration: 3.0
      flash_rate: 2
    audio:
      volume: 0.8
      critical_sound: "sounds/alarm.wav"
      warning_sound: "sounds/beep.wav"
    haptic:
      enabled: false

recording:
  enabled: true
  triggers:
    risk_threshold: 0.7
    ttc_threshold: 1.5
  storage_path: "scenarios/"
  max_duration: 30.0

visualization:
  enabled: true
  port: 8080
  update_rate: 30
```

## Data Models

### Coordinate Systems

**Vehicle Coordinate Frame:**
- Origin: Center of rear axle
- X-axis: Forward (front of vehicle)
- Y-axis: Left
- Z-axis: Up

**BEV Coordinate Frame:**
- Origin: Center of image
- X-axis: Right in image
- Y-axis: Up in image (forward in vehicle frame)
- Scale: 0.1 meters per pixel

**Camera Coordinate Frame:**
- Origin: Camera optical center
- Z-axis: Along optical axis (forward)
- X-axis: Right
- Y-axis: Down

### Calibration Data

```yaml
# configs/calibration/front_left.yaml
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

### Scenario Recording Format

```json
{
  "metadata": {
    "timestamp": "2024-11-15T10:30:45.123Z",
    "duration": 15.5,
    "trigger": "high_risk",
    "location": "optional_gps"
  },
  "files": {
    "interior": "interior.mp4",
    "front_left": "front_left.mp4",
    "front_right": "front_right.mp4",
    "bev": "bev.mp4"
  },
  "annotations": "annotations.json"
}
```

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

## Error Handling

### Camera Failures

**Strategy:** Graceful degradation with reduced coverage

```python
class CameraManager:
    def get_frame_bundle(self) -> Optional[CameraBundle]:
        try:
            frames = self._capture_all()
            return self._synchronize(frames)
        except CameraDisconnectError as e:
            self.logger.warning(f"Camera {e.camera_id} disconnected")
            # Continue with remaining cameras
            return self._capture_partial()
        except SyncError as e:
            self.logger.error(f"Sync failed: {e}")
            # Return None, skip this frame
            return None
```

### Model Inference Errors

**Strategy:** Automatic recovery with fallback

```python
class SemanticSegmentor:
    def segment(self, bev_image: np.ndarray) -> SegmentationOutput:
        try:
            return self._inference(bev_image)
        except RuntimeError as e:
            self.logger.error(f"Inference failed: {e}")
            self.error_count += 1
            
            if self.error_count > 3:
                # Reload model
                self._reload_model()
                self.error_count = 0
            
            # Return previous frame result
            return self.last_valid_output
```

### System Crash Recovery

**Strategy:** State persistence and restoration

```python
class SentinelSystem:
    def __init__(self):
        self.state_file = "state/system_state.pkl"
        self._restore_state()
    
    def _restore_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                self.state = pickle.load(f)
            self.logger.info("State restored from checkpoint")
    
    def _save_state(self):
        with open(self.state_file, 'wb') as f:
            pickle.dump(self.state, f)
```

## Testing Strategy

### Unit Testing

**Approach:** Test each module in isolation with mock inputs

```python
# tests/test_bev_generation.py
def test_bev_generation():
    # Given
    generator = BEVGenerator(config)
    frames = load_test_frames()
    
    # When
    output = generator.generate(frames)
    
    # Then
    assert output.image.shape == (640, 640, 3)
    assert output.mask.shape == (640, 640)
    assert np.any(output.mask)  # Some valid regions exist

def test_risk_calculation():
    # Given
    engine = ContextualIntelligence(config)
    hazard = create_test_hazard(ttc=1.5)
    driver = create_test_driver(looking_away=True)
    
    # When
    risk = engine._calculate_contextual_risk(hazard, driver)
    
    # Then
    assert risk.contextual_score > 0.7
    assert not risk.driver_aware
```

### Integration Testing

**Approach:** Test module interactions with real data flow

```python
# tests/test_integration.py
def test_end_to_end_pipeline():
    # Given
    system = SentinelSystem(config)
    test_scenario = load_test_scenario()
    
    # When
    for frame_bundle in test_scenario:
        alerts = system.process(frame_bundle)
    
    # Then
    assert len(alerts) > 0
    assert all(a.urgency in ['info', 'warning', 'critical'] for a in alerts)
```

### Performance Testing

**Approach:** Benchmark latency and throughput

```python
# tests/test_performance.py
def test_latency_requirements():
    system = SentinelSystem(config)
    latencies = []
    
    for _ in range(100):
        start = time.time()
        system.process(get_frame_bundle())
        latencies.append(time.time() - start)
    
    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 0.1  # 100ms requirement
```

### Validation Testing

**Approach:** Evaluate accuracy on labeled datasets

```python
# tests/test_validation.py
def test_segmentation_accuracy():
    segmentor = SemanticSegmentor(config)
    val_dataset = load_validation_dataset()
    
    ious = []
    for image, ground_truth in val_dataset:
        pred = segmentor.segment(image)
        iou = calculate_miou(pred.class_map, ground_truth)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    assert mean_iou >= 0.75  # 75% requirement
```

## Performance Optimization

### GPU Memory Management

- Use FP16 mixed precision for inference
- Batch size = 1 for minimal latency
- Pre-allocate tensors to avoid dynamic allocation
- Clear cache periodically

### CPU Optimization

- Multi-threaded camera capture
- Parallel processing of independent modules (DMS + Perception)
- Use numpy vectorized operations
- Minimize Python loops in hot paths

### Latency Budget Allocation

| Module | Target Latency | Strategy |
|--------|---------------|----------|
| Camera Capture | 5ms | Hardware buffering |
| BEV Generation | 15ms | GPU-accelerated warping |
| Semantic Seg | 15ms | TensorRT optimization |
| Object Detection | 20ms/cam | Parallel inference |
| DMS | 25ms | Model quantization |
| Risk Assessment | 10ms | Vectorized numpy |
| Alert Generation | 5ms | Simple logic |
| **Total** | **95ms** | **Buffer: 5ms** |

## Deployment Considerations

### Hardware Setup

- Mount cameras with stable fixtures
- Ensure USB 3.0 connections for bandwidth
- GPU with compute capability ≥ 7.0
- SSD for fast model loading

### Calibration Procedure

1. Print calibration checkerboard pattern
2. Run calibration script: `python scripts/calibrate.py --cameras 3`
3. Capture 20+ images per camera at various angles
4. Script computes intrinsics, extrinsics, homography
5. Save calibration files to `configs/calibration/`

### Model Deployment

- Convert PyTorch models to ONNX for portability
- Optimize with TensorRT for NVIDIA GPUs
- Store models in `models/` directory
- Download script: `python scripts/download_models.py`

### Docker Deployment

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "main.py", "--config", "configs/default.yaml"]
```

Build and run:
```bash
docker build -t sentinel:latest .
docker run --gpus all -v /dev/video0:/dev/video0 sentinel:latest
```

## Security and Privacy

### Data Privacy

- All processing happens locally (no cloud transmission)
- Driver video stored only during triggered scenarios
- Encryption at rest for recorded scenarios
- Configurable data retention policy

### Access Control

- Dashboard requires authentication
- API endpoints protected with tokens
- Configuration files have restricted permissions

## PyQt6 GUI Architecture

### Main Window Structure

```
SENTINELMainWindow
├── Menu Bar (File, System, View, Tools, Analytics, Help)
├── Tool Bar (Quick actions: Start, Stop, Record, Screenshot)
├── Central Widget: LiveMonitorWidget
│   ├── Camera Grid (2x2)
│   │   ├── Interior Camera Display
│   │   ├── Front Left Camera Display
│   │   ├── Front Right Camera Display
│   │   └── BEV Canvas (interactive)
│   └── Status Panels (horizontal layout)
│       ├── Driver State Panel
│       ├── Risk Assessment Panel
│       └── Alerts Panel
├── Dock Widgets (dockable/floating)
│   ├── Performance Monitor (FPS, latency, resources)
│   ├── Scenarios Browser (recorded events)
│   ├── Configuration Editor (live tuning)
│   ├── Map View (HD map overlay)
│   └── Analytics Dashboard (trip statistics)
└── Status Bar (system status, connection indicators)
```

### GUI Components

**1. LiveMonitorWidget**
- 2x2 grid of VideoDisplayWidget for camera feeds
- Interactive BEVCanvas with QGraphicsScene
- Real-time updates at 30 FPS using QTimer
- GPU-accelerated rendering with QOpenGLWidget

**2. DriverStatePanel**
- CircularGaugeWidget for readiness score
- GazeDirectionWidget with 3D head visualization
- Metrics grid with labeled values
- StatusIndicator widgets with color coding
- PyQtGraph for trend charts

**3. RiskAssessmentPanel**
- CircularGaugeWidget for overall risk
- QListWidget with custom HazardListItem widgets
- ZoneRiskRadarChart (custom QPainter rendering)
- TTCDisplayWidget with countdown timer

**4. AlertsPanel**
- QTextEdit with HTML formatting for alert history
- QSoundEffect for audio alerts
- Alert statistics display
- Export and filtering controls

**5. Performance Dock Widget**
- PyQtGraph PlotWidget for FPS and latency
- ModuleBreakdownWidget (stacked bar chart)
- ResourceUsageWidget (GPU/CPU gauges)

**6. Scenarios Dock Widget**
- QListWidget with scenario thumbnails
- Search and filter controls
- Playback controls (play, pause, step, speed)
- Export and delete actions

**7. Configuration Dock Widget**
- QTabWidget with category tabs
- LabeledSlider widgets for parameters
- Save/Reset buttons
- Real-time parameter updates

### Threading Model

```python
Main Thread (GUI)
├── QTimer (30 Hz) → Update displays
├── Worker Thread 1 → SENTINEL system processing
├── Worker Thread 2 → Video encoding for recording
└── Worker Thread 3 → Cloud sync operations
```

### Signal/Slot Communication

```python
# Worker thread signals
class SentinelWorker(QThread):
    frame_ready = pyqtSignal(dict)  # Camera frames
    bev_ready = pyqtSignal(np.ndarray)  # BEV image
    detections_ready = pyqtSignal(list)  # 3D detections
    driver_state_ready = pyqtSignal(object)  # DriverState
    risks_ready = pyqtSignal(object)  # RiskAssessment
    alerts_ready = pyqtSignal(list)  # Alerts
    performance_ready = pyqtSignal(dict)  # Metrics
```

## Advanced Features Architecture

### Trajectory Prediction Module

```python
class TrajectoryPredictor:
    """Multi-hypothesis trajectory prediction"""
    
    Components:
    - LSTM Model: Learned prediction from historical data
    - Physics Models: CV, CA, CT models for different scenarios
    - Ensemble: Weighted combination of predictions
    - Uncertainty: Covariance estimation for confidence bounds
    - Collision Probability: Mahalanobis distance calculation
```

**Data Flow:**
1. Object detection history (last 30 frames)
2. Scene context (nearby objects, road geometry)
3. LSTM inference → learned trajectories
4. Physics models → analytical trajectories
5. Ensemble → ranked hypotheses (top 3)
6. Uncertainty quantification → confidence bounds
7. Collision probability → risk assessment input

### Driver Behavior Profiling Module

```python
class DriverBehaviorProfiler:
    """Learn and adapt to individual drivers"""
    
    Components:
    - Face Recognition: Identify driver from face embedding
    - Metrics Tracking: Reaction time, following distance, lane changes
    - Style Classification: Aggressive/Normal/Cautious
    - Threshold Adaptation: Personalized alert thresholds
    - Report Generation: Safety scores and recommendations
```

**Profile Storage:**
```json
{
  "driver_id": "hash_of_face_embedding",
  "total_distance": 1500.5,
  "total_time": 45.2,
  "driving_style": "cautious",
  "metrics": {
    "reaction_time": [0.8, 0.7, 0.9],
    "following_distance": [25.0, 28.0, 30.0],
    "risk_tolerance": 0.3
  },
  "safety_score": 85,
  "last_updated": "2024-11-16T10:30:00Z"
}
```

### HD Map Manager Module

```python
class HDMapManager:
    """High-definition map integration"""
    
    Components:
    - Map Parser: OpenDRIVE/Lanelet2 format support
    - Localization: GPS + visual odometry + map matching
    - Feature Query: Upcoming signs, lights, intersections
    - Path Prediction: A* routing through lane graph
    - BEV Overlay: Render map features on BEV
```

**Map Layers:**
- Lane geometry (centerlines, boundaries)
- Traffic signs (stop, yield, speed limit)
- Traffic lights (position, state)
- Crosswalks and pedestrian zones
- Road boundaries and curbs
- Speed limits per lane

### CAN Bus Interface Module

```python
class CANBusInterface:
    """Vehicle CAN bus integration"""
    
    Components:
    - SocketCAN Connection: Linux CAN interface
    - DBC Parser: Message decoding from DBC file
    - Telemetry Reader: Speed, steering, brake, throttle
    - Command Sender: Intervention commands (if enabled)
    - Logger: Record all CAN traffic
```

**Telemetry Integration:**
- Vehicle speed → TTC calculation
- Steering angle → trajectory prediction
- Brake pressure → intervention detection
- Turn signals → path prediction
- Gear position → context awareness

### Cloud Sync Module

```python
class FleetManager:
    """Cloud synchronization and fleet management"""
    
    Components:
    - API Client: REST API for cloud backend
    - Trip Uploader: Anonymized trip summaries
    - Scenario Uploader: High-risk events (with consent)
    - Model Downloader: OTA model updates
    - Profile Sync: Driver profiles across vehicles
```

**API Endpoints:**
- POST /trips → Upload trip summary
- POST /scenarios → Upload scenario
- GET /models/{name}/latest → Download model
- GET /fleet/{id}/stats → Fleet statistics
- PUT /drivers/{id}/profile → Sync profile

## Updated Data Structures

```python
@dataclass
class Trajectory:
    """Predicted trajectory with uncertainty"""
    points: List[Tuple[float, float, float]]  # (x, y, z) positions
    timestamps: List[float]  # Time for each point
    uncertainty: List[np.ndarray]  # Covariance matrices
    confidence: float  # Overall confidence score
    model: str  # 'lstm', 'cv', 'ca', 'ct'

@dataclass
class DriverProfile:
    """Individual driver behavior profile"""
    driver_id: str
    face_embedding: np.ndarray
    total_distance: float
    total_time: float
    driving_style: str  # 'aggressive', 'normal', 'cautious'
    metrics: Dict[str, List[float]]
    safety_score: float
    attention_score: float
    eco_score: float
    last_updated: datetime

@dataclass
class MapFeature:
    """HD map feature"""
    feature_id: str
    type: str  # 'lane', 'sign', 'light', 'crosswalk'
    position: Tuple[float, float, float]
    attributes: Dict[str, Any]
    geometry: List[Tuple[float, float]]  # Polyline

@dataclass
class VehicleTelemetry:
    """CAN bus telemetry"""
    timestamp: float
    speed: float  # m/s
    steering_angle: float  # radians
    brake_pressure: float  # bar
    throttle_position: float  # 0-1
    gear: int
    turn_signal: str  # 'left', 'right', 'none'
```

## Configuration Extensions

```yaml
# GUI configuration
gui:
  theme: "dark"  # 'dark' or 'light'
  accent_color: "#00aaff"
  window_geometry: [100, 100, 1920, 1080]
  update_rate: 30  # Hz
  enable_animations: true

# Trajectory prediction
trajectory_prediction:
  enabled: true
  horizon: 5.0  # seconds
  num_hypotheses: 3
  lstm_model: "models/trajectory_lstm.pth"
  uncertainty_estimation: true

# Driver profiling
driver_profiling:
  enabled: true
  face_recognition_threshold: 0.6
  profile_storage: "profiles/"
  auto_adapt_thresholds: true

# HD map
hd_map:
  enabled: false
  map_file: "maps/city_map.xodr"
  format: "opendrive"  # 'opendrive' or 'lanelet2'
  localization_method: "gps_visual"

# CAN bus
can_bus:
  enabled: false
  interface: "socketcan"
  channel: "can0"
  dbc_file: "configs/vehicle.dbc"
  enable_control: false

# Cloud sync
cloud:
  enabled: false
  api_url: "https://api.sentinel-fleet.com"
  api_key: "your_api_key"
  vehicle_id: "vehicle_001"
  fleet_id: "fleet_alpha"
  sync_interval: 300  # seconds
  upload_scenarios: true
```

## Future Enhancements

### Phase 2 (Completed in this spec)
- ✓ CAN bus integration for vehicle telemetry
- ✓ HD map integration with lane-level localization
- ✓ Cloud backend for fleet analytics
- ✓ LSTM-based trajectory prediction
- ✓ Driver behavior profiling

### Phase 3 (Future)
- Depth cameras for improved 3D perception
- Radar/LiDAR sensor fusion
- Mobile app for configuration
- V2V communication

### Phase 4 (Production)
- ASIL-B safety certification
- Edge TPU optimization
- 5G connectivity
- ADAS Level 2+ integration
