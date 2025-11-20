# Project Structure

## Directory Organization

```
sentinel/
├── src/                          # Source code
│   ├── camera/                   # Camera management
│   │   ├── capture.py           # Multi-threaded camera capture
│   │   ├── sync.py              # Timestamp synchronization
│   │   ├── calibration.py       # Calibration loading
│   │   └── manager.py           # CameraManager implementation
│   ├── perception/               # Perception pipeline
│   │   ├── bev/                 # Bird's eye view generation
│   │   │   ├── transformer.py   # Perspective transformation
│   │   │   ├── stitcher.py      # Multi-view stitching
│   │   │   └── generator.py     # BEVGenerator implementation
│   │   ├── segmentation/        # Semantic segmentation
│   │   │   ├── model.py         # Model wrapper
│   │   │   ├── smoother.py      # Temporal smoothing
│   │   │   └── segmentor.py     # SemanticSegmentor implementation
│   │   └── detection/           # Object detection
│   │       ├── detector_2d.py   # 2D detection per camera
│   │       ├── estimator_3d.py  # 3D bounding box estimation
│   │       ├── fusion.py        # Multi-view fusion
│   │       ├── tracker.py       # DeepSORT tracking
│   │       └── detector.py      # ObjectDetector implementation
│   ├── dms/                      # Driver monitoring system
│   │   ├── face.py              # Face detection & landmarks
│   │   ├── gaze.py              # Gaze estimation
│   │   ├── pose.py              # Head pose calculation
│   │   ├── drowsiness.py        # Drowsiness detection
│   │   ├── distraction.py       # Distraction classification
│   │   └── monitor.py           # DMS implementation
│   ├── intelligence/             # Contextual intelligence
│   │   ├── scene_graph.py       # Scene graph builder
│   │   ├── attention.py         # Attention mapping
│   │   ├── ttc.py               # Time-to-collision
│   │   ├── trajectory.py        # Trajectory prediction (basic)
│   │   ├── risk.py              # Risk assessment
│   │   └── engine.py            # ContextualIntelligence implementation
│   ├── advanced/                 # Advanced features
│   │   ├── trajectory_predictor.py  # LSTM + physics trajectory prediction
│   │   ├── behavior_profiler.py     # Driver behavior profiling
│   │   ├── hd_map_manager.py        # HD map integration
│   │   ├── can_interface.py         # CAN bus interface
│   │   └── fleet_manager.py         # Cloud synchronization
│   ├── alerts/                   # Alert system
│   │   ├── generator.py         # Alert generation logic
│   │   ├── suppression.py       # Alert suppression
│   │   └── system.py            # AlertSystem implementation
│   ├── recording/                # Scenario recording
│   │   ├── trigger.py           # Recording triggers
│   │   ├── recorder.py          # Frame recording
│   │   └── exporter.py          # Scenario export
│   ├── visualization/            # Dashboard (web-based)
│   │   ├── backend/             # FastAPI server
│   │   └── frontend/            # React application
│   ├── gui/                      # PyQt6 GUI application
│   │   ├── main_window.py       # Main application window
│   │   ├── widgets/             # Custom widgets
│   │   │   ├── live_monitor.py  # Central monitoring widget
│   │   │   ├── driver_state_panel.py
│   │   │   ├── risk_panel.py
│   │   │   ├── alerts_panel.py
│   │   │   ├── bev_canvas.py    # Interactive BEV display
│   │   │   ├── gauges.py        # Circular gauges, bars
│   │   │   └── charts.py        # PyQtGraph charts
│   │   ├── dock_widgets/        # Dockable panels
│   │   │   ├── performance.py   # Performance monitoring
│   │   │   ├── scenarios.py     # Scenario browser
│   │   │   ├── configuration.py # Configuration editor
│   │   │   ├── map_view.py      # HD map display
│   │   │   └── analytics.py     # Analytics dashboard
│   │   ├── dialogs/             # Modal dialogs
│   │   │   ├── settings_dialog.py
│   │   │   ├── calibration_dialog.py
│   │   │   ├── scenario_replay_dialog.py
│   │   │   └── alert_builder_dialog.py
│   │   ├── workers/             # Background threads
│   │   │   ├── sentinel_worker.py
│   │   │   └── cloud_worker.py
│   │   ├── themes/              # Style sheets
│   │   │   ├── dark.qss
│   │   │   └── light.qss
│   │   └── resources/           # Icons, images, fonts
│   │       ├── icons/
│   │       └── fonts/
│   ├── core/                     # Core infrastructure
│   │   ├── config.py            # Configuration management
│   │   ├── logging.py           # Logging setup
│   │   ├── data_structures.py   # Dataclasses
│   │   └── interfaces.py        # Abstract base classes
│   ├── main.py                   # System orchestration (headless)
│   └── gui_main.py               # GUI application entry point
├── configs/                      # Configuration files
│   ├── default.yaml             # Default system config
│   └── calibration/             # Camera calibration data
│       ├── interior.yaml
│       ├── front_left.yaml
│       └── front_right.yaml
├── models/                       # Pretrained models
│   ├── bev_segmentation.pth
│   ├── yolov8m_automotive.pt
│   ├── l2cs_gaze.pth
│   ├── drowsiness_model.pth
│   └── distraction_clf.pth
├── scenarios/                    # Recorded scenarios
│   └── {timestamp}/
│       ├── metadata.json
│       ├── annotations.json
│       └── *.mp4
├── tests/                        # Test suite
│   ├── test_camera.py
│   ├── test_bev.py
│   ├── test_segmentation.py
│   ├── test_detection.py
│   ├── test_dms.py
│   ├── test_intelligence.py
│   ├── test_alerts.py
│   ├── test_integration.py
│   ├── test_performance.py
│   └── test_validation.py
├── scripts/                      # Utility scripts
│   ├── calibrate.py             # Camera calibration
│   └── download_models.py       # Model download
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker configuration
└── README.md                     # Project documentation
```

## Module Architecture

### Data Flow
1. **Camera Management** → Synchronized frame bundles
2. **Perception Pipeline** (parallel):
   - BEV Generation → Semantic Segmentation
   - Multi-View Detection → Object Tracking
3. **Driver Monitoring** (parallel) → Driver state
4. **Contextual Intelligence** → Risk assessment
5. **Alert System** → Multi-modal alerts
6. **Recording & Visualization** → Storage and display

### Key Interfaces

All modules implement abstract base classes defined in `src/core/interfaces.py`:
- `ICameraManager`: Camera capture and synchronization
- `IBEVGenerator`: BEV transformation
- `ISemanticSegmentor`: Pixel classification
- `IObjectDetector`: 3D object detection and tracking
- `IDMS`: Driver state monitoring
- `IContextualIntelligence`: Risk assessment
- `IAlertSystem`: Alert generation and dispatch

### Configuration-Driven Design

All system parameters are externalized to YAML files:
- Camera settings (device indices, resolution, FPS)
- Model paths and inference parameters
- Risk assessment thresholds and weights
- Alert urgency levels and modalities

No hardcoded parameters in source code.

## Coordinate Systems

- **Vehicle Frame**: Origin at rear axle center, X forward, Y left, Z up
- **BEV Frame**: Top-down view, 640x640 pixels, 0.1m per pixel
- **Camera Frame**: Standard pinhole model with calibrated intrinsics/extrinsics
