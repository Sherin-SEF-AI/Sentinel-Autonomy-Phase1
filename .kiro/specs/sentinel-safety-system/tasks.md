# Implementation Plan

- [x] 1. Project setup and core infrastructure
  - Create project directory structure (src/, configs/, models/, tests/, scripts/)
  - Set up Python package with __init__.py files
  - Create requirements.txt with all dependencies
  - Implement ConfigManager class for YAML configuration loading
  - Create logging infrastructure with configurable levels
  - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 2. Define core data structures and interfaces
  - [x] 2.1 Implement data classes
    - Create dataclasses for CameraBundle, BEVOutput, SegmentationOutput
    - Create dataclasses for Detection2D, Detection3D
    - Create dataclasses for DriverState with all sub-dictionaries
    - Create dataclasses for Hazard, Risk, RiskAssessment
    - Create dataclasses for Alert
    - _Requirements: 1.2, 2.3, 3.2, 4.2, 5.6, 6.1, 7.1_
  
  - [x] 2.2 Define module interfaces
    - Create abstract base classes: ICameraManager, IBEVGenerator, ISemanticSegmentor
    - Create abstract base classes: IObjectDetector, IDMS, IContextualIntelligence, IAlertSystem
    - Define method signatures for all interfaces
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1_

- [x] 3. Implement Camera Management Module
  - [x] 3.1 Create CameraCapture class
    - Implement multi-threaded camera capture from USB devices
    - Add camera initialization with resolution and FPS configuration
    - Implement frame buffering with thread-safe circular buffer
    - Add camera health monitoring and disconnection detection
    - _Requirements: 1.1, 1.3, 1.4, 11.2_
  
  - [x] 3.2 Implement timestamp synchronization
    - Create TimestampSync class for software-based alignment
    - Implement synchronization logic with ±5ms tolerance
    - Add frame dropping for out-of-sync frames
    - _Requirements: 1.2_
  
  - [x] 3.3 Add calibration loading
    - Create CalibrationLoader class to read YAML calibration files
    - Load intrinsic parameters (fx, fy, cx, cy, distortion)
    - Load extrinsic parameters (translation, rotation)
    - Load homography matrices for BEV transformation
    - _Requirements: 1.5, 12.1_
  
  - [x] 3.4 Implement CameraManager class
    - Integrate CameraCapture, TimestampSync, and CalibrationLoader
    - Implement ICameraManager interface methods (start, stop, get_frame_bundle, is_healthy)
    - Add automatic reconnection logic for disconnected cameras
    - Implement graceful degradation when one camera fails
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 11.2_

- [x] 4. Implement BEV Generation Module
  - [x] 4.1 Create PerspectiveTransformer class
    - Implement inverse perspective mapping using homography matrices
    - Apply camera undistortion using intrinsic parameters
    - Warp frames to BEV coordinate system
    - _Requirements: 2.1, 2.4_
  
  - [x] 4.2 Implement ViewStitcher class
    - Identify overlapping regions between multiple BEV views
    - Implement multi-band blending using Laplacian pyramid
    - Composite final BEV image at 640x640 resolution
    - _Requirements: 2.2, 2.3_
  
  - [x] 4.3 Create MaskGenerator class
    - Generate valid region masks excluding vehicle body and sky
    - Apply masks to BEV output
    - _Requirements: 2.4_
  
  - [x] 4.4 Implement BEVGenerator class
    - Integrate PerspectiveTransformer, ViewStitcher, and MaskGenerator
    - Implement IBEVGenerator interface
    - Optimize for 15ms processing time target
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 10.1_

- [x] 5. Implement Semantic Segmentation Module
  - [x] 5.1 Create model wrapper for BEV segmentation
    - Load pretrained BEVFormer-Tiny or similar model
    - Implement model inference with FP16 precision
    - Add GPU memory management
    - _Requirements: 3.1, 10.3_
  
  - [x] 5.2 Implement temporal smoothing
    - Create TemporalSmoother class with exponential moving average
    - Apply smoothing with alpha=0.7 to reduce flicker
    - _Requirements: 3.3_
  
  - [x] 5.3 Implement SemanticSegmentor class
    - Integrate model wrapper and temporal smoother
    - Implement ISemanticSegmentor interface
    - Output class map (640x640 int8) and confidence map (640x640 float32)
    - Optimize for 15ms inference time
    - Add error recovery for inference failures
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 10.1, 11.3_
  
  - [x] 5.4 Validate segmentation accuracy
    - Load validation dataset
    - Calculate mIoU metric
    - Verify mIoU ≥ 75%
    - _Requirements: 3.1_

- [x] 6. Implement Multi-View Object Detection Module
  - [x] 6.1 Create 2D object detector
    - Load YOLOv8 model with automotive fine-tuning
    - Implement per-camera detection with confidence and NMS thresholds
    - Output Detection2D objects with bbox, class, confidence
    - Optimize for 20ms per camera processing time
    - _Requirements: 4.1, 10.1_
  
  - [x] 6.2 Implement 3D bounding box estimation
    - Create 3D projection from 2D detections using camera extrinsics
    - Estimate 3D bbox (x, y, z, w, h, l, θ) in vehicle coordinate frame
    - _Requirements: 4.2_
  
  - [x] 6.3 Create multi-view fusion
    - Implement Hungarian algorithm for detection association
    - Fuse detections from multiple cameras with IoU threshold 0.3
    - Apply confidence weighting
    - Output unified Detection3D list
    - _Requirements: 4.3_
  
  - [x] 6.4 Implement object tracking
    - Integrate DeepSORT tracker
    - Maintain consistent track IDs for minimum 30 frames
    - Configure max_age=30, min_hits=3, iou_threshold=0.3
    - _Requirements: 4.4_
  
  - [x] 6.5 Add velocity estimation
    - Calculate velocity vectors from position changes across frames
    - Apply Kalman filtering for smoothing
    - _Requirements: 4.5_
  
  - [x] 6.6 Implement ObjectDetector class
    - Integrate 2D detector, 3D estimation, fusion, tracking, and velocity estimation
    - Implement IObjectDetector interface
    - Add error recovery for inference failures
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 11.3_

- [x] 7. Implement Driver Monitoring System (DMS)
  - [x] 7.1 Create face detection and landmark extraction
    - Integrate MediaPipe Face Detection and Face Mesh
    - Extract 68+ facial landmarks
    - _Requirements: 5.1_
  
  - [x] 7.2 Implement gaze estimation
    - Load L2CS-Net model for gaze estimation
    - Estimate gaze pitch and yaw angles
    - Map gaze to attention zones (8 zones)
    - Validate gaze error < 5 degrees
    - _Requirements: 5.2_
  
  - [x] 7.3 Implement head pose calculation
    - Calculate head pose (roll, pitch, yaw) from facial landmarks
    - Use PnP algorithm with 3D face model
    - _Requirements: 5.3_
  
  - [x] 7.4 Create drowsiness detection
    - Calculate Eye Aspect Ratio (EAR) for both eyes
    - Compute PERCLOS over 60-frame window (2 seconds at 30fps)
    - Detect PERCLOS > 80% for 3+ seconds
    - Implement yawn detection with frequency tracking (>3 per minute)
    - Detect head nodding events (>2 per minute)
    - Detect micro-sleep events (eyes closed >2 seconds)
    - _Requirements: 5.4_
  
  - [x] 7.5 Implement distraction classification
    - Load MobileNetV3 distraction classifier
    - Classify distraction types (phone, passenger, controls, eyes off road, hands off wheel)
    - Track distraction duration
    - Flag distraction when eyes off road >2 seconds
    - _Requirements: 5.5_
  
  - [x] 7.6 Calculate driver readiness score
    - Implement weighted sum: alertness (0.4) + attention (0.3) + distraction (0.3)
    - Output readiness score 0-100
    - _Requirements: 5.6_
  
  - [x] 7.7 Implement DMS class
    - Integrate all DMS components
    - Implement IDMS interface
    - Optimize for 25ms processing time
    - Output complete DriverState dataclass
    - Add error recovery
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 10.1, 11.3_

- [x] 8. Implement Contextual Intelligence Engine
  - [x] 8.1 Create scene graph builder
    - Build spatial representation of all detected objects
    - Store object relationships and spatial proximity
    - _Requirements: 6.1_
  
  - [x] 8.2 Implement attention mapping
    - Map driver gaze to 8 spatial zones around vehicle
    - Define zone boundaries (front: -30° to 30°, front-left: 30° to 75°, etc.)
    - Determine which zones driver is attending to
    - _Requirements: 6.2_
  
  - [x] 8.3 Create TTC calculator
    - Implement constant velocity TTC calculation
    - Add safety margin of 1.5 meters
    - Calculate TTC for each detected object
    - _Requirements: 6.3_
  
  - [x] 8.4 Implement trajectory prediction
    - Predict object trajectories 3 seconds ahead
    - Use linear prediction with 0.1 second time steps
    - Generate trajectory waypoints
    - _Requirements: 6.4_
  
  - [x] 8.5 Create base risk calculator
    - Implement weighted risk: TTC (0.4) + trajectory conflict (0.3) + vulnerability (0.2) + relative speed (0.1)
    - Assign vulnerability scores (pedestrian > cyclist > vehicle)
    - Output base risk score 0-1
    - _Requirements: 6.5_
  
  - [x] 8.6 Implement contextual risk assessment
    - Calculate awareness penalty (2.0 if not looking, 1.0 if looking)
    - Calculate capacity factor (2.0 - readiness/100)
    - Compute contextual risk = base_risk × awareness_penalty × capacity_factor
    - Categorize urgency levels (low, medium, high, critical)
    - _Requirements: 6.5_
  
  - [x] 8.7 Create risk prioritization
    - Sort risks by contextual score
    - Select top 3 threats
    - Detect attention-risk mismatches
    - _Requirements: 6.6, 6.7_
  
  - [x] 8.8 Implement ContextualIntelligence class
    - Integrate all risk assessment components
    - Implement IContextualIntelligence interface
    - Optimize for 10ms processing time
    - Output RiskAssessment dataclass
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 10.1_

- [x] 9. Implement Alert & Action System
  - [x] 9.1 Create alert generation logic
    - Implement urgency-based alert generation (INFO, WARNING, CRITICAL)
    - Generate CRITICAL alerts for risk > 0.9 with visual flash, audio alarm, haptic pulse
    - Generate WARNING alerts for risk > 0.7 when driver unaware
    - Generate INFO alerts for risk > 0.5 when cognitive load < 0.7
    - Adapt alert timing based on driver cognitive load
    - _Requirements: 7.1, 7.3, 7.4_
  
  - [x] 9.2 Implement alert suppression
    - Track alert history per hazard
    - Suppress duplicate alerts within 5 second window
    - Limit to maximum 2 simultaneous alerts
    - _Requirements: 7.5_
  
  - [x] 9.3 Add alert logging
    - Log all alerts with timestamp and context
    - Store alert history for analysis
    - _Requirements: 7.6_
  
  - [x] 9.4 Create multi-modal dispatch
    - Implement visual alert rendering
    - Implement audio alert playback
    - Add placeholder for haptic feedback
    - _Requirements: 7.2_
  
  - [x] 9.5 Implement AlertSystem class
    - Integrate alert generation, suppression, logging, and dispatch
    - Implement IAlertSystem interface
    - Output list of Alert objects
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 10. Implement Data Recording Module
  - [x] 10.1 Create recording trigger logic
    - Trigger on risk score > 0.7
    - Trigger on driver distraction during hazard
    - Trigger on system intervention
    - Trigger on TTC < 1.5 seconds
    - _Requirements: 8.1_
  
  - [x] 10.2 Implement frame recording
    - Record all camera frames (interior, front_left, front_right)
    - Record BEV output
    - Record object detections
    - Record driver state
    - Record risk scores
    - _Requirements: 8.2_
  
  - [x] 10.3 Create scenario export
    - Export camera feeds as MP4 videos
    - Export annotations as JSON
    - Create metadata JSON with timestamp, duration, trigger
    - Organize in scenarios/{timestamp}/ directory structure
    - _Requirements: 8.4_
  
  - [x] 10.4 Implement playback support
    - Load recorded scenarios from disk
    - Support frame-by-frame navigation
    - _Requirements: 8.3_
  
  - [x] 10.5 Implement ScenarioRecorder class
    - Integrate trigger logic, recording, export, and playback
    - Implement recording API (start_recording, stop_recording, save_frame, export_scenario)
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 11. Implement Visualization Dashboard
  - [x] 11.1 Create FastAPI backend
    - Set up FastAPI application
    - Create WebSocket endpoint for real-time data streaming
    - Implement REST endpoints for configuration and playback
    - _Requirements: 9.1, 9.2_
  
  - [x] 11.2 Implement real-time data streaming
    - Stream BEV images with semantic overlay
    - Stream 3D bounding boxes
    - Stream driver state metrics
    - Stream risk scores
    - Stream system performance metrics (FPS, latency)
    - Update at 30 Hz
    - _Requirements: 9.1_
  
  - [x] 11.3 Create web frontend
    - Build React application with Three.js for 3D visualization
    - Implement live view with BEV, camera feeds, DMS output
    - Create multi-view layout
    - Display driver attention heatmap
    - Display risk panel with top-3 hazards
    - Display performance graphs
    - _Requirements: 9.1, 9.2_
  
  - [x] 11.4 Add playback interface
    - Create scenario browser
    - Implement frame-by-frame scrubbing controls
    - Add annotation overlay toggle
    - _Requirements: 9.3_

- [x] 12. Implement main system orchestration
  - [x] 12.1 Create SentinelSystem main class
    - Initialize all modules (CameraManager, BEVGenerator, SemanticSegmentor, ObjectDetector, DMS, ContextualIntelligence, AlertSystem, ScenarioRecorder)
    - Load configuration from YAML
    - Set up logging
    - _Requirements: 12.1, 12.2, 12.3, 12.4_
  
  - [x] 12.2 Implement main processing loop
    - Get synchronized camera bundle
    - Process DMS pipeline in parallel with perception pipeline
    - Generate BEV from external cameras
    - Run semantic segmentation on BEV
    - Detect and track objects
    - Assess contextual risks
    - Generate and dispatch alerts
    - Record scenarios when triggered
    - Stream data to visualization dashboard
    - _Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1_
  
  - [x] 12.3 Add performance monitoring
    - Track FPS and latency per module
    - Monitor GPU memory usage
    - Monitor CPU usage
    - Log performance metrics
    - _Requirements: 9.1, 10.1, 10.2, 10.3, 10.4_
  
  - [x] 12.4 Implement graceful shutdown
    - Stop all camera capture threads
    - Save system state
    - Close all resources
    - _Requirements: 11.4_
  
  - [x] 12.5 Add state persistence and recovery
    - Save system state periodically
    - Restore state on startup after crash
    - Recover within 2 seconds
    - _Requirements: 11.4_

- [x] 13. Create calibration tooling
  - [x] 13.1 Implement camera calibration script
    - Create calibration script that captures checkerboard images
    - Compute camera intrinsics using OpenCV
    - Compute camera extrinsics relative to vehicle frame
    - Compute homography matrices for BEV transformation
    - Save calibration data to YAML files
    - _Requirements: 1.5, 12.1_
  
  - [x] 13.2 Add calibration validation
    - Visualize undistorted images
    - Visualize BEV transformation
    - Verify calibration quality
    - _Requirements: 1.5_

- [x] 14. Create deployment scripts
  - [x] 14.1 Create model download script
    - Download pretrained models (BEV segmentation, YOLOv8, L2CS-Net, drowsiness, distraction)
    - Verify model checksums
    - Place models in models/ directory
    - _Requirements: 3.1, 4.1, 5.2_
  
  - [x] 14.2 Create Docker configuration
    - Write Dockerfile with CUDA base image
    - Install dependencies
    - Configure GPU access
    - Set up entry point
    - _Requirements: 12.1_
  
  - [x] 14.3 Create installation documentation
    - Document hardware requirements
    - Document software dependencies
    - Provide installation steps
    - Include troubleshooting guide
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

- [x] 15. System integration and validation
  - [x] 15.1 Perform end-to-end integration testing
    - Test complete pipeline with real camera feeds
    - Verify data flow between all modules
    - Check timing constraints
    - _Requirements: 10.1, 10.2_
  
  - [x] 15.2 Validate performance requirements
    - Measure end-to-end latency (target: <100ms at p95)
    - Measure throughput (target: ≥30 FPS)
    - Measure GPU memory usage (target: ≤8GB)
    - Measure CPU usage (target: ≤60% on 8-core)
    - _Requirements: 10.1, 10.2, 10.3, 10.4_
  
  - [x] 15.3 Validate reliability requirements
    - Test camera disconnection and reconnection
    - Test graceful degradation with one camera failure
    - Test automatic recovery from inference errors
    - Test crash recovery and state restoration
    - _Requirements: 11.1, 11.2, 11.3, 11.4_
  
  - [x] 15.4 Validate accuracy requirements
    - Measure BEV segmentation mIoU (target: ≥75%)
    - Measure object detection mAP (target: ≥80%)
    - Measure gaze estimation error (target: <5°)
    - Measure risk prediction accuracy (target: ≥85%)
    - _Requirements: 3.1, 4.1, 5.2, 6.5_

- [x] 16. Implement PyQt6 GUI foundation
  - [x] 16.1 Create main window structure
    - Implement SENTINELMainWindow class with QMainWindow
    - Create menu bar with File, System, View, Tools, Analytics, Help menus
    - Add toolbar with quick action buttons (Start, Stop, Record, Screenshot)
    - Implement status bar with system status indicators
    - Add keyboard shortcuts (F5 start, F6 stop, F11 fullscreen, Ctrl+Q quit)
    - _Requirements: 13.1, 13.4, 13.5_
  
  - [x] 16.2 Implement central monitoring widget
    - Create LiveMonitorWidget with 2x2 camera grid layout
    - Implement VideoDisplayWidget for camera feed display
    - Add QTimer for 30 FPS updates
    - Convert numpy arrays to QPixmap for display
    - Implement responsive layout with QSplitter
    - _Requirements: 13.2, 13.3_
  
  - [x] 16.3 Create theme system
    - Design dark theme QSS stylesheet
    - Design light theme QSS stylesheet
    - Implement theme switching functionality
    - Add configurable accent colors
    - Apply modern styling to all widgets
    - _Requirements: 13.7_
  
  - [x] 16.4 Implement window state persistence
    - Save window geometry on close
    - Save dock widget positions and visibility
    - Save user preferences (theme, shortcuts)
    - Restore state on application startup
    - _Requirements: 13.5_
  
  - [x] 16.5 Add multi-monitor support
    - Detect available monitors
    - Allow window dragging across monitors
    - Support floating dock widgets on secondary monitors
    - Persist monitor-specific layouts
    - _Requirements: 13.6_

- [x] 17. Implement interactive BEV canvas
  - [x] 17.1 Create BEVCanvas widget
    - Implement QGraphicsScene for BEV rendering
    - Add QGraphicsView with zoom and pan controls
    - Display BEV image as background pixmap
    - Implement mouse wheel zoom and drag pan
    - _Requirements: 14.1, 14.6_
  
  - [x] 17.2 Add object overlays
    - Draw 3D bounding boxes on detected objects
    - Color code boxes by object class
    - Display track IDs and confidence scores
    - Implement click detection for object selection
    - Show detailed object info on click
    - _Requirements: 14.1, 14.2_
  
  - [x] 17.3 Implement trajectory visualization
    - Draw predicted trajectories as polylines
    - Render uncertainty bounds as semi-transparent regions
    - Color code trajectories by collision probability
    - Animate trajectory updates smoothly
    - _Requirements: 14.3_
  
  - [x] 17.4 Add attention zone overlay
    - Draw 8 attention zones as sectors
    - Highlight zones based on driver gaze
    - Color code zones by risk level
    - Show zone labels and risk scores
    - _Requirements: 14.4_
  
  - [x] 17.5 Implement distance grid
    - Draw concentric circles at 5-meter intervals
    - Add radial lines for angular reference
    - Display distance labels
    - Make grid toggleable
    - _Requirements: 14.5_
  
  - [x] 17.6 Add screenshot and recording
    - Implement screenshot capture to PNG
    - Add video recording to MP4
    - Include timestamp and annotations
    - Provide save dialog with filename
    - _Requirements: 14.7_

- [x] 18. Implement driver state panel
  - [x] 18.1 Create circular gauge widget
    - Implement CircularGaugeWidget with custom QPainter
    - Draw background arc with gradient
    - Draw value arc with color zones (green/yellow/red)
    - Add needle pointer and center value text
    - Animate value changes smoothly
    - _Requirements: 15.1_
  
  - [x] 18.2 Implement gaze direction widget
    - Create GazeDirectionWidget with 3D head visualization
    - Draw head outline with pitch and yaw indicators
    - Show gaze vector as arrow
    - Highlight current attention zone
    - _Requirements: 15.2_
  
  - [x] 18.3 Add metrics display grid
    - Create labeled metric displays for alertness, attention, blink rate, head pose
    - Format values with units
    - Update values in real-time
    - Color code based on thresholds
    - _Requirements: 15.3_
  
  - [x] 18.4 Implement status indicators
    - Create StatusIndicator widget with icon and label
    - Add color-coded states (green OK, yellow warning, red critical)
    - Implement for drowsiness, distraction, eyes-on-road
    - Add pulsing animation for warnings
    - _Requirements: 15.4_
  
  - [x] 18.5 Add trend graphs
    - Use PyQtGraph for real-time plotting
    - Plot metrics over last 60 seconds
    - Add scrolling time axis
    - Show threshold lines
    - _Requirements: 15.5_
  
  - [x] 18.6 Implement warning animations
    - Add pulsing effect for threshold crossings
    - Flash indicator colors
    - Trigger sound effects
    - _Requirements: 15.6_

- [x] 19. Implement risk assessment panel
  - [x] 19.1 Create overall risk gauge
    - Implement CircularGaugeWidget for risk score
    - Set critical threshold at 0.7
    - Color code zones (green <0.5, yellow 0.5-0.7, red >0.7)
    - Animate risk changes
    - _Requirements: 16.1_
  
  - [x] 19.2 Implement hazards list
    - Create HazardListItem custom widget
    - Display hazard icon, type, zone, TTC
    - Show attention status (attended/unattended)
    - Add risk score progress bar
    - Update list with top 3 hazards
    - _Requirements: 16.2, 16.5_
  
  - [x] 19.3 Create zone risk radar chart
    - Implement ZoneRiskRadarChart with custom QPainter
    - Draw octagonal grid for 8 zones
    - Plot risk values as filled polygon
    - Add zone labels
    - Highlight high-risk zones
    - _Requirements: 16.3_
  
  - [x] 19.4 Add TTC display
    - Create TTCDisplayWidget with countdown timer
    - Color code by urgency (green >3s, yellow 1.5-3s, red <1.5s)
    - Show minimum TTC across all objects
    - Animate countdown
    - _Requirements: 16.4_
  
  - [x] 19.5 Implement risk timeline
    - Use PyQtGraph for historical risk plot
    - Plot risk score over last 5 minutes
    - Mark alert events on timeline
    - Add scrolling and zoom
    - _Requirements: 16.6_

- [x] 20. Implement alerts panel
  - [x] 20.1 Create alert display
    - Use QTextEdit with HTML formatting
    - Color code alerts by urgency (red critical, yellow warning, blue info)
    - Add timestamps and icons
    - Auto-scroll to latest alert
    - _Requirements: 13.2, 13.3_
  
  - [x] 20.2 Add audio alerts
    - Implement QSoundEffect for alert sounds
    - Load different sounds for critical, warning, info
    - Add mute/unmute toggle button
    - Control volume from settings
    - _Requirements: 13.2_
  
  - [x] 20.3 Implement alert controls
    - Add clear history button
    - Add export log button (save to text file)
    - Add false positive marking
    - Implement alert filtering
    - _Requirements: 13.3_
  
  - [x] 20.4 Add alert statistics
    - Display total alerts count
    - Show critical alerts count
    - Track false positives
    - Update statistics in real-time
    - _Requirements: 13.3_
  
  - [x] 20.5 Implement critical alert effects
    - Flash window/screen for critical alerts
    - Shake animation for main window
    - Bring window to front
    - Play urgent sound
    - _Requirements: 13.3_

- [x] 21. Implement performance monitoring dock
  - [x] 21.1 Create FPS graph
    - Use PyQtGraph PlotWidget
    - Plot FPS over last 60 seconds
    - Add 30 FPS target line
    - Color code below threshold
    - _Requirements: 17.1_
  
  - [x] 21.2 Create latency graph
    - Plot end-to-end latency over time
    - Add 100ms threshold line
    - Show p95 latency value
    - Color code violations
    - _Requirements: 17.2_
  
  - [x] 21.3 Implement module breakdown
    - Create stacked bar chart for module timings
    - Show time spent in each pipeline stage
    - Update at 1 Hz
    - Add tooltips with exact values
    - _Requirements: 17.3_
  
  - [x] 21.4 Add resource usage displays
    - Create GPU memory gauge (max 8GB)
    - Create CPU usage gauge (max 60%)
    - Show current and peak values
    - Color code based on thresholds
    - _Requirements: 17.4, 17.5_
  
  - [x] 21.5 Implement performance logging
    - Log performance metrics to file
    - Export performance reports
    - Generate performance summary
    - _Requirements: 17.6_

- [x] 22. Implement scenarios dock widget
  - [x] 22.1 Create scenarios list
    - Display all recorded scenarios in QListWidget
    - Show thumbnail, timestamp, duration, trigger type
    - Implement custom list item widget
    - Sort by timestamp (newest first)
    - _Requirements: 18.1_
  
  - [x] 22.2 Add search and filtering
    - Implement search box for text filtering
    - Add filter combo box (All, Critical, High Risk, Near Miss, Distracted)
    - Filter scenarios in real-time
    - _Requirements: 18.2_
  
  - [x] 22.3 Create scenario replay dialog
    - Implement modal dialog for playback
    - Show synchronized video players for all cameras
    - Display annotations overlay
    - Open on double-click
    - _Requirements: 18.3_
  
  - [x] 22.4 Implement playback controls
    - Add play/pause button
    - Add step forward/backward buttons
    - Add speed slider (0.25x to 2x)
    - Add timeline scrubber
    - Show current frame number and timestamp
    - _Requirements: 18.4, 18.5_
  
  - [x] 22.5 Add scenario actions
    - Implement export to MP4 and JSON
    - Add delete with confirmation dialog
    - Add share functionality
    - _Requirements: 18.6, 18.7_

- [x] 23. Implement configuration dock widget
  - [x] 23.1 Create tabbed configuration interface
    - Implement QTabWidget with tabs for Cameras, Detection, DMS, Risk, Alerts
    - Load current configuration values
    - Display in organized layout
    - _Requirements: 19.1_
  
  - [x] 23.2 Implement parameter controls
    - Create LabeledSlider widget with value display
    - Show min, max, current value, and units
    - Connect sliders to configuration values
    - Validate input ranges
    - _Requirements: 19.2, 19.5_
  
  - [x] 23.3 Add real-time parameter updates
    - Apply non-critical parameter changes immediately
    - Update system without restart
    - Show which parameters require restart
    - _Requirements: 19.3_
  
  - [x] 23.4 Implement save and reset
    - Add save button to persist changes to YAML
    - Add reset button to restore defaults
    - Show unsaved changes indicator
    - Confirm before discarding changes
    - _Requirements: 19.4_
  
  - [x] 23.5 Add profile management
    - Implement import configuration profile
    - Implement export configuration profile
    - Save named presets
    - _Requirements: 19.6_

- [x] 24. Implement worker thread integration
  - [x] 24.1 Create SentinelWorker thread
    - Implement QThread subclass for SENTINEL system
    - Run main processing loop in background
    - Define signals for all data outputs
    - Handle thread lifecycle (start, stop, cleanup)
    - _Requirements: 13.1_
  
  - [x] 24.2 Connect signals to GUI slots
    - Connect frame_ready signal to video displays
    - Connect bev_ready signal to BEV canvas
    - Connect detections_ready signal to overlays
    - Connect driver_state_ready signal to driver panel
    - Connect risks_ready signal to risk panel
    - Connect alerts_ready signal to alerts panel
    - Connect performance_ready signal to performance dock
    - _Requirements: 13.3_
  
  - [x] 24.3 Implement thread-safe data passing
    - Use Qt signals for cross-thread communication
    - Deep copy data before emitting signals
    - Handle signal queuing and buffering
    - Prevent GUI blocking
    - _Requirements: 13.3_
  
  - [x] 24.4 Add error handling
    - Catch exceptions in worker thread
    - Emit error signals to GUI
    - Display error dialogs
    - Implement automatic recovery
    - _Requirements: 11.3_

- [x] 25. Implement advanced trajectory prediction
  - [x] 25.1 Create LSTM trajectory model
    - Design LSTM architecture for trajectory prediction
    - Implement model training pipeline
    - Train on historical trajectory data
    - Save trained model weights
    - _Requirements: 20.2_
  
  - [x] 25.2 Implement physics-based models
    - Create constant velocity model
    - Create constant acceleration model
    - Create constant turn rate model
    - Select model based on object type and history
    - _Requirements: 20.2_
  
  - [x] 25.3 Create trajectory predictor class
    - Implement TrajectoryPredictor with LSTM and physics models
    - Extract motion history from tracked objects
    - Extract scene context features
    - Generate multiple trajectory hypotheses
    - _Requirements: 20.1_
  
  - [x] 25.4 Implement uncertainty estimation
    - Calculate covariance matrices for predictions
    - Estimate confidence bounds
    - Propagate uncertainty through time
    - _Requirements: 20.3_
  
  - [x] 25.5 Add collision probability calculation
    - Implement Mahalanobis distance computation
    - Calculate collision probability between trajectories
    - Consider uncertainty ellipses
    - Output probability scores
    - _Requirements: 20.4_
  
  - [x] 25.6 Integrate with risk assessment
    - Pass trajectory predictions to contextual intelligence
    - Use collision probabilities in risk calculation
    - Update risk scores based on predicted conflicts
    - _Requirements: 20.1_
  
  - [x] 25.7 Add trajectory visualization
    - Draw predicted trajectories in BEV canvas
    - Render uncertainty bounds as transparent regions
    - Color code by collision probability
    - Show multiple hypotheses
    - _Requirements: 20.5_
  
  - [x] 25.8 Optimize performance
    - Profile trajectory prediction latency
    - Optimize LSTM inference
    - Parallelize physics models
    - Target <5ms per object
    - _Requirements: 20.6_

- [x] 26. Implement driver behavior profiling
  - [x] 26.1 Create face recognition system
    - Extract face embeddings using FaceNet or similar
    - Implement face matching with threshold
    - Store driver face embeddings
    - Identify driver at session start
    - _Requirements: 21.1_
  
  - [x] 26.2 Implement metrics tracking
    - Track reaction time from alert to action
    - Track following distance preferences
    - Track lane change frequency
    - Track speed profile statistics
    - Track risk tolerance from near-miss events
    - _Requirements: 21.2_
  
  - [x] 26.3 Create driving style classifier
    - Analyze metrics to classify style
    - Categorize as aggressive, normal, or cautious
    - Update classification over time
    - _Requirements: 21.3_
  
  - [x] 26.4 Implement threshold adaptation
    - Calculate personalized TTC threshold based on reaction time
    - Adjust following distance for driving style
    - Adapt alert sensitivity
    - Apply 1.5x safety margin
    - _Requirements: 21.4_
  
  - [x] 26.5 Create driver report generator
    - Calculate safety score (0-100)
    - Calculate attention score (0-100)
    - Calculate eco-driving score (0-100)
    - Generate recommendations
    - Analyze trends over time
    - _Requirements: 21.5_
  
  - [x] 26.6 Implement profile persistence
    - Save driver profiles to JSON files
    - Load profiles on driver identification
    - Update profiles after each session
    - Support multiple driver profiles
    - _Requirements: 21.6_
  
  - [x] 26.7 Add profile management GUI
    - Display current driver profile
    - Show driver statistics and scores
    - Visualize behavior trends
    - Allow profile reset
    - _Requirements: 21.5_

- [x] 27. Implement HD map integration
  - [x] 27.1 Create map parser
    - Implement OpenDRIVE format parser
    - Implement Lanelet2 format parser
    - Extract lane geometry (centerlines, boundaries)
    - Extract traffic signs and lights
    - Extract crosswalks and road boundaries
    - _Requirements: 22.1_
  
  - [x] 27.2 Implement map matching
    - Perform GPS-based coarse localization
    - Implement visual odometry for precise position
    - Match position to lane using geometry
    - Calculate lateral offset from lane center
    - _Requirements: 22.2_
  
  - [x] 27.3 Create feature query system
    - Query upcoming features within lookahead distance
    - Return signs, lights, intersections
    - Include feature attributes (speed limit, sign type)
    - _Requirements: 22.3_
  
  - [x] 27.4 Implement path prediction
    - Build lane graph from map
    - Implement A* routing algorithm
    - Use turn signals and destination for prediction
    - Output predicted lane sequence
    - _Requirements: 22.4_
  
  - [x] 27.5 Add BEV map overlay
    - Render lane boundaries on BEV
    - Draw crosswalks and road markings
    - Display traffic sign icons
    - Show traffic light states
    - Make overlay toggleable
    - _Requirements: 22.5_
  
  - [x] 27.6 Create map view dock widget
    - Display map with OpenStreetMap background
    - Show current vehicle position and heading
    - Highlight current lane
    - Display upcoming features list
    - _Requirements: 22.5_
  
  - [x] 27.7 Optimize map queries
    - Implement spatial indexing (R-tree)
    - Cache nearby map features
    - Target <5ms query time
    - _Requirements: 22.6_

- [x] 28. Implement CAN bus integration
  - [x] 28.1 Create CAN interface
    - Implement SocketCAN connection for Linux
    - Handle CAN bus initialization
    - Implement reconnection logic
    - Add error handling
    - _Requirements: 23.1_
  
  - [x] 28.2 Implement DBC parser
    - Parse DBC file format
    - Extract message definitions
    - Extract signal definitions
    - Build decoding tables
    - _Requirements: 23.3_
  
  - [x] 28.3 Create telemetry reader
    - Read CAN messages at 100 Hz
    - Decode speed, steering, brake, throttle
    - Decode gear and turn signals
    - Store in VehicleTelemetry dataclass
    - _Requirements: 23.2_
  
  - [x] 28.4 Implement command sender
    - Encode brake intervention commands
    - Encode steering intervention commands
    - Send via CAN bus
    - Add safety checks and limits
    - _Requirements: 23.4_
  
  - [x] 28.5 Integrate with risk assessment
    - Pass vehicle speed to TTC calculation
    - Use steering angle in trajectory prediction
    - Detect braking events
    - Use turn signals for path prediction
    - _Requirements: 23.5_
  
  - [x] 28.6 Add telemetry visualization
    - Create vehicle telemetry dashboard
    - Display speedometer gauge
    - Display steering angle indicator
    - Display brake/throttle bars
    - Show gear and turn signals
    - _Requirements: 23.2_
  
  - [x] 28.7 Implement CAN logging
    - Log all CAN messages to file
    - Include timestamps
    - Support playback from logs
    - _Requirements: 23.6_

- [x] 29. Implement cloud synchronization
  - [x] 29.1 Create API client
    - Implement REST API client with requests library
    - Add authentication with API key
    - Handle connection errors and retries
    - Implement rate limiting
    - _Requirements: 24.1_
  
  - [x] 29.2 Implement trip uploader
    - Collect trip summary data
    - Anonymize GPS coordinates
    - Upload every 5 minutes
    - Handle offline queueing
    - _Requirements: 24.1_
  
  - [x] 29.3 Create scenario uploader
    - Check user consent for scenario upload
    - Compress scenario videos
    - Upload high-risk scenarios
    - Track upload status
    - _Requirements: 24.2_
  
  - [x] 29.4 Implement model downloader
    - Check for model updates every 24 hours
    - Download new model versions
    - Verify model signatures
    - Install models atomically
    - _Requirements: 24.3_
  
  - [x] 29.5 Add profile synchronization
    - Upload driver profiles to cloud
    - Download profiles from cloud
    - Merge profiles across vehicles
    - Encrypt profile data
    - _Requirements: 24.4_
  
  - [x] 29.6 Create fleet statistics viewer
    - Fetch fleet-wide statistics
    - Display aggregate metrics
    - Show vehicle rankings
    - Visualize trends
    - _Requirements: 24.5_
  
  - [x] 29.7 Implement offline support
    - Queue operations when offline
    - Detect connectivity changes
    - Auto-sync when online
    - Show sync status in GUI
    - _Requirements: 24.6_
  
  - [x] 29.8 Add cloud settings GUI
    - Configure API endpoint and credentials
    - Enable/disable cloud sync
    - Set sync interval
    - Manage consent preferences
    - _Requirements: 24.1_

- [x] 30. Create analytics and reporting
  - [x] 30.1 Implement trip analytics
    - Track trip duration, distance, average speed
    - Calculate trip safety score
    - Count alerts by type
    - Identify high-risk segments
    - _Requirements: 21.5_
  
  - [x] 30.2 Create risk heatmap
    - Generate spatial risk heatmap
    - Aggregate risk by location
    - Visualize on map
    - Export heatmap images
    - _Requirements: 16.6_
  
  - [x] 30.3 Implement driver behavior reports
    - Generate PDF reports with charts
    - Include safety scores and trends
    - Add recommendations
    - Export to Excel format
    - _Requirements: 21.5_
  
  - [x] 30.4 Create analytics dashboard
    - Display trip statistics
    - Show driver performance metrics
    - Plot trends over time
    - Compare against fleet averages
    - _Requirements: 17.1_
  
  - [x] 30.5 Add export functionality
    - Export data to CSV
    - Export reports to PDF
    - Export visualizations to PNG
    - _Requirements: 19.6_

- [-] 31. Testing and validation for GUI and advanced features
  - [x] 31.1 Test GUI components
    - Unit test all custom widgets
    - Test signal/slot connections
    - Test thread safety
    - Verify 30 FPS rendering
    - _Requirements: 13.3_
  
  - [x] 31.2 Test trajectory prediction
    - Validate LSTM model accuracy
    - Test physics models
    - Verify uncertainty estimation
    - Measure collision probability accuracy
    - Benchmark performance (<5ms)
    - _Requirements: 20.2, 20.3, 20.4, 20.6_
  
  - [x] 31.3 Test driver profiling
    - Validate face recognition accuracy (>95%)
    - Test metrics tracking
    - Verify style classification
    - Test threshold adaptation
    - _Requirements: 21.1, 21.2, 21.3, 21.4_
  
  - [x] 31.4 Test HD map integration
    - Validate map parsing
    - Test map matching accuracy (0.2m)
    - Verify feature queries
    - Benchmark query performance (<5ms)
    - _Requirements: 22.1, 22.2, 22.3, 22.6_
  
  - [x] 31.5 Test CAN bus integration
    - Test CAN connection and reconnection
    - Validate message decoding
    - Test telemetry reading at 100 Hz
    - Verify command sending
    - _Requirements: 23.1, 23.2, 23.3, 23.4_
  
  - [x] 31.6 Test cloud synchronization
    - Test API client with mock server
    - Verify data upload and download
    - Test offline queueing
    - Validate encryption
    - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.6_
  
  - [x] 31.7 Perform usability testing
    - Test GUI responsiveness
    - Verify keyboard shortcuts
    - Test multi-monitor support
    - Validate accessibility features
    - _Requirements: 13.4, 13.5, 13.6_
  
  - [ ] 31.8 Integration testing
    - Test GUI with full SENTINEL system
    - Verify all data flows correctly
    - Test under load (30 FPS sustained)
    - Validate memory usage
    - _Requirements: 13.3, 17.1, 17.2_
