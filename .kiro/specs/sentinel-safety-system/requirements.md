# Requirements Document

## Introduction

SENTINEL is a contextual safety intelligence platform for vehicles that combines 360° Bird's Eye View (BEV) perception, multi-camera sensor fusion, and driver monitoring to create a holistic safety system. The system correlates environmental threats with driver awareness to provide intelligent, context-aware safety interventions, aiming to prevent accidents by understanding both the environment and driver state in real-time.

## Glossary

- **SENTINEL System**: The complete contextual safety intelligence platform including all hardware and software components
- **BEV (Bird's Eye View)**: A top-down perspective transformation of camera views showing the vehicle's surroundings
- **DMS (Driver Monitoring System)**: The subsystem that monitors driver state including attention, drowsiness, and distraction
- **TTC (Time To Collision)**: The estimated time until a potential collision with a detected object
- **PERCLOS**: Percentage of eye closure over time, used as a drowsiness indicator
- **Contextual Risk Score**: A calculated risk value that combines environmental hazards with driver awareness state
- **Scene Graph**: A spatial representation of all detected objects and their relationships
- **Attention Zone**: One of eight spatial zones around the vehicle (front, front-left, left, rear-left, rear, rear-right, right, front-right)
- **Camera Bundle**: A synchronized set of frames from all cameras with aligned timestamps
- **Homography Matrix**: A transformation matrix used to convert camera perspective to BEV perspective

## Requirements

### Requirement 1: Multi-Camera Management

**User Story:** As a system operator, I want the system to manage multiple cameras simultaneously, so that I can capture comprehensive views of both the environment and driver.

#### Acceptance Criteria

1. WHEN the SENTINEL System starts, THE SENTINEL System SHALL initialize three USB cameras (interior, front-left, front-right) with configured resolution and frame rate
2. WHEN capturing frames from multiple cameras, THE SENTINEL System SHALL synchronize timestamps across all cameras within 5 milliseconds tolerance
3. IF a camera disconnects during operation, THEN THE SENTINEL System SHALL detect the disconnection within 1 second and log the event
4. WHEN a disconnected camera reconnects, THE SENTINEL System SHALL automatically reinitialize the camera and resume capture
5. WHERE camera calibration data exists, THE SENTINEL System SHALL load intrinsic and extrinsic parameters from persistent storage at startup

### Requirement 2: Bird's Eye View Generation

**User Story:** As a perception engineer, I want external camera views transformed into a unified top-down perspective, so that I can analyze the vehicle's surroundings in a single coherent view.

#### Acceptance Criteria

1. WHEN receiving frames from external cameras, THE SENTINEL System SHALL transform each frame to BEV perspective using homography matrices
2. WHEN multiple BEV views overlap, THE SENTINEL System SHALL blend the overlapping regions using multi-band blending to create seamless transitions
3. THE SENTINEL System SHALL output BEV images at 640x640 pixel resolution with 0.1 meters per pixel scale
4. WHEN generating BEV output, THE SENTINEL System SHALL create a valid region mask that excludes vehicle body and sky areas
5. WHEN BEV generation completes, THE SENTINEL System SHALL complete the transformation within 15 milliseconds

### Requirement 3: Semantic Segmentation

**User Story:** As a perception engineer, I want each pixel in the BEV classified by type, so that I can understand the drivable areas and identify different object categories.

#### Acceptance Criteria

1. WHEN processing a BEV image, THE SENTINEL System SHALL classify each pixel into one of nine classes (road, lane_marking, vehicle, pedestrian, cyclist, obstacle, parking_space, curb, vegetation)
2. WHEN segmentation completes, THE SENTINEL System SHALL output per-pixel class probabilities as confidence values
3. WHEN processing consecutive frames, THE SENTINEL System SHALL apply temporal smoothing with alpha 0.7 to reduce segmentation flicker
4. THE SENTINEL System SHALL complete semantic segmentation inference within 15 milliseconds on the target GPU
5. WHEN evaluated on validation data, THE SENTINEL System SHALL achieve mean Intersection over Union of 75 percent or higher

### Requirement 4: Multi-View Object Detection

**User Story:** As a safety engineer, I want objects detected and tracked in 3D space from multiple camera views, so that I can monitor potential hazards around the vehicle.

#### Acceptance Criteria

1. WHEN processing camera frames, THE SENTINEL System SHALL detect objects of classes (vehicle, pedestrian, cyclist, traffic_sign, traffic_light) in each camera view
2. WHEN objects are detected, THE SENTINEL System SHALL estimate 3D bounding boxes with position (x, y, z), dimensions (w, h, l), and orientation (θ) in vehicle coordinate frame
3. WHEN detections exist from multiple cameras, THE SENTINEL System SHALL fuse detections into a unified object list using Hungarian algorithm with 0.3 IoU threshold
4. WHEN tracking objects across frames, THE SENTINEL System SHALL maintain consistent track IDs for at least 30 frames
5. WHEN objects move between frames, THE SENTINEL System SHALL estimate velocity vectors for each tracked object
6. THE SENTINEL System SHALL complete object detection per camera within 20 milliseconds

### Requirement 5: Driver State Monitoring

**User Story:** As a safety engineer, I want continuous monitoring of driver attention and alertness, so that I can assess whether the driver is aware of potential hazards.

#### Acceptance Criteria

1. WHEN processing interior camera frames, THE SENTINEL System SHALL detect the driver face and extract 68 or more facial landmarks
2. WHEN facial landmarks are detected, THE SENTINEL System SHALL estimate gaze direction with pitch and yaw angles in vehicle coordinate frame with error less than 5 degrees
3. WHEN analyzing driver state, THE SENTINEL System SHALL calculate head pose with roll, pitch, and yaw angles
4. IF PERCLOS exceeds 80 percent for 3 or more seconds, THEN THE SENTINEL System SHALL flag drowsiness indicator
5. IF yawning frequency exceeds 3 per minute, THEN THE SENTINEL System SHALL flag drowsiness indicator
6. IF eyes remain closed for more than 2 seconds, THEN THE SENTINEL System SHALL detect micro-sleep event
7. WHEN detecting distraction, THE SENTINEL System SHALL classify distraction type (phone usage, looking at passenger, adjusting controls, eyes off road, hands off wheel)
8. IF eyes are off road for more than 2 seconds, THEN THE SENTINEL System SHALL flag distraction
9. WHEN driver state analysis completes, THE SENTINEL System SHALL output a driver readiness score from 0 to 100
10. THE SENTINEL System SHALL complete DMS processing within 25 milliseconds

### Requirement 6: Contextual Risk Assessment

**User Story:** As a safety engineer, I want environmental hazards correlated with driver awareness, so that I can identify situations where the driver may not be aware of critical threats.

#### Acceptance Criteria

1. WHEN objects are detected, THE SENTINEL System SHALL maintain a spatial scene graph representing all detected objects and their relationships
2. WHEN driver gaze is estimated, THE SENTINEL System SHALL map driver attention to one of eight spatial zones around the vehicle
3. WHEN objects are tracked, THE SENTINEL System SHALL calculate Time-To-Collision for each object using constant velocity model
4. WHEN assessing future threats, THE SENTINEL System SHALL predict object trajectories 3 seconds ahead with 0.1 second time steps
5. WHEN calculating contextual risk, THE SENTINEL System SHALL compute risk score as base hazard score multiplied by awareness penalty multiplied by capacity factor, where awareness penalty equals 2.0 if driver not looking at hazard zone else 1.0, and capacity factor equals 2.0 minus driver readiness divided by 100
6. WHEN multiple risks exist, THE SENTINEL System SHALL prioritize risks and output the top three threats
7. WHEN a hazard exists in an unattended zone, THE SENTINEL System SHALL detect the attention-risk mismatch
8. THE SENTINEL System SHALL complete risk assessment within 10 milliseconds

### Requirement 7: Alert Generation and Dispatch

**User Story:** As a driver, I want timely alerts about hazards I may not be aware of, so that I can take corrective action to avoid accidents.

#### Acceptance Criteria

1. WHEN risks are identified, THE SENTINEL System SHALL generate alerts with urgency levels (INFO, WARNING, CRITICAL)
2. WHEN generating alerts, THE SENTINEL System SHALL support multi-modal output including visual, audio, and haptic modalities
3. WHEN driver cognitive load is high, THE SENTINEL System SHALL adapt alert timing based on driver readiness score
4. IF an alert was generated for the same hazard within 5 seconds, THEN THE SENTINEL System SHALL suppress the redundant alert
5. WHEN an alert is generated, THE SENTINEL System SHALL log the alert with timestamp and context data
6. WHEN risk score exceeds 0.9, THE SENTINEL System SHALL generate CRITICAL alert with visual flash, audio alarm, and haptic pulse
7. WHEN risk score exceeds 0.7 and driver is not aware, THE SENTINEL System SHALL generate WARNING alert with visual HUD and audio beep

### Requirement 8: Scenario Recording

**User Story:** As a system analyst, I want critical scenarios automatically recorded, so that I can analyze system behavior and improve safety algorithms.

#### Acceptance Criteria

1. IF contextual risk score exceeds 0.7, THEN THE SENTINEL System SHALL trigger scenario recording
2. IF driver is distracted during a hazard event, THEN THE SENTINEL System SHALL trigger scenario recording
3. IF a system intervention occurs, THEN THE SENTINEL System SHALL trigger scenario recording
4. IF Time-To-Collision falls below 1.5 seconds, THEN THE SENTINEL System SHALL trigger scenario recording
5. WHEN recording a scenario, THE SENTINEL System SHALL save all camera frames, BEV output, object detections, driver state, and risk scores
6. WHEN a scenario is recorded, THE SENTINEL System SHALL export the scenario in JSON format with accompanying video files
7. WHEN playback is requested, THE SENTINEL System SHALL support frame-by-frame analysis of recorded scenarios

### Requirement 9: Real-Time Visualization

**User Story:** As a system operator, I want a real-time dashboard showing system state, so that I can monitor performance and verify correct operation.

#### Acceptance Criteria

1. THE SENTINEL System SHALL display real-time stitched BEV with semantic segmentation overlay
2. THE SENTINEL System SHALL display 3D bounding boxes overlaid on detected objects
3. THE SENTINEL System SHALL display driver attention heatmap showing gaze direction
4. THE SENTINEL System SHALL display risk scores for each identified hazard
5. THE SENTINEL System SHALL display driver state metrics including alertness, gaze direction, and distraction type
6. THE SENTINEL System SHALL display system performance metrics including frames per second and latency per module
7. WHEN displaying multiple views, THE SENTINEL System SHALL support multi-view layout showing BEV, raw camera feeds, and DMS output simultaneously
8. WHERE recorded scenarios exist, THE SENTINEL System SHALL provide replay controls for frame-by-frame review

### Requirement 10: System Performance

**User Story:** As a system engineer, I want the system to process data in real-time with low latency, so that alerts can be generated quickly enough to be actionable.

#### Acceptance Criteria

1. THE SENTINEL System SHALL process the complete pipeline from camera capture to alert generation within 100 milliseconds at 95th percentile
2. THE SENTINEL System SHALL maintain processing throughput of 30 frames per second or higher
3. WHILE operating, THE SENTINEL System SHALL consume no more than 8 gigabytes of GPU memory
4. WHILE operating on an 8-core processor, THE SENTINEL System SHALL consume no more than 60 percent CPU utilization

### Requirement 11: System Reliability

**User Story:** As a system operator, I want the system to handle failures gracefully, so that partial sensor failures do not cause complete system shutdown.

#### Acceptance Criteria

1. THE SENTINEL System SHALL maintain uptime of 99.9 percent or higher during operation
2. IF one camera fails, THEN THE SENTINEL System SHALL continue operating with reduced coverage from remaining cameras
3. IF a model inference error occurs, THEN THE SENTINEL System SHALL recover automatically and log the error
4. IF the system crashes, THEN THE SENTINEL System SHALL restore previous state upon restart within 2 seconds

### Requirement 12: System Configuration

**User Story:** As a system integrator, I want all system parameters configurable via files, so that I can adapt the system to different vehicle configurations without code changes.

#### Acceptance Criteria

1. THE SENTINEL System SHALL load camera configuration including device indices, resolution, frame rate, and calibration paths from YAML configuration file
2. THE SENTINEL System SHALL load model configuration including architecture, weights paths, and inference parameters from YAML configuration file
3. THE SENTINEL System SHALL load risk assessment thresholds and weights from YAML configuration file
4. THE SENTINEL System SHALL load alert configuration including urgency thresholds and modality settings from YAML configuration file
5. WHERE configuration parameters are updated, THE SENTINEL System SHALL support real-time parameter tuning without system restart for non-critical parameters

### Requirement 13: PyQt6 GUI Application

**User Story:** As a system operator, I want a professional desktop GUI application, so that I can monitor and control the SENTINEL system with an intuitive interface.

#### Acceptance Criteria

1. THE SENTINEL System SHALL provide a PyQt6-based desktop application with main window, menu bar, toolbar, and status bar
2. THE SENTINEL System SHALL display live camera feeds in a 2x2 grid layout with interior, front-left, front-right, and BEV views
3. THE SENTINEL System SHALL render all video displays at 30 frames per second with GPU acceleration
4. THE SENTINEL System SHALL provide keyboard shortcuts for all major actions including start system (F5), stop system (F6), and full screen (F11)
5. THE SENTINEL System SHALL persist window layout and user preferences across application restarts
6. THE SENTINEL System SHALL support multi-monitor configurations with draggable dock widgets
7. THE SENTINEL System SHALL apply a modern dark theme with configurable accent colors
8. THE SENTINEL System SHALL display tooltips on all interactive elements with descriptive help text

### Requirement 14: Interactive BEV Canvas

**User Story:** As a system operator, I want an interactive bird's eye view display, so that I can inspect detected objects and understand spatial relationships.

#### Acceptance Criteria

1. THE SENTINEL System SHALL render BEV image with overlaid 3D bounding boxes for all detected objects
2. WHEN user clicks on a detected object in BEV, THE SENTINEL System SHALL display detailed information including class, confidence, velocity, and track ID
3. THE SENTINEL System SHALL draw predicted trajectories for all tracked objects with uncertainty bounds
4. THE SENTINEL System SHALL highlight attention zones with color coding based on driver gaze direction
5. THE SENTINEL System SHALL overlay a distance grid with 5-meter spacing on the BEV display
6. THE SENTINEL System SHALL support zoom and pan controls with mouse wheel and drag gestures
7. THE SENTINEL System SHALL provide screenshot and video recording functionality for BEV display

### Requirement 15: Driver State Visualization

**User Story:** As a safety analyst, I want comprehensive driver state visualization, so that I can assess driver readiness and attention patterns.

#### Acceptance Criteria

1. THE SENTINEL System SHALL display driver readiness score as a circular gauge with color-coded zones (green above 70, yellow 50-70, red below 50)
2. THE SENTINEL System SHALL visualize driver gaze direction as a 3D head model with pitch and yaw indicators
3. THE SENTINEL System SHALL display real-time metrics for alertness, attention, blink rate, and head pose in a grid layout
4. THE SENTINEL System SHALL show status indicators for drowsiness, distraction, and eyes-on-road with color-coded states (green OK, yellow warning, red critical)
5. THE SENTINEL System SHALL plot trend graphs for driver metrics over the last 60 seconds
6. WHEN driver state thresholds are crossed, THE SENTINEL System SHALL animate warning indicators with pulsing effects

### Requirement 16: Risk Assessment Dashboard

**User Story:** As a safety engineer, I want a comprehensive risk assessment dashboard, so that I can monitor hazards and evaluate system performance.

#### Acceptance Criteria

1. THE SENTINEL System SHALL display overall risk score as a circular gauge with critical threshold at 0.7
2. THE SENTINEL System SHALL list the top 3 active hazards with icons, zone information, TTC, and attention status
3. THE SENTINEL System SHALL render an 8-zone radar chart showing risk distribution around the vehicle
4. THE SENTINEL System SHALL display minimum time-to-collision with countdown timer and color coding
5. THE SENTINEL System SHALL highlight hazards in unattended zones with warning icons
6. THE SENTINEL System SHALL plot historical risk timeline for the last 5 minutes

### Requirement 17: Performance Monitoring

**User Story:** As a system engineer, I want real-time performance monitoring, so that I can identify bottlenecks and optimize system performance.

#### Acceptance Criteria

1. THE SENTINEL System SHALL display frame rate graph showing FPS over the last 60 seconds with target line at 30 FPS
2. THE SENTINEL System SHALL display pipeline latency graph showing end-to-end processing time with 100ms threshold line
3. THE SENTINEL System SHALL show module breakdown with stacked bar chart indicating time spent in each pipeline stage
4. THE SENTINEL System SHALL display GPU memory usage as percentage with 8GB maximum threshold
5. THE SENTINEL System SHALL display CPU usage as percentage with 60 percent threshold on 8-core processor
6. THE SENTINEL System SHALL update performance metrics at 1 Hz refresh rate

### Requirement 18: Scenario Management

**User Story:** As a system analyst, I want to browse and replay recorded scenarios, so that I can analyze critical events and improve algorithms.

#### Acceptance Criteria

1. THE SENTINEL System SHALL display a list of all recorded scenarios with timestamp, duration, and trigger type
2. THE SENTINEL System SHALL support search and filtering of scenarios by trigger type (critical, high risk, near miss, distracted driver)
3. WHEN user double-clicks a scenario, THE SENTINEL System SHALL open the scenario replay dialog
4. THE SENTINEL System SHALL provide playback controls including play, pause, step forward, step backward, and speed adjustment
5. THE SENTINEL System SHALL display synchronized video playback of all camera feeds with annotations overlay
6. THE SENTINEL System SHALL support export of scenarios to external formats (MP4 video, JSON annotations)
7. THE SENTINEL System SHALL allow deletion of scenarios with confirmation dialog

### Requirement 19: Configuration Interface

**User Story:** As a system operator, I want to adjust system parameters through the GUI, so that I can tune the system without editing configuration files.

#### Acceptance Criteria

1. THE SENTINEL System SHALL provide a configuration dialog with tabbed interface for cameras, detection, DMS, risk assessment, and alerts
2. THE SENTINEL System SHALL display all tunable parameters as labeled sliders with current value, min, max, and units
3. WHEN user adjusts a parameter, THE SENTINEL System SHALL apply the change in real-time for non-critical parameters
4. THE SENTINEL System SHALL provide save and reset buttons to persist or discard configuration changes
5. THE SENTINEL System SHALL validate parameter values and display error messages for invalid inputs
6. THE SENTINEL System SHALL support import and export of configuration profiles

### Requirement 20: Advanced Trajectory Prediction

**User Story:** As a safety engineer, I want multi-hypothesis trajectory prediction, so that I can anticipate multiple possible future paths for detected objects.

#### Acceptance Criteria

1. WHEN tracking an object, THE SENTINEL System SHALL predict up to 3 trajectory hypotheses for 5 seconds ahead with 0.1 second time steps
2. THE SENTINEL System SHALL use LSTM-based prediction model combined with physics-based models (constant velocity, constant acceleration, constant turn rate)
3. THE SENTINEL System SHALL calculate uncertainty bounds for each predicted trajectory using covariance estimation
4. THE SENTINEL System SHALL compute collision probability between ego vehicle trajectory and object trajectories using Mahalanobis distance
5. THE SENTINEL System SHALL visualize predicted trajectories in BEV display with confidence-based transparency
6. THE SENTINEL System SHALL complete trajectory prediction within 5 milliseconds per object

### Requirement 21: Driver Behavior Profiling

**User Story:** As a fleet manager, I want individual driver behavior profiling, so that I can provide personalized coaching and adapt safety thresholds.

#### Acceptance Criteria

1. WHEN a driver starts a session, THE SENTINEL System SHALL identify the driver using face recognition with 95 percent accuracy
2. THE SENTINEL System SHALL track driver-specific metrics including reaction time, following distance preference, lane change frequency, and risk tolerance
3. THE SENTINEL System SHALL classify driving style as aggressive, normal, or cautious based on behavioral patterns
4. THE SENTINEL System SHALL adapt alert thresholds based on driver reaction time with 1.5x safety margin
5. THE SENTINEL System SHALL generate comprehensive driver behavior reports including safety score, attention score, and recommendations
6. THE SENTINEL System SHALL persist driver profiles across sessions with automatic profile updates

### Requirement 22: HD Map Integration

**User Story:** As a localization engineer, I want HD map integration, so that I can achieve lane-level positioning and anticipate upcoming road features.

#### Acceptance Criteria

1. THE SENTINEL System SHALL load HD maps in OpenDRIVE or Lanelet2 format with lane geometry, traffic signs, traffic lights, and crosswalks
2. THE SENTINEL System SHALL perform map matching to determine current lane ID and lateral offset from lane center with 0.2 meter accuracy
3. THE SENTINEL System SHALL query upcoming map features within 100 meters lookahead including signs, lights, and intersections
4. THE SENTINEL System SHALL predict driver intended path through road network using turn signals and destination
5. THE SENTINEL System SHALL overlay HD map features on BEV display with lane boundaries, crosswalks, and traffic signs
6. THE SENTINEL System SHALL complete map matching and feature query within 5 milliseconds

### Requirement 23: CAN Bus Integration

**User Story:** As a vehicle integrator, I want CAN bus integration, so that I can access vehicle telemetry and send intervention commands.

#### Acceptance Criteria

1. THE SENTINEL System SHALL connect to vehicle CAN bus using SocketCAN interface on Linux
2. THE SENTINEL System SHALL read vehicle telemetry including speed, steering angle, brake pressure, throttle position, gear, and turn signals at 100 Hz
3. THE SENTINEL System SHALL decode CAN messages based on configurable DBC file
4. WHERE vehicle supports control, THE SENTINEL System SHALL send brake and steering intervention commands via CAN bus
5. THE SENTINEL System SHALL integrate vehicle speed and steering angle into risk assessment calculations
6. THE SENTINEL System SHALL log all CAN bus communication for debugging and analysis

### Requirement 24: Cloud Synchronization

**User Story:** As a fleet manager, I want cloud synchronization, so that I can aggregate data across vehicles and deploy model updates.

#### Acceptance Criteria

1. THE SENTINEL System SHALL upload anonymized trip summaries to cloud backend every 5 minutes including duration, distance, alert count, and average risk score
2. THE SENTINEL System SHALL upload high-risk scenarios to cloud for model improvement with driver consent
3. THE SENTINEL System SHALL check for model updates every 24 hours and download new versions with signature verification
4. THE SENTINEL System SHALL sync driver profiles across vehicles in the fleet with encrypted transmission
5. THE SENTINEL System SHALL retrieve fleet-wide statistics including average safety score and total miles driven
6. THE SENTINEL System SHALL support offline operation with automatic sync when connectivity is restored
