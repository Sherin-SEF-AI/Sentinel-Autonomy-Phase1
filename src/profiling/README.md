# Driver Behavior Profiling Module

This module provides driver identification, behavior tracking, and personalized safety threshold adaptation for the SENTINEL system.

## Overview

The driver behavior profiling system identifies individual drivers using face recognition, tracks their driving behavior over time, classifies their driving style, and adapts safety thresholds to provide personalized safety interventions.

## Components

### 1. Face Recognition System (`face_recognition.py`)

Identifies drivers using face embeddings extracted from the interior camera.

**Features:**
- Face detection using OpenCV DNN or Haar cascades
- Face embedding extraction (FaceNet-style)
- Face matching with configurable similarity threshold (default: 0.6)
- Automatic driver ID generation for new drivers

**Usage:**
```python
from src.profiling import FaceRecognitionSystem

config = {
    'recognition_threshold': 0.6,
    'embedding_size': 128
}
face_recognition = FaceRecognitionSystem(config)

# Extract embedding from frame
embedding = face_recognition.extract_face_embedding(frame)

# Match against stored embeddings
driver_id, similarity = face_recognition.match_face(embedding, stored_embeddings)
```

### 2. Metrics Tracker (`metrics_tracker.py`)

Tracks driver behavior metrics throughout a driving session.

**Metrics Tracked:**
- Reaction time: Time from alert to driver action
- Following distance: Distance to vehicle ahead
- Lane change frequency: Number of lane changes per hour
- Speed profile: Average, max, variance
- Risk tolerance: Behavior during high-risk situations

**Usage:**
```python
from src.profiling import MetricsTracker

tracker = MetricsTracker(config)
tracker.start_session(timestamp)

# Update metrics during driving
tracker.update(timestamp, speed=25.0, following_distance=30.0, lane_id=2, risk_score=0.3)

# Record alerts and actions
tracker.record_alert(alert_id=1, timestamp=t1)
tracker.record_driver_action(alert_id=1, timestamp=t2, action_type='brake')

# Get summary
summary = tracker.get_summary()
```

### 3. Driving Style Classifier (`style_classifier.py`)

Classifies driving style as aggressive, normal, or cautious based on behavior metrics.

**Classification Criteria:**
- Reaction time (faster = more aggressive)
- Following distance (shorter = more aggressive)
- Lane change frequency (higher = more aggressive)
- Speed variance (higher = more aggressive)
- Risk tolerance (higher = more aggressive)

**Usage:**
```python
from src.profiling import DrivingStyleClassifier, DrivingStyle

classifier = DrivingStyleClassifier(config)
metrics = tracker.get_summary()

# Classify driving style
style = classifier.classify(metrics)  # Returns DrivingStyle enum

# Get description
description = classifier.get_style_description(style)
```

### 4. Threshold Adapter (`threshold_adapter.py`)

Adapts safety thresholds based on driver profile with 1.5x safety margin.

**Adapted Thresholds:**
- TTC threshold: Based on reaction time × 1.5
- Following distance: Based on driving style
- Alert sensitivity: Based on risk tolerance

**Usage:**
```python
from src.profiling import ThresholdAdapter

adapter = ThresholdAdapter(config)

# Adapt thresholds
adapted = adapter.adapt_thresholds(metrics, driving_style)

# Get current thresholds
thresholds = adapter.get_adapted_thresholds()
# Returns: {'ttc_threshold': 2.5, 'following_distance': 28.0, 'alert_sensitivity': 0.75}
```

### 5. Report Generator (`report_generator.py`)

Generates comprehensive driver behavior reports with scores and recommendations.

**Scores Calculated:**
- Safety score (0-100): Based on near-misses, risk tolerance, following distance
- Attention score (0-100): Based on reaction time consistency
- Eco-driving score (0-100): Based on speed variance, lane changes

**Usage:**
```python
from src.profiling import DriverReportGenerator

generator = DriverReportGenerator(config)

# Generate report
report = generator.generate_report(metrics, driving_style, driver_id)

# Export as text
text_report = generator.export_report_text(report)
print(text_report)
```

### 6. Profile Manager (`profile_manager.py`)

Manages driver profiles with persistence to JSON files.

**Features:**
- Driver identification from face
- Profile creation and loading
- Session tracking and profile updates
- Profile persistence to disk
- Multiple driver support

**Usage:**
```python
from src.profiling import ProfileManager

manager = ProfileManager(config)

# Identify driver from frame
driver_id = manager.identify_driver(frame)

# Start session
manager.start_session(driver_id, timestamp)

# Update metrics during session
tracker = manager.get_metrics_tracker()
tracker.update(...)

# End session (automatically updates profile)
manager.end_session(timestamp)

# Get adapted thresholds for driver
thresholds = manager.get_adapted_thresholds(driver_id)

# Get profile
profile = manager.get_profile(driver_id)
```

## Profile Data Structure

Driver profiles are stored as JSON files in the `profiles/` directory:

```json
{
  "driver_id": "driver_abc123",
  "face_embedding": [0.123, -0.456, ...],
  "total_distance": 15000.5,
  "total_time": 45000.0,
  "driving_style": "normal",
  "avg_reaction_time": 1.2,
  "avg_following_distance": 28.5,
  "avg_lane_change_freq": 4.2,
  "avg_speed": 22.5,
  "risk_tolerance": 0.45,
  "safety_score": 85.0,
  "attention_score": 88.0,
  "eco_score": 82.0,
  "session_count": 15,
  "last_updated": "2024-11-16T10:30:00",
  "created_at": "2024-10-01T08:00:00"
}
```

## GUI Integration

The `DriverProfileDock` widget displays driver profile information in the GUI:

**Features:**
- Current driver information display
- Performance scores with progress bars
- Score trends over time (line graph)
- Detailed behavior statistics
- Profile reset and delete controls

**Usage:**
```python
from src.gui.widgets import DriverProfileDock

# Create dock widget
profile_dock = DriverProfileDock()

# Update with profile data
profile_dock.update_profile(profile_data)

# Update trends
profile_dock.update_trends(history)

# Connect signals
profile_dock.profile_reset_requested.connect(on_reset)
profile_dock.profile_deleted_requested.connect(on_delete)
```

## Configuration

Example configuration in `configs/default.yaml`:

```yaml
driver_profiling:
  enabled: true
  profiles_dir: "profiles/"
  auto_save: true
  
  face_recognition:
    recognition_threshold: 0.6
    embedding_size: 128
  
  metrics_tracker:
    # No specific config needed
  
  style_classifier:
    # Uses default thresholds
  
  threshold_adapter:
    base_ttc_threshold: 2.0
    base_following_distance: 25.0
    base_alert_sensitivity: 0.7
  
  report_generator:
    # No specific config needed
```

## Integration with SENTINEL System

The profiling system integrates with the main SENTINEL system:

1. **Driver Identification**: At session start, identify driver from interior camera
2. **Metrics Tracking**: Throughout session, track behavior metrics
3. **Threshold Adaptation**: Use adapted thresholds in risk assessment
4. **Profile Updates**: At session end, update driver profile
5. **Report Generation**: Generate behavior reports for analysis

## Safety Considerations

- **1.5x Safety Margin**: All adapted thresholds include a 1.5x safety margin
- **Threshold Clamping**: Thresholds are clamped to safe ranges
- **Default Fallback**: Unknown drivers use default (conservative) thresholds
- **Privacy**: Face embeddings are hashed, no raw images stored

## Testing

Run tests for the profiling module:

```bash
pytest tests/unit/test_profiling.py
```

## Future Enhancements

- Deep learning-based face recognition (FaceNet, ArcFace)
- More sophisticated driving style classification (ML-based)
- Cloud synchronization of driver profiles
- Fleet-wide driver analytics
- Personalized coaching recommendations
