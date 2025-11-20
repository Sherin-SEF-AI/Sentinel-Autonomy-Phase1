# Driver Profiling Module - Logging Summary

## Overview

The Driver Profiling Module provides comprehensive logging for driver identification, behavior tracking, and personalized safety threshold adaptation. All components use structured logging with appropriate levels for real-time performance monitoring.

## Logging Configuration

### Logger Hierarchy

```yaml
src.profiling:                          # Root profiling logger (INFO)
  ├── src.profiling.face_recognition    # Face recognition (INFO)
  ├── src.profiling.metrics_tracker     # Metrics tracking (DEBUG)
  ├── src.profiling.style_classifier    # Style classification (INFO)
  ├── src.profiling.threshold_adapter   # Threshold adaptation (INFO)
  ├── src.profiling.report_generator    # Report generation (INFO)
  └── src.profiling.profile_manager     # Profile management (INFO)
```

### Log Levels

- **DEBUG**: Detailed metrics updates, threshold calculations, feature scoring
- **INFO**: Driver identification, session events, profile updates, classifications
- **WARNING**: Missing data, identification failures, insufficient metrics
- **ERROR**: Face detection failures, profile loading errors, file I/O errors

## Component Logging Details

### 1. Face Recognition System (`face_recognition.py`)

**Key Events Logged:**

```python
# Initialization
logger.info(f"FaceRecognitionSystem initialized with threshold={self.recognition_threshold}")

# Face detection
logger.info("Face detector initialized successfully")
logger.warning(f"Face detection model not found at {model_path}, using fallback")

# Driver matching
logger.info(f"Driver matched: {best_match_id} (similarity={best_similarity:.3f})")
logger.info(f"No driver match found (best similarity={best_similarity:.3f})")

# Errors
logger.error(f"Failed to initialize face detector: {e}")
logger.error(f"Failed to extract face embedding: {e}")
```

**Performance Considerations:**
- Face detection: ~10-20ms per frame
- Embedding extraction: ~5-10ms
- Matching against N profiles: ~1ms per profile

### 2. Metrics Tracker (`metrics_tracker.py`)

**Key Events Logged:**

```python
# Session management
logger.info("MetricsTracker initialized")
logger.info(f"Metrics tracking session started at {timestamp}")
logger.info(f"Metrics tracking session ended. Duration: {self.session_duration:.1f}s")

# Metric updates (DEBUG level)
logger.debug(f"Lane change detected at {timestamp}")

# Alert tracking
logger.debug(f"Alert {alert_id} recorded at {timestamp}")
logger.info(f"Reaction time recorded: {reaction_time:.3f}s for action '{action_type}'")

# Near-miss events
logger.info(f"Near-miss event recorded: TTC={ttc:.2f}s, risk={risk_score:.3f}")

# Reset
logger.info("Metrics tracker reset")
```

**Metrics Tracked:**
- Reaction time: Alert → driver action latency
- Following distance: Distance to vehicle ahead
- Lane change frequency: Changes per hour
- Speed profile: Mean, max, variance
- Risk tolerance: Behavior during high-risk situations

### 3. Driving Style Classifier (`style_classifier.py`)

**Key Events Logged:**

```python
# Initialization
logger.info("DrivingStyleClassifier initialized")

# Classification
logger.info(f"Driving style classified: {smoothed_style.value} (score={weighted_score:.3f})")
logger.warning("Insufficient data for style classification")

# Reset
logger.info("Driving style classifier reset")
```

**Classification Categories:**
- **Aggressive**: Fast reactions, short following distances, high risk tolerance
- **Normal**: Balanced behavior, moderate risk tolerance
- **Cautious**: Longer distances, fewer lane changes, low risk tolerance
- **Unknown**: Insufficient data

**Feature Weights:**
- Reaction time: 20%
- Following distance: 25%
- Lane change frequency: 15%
- Speed variance: 15%
- Risk tolerance: 25%

### 4. Threshold Adapter (`threshold_adapter.py`)

**Key Events Logged:**

```python
# Initialization
logger.info(f"ThresholdAdapter initialized with base TTC={self.base_ttc_threshold}s, "
           f"following_distance={self.base_following_distance}m, "
           f"alert_sensitivity={self.base_alert_sensitivity}")

# Threshold adaptation
logger.info(f"Thresholds adapted: TTC={adapted['ttc_threshold']:.2f}s, "
           f"following_distance={adapted['following_distance']:.1f}m, "
           f"alert_sensitivity={adapted['alert_sensitivity']:.2f}")

# Detailed calculations (DEBUG)
logger.debug(f"TTC threshold adapted from {self.base_ttc_threshold:.2f}s to {adapted_ttc:.2f}s "
            f"based on reaction time {reaction_time:.2f}s")
logger.debug(f"Following distance adapted from {self.base_following_distance:.1f}m to {adapted_distance:.1f}m "
            f"for {driving_style.value} style")

# Reset
logger.info("Thresholds reset to defaults")
```

**Adaptation Rules:**
- TTC threshold = reaction_time × 1.5 (safety margin)
- Following distance: Style-based multiplier (0.8-1.3×)
- Alert sensitivity: Inverse of risk tolerance

### 5. Report Generator (`report_generator.py`)

**Key Events Logged:**

```python
# Initialization
logger.info("DriverReportGenerator initialized")

# Report generation
logger.info(f"Report generated for {driver_id}: "
           f"Safety={safety_score:.1f}, Attention={attention_score:.1f}, Eco={eco_score:.1f}")
```

**Report Components:**
- **Safety Score** (0-100): Based on near-misses, risk tolerance, following distance
- **Attention Score** (0-100): Based on reaction time consistency
- **Eco Score** (0-100): Based on speed variance, lane changes
- **Recommendations**: Personalized improvement suggestions
- **Trends**: Score changes over time

### 6. Profile Manager (`profile_manager.py`)

**Key Events Logged:**

```python
# Initialization
logger.info(f"ProfileManager initialized with {len(self.profiles)} profiles")
logger.info(f"Loaded profile: {profile.driver_id}")

# Driver identification
logger.info(f"Driver identified: {driver_id} (similarity={similarity:.3f})")
logger.info(f"New driver profile created: {new_driver_id}")
logger.warning(f"Profile {driver_id} not found")

# Session management
logger.info(f"Session started for driver {driver_id}")
logger.info(f"Session ended for driver {self.active_profile_id}")

# Profile updates
logger.info(f"Profile updated for {self.active_profile_id}: "
           f"sessions={profile.session_count}, "
           f"safety={profile.safety_score:.1f}, "
           f"style={profile.driving_style}")

# File operations
logger.info(f"Profile saved: {driver_id}")
logger.info(f"Profile deleted: {driver_id}")

# Errors
logger.error(f"Failed to load profile from {profile_file}: {e}")
logger.error(f"Failed to save profile {driver_id}: {e}")
```

**Profile Storage:**
- Location: `profiles/{driver_id}.json`
- Format: JSON with face embedding, metrics, scores
- Auto-save: Configurable after each session

## Integration with SENTINEL System

### Data Flow

```
Camera Frame → Face Recognition → Driver Identification
                                         ↓
                                  Profile Manager
                                         ↓
Vehicle Telemetry → Metrics Tracker → Session Metrics
                                         ↓
                              Style Classifier → Driving Style
                                         ↓
                              Threshold Adapter → Personalized Thresholds
                                         ↓
                              Report Generator → Driver Report
                                         ↓
                              Profile Manager → Updated Profile
```

### Performance Impact

**Target Latencies:**
- Face recognition: <20ms (once per session start)
- Metrics update: <1ms (per frame)
- Style classification: <5ms (end of session)
- Threshold adaptation: <2ms (end of session)
- Report generation: <10ms (end of session)

**Total Session Overhead:**
- Session start: ~20ms (face recognition)
- Per-frame: ~1ms (metrics tracking)
- Session end: ~20ms (classification + reporting)

## Verification

Run the verification script to test logging:

```bash
python scripts/verify_profiling_logging.py
```

**Expected Output:**
- All components initialize successfully
- Face recognition detects/matches faces
- Metrics tracker records session data
- Style classifier determines driving style
- Threshold adapter personalizes settings
- Report generator creates comprehensive reports
- Profile manager saves/loads profiles

**Log Files:**
- `logs/sentinel.log`: All profiling events
- `logs/errors.log`: Error-level events only

## Configuration Example

```yaml
# configs/default.yaml
driver_profiling:
  enabled: true
  profiles_dir: "profiles/"
  auto_save: true
  
  face_recognition:
    recognition_threshold: 0.6
    embedding_size: 128
  
  threshold_adapter:
    base_ttc_threshold: 2.0
    base_following_distance: 25.0
    base_alert_sensitivity: 0.7
```

## Troubleshooting

### Common Issues

1. **No face detected**
   - Check camera positioning
   - Verify lighting conditions
   - Review face detector logs

2. **Insufficient data for classification**
   - Need minimum 3 reaction time samples
   - Need minimum 10 following distance samples
   - Extend session duration

3. **Profile not saving**
   - Check profiles directory permissions
   - Verify disk space
   - Review file I/O error logs

4. **Threshold adaptation not working**
   - Verify profile has sufficient session data
   - Check base threshold configuration
   - Review adaptation calculation logs

## Best Practices

1. **Log Level Selection**
   - Use INFO for production monitoring
   - Use DEBUG for development/troubleshooting
   - Avoid DEBUG in real-time operation (performance impact)

2. **Performance Monitoring**
   - Track face recognition latency
   - Monitor metrics update frequency
   - Measure profile save duration

3. **Data Privacy**
   - Face embeddings are hashed for IDs
   - No raw images stored in profiles
   - Profiles stored locally only

4. **Profile Management**
   - Regular profile backups recommended
   - Periodic cleanup of inactive profiles
   - Version control for profile format changes

## Summary

The Driver Profiling Module provides comprehensive logging for:
- ✓ Driver identification via face recognition
- ✓ Behavior metrics tracking (reaction time, following distance, etc.)
- ✓ Driving style classification (aggressive/normal/cautious)
- ✓ Personalized safety threshold adaptation
- ✓ Comprehensive driver reports with scores and recommendations
- ✓ Persistent profile storage and management

All logging follows SENTINEL's performance requirements with minimal overhead (<1ms per frame) and detailed session-level reporting.
