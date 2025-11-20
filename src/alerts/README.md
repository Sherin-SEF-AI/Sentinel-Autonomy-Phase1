# Alert & Action System

The Alert & Action System is responsible for generating, managing, and dispatching context-aware safety alerts based on risk assessment and driver state.

## Overview

The system integrates four key components:
1. **Alert Generation**: Creates alerts based on risk scores and driver awareness
2. **Alert Suppression**: Prevents alert fatigue through intelligent filtering
3. **Alert Logging**: Records all alerts for analysis and debugging
4. **Multi-Modal Dispatch**: Delivers alerts through visual, audio, and haptic channels

## Architecture

```
RiskAssessment + DriverState
         ↓
   AlertGenerator
         ↓
  AlertSuppressor
         ↓
    AlertLogger
         ↓
  AlertDispatcher
         ↓
   Visual/Audio/Haptic Output
```

## Components

### AlertGenerator

Generates alerts based on contextual risk scores and driver state.

**Alert Levels:**
- **CRITICAL** (risk > 0.9): Visual flash + audio alarm + haptic pulse
- **WARNING** (risk > 0.7, driver unaware): Visual HUD + audio beep
- **INFO** (risk > 0.5, low cognitive load): Visual display only

**Cognitive Load Adaptation:**
- INFO alerts only generated when driver cognitive load < 0.7
- Cognitive load = 1.0 - (readiness_score / 100)

### AlertSuppressor

Prevents alert fatigue through intelligent suppression logic.

**Features:**
- Duplicate suppression within 5-second window per hazard
- Maximum 2 simultaneous alerts
- Priority-based filtering (critical > warning > info)

### AlertLogger

Logs all alerts with timestamp and context for analysis.

**Capabilities:**
- Dedicated alert log file (`logs/alerts.log`)
- In-memory history for quick access
- Statistics calculation (total, by urgency, unique hazards)
- Filtering by urgency or hazard ID

### AlertDispatcher

Dispatches alerts through multiple modalities.

**Modalities:**
- **Visual**: On-screen display with color coding and positioning
- **Audio**: Sound playback (alarm for critical, beep for warning)
- **Haptic**: Vibration patterns (placeholder for hardware integration)

**Visual Alert Properties:**
- Display duration: 3 seconds (configurable)
- Flash rate: 2 Hz for critical alerts
- Color coding: Red (critical), Orange (warning), Blue (info)
- Position: Center (critical), Top (warning), Bottom (info)

## Usage

### Basic Usage

```python
from src.alerts import AlertSystem
from src.core.config import ConfigManager

# Initialize
config_manager = ConfigManager('configs/default.yaml')
alert_system = AlertSystem(config_manager.get_section('alerts'))

# Process risks
alerts = alert_system.process(risk_assessment, driver_state)

# Get active alerts
active = alert_system.get_active_alerts()

# Get statistics
stats = alert_system.get_alert_statistics()
```

### Configuration

```yaml
alerts:
  suppression:
    duplicate_window: 5.0      # Seconds
    max_simultaneous: 2        # Maximum concurrent alerts
  
  escalation:
    critical_threshold: 0.9    # Risk score threshold
    high_threshold: 0.7
    medium_threshold: 0.5
  
  modalities:
    visual:
      display_duration: 3.0    # Seconds
      flash_rate: 2            # Hz for critical alerts
    audio:
      volume: 0.8
      critical_sound: "sounds/alarm.wav"
      warning_sound: "sounds/beep.wav"
    haptic:
      enabled: false           # Hardware integration required
```

## Alert Generation Logic

### Critical Alerts (risk > 0.9)
- Generated regardless of driver awareness
- All modalities activated (visual + audio + haptic)
- Highest priority, never suppressed

### Warning Alerts (risk > 0.7)
- Only generated when driver is NOT aware of hazard
- Visual + audio modalities
- Medium priority

### Info Alerts (risk > 0.5)
- Only generated when cognitive load < 0.7
- Visual modality only
- Lowest priority, may be suppressed

## Alert Message Format

Messages include:
- Urgency level (CRITICAL/WARNING/INFO)
- Hazard type (vehicle, pedestrian, cyclist, etc.)
- Spatial zone (front, left, rear, etc.)
- Time-to-collision (TTC)
- Driver awareness status (for warnings)

**Examples:**
```
CRITICAL: pedestrian ahead in front zone! TTC: 0.8s
WARNING: vehicle in left zone (not looking). TTC: 1.5s
INFO: cyclist detected in front-right zone. TTC: 3.0s
```

## Alert Suppression Rules

1. **Duplicate Suppression**: Same hazard within 5 seconds
2. **Simultaneous Limit**: Maximum 2 alerts at once
3. **Priority Ordering**: Critical > Warning > Info
4. **History Cleanup**: Expired entries removed automatically

## Alert Logging

All alerts are logged with:
- Timestamp (Unix time + ISO format)
- Urgency level
- Message content
- Hazard ID
- Modalities used
- Context (driver state, risk scores)

**Log Format:**
```json
{
  "timestamp": 1700000000.123,
  "datetime": "2024-11-15T10:30:00.123",
  "urgency": "warning",
  "message": "WARNING: vehicle in left zone (not looking). TTC: 1.5s",
  "hazard_id": 42,
  "modalities": ["visual", "audio"],
  "dismissed": false,
  "context": {
    "driver_readiness": 65.0,
    "driver_attention_zone": "front",
    "num_hazards": 3
  }
}
```

## Testing

Run tests:
```bash
pytest tests/test_alerts.py -v
```

Run example:
```bash
python3 examples/alerts_example.py
```

## Performance

- Alert generation: < 1ms
- Suppression logic: < 1ms
- Total processing: < 5ms
- No blocking operations

## Future Enhancements

1. **Adaptive Timing**: Adjust alert timing based on driver reaction time
2. **Personalization**: Learn driver preferences and adapt alert style
3. **Haptic Integration**: Connect to steering wheel or seat haptic actuators
4. **Voice Alerts**: Add text-to-speech for critical situations
5. **Alert Effectiveness**: Track driver response to optimize alert strategy

## Requirements Mapping

- **Requirement 7.1**: Alert generation with urgency levels
- **Requirement 7.2**: Multi-modal dispatch (visual, audio, haptic)
- **Requirement 7.3**: Cognitive load adaptation
- **Requirement 7.4**: Alert timing adaptation
- **Requirement 7.5**: Alert suppression logic
- **Requirement 7.6**: Alert logging with context
