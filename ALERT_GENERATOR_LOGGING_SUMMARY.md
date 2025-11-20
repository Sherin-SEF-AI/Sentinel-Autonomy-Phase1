# Alert Generator Logging Implementation Summary

## Overview
Comprehensive logging has been added to `src/alerts/generator.py` to track alert generation, risk evaluation, and performance metrics in the SENTINEL alert system.

## Changes Made

### 1. Logging Setup
- Added `import logging` and module-level logger: `logger = logging.getLogger(__name__)`
- Logger name: `src.alerts.generator`

### 2. Initialization Logging
**Location:** `AlertGenerator.__init__()`
- **INFO level**: Logs initialization with all threshold configurations
- **Metrics tracked**: critical_threshold, high_threshold, medium_threshold, cognitive_load_threshold
- **Statistics added**: Total alerts generated and breakdown by urgency level

### 3. Alert Generation Logging
**Location:** `generate_alerts()`
- **DEBUG level (entry)**: Logs start of alert generation with input parameters
  - Number of risks to evaluate
  - Driver readiness score
  - Calculated cognitive load
- **INFO level**: Logs each alert generated with full details
  - Alert urgency level
  - Hazard ID
  - Modalities (visual, audio, haptic)
  - Complete alert message
- **DEBUG level (exit)**: Logs completion with performance metrics
  - Number of alerts generated
  - Processing duration in milliseconds
- **WARNING level**: Logs performance issues when exceeding 5ms target

### 4. Risk Evaluation Logging
**Location:** `_generate_alert_for_risk()`
- **DEBUG level**: Logs each risk evaluation with:
  - Hazard ID and type
  - Contextual risk score
  - Driver awareness state
  - Time-to-collision (TTC)
- **DEBUG level**: Logs alert triggering decisions:
  - Critical alerts: When contextual score exceeds threshold
  - Warning alerts: When score exceeds threshold AND driver unaware
  - Info alerts: When score exceeds threshold AND cognitive load is low
- **DEBUG level**: Logs alert suppression reasons:
  - Warning suppressed when driver is aware
  - Info suppressed when cognitive load is high

### 5. Message Formatting Logging
**Location:** `_format_message()`
- **DEBUG level**: Logs formatted alert messages for traceability

### 6. Statistics Method
**Added:** `get_statistics()` method to retrieve:
- Total alerts generated
- Breakdown by urgency level (critical, warning, info)

## Logging Configuration

### Updated `configs/logging.yaml`
```yaml
src.alerts.generator:
  level: DEBUG
  handlers: [file_alerts, file_all]
  propagate: false
```

**Log destinations:**
- `logs/alerts.log` - Alert-specific logs
- `logs/sentinel.log` - System-wide logs
- Console output at INFO level

## Performance Considerations

### Target Latency: 5ms
- Alert generation is lightweight (simple threshold checks)
- Logging overhead minimal with DEBUG level in production
- Performance warnings logged when exceeding target

### Log Volume Management
- DEBUG logs provide detailed decision tracing for development
- INFO logs capture key events (alerts generated)
- WARNING logs flag performance issues
- Production can use INFO level to reduce volume

## Key Logging Patterns

### 1. Alert Generation Flow
```
DEBUG: Alert generation started: num_risks=3, driver_readiness=85.0, cognitive_load=0.15
DEBUG: Evaluating risk: hazard_id=42, type=vehicle, contextual_score=0.95, driver_aware=False, ttc=1.2s
DEBUG: Critical alert triggered: contextual_score=0.95 > threshold=0.9
DEBUG: Alert message formatted: urgency=critical, message='CRITICAL: vehicle ahead in front zone! TTC: 1.2s'
INFO: Alert generated: urgency=critical, hazard_id=42, modalities=['visual', 'audio', 'haptic'], message='CRITICAL: vehicle ahead in front zone! TTC: 1.2s'
DEBUG: Alert generation completed: alerts_generated=1, duration=2.34ms
```

### 2. Alert Suppression
```
DEBUG: Evaluating risk: hazard_id=15, type=pedestrian, contextual_score=0.75, driver_aware=True, ttc=3.5s
DEBUG: Warning alert suppressed: driver_aware=True
```

### 3. Performance Warning
```
WARNING: Alert generation exceeded target: duration=6.23ms, target=5.0ms
```

## Integration Points

### Upstream Dependencies
- **RiskAssessment**: Provides top risks with contextual scores
- **DriverState**: Provides readiness score for cognitive load calculation

### Downstream Consumers
- **AlertSuppressor**: Receives generated alerts for deduplication
- **AlertDispatcher**: Receives alerts for multi-modal delivery
- **AlertLogger**: Logs alerts to persistent storage

## Testing Recommendations

### Unit Tests
```python
def test_alert_generation_logging(caplog):
    """Test that alert generation logs appropriately."""
    generator = AlertGenerator(config)
    
    with caplog.at_level(logging.DEBUG):
        alerts = generator.generate_alerts(risks, driver_state)
    
    assert "Alert generation started" in caplog.text
    assert "Alert generated" in caplog.text
    assert "Alert generation completed" in caplog.text
```

### Performance Tests
```python
def test_alert_generation_performance():
    """Test alert generation meets latency target."""
    generator = AlertGenerator(config)
    
    start = time.time()
    alerts = generator.generate_alerts(risks, driver_state)
    duration_ms = (time.time() - start) * 1000
    
    assert duration_ms < 5.0  # 5ms target
```

## Monitoring and Observability

### Key Metrics to Track
1. **Alert generation rate**: Alerts per second
2. **Alert distribution**: Breakdown by urgency level
3. **Processing latency**: P50, P95, P99 durations
4. **Suppression rate**: Percentage of risks not generating alerts

### Log Analysis Queries
```bash
# Count alerts by urgency
grep "Alert generated" logs/alerts.log | grep -o "urgency=[a-z]*" | sort | uniq -c

# Find performance issues
grep "exceeded target" logs/alerts.log

# Track cognitive load impact
grep "cognitive_load" logs/alerts.log | grep -o "cognitive_load=[0-9.]*"
```

## Benefits

1. **Decision Traceability**: Every alert decision is logged with reasoning
2. **Performance Monitoring**: Real-time tracking of processing latency
3. **Debugging Support**: Detailed logs for troubleshooting alert logic
4. **Statistics Tracking**: Built-in counters for system health monitoring
5. **Production Ready**: Configurable log levels for different environments

## Status
✅ **Complete** - Logging fully implemented and tested
- All key operations logged
- Performance metrics tracked
- Configuration updated
- No diagnostic errors
