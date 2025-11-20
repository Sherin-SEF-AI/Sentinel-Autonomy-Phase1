# Alerts Panel Logging Quick Reference

## Logger Configuration

```python
# Logger name
logger = logging.getLogger('sentinel.gui.alerts_panel')

# Configuration in configs/logging.yaml
sentinel.gui.alerts_panel:
  level: DEBUG
  handlers: [file_all]
  propagate: false
```

## Key Logging Points

### Initialization
```python
# DEBUG: Component setup
self.logger.debug("AlertsPanel initialization started")
self.logger.debug("Audio settings initialized: enabled={enabled}, volume={volume}")
self.logger.debug("Flash timer initialized for critical alerts")

# INFO: Completion
self.logger.info("AlertsPanel initialized successfully")
```

### Alert Addition
```python
# DEBUG: Processing steps
self.logger.debug(f"Adding alert: urgency={urgency}, hazard_id={hazard_id}")
self.logger.debug(f"Statistics updated: total={total}, critical={critical}")
self.logger.debug(f"Alert entry created: id={id}, history_size={size}")

# INFO: Success
self.logger.info(f"Alert added successfully: urgency={urgency}, message='{message}'")
```

### Audio Playback
```python
# DEBUG: Audio operations
self.logger.debug(f"Attempting to play alert sound: urgency={urgency}")
self.logger.debug("Audio alert skipped: audio disabled")

# INFO: Playback
self.logger.info(f"Playing {urgency} alert sound from {sound_file}")

# WARNING: Missing resources
self.logger.warning(f"No audio player found for urgency: {urgency}")

# ERROR: Failures
self.logger.error(f"Failed to play sound file {sound_file}: {e}")
```

### Critical Effects
```python
# DEBUG: Effect triggers
self.logger.debug("Triggering critical alert visual effects")
self.logger.debug("Flash timer started: interval=200ms")
self.logger.debug("Window brought to front and activated")

# INFO: Success
self.logger.info("Critical alert effects triggered successfully")
```

### User Interactions
```python
# INFO: State changes
self.logger.info("Audio alerts muted")
self.logger.info("Audio alerts unmuted")
self.logger.info(f"Alert filter changed: {filter_text}")

# DEBUG: Adjustments
self.logger.debug(f"Volume changed to {value}%")
```

### Display Management
```python
# DEBUG: Refresh operations
self.logger.debug(f"Refreshing display with filter: {filter_text}")
self.logger.debug(f"Display refreshed: {filtered_count}/{total} alerts shown")

# INFO: History operations
self.logger.info(f"Clearing alert history: {count} alerts")
self.logger.info(f"Alert history cleared: {count} alerts removed")
```

### Export Operations
```python
# INFO: Export start
self.logger.info(f"Exporting alert log to file: {filename}")

# INFO: Success
self.logger.info(f"Alert log exported successfully: {filename}, {count} alerts")

# ERROR: Failures
self.logger.error(f"Failed to export alert log to {filename}: {e}", exc_info=True)
```

### False Positive Marking
```python
# DEBUG: Operation start
self.logger.debug(f"Marking alert as false positive: id={id}")

# INFO: Success
self.logger.info(f"Alert marked as false positive: id={id}, total_fp={count}")

# WARNING: Not found
self.logger.warning(f"Alert not found for false positive marking: id={id}")
```

## Log Levels Guide

| Level | Usage | Examples |
|-------|-------|----------|
| DEBUG | Detailed operation steps | Component initialization, display updates, filter changes |
| INFO | State changes, successful operations | Alert added, audio state changed, export completed |
| WARNING | Recoverable issues | Alert not found, missing audio player, window activation failed |
| ERROR | Failures requiring attention | Audio playback failed, export failed |

## Common Log Patterns

### Operation Start-Complete Pattern
```python
self.logger.debug(f"Starting operation: param={value}")
# ... operation code ...
self.logger.info(f"Operation completed: result={result}")
```

### Try-Except with Logging
```python
try:
    # operation
    self.logger.info("Operation successful")
except Exception as e:
    self.logger.error(f"Operation failed: {e}", exc_info=True)
```

### State Transition Logging
```python
old_state = self.current_state
self.current_state = new_state
self.logger.info(f"State changed: {old_state} -> {new_state}")
```

## Verification

```bash
# Run verification script
python scripts/verify_alerts_panel_logging_simple.py

# Expected: 39/39 checks passed
```

## Performance Impact

- **DEBUG level**: ~31 log statements (development only)
- **INFO level**: ~17 log statements (production)
- **Overhead**: < 0.1ms per log statement
- **Total impact**: < 1% of processing time

## Monitoring Tips

### Key Metrics to Track

1. **Alert frequency**: Count of alerts per urgency level
2. **False positive rate**: FP count / total alerts
3. **Audio failures**: Failed playback attempts
4. **Export operations**: Success/failure rate

### Log Analysis Commands

```bash
# Count alerts by urgency
grep "Alert added successfully" logs/sentinel.log | grep -o "urgency=[a-z]*" | sort | uniq -c

# Find audio failures
grep "Failed to play sound" logs/sentinel.log

# Track false positives
grep "marked as false positive" logs/sentinel.log | wc -l

# Monitor export operations
grep "Alert log exported" logs/sentinel.log
```

## Troubleshooting

### No logs appearing
```bash
# Check logger configuration
grep "sentinel.gui.alerts_panel" configs/logging.yaml

# Verify log directory exists
ls -la logs/

# Check log file permissions
ls -la logs/sentinel.log
```

### Too many DEBUG logs
```yaml
# Change to INFO level in configs/logging.yaml
sentinel.gui.alerts_panel:
  level: INFO  # Changed from DEBUG
```

### Missing specific logs
```python
# Verify logger is initialized
self.logger = logging.getLogger('sentinel.gui.alerts_panel')

# Check log level
self.logger.setLevel(logging.DEBUG)
```

## Related Files

- **Source**: `src/gui/widgets/alerts_panel.py`
- **Config**: `configs/logging.yaml`
- **Verification**: `scripts/verify_alerts_panel_logging_simple.py`
- **Documentation**: `ALERTS_PANEL_LOGGING_SUMMARY.md`
