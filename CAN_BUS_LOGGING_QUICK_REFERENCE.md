# CAN Bus Logging Quick Reference

## Module Overview

The CAN bus module provides vehicle telemetry integration and control commands via SocketCAN on Linux.

## Logger Configuration

### Log Levels

| Module | Level | Rationale |
|--------|-------|-----------|
| `src.can.interface` | INFO | Connection events, errors |
| `src.can.dbc_parser` | INFO | DBC file loading, parsing |
| `src.can.telemetry_reader` | DEBUG | High-frequency updates (100 Hz) |
| `src.can.command_sender` | INFO | Safety-critical commands |
| `src.can.logger` | INFO | CAN traffic logging operations |

### Production vs Development

**Production (INFO level):**
- ~30 log entries/minute
- Connection events only
- Control commands logged
- Minimal performance impact

**Development (DEBUG level):**
- ~700 log entries/second
- All telemetry updates
- All CAN frames logged
- Useful for debugging

## Common Log Messages

### CANInterface

```python
# Connection
INFO: "Initializing CAN interface on can0"
INFO: "Connected to CAN bus on can0"
INFO: "Disconnected from CAN bus on can0"
INFO: "Attempting to reconnect to CAN bus on can0"

# Frame I/O (DEBUG)
DEBUG: "Sent CAN frame: ID=0x100, data=0102030405060708"
DEBUG: "Received CAN frame: ID=0x200, data=1122334455667788"

# Errors
ERROR: "Failed to connect to CAN bus: [Errno 19] No such device"
ERROR: "Failed to send CAN frame: connection lost"
WARNING: "Cannot send frame: not connected to CAN bus"
```

### DBCParser

```python
# Loading
INFO: "Loading DBC file: configs/vehicle.dbc"
INFO: "Loaded 6 messages from DBC file"

# Errors
ERROR: "DBC file not found: configs/vehicle.dbc"
ERROR: "Failed to parse DBC file: invalid format"
ERROR: "Failed to decode signal Speed: invalid data length"
```

### TelemetryReader

```python
# Lifecycle
INFO: "Initialized telemetry reader at 100.0 Hz"
INFO: "Started telemetry reader thread"
INFO: "Telemetry reading loop started"
INFO: "Telemetry reading loop stopped"
INFO: "Stopped telemetry reader thread"

# Updates (DEBUG - 100 Hz)
DEBUG: "Updated speed: 25.50 m/s"
DEBUG: "Updated steering: 0.125 rad"
DEBUG: "Updated brake: 2.50 bar"
DEBUG: "Updated throttle: 0.65"
DEBUG: "Updated gear: 4"
DEBUG: "Updated turn signal: left"

# Errors
ERROR: "Error in telemetry reading loop: connection lost"
```

### CommandSender

```python
# Safety
WARNING: "Control commands ENABLED - vehicle can be controlled via CAN bus"
INFO: "Control commands DISABLED - commands will be logged but not sent"

# Commands
INFO: "Sent brake command: 0.75"
INFO: "Sent steering command: 0.250 rad"
WARNING: "Sending EMERGENCY STOP command"
INFO: "Releasing vehicle control"

# Watchdog
WARNING: "Brake command watchdog timeout - resetting"
WARNING: "Steering command watchdog timeout - resetting"

# Errors
ERROR: "Failed to encode brake command message (ID: 0x700)"
ERROR: "Failed to send brake command"

# Debug (when disabled)
DEBUG: "Brake command not sent (control disabled): 0.50"
DEBUG: "Steering command not sent (control disabled): 0.250 rad"
```

### CANLogger

```python
# Lifecycle
INFO: "CAN Logger initialized: log_file=logs/can_traffic.log, enabled=True"
INFO: "Started CAN logging thread"
INFO: "CAN logging loop started"
INFO: "CAN logging loop stopped"
INFO: "Stopped CAN logging. Total messages: 15234"

# Playback
INFO: "CAN Playback initialized: log_file=logs/can_traffic.log"
INFO: "Loaded 15234 CAN messages from log"

# Errors
ERROR: "Failed to start CAN logging: disk full"
ERROR: "Failed to log CAN message: I/O error"
ERROR: "Failed to load CAN log: file corrupted"
```

## Performance Impact

### Log Volume Estimates

| Configuration | Entries/Second | Entries/Minute | Impact |
|--------------|----------------|----------------|--------|
| INFO (Production) | ~0.5 | ~30 | Minimal |
| DEBUG (Development) | ~700 | ~42,000 | Moderate |

### Recommendations

1. **Production**: Use INFO level
   - Minimal overhead
   - Captures important events
   - Suitable for 30+ FPS operation

2. **Development**: Use DEBUG level
   - Full telemetry visibility
   - Useful for integration testing
   - May impact performance slightly

3. **Troubleshooting**: Use DEBUG level temporarily
   - Enable for specific debugging sessions
   - Disable after issue resolved
   - Monitor disk space

## Verification

```bash
# Run verification script
python3 scripts/verify_can_logging_simple.py

# Check log output
tail -f logs/sentinel.log | grep "src.can"

# Count CAN log entries
grep "src.can" logs/sentinel.log | wc -l
```

## Log File Locations

- **Main log**: `logs/sentinel.log` - All system logs
- **Error log**: `logs/errors.log` - Errors only
- **CAN traffic**: `logs/can_traffic.log` - Raw CAN messages (optional)

## Integration Example

```python
import logging
from src.can import CANInterface, DBCParser, TelemetryReader

# Logger is automatically configured from logging.yaml
logger = logging.getLogger(__name__)

# Create CAN interface
can_interface = CANInterface(channel='can0')
can_interface.connect()  # Logs: "Connected to CAN bus on can0"

# Load DBC file
dbc_parser = DBCParser('configs/vehicle.dbc')  # Logs: "Loaded 6 messages..."

# Start telemetry reader
telemetry = TelemetryReader(can_interface, dbc_parser)
telemetry.start()  # Logs: "Started telemetry reader thread"

# Get telemetry (updates logged at DEBUG level)
data = telemetry.get_telemetry()
logger.info(f"Current speed: {data.speed:.2f} m/s")
```

## Safety Notes

1. **Control commands are logged at INFO level** - Always visible for audit trail
2. **Watchdog timeouts are logged as warnings** - Indicates potential safety issues
3. **Emergency stops are logged as warnings** - Critical safety events
4. **Control enable/disable logged** - State changes tracked

## Troubleshooting

### No CAN log entries

1. Check logging configuration: `configs/logging.yaml`
2. Verify logger level: Should be INFO or DEBUG
3. Check log file permissions: `logs/` directory writable

### Too many log entries

1. Change telemetry_reader level from DEBUG to INFO
2. Disable frame-level logging (DEBUG → INFO for interface)
3. Increase log rotation size in `logging.yaml`

### Missing telemetry updates

1. Enable DEBUG level for `src.can.telemetry_reader`
2. Check CAN connection status in logs
3. Verify DBC file loaded correctly

## Summary

✓ All CAN modules have comprehensive logging  
✓ Appropriate log levels for real-time performance  
✓ Safety-critical operations always logged  
✓ High-frequency operations use DEBUG level  
✓ Production-ready with minimal overhead
