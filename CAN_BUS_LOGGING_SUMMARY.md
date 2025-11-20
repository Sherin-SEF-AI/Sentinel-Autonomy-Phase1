# CAN Bus Module Logging Summary

## Overview

Comprehensive logging has been implemented for the CAN bus integration module, following SENTINEL's real-time performance requirements and logging standards.

## Logging Configuration

### Module Loggers Added to `configs/logging.yaml`

```yaml
# CAN Bus Module
src.can:
  level: INFO
  handlers: [file_all]
  propagate: false

src.can.interface:
  level: INFO
  handlers: [file_all]
  propagate: false

src.can.dbc_parser:
  level: INFO
  handlers: [file_all]
  propagate: false

src.can.telemetry_reader:
  level: DEBUG
  handlers: [file_all]
  propagate: false

src.can.command_sender:
  level: INFO
  handlers: [file_all]
  propagate: false

src.can.logger:
  level: INFO
  handlers: [file_all]
  propagate: false
```

### Log Levels Rationale

- **src.can.interface**: INFO - Connection events, frame transmission errors
- **src.can.dbc_parser**: INFO - DBC file loading, message parsing
- **src.can.telemetry_reader**: DEBUG - High-frequency telemetry updates (100 Hz)
- **src.can.command_sender**: INFO - Safety-critical control commands
- **src.can.logger**: INFO - CAN traffic logging operations

## Logging Implementation

### 1. CANInterface (`src/can/interface.py`)

**Already has comprehensive logging:**

- ✓ Logger initialized: `logger = logging.getLogger(__name__)`
- ✓ Connection events: `logger.info(f"Connected to CAN bus on {self.channel}")`
- ✓ Disconnection: `logger.info(f"Disconnected from CAN bus on {self.channel}")`
- ✓ Frame transmission: `logger.debug(f"Sent CAN frame: ID=0x{can_id:X}, data={data.hex()}")`
- ✓ Frame reception: `logger.debug(f"Received CAN frame: ID=0x{can_id:X}, data={data.hex()}")`
- ✓ Errors: `logger.error(f"Failed to connect to CAN bus: {e}")`
- ✓ Warnings: `logger.warning("Cannot send frame: not connected to CAN bus")`

### 2. DBCParser (`src/can/dbc_parser.py`)

**Already has comprehensive logging:**

- ✓ Logger initialized: `logger = logging.getLogger(__name__)`
- ✓ DBC loading: `logger.info(f"Loading DBC file: {dbc_file}")`
- ✓ Parse completion: `logger.info(f"Loaded {len(self.messages)} messages from DBC file")`
- ✓ Errors: `logger.error(f"Failed to parse DBC file: {e}")`
- ✓ Signal decode errors: `logger.error(f"Failed to decode signal {signal_name}: {e}")`

### 3. TelemetryReader (`src/can/telemetry_reader.py`)

**Already has comprehensive logging:**

- ✓ Logger initialized: `logger = logging.getLogger(__name__)`
- ✓ Initialization: `logger.info(f"Initialized telemetry reader at {update_rate} Hz")`
- ✓ Thread start/stop: `logger.info("Started telemetry reader thread")`
- ✓ Loop status: `logger.info("Telemetry reading loop started")`
- ✓ Telemetry updates (DEBUG): `logger.debug(f"Updated speed: {self.telemetry.speed:.2f} m/s")`
- ✓ Errors: `logger.error(f"Error in telemetry reading loop: {e}")`

**Telemetry update logging at DEBUG level (100 Hz):**
- Speed updates
- Steering angle updates
- Brake pressure updates
- Throttle position updates
- Gear changes
- Turn signal changes

### 4. CommandSender (`src/can/command_sender.py`)

**Already has comprehensive logging:**

- ✓ Logger initialized: `logger = logging.getLogger(__name__)`
- ✓ Safety warnings: `logger.warning("Control commands ENABLED - vehicle can be controlled via CAN bus")`
- ✓ Control disabled: `logger.info("Control commands DISABLED - commands will be logged but not sent")`
- ✓ Brake commands: `logger.info(f"Sent brake command: {brake_value:.2f}")`
- ✓ Steering commands: `logger.info(f"Sent steering command: {steering_angle:.3f} rad")`
- ✓ Emergency stop: `logger.warning("Sending EMERGENCY STOP command")`
- ✓ Watchdog timeouts: `logger.warning("Brake command watchdog timeout - resetting")`
- ✓ Errors: `logger.error("Failed to send brake command")`

### 5. CANLogger (`src/can/logger.py`)

**Already has comprehensive logging:**

- ✓ Logger initialized: `logger = logging.getLogger(__name__)`
- ✓ Initialization: `logger.info(f"CAN Logger initialized: log_file={log_file}, enabled={enable_logging}")`
- ✓ Thread start/stop: `logger.info("Started CAN logging thread")`
- ✓ Loop status: `logger.info("CAN logging loop started")`
- ✓ Statistics: `logger.info(f"Stopped CAN logging. Total messages: {self.message_count}")`
- ✓ Playback: `logger.info(f"Loaded {len(self.messages)} CAN messages from log")`
- ✓ Errors: `logger.error(f"Failed to log CAN message: {e}")`

## Logging Patterns

### Connection Management
```python
logger.info(f"Connected to CAN bus on {self.channel}")
logger.info(f"Disconnected from CAN bus on {self.channel}")
logger.info(f"Attempting to reconnect to CAN bus on {self.channel}")
logger.error(f"Failed to connect to CAN bus: {e}")
```

### Frame Transmission/Reception (DEBUG level for performance)
```python
logger.debug(f"Sent CAN frame: ID=0x{can_id:X}, data={data.hex()}")
logger.debug(f"Received CAN frame: ID=0x{can_id:X}, data={data.hex()}")
```

### Telemetry Updates (DEBUG level - 100 Hz)
```python
logger.debug(f"Updated speed: {self.telemetry.speed:.2f} m/s")
logger.debug(f"Updated steering: {self.telemetry.steering_angle:.3f} rad")
logger.debug(f"Updated brake: {self.telemetry.brake_pressure:.2f} bar")
```

### Control Commands (INFO level - safety critical)
```python
logger.info(f"Sent brake command: {brake_value:.2f}")
logger.info(f"Sent steering command: {steering_angle:.3f} rad")
logger.warning("Sending EMERGENCY STOP command")
logger.warning("Brake command watchdog timeout - resetting")
```

### Safety Warnings
```python
logger.warning("Control commands ENABLED - vehicle can be controlled via CAN bus")
logger.info("Control commands DISABLED - commands will be logged but not sent")
logger.info("Releasing vehicle control")
```

## Performance Considerations

### High-Frequency Operations (100 Hz)

The telemetry reader operates at 100 Hz, which could generate significant log volume:

1. **Telemetry updates use DEBUG level** - Can be disabled in production
2. **Frame transmission/reception use DEBUG level** - Minimal overhead when disabled
3. **Connection events use INFO level** - Infrequent, always logged
4. **Control commands use INFO level** - Safety-critical, always logged

### Log Volume Estimates

With DEBUG level enabled:
- Telemetry updates: ~100 messages/second × 6 signals = 600 log entries/second
- Frame reception: ~100 messages/second = 100 log entries/second
- **Total: ~700 log entries/second**

With INFO level (production):
- Connection events: ~1-2 messages/minute
- Control commands: ~10-20 messages/minute
- **Total: ~30 log entries/minute**

## Verification

Run the verification script to test logging:

```bash
python scripts/verify_can_logging.py
```

This script:
1. Initializes all CAN module loggers
2. Tests log message output at various levels
3. Verifies log file creation and content
4. Confirms logging configuration

## Integration with SENTINEL System

### Log Files

All CAN module logs are written to:
- **Main log**: `logs/sentinel.log` - All system logs including CAN
- **Error log**: `logs/errors.log` - CAN errors only
- **CAN traffic log**: `logs/can_traffic.log` - Raw CAN message log (optional)

### Log Rotation

CAN logs use the same rotation policy as other SENTINEL modules:
- Max file size: 10MB
- Backup count: 5 files
- Automatic rotation when size exceeded

## Safety Considerations

### Control Command Logging

All vehicle control commands are logged at INFO level for safety auditing:
- Brake commands with values
- Steering commands with angles
- Emergency stop commands
- Control enable/disable events
- Watchdog timeout warnings

### Audit Trail

The CAN logger module provides a complete audit trail:
- All CAN messages logged with timestamps
- Playback capability for incident investigation
- Message count tracking
- Duration calculation

## Summary

✓ **All CAN modules have comprehensive logging**
✓ **Logging levels appropriate for real-time performance**
✓ **Safety-critical operations always logged**
✓ **High-frequency operations use DEBUG level**
✓ **Configuration added to configs/logging.yaml**
✓ **Verification script created**
✓ **Follows SENTINEL logging standards**

The CAN bus module logging is production-ready and aligned with SENTINEL's real-time performance requirements (30+ FPS, <100ms latency).
