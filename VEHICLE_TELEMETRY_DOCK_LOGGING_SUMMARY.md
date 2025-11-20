# Vehicle Telemetry Dock Logging Summary

## Overview

Comprehensive logging has been implemented for the Vehicle Telemetry Dock widget (`src/gui/widgets/vehicle_telemetry_dock.py`), which displays real-time vehicle telemetry from the CAN bus interface.

## Logging Configuration

### Module Logger
- **Logger Name**: `src.gui.widgets.vehicle_telemetry_dock`
- **Log Level**: INFO (configurable in `configs/logging.yaml`)
- **Handlers**: Console and file (`logs/sentinel.log`)

### Sub-component Loggers
- `src.gui.widgets.vehicle_telemetry_dock.SteeringIndicator`
- `src.gui.widgets.vehicle_telemetry_dock.BarIndicator`
- `src.gui.widgets.vehicle_telemetry_dock.GearIndicator`
- `src.gui.widgets.vehicle_telemetry_dock.TurnSignalIndicator`

## Logging Points

### 1. Initialization (DEBUG Level)
```python
# Main dock initialization
"Vehicle Telemetry Dock initialized"
"Setting up Vehicle Telemetry Dock UI"
"Vehicle Telemetry Dock UI setup completed"

# Sub-component initialization
"SteeringIndicator initialized"
"BarIndicator initialized: label={label}"
"GearIndicator initialized"
"TurnSignalIndicator initialized"
```

### 2. Telemetry Updates (DEBUG Level)
```python
# Comprehensive telemetry data
"Telemetry update received: speed={speed}m/s, steering={angle}rad, brake={pressure}bar, throttle={position}, gear={gear}, signal={signal}"

# Update completion
"Telemetry display updated successfully"
```

### 3. State Changes (DEBUG Level)
```python
# Steering angle changes
"Steering angle updated: {angle} rad ({degrees}°)"

# Gear changes
"Gear changed: {old_gear} -> {new_gear} ({gear_str})"

# Turn signal changes
"Turn signal changed: {old_signal} -> {new_signal}"

# Value clamping
"{label} value clamped: {original} -> {clamped}"
```

### 4. Error Handling (ERROR Level)
```python
# Telemetry update errors
"Telemetry display update failed: {error}"
```

## Log Message Format

All log messages follow the pattern:
- **Past tense** for completed actions
- **Present tense** for state descriptions
- **Include relevant context**: values, units, state transitions
- **Concise but informative**

## Example Log Output

```
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock - INFO - Vehicle Telemetry Dock initialized
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock - DEBUG - Setting up Vehicle Telemetry Dock UI
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock.SteeringIndicator - DEBUG - SteeringIndicator initialized
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock.BarIndicator - DEBUG - BarIndicator initialized: label=Brake
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock.BarIndicator - DEBUG - BarIndicator initialized: label=Throttle
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock.GearIndicator - DEBUG - GearIndicator initialized
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock.TurnSignalIndicator - DEBUG - TurnSignalIndicator initialized
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock - DEBUG - Vehicle Telemetry Dock UI setup completed
2024-11-18 10:30:46 - src.gui.widgets.vehicle_telemetry_dock - DEBUG - Telemetry update received: speed=15.50m/s, steering=0.200rad, brake=2.50bar, throttle=0.60, gear=3, signal=left
2024-11-18 10:30:46 - src.gui.widgets.vehicle_telemetry_dock.SteeringIndicator - DEBUG - Steering angle updated: 0.200 rad (11.5°)
2024-11-18 10:30:46 - src.gui.widgets.vehicle_telemetry_dock - DEBUG - Telemetry display updated successfully
2024-11-18 10:30:47 - src.gui.widgets.vehicle_telemetry_dock.GearIndicator - DEBUG - Gear changed: 3 -> 4 (4)
2024-11-18 10:30:48 - src.gui.widgets.vehicle_telemetry_dock.TurnSignalIndicator - DEBUG - Turn signal changed: left -> right
```

## Performance Considerations

### Real-time Requirements
- **Target**: 30+ FPS GUI updates
- **Logging Impact**: Minimal (DEBUG logs only in development)

### Optimization Strategies
1. **DEBUG level for frequent updates**: Telemetry updates logged at DEBUG level to avoid performance impact in production
2. **INFO level for initialization**: One-time events logged at INFO level
3. **Conditional logging**: State changes only logged when values actually change
4. **No logging in paint events**: Paint methods have no logging to avoid performance degradation

## Integration with CAN Bus Module

The Vehicle Telemetry Dock receives data from:
- `src.can.telemetry_reader.TelemetryReader` via signal/slot mechanism
- Updates at CAN bus rate (typically 100 Hz)
- Displays 6 key telemetry values:
  1. Speed (m/s)
  2. Steering angle (radians)
  3. Brake pressure (bar)
  4. Throttle position (0-1)
  5. Gear position
  6. Turn signal state

## Verification

Run the verification script to test logging:

```bash
python scripts/verify_vehicle_telemetry_logging.py
```

Expected output:
- ✓ Initialization logging verified
- ✓ Telemetry update logging verified
- ✓ State change logging verified
- ✓ All verifications passed

## Configuration

To adjust logging levels, edit `configs/logging.yaml`:

```yaml
src.gui.widgets.vehicle_telemetry_dock:
  level: INFO  # Change to DEBUG for detailed telemetry logs
  handlers: [file_all]
  propagate: false
```

## Related Components

- **CAN Bus Interface**: `src/can/interface.py`
- **Telemetry Reader**: `src/can/telemetry_reader.py`
- **Data Structures**: `src/core/data_structures.py` (VehicleTelemetry)
- **Circular Gauge**: `src/gui/widgets/circular_gauge.py`

## Status

✅ **COMPLETE** - All logging implemented and verified
- Initialization logging: ✓
- Telemetry update logging: ✓
- State change logging: ✓
- Error handling: ✓
- Configuration: ✓
- Verification script: ✓
