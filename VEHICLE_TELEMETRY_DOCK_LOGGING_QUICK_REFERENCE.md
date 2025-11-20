# Vehicle Telemetry Dock Logging - Quick Reference

## Logger Configuration

**Module**: `src.gui.widgets.vehicle_telemetry_dock`  
**Level**: INFO (DEBUG for detailed telemetry logs)  
**Config**: `configs/logging.yaml`

```yaml
src.gui.widgets.vehicle_telemetry_dock:
  level: INFO
  handlers: [file_all]
  propagate: false
```

## Key Logging Points

### Initialization
```python
logger.info("Vehicle Telemetry Dock initialized")
logger.debug("Setting up Vehicle Telemetry Dock UI")
logger.debug("Vehicle Telemetry Dock UI setup completed")
```

### Telemetry Updates
```python
logger.debug(
    f"Telemetry update received: speed={speed}m/s, "
    f"steering={angle}rad, brake={pressure}bar, "
    f"throttle={position}, gear={gear}, signal={signal}"
)
logger.debug("Telemetry display updated successfully")
```

### State Changes
```python
logger.debug(f"Steering angle updated: {angle:.3f} rad ({degrees:.1f}°)")
logger.debug(f"Gear changed: {old} -> {new} ({gear_str})")
logger.debug(f"Turn signal changed: {old} -> {new}")
```

### Error Handling
```python
logger.error(f"Telemetry display update failed: {error}", exc_info=True)
```

## Sub-Components

Each visual indicator has its own logger:
- `SteeringIndicator` - Steering wheel visualization
- `BarIndicator` - Brake/throttle bars
- `GearIndicator` - Gear position display
- `TurnSignalIndicator` - Turn signal arrows

## Verification

```bash
# Quick verification
python scripts/verify_vehicle_telemetry_logging_simple.py

# Full GUI test (requires display)
python scripts/verify_vehicle_telemetry_logging.py
```

## Integration

Receives telemetry from:
- **CAN Bus**: `src.can.telemetry_reader.TelemetryReader`
- **Data Structure**: `VehicleTelemetry` from `src.core.data_structures`
- **Update Rate**: 100 Hz (CAN bus rate)

## Performance

- **DEBUG logs**: Only in development (minimal production overhead)
- **No logging in paint events**: Maintains 30+ FPS GUI performance
- **Conditional logging**: State changes only logged when values change

## Example Output

```
2024-11-18 10:30:45 - src.gui.widgets.vehicle_telemetry_dock - INFO - Vehicle Telemetry Dock initialized
2024-11-18 10:30:46 - src.gui.widgets.vehicle_telemetry_dock - DEBUG - Telemetry update received: speed=15.50m/s, steering=0.200rad, brake=2.50bar, throttle=0.60, gear=3, signal=left
2024-11-18 10:30:47 - src.gui.widgets.vehicle_telemetry_dock.GearIndicator - DEBUG - Gear changed: 3 -> 4 (4)
```

## Status

✅ **COMPLETE** - All logging implemented and verified
