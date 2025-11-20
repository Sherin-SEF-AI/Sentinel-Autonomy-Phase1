# CAN Bus Integration - Quick Reference

## Quick Start

### 1. Setup Virtual CAN (for testing)
```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 up
```

### 2. Basic Usage
```python
from src.can import CANInterface, DBCParser, TelemetryReader

# Initialize
can = CANInterface(channel='vcan0')
dbc = DBCParser('configs/vehicle.dbc')
reader = TelemetryReader(can, dbc)

# Start
can.connect()
reader.start()

# Read
telemetry = reader.get_telemetry()
print(f"Speed: {telemetry.speed} m/s")

# Stop
reader.stop()
can.disconnect()
```

### 3. Run Example
```bash
python examples/can_example.py
```

## Configuration

Enable in `configs/default.yaml`:
```yaml
can_bus:
  enabled: true
  channel: "vcan0"  # or "can0" for real hardware
  dbc_file: "configs/vehicle.dbc"
```

## Testing

### Send Test Messages
```bash
# Speed: 25 m/s (0x09C4 = 2500 in hex, scale 0.01)
cansend vcan0 100#C409000000000000

# Steering: 0.1 rad left (0x0064 = 100, scale 0.001)
cansend vcan0 200#6400000000000000

# Brake: 5 bar (0x0032 = 50, scale 0.1)
cansend vcan0 300#3200000000000000
```

### Monitor Traffic
```bash
candump vcan0
```

## Message IDs

| ID (Hex) | ID (Dec) | Message | Signals |
|----------|----------|---------|---------|
| 0x100 | 256 | VehicleSpeed | Speed (m/s) |
| 0x200 | 512 | SteeringAngle | Angle (rad) |
| 0x300 | 768 | BrakePressure | Pressure (bar) |
| 0x400 | 1024 | ThrottlePosition | Position (0-1) |
| 0x500 | 1280 | GearPosition | Gear (-1 to 6) |
| 0x600 | 1536 | TurnSignal | Signal (0/1/2) |
| 0x700 | 1792 | BrakeCommand | Command (0-1) |
| 0x800 | 2048 | SteeringCommand | Command (rad) |

## Safety

⚠️ **Control commands are DISABLED by default**

To enable (only in controlled test environment):
```python
command_sender.set_enable_control(True)
```

Safety limits:
- Max brake: 1.0 (100%)
- Max steering: 0.5 rad (~28°)
- Watchdog: 0.5 seconds

## Troubleshooting

### "No such device" error
```bash
# Check interface exists
ip link show can0

# Bring up interface
sudo ip link set can0 up
```

### Permission denied
```bash
# Add user to can group
sudo usermod -a -G can $USER

# Or run with sudo (not recommended)
sudo python examples/can_example.py
```

### No messages received
- Check CAN bus bitrate matches vehicle
- Verify physical connections
- Use `candump` to verify messages on bus

## Integration with SENTINEL

### In Main System
```python
# Initialize
if config.can_bus.enabled:
    can_interface = CANInterface(channel=config.can_bus.channel)
    dbc_parser = DBCParser(config.can_bus.dbc_file)
    telemetry_reader = TelemetryReader(can_interface, dbc_parser)
    
    can_interface.connect()
    telemetry_reader.start()

# In processing loop
telemetry = telemetry_reader.get_telemetry()

# Pass to risk assessment
risks = intelligence_engine.assess(
    detections=detections,
    driver_state=driver_state,
    bev_seg=bev_seg,
    vehicle_telemetry=telemetry  # ← CAN data
)
```

### In GUI
```python
# Add telemetry dock
from src.gui.widgets.vehicle_telemetry_dock import VehicleTelemetryDock

telemetry_dock = QDockWidget("Vehicle Telemetry")
telemetry_widget = VehicleTelemetryDock()
telemetry_dock.setWidget(telemetry_widget)
main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, telemetry_dock)

# Connect signal
worker.telemetry_ready.connect(telemetry_widget.update_telemetry)
```

## Logging

### Enable Logging
```python
from src.can import CANLogger

logger = CANLogger(can_interface, log_file='logs/can_traffic.log')
logger.start()
# ... run system ...
logger.stop()
```

### Playback
```python
from src.can import CANPlayback

playback = CANPlayback('logs/can_traffic.log')
playback.load()

# Get messages
for i in range(playback.get_message_count()):
    timestamp, msg_id, data = playback.get_next_message()
    print(f"[{timestamp}] 0x{msg_id:X}: {data.hex()}")
```

## Performance

- Telemetry reading: **100 Hz**
- Message decoding: **<1 ms**
- Command latency: **<5 ms**
- Logging overhead: **Minimal** (async)

## Files

- **Module**: `src/can/`
- **Config**: `configs/default.yaml` (can_bus section)
- **DBC**: `configs/vehicle.dbc`
- **Example**: `examples/can_example.py`
- **Docs**: `src/can/README.md`
- **Summary**: `.kiro/specs/sentinel-safety-system/TASK_28_SUMMARY.md`

## References

- [SocketCAN Documentation](https://www.kernel.org/doc/html/latest/networking/can.html)
- [DBC File Format](https://www.csselectronics.com/pages/can-dbc-file-database-intro)
- [can-utils Tools](https://github.com/linux-can/can-utils)
