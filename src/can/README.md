# CAN Bus Integration Module

This module provides CAN bus communication capabilities for the SENTINEL system, enabling integration with vehicle telemetry and control systems.

## Overview

The CAN bus integration allows SENTINEL to:
- Read vehicle telemetry (speed, steering, brake, throttle, gear, turn signals)
- Send intervention commands (brake, steering) when enabled
- Log all CAN traffic for debugging and analysis
- Integrate vehicle data into risk assessment and trajectory prediction

## Components

### CANInterface (`interface.py`)
Low-level SocketCAN interface for Linux systems.
- Manages CAN bus connection and reconnection
- Sends and receives raw CAN frames
- Handles errors and automatic recovery

### DBCParser (`dbc_parser.py`)
Parses DBC (CAN database) files to decode CAN messages.
- Extracts message definitions
- Extracts signal definitions with scaling and offsets
- Builds decoding tables for efficient message parsing

### TelemetryReader (`telemetry_reader.py`)
Reads and decodes vehicle telemetry from CAN bus.
- Continuously reads CAN messages at 100 Hz
- Decodes messages using DBC definitions
- Outputs VehicleTelemetry dataclass

### CommandSender (`command_sender.py`)
Sends control commands to vehicle via CAN bus.
- Encodes brake intervention commands
- Encodes steering intervention commands
- Applies safety checks and limits
- Requires explicit enable flag for safety

## Usage

### Basic Telemetry Reading

```python
from src.can import CANInterface, DBCParser, TelemetryReader

# Initialize components
can_interface = CANInterface(channel='can0')
dbc_parser = DBCParser('configs/vehicle.dbc')
telemetry_reader = TelemetryReader(can_interface, dbc_parser)

# Connect and start reading
can_interface.connect()
telemetry_reader.start()

# Get latest telemetry
telemetry = telemetry_reader.get_telemetry()
print(f"Speed: {telemetry.speed} m/s")
print(f"Steering: {telemetry.steering_angle} rad")

# Stop and disconnect
telemetry_reader.stop()
can_interface.disconnect()
```

### Sending Commands (if enabled)

```python
from src.can import CommandSender

# Initialize command sender (control disabled by default)
command_sender = CommandSender(can_interface, dbc_parser, enable_control=False)

# Enable control (requires explicit flag for safety)
command_sender.enable_control = True

# Send brake command (0-1 range)
command_sender.send_brake_command(0.3)  # 30% braking

# Send steering command (radians)
command_sender.send_steering_command(0.1)  # 0.1 rad left
```

### Integration with SENTINEL

```python
# In main system orchestration
if config.can_bus.enabled:
    can_interface = CANInterface(channel=config.can_bus.channel)
    dbc_parser = DBCParser(config.can_bus.dbc_file)
    telemetry_reader = TelemetryReader(can_interface, dbc_parser)
    
    can_interface.connect()
    telemetry_reader.start()
    
    # In processing loop
    telemetry = telemetry_reader.get_telemetry()
    
    # Use in risk assessment
    risks = intelligence_engine.assess(
        detections=detections,
        driver_state=driver_state,
        bev_seg=bev_seg,
        vehicle_telemetry=telemetry  # Pass vehicle data
    )
```

## Configuration

Add to `configs/default.yaml`:

```yaml
can_bus:
  enabled: false  # Set to true to enable CAN integration
  interface: "socketcan"
  channel: "can0"  # CAN interface name
  dbc_file: "configs/vehicle.dbc"  # DBC file path
  enable_control: false  # Enable sending control commands
  log_traffic: true  # Log all CAN messages
  log_file: "logs/can_traffic.log"
```

## DBC File Format

The DBC file defines CAN message and signal structures. Example:

```
BO_ 100 VehicleSpeed: 8 Vector__XXX
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "m/s" Vector__XXX

BO_ 200 SteeringAngle: 8 Vector__XXX
 SG_ Angle : 0|16@1- (0.001,-32.768) [-32.768|32.767] "rad" Vector__XXX

BO_ 300 BrakePressure: 8 Vector__XXX
 SG_ Pressure : 0|16@1+ (0.1,0) [0|6553.5] "bar" Vector__XXX
```

## Linux Setup

### Install CAN utilities

```bash
sudo apt-get install can-utils
```

### Configure CAN interface

```bash
# Bring up CAN interface at 500 kbps
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up

# For testing with virtual CAN
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set vcan0 up
```

### Test CAN communication

```bash
# Send test frame
cansend vcan0 100#0102030405060708

# Monitor CAN traffic
candump vcan0
```

## Safety Considerations

- Control commands are **disabled by default**
- Explicit `enable_control=True` flag required
- Safety limits applied to all commands
- Watchdog timer for command timeout
- Fail-safe behavior on communication loss

## Performance

- Telemetry reading: 100 Hz
- Message decoding: <1 ms per message
- Command latency: <5 ms

## Troubleshooting

### "No such device" error
- Check that CAN interface exists: `ip link show can0`
- Verify interface is up: `sudo ip link set can0 up`

### Permission denied
- Add user to `can` group: `sudo usermod -a -G can $USER`
- Or run with sudo (not recommended for production)

### No messages received
- Check CAN bus bitrate matches vehicle
- Verify physical CAN connections
- Use `candump` to verify messages on bus

## References

- [SocketCAN Documentation](https://www.kernel.org/doc/html/latest/networking/can.html)
- [DBC File Format](https://www.csselectronics.com/pages/can-dbc-file-database-intro)
- [python-can Library](https://python-can.readthedocs.io/)
