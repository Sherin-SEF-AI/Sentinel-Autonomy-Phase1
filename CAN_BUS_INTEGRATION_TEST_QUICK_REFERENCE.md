# CAN Bus Integration Testing - Quick Reference

## Test Overview

Comprehensive integration tests for CAN bus system covering connection, message decoding, telemetry reading at 100 Hz, and command sending.

**Test File**: `tests/unit/test_can_integration.py`  
**Total Tests**: 32  
**Status**: ✅ All passing

## Quick Test Commands

### Run All CAN Integration Tests
```bash
python3 -m pytest tests/unit/test_can_integration.py -v
```

### Run Specific Test Categories

#### Connection Tests (5 tests)
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestCANConnectionAndReconnection -v
```

#### Message Decoding Tests (6 tests)
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestMessageDecoding -v
```

#### Telemetry Reading Tests (6 tests)
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestTelemetryReading -v
```

#### Command Sending Tests (10 tests)
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestCommandSending -v
```

#### Performance Tests (3 tests)
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestCANIntegrationPerformance -v
```

#### End-to-End Tests (3 tests)
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestCANEndToEndIntegration -v
```

### Run Specific Test
```bash
python3 -m pytest tests/unit/test_can_integration.py::TestTelemetryReading::test_telemetry_update_rate_100hz -v
```

### Run with Coverage
```bash
python3 -m pytest tests/unit/test_can_integration.py --cov=src.can --cov-report=html
```

## Test Categories

### 1. Connection & Reconnection (Req 23.1)
- ✅ Initial connection success
- ✅ Connection failure handling
- ✅ Reconnection after disconnect
- ✅ Reconnection after error
- ✅ Multiple reconnection attempts

### 2. Message Decoding (Req 23.3)
- ✅ Little-endian unsigned signals
- ✅ Big-endian signed signals
- ✅ Scale and offset encoding
- ✅ Multiple signals per message
- ✅ DBC file parsing

### 3. Telemetry Reading (Req 23.2)
- ✅ Reader initialization
- ✅ Thread start/stop
- ✅ 100 Hz update rate
- ✅ Data updates
- ✅ All fields (speed, steering, brake, throttle, gear, turn signal)
- ✅ Sustained 100 Hz performance

### 4. Command Sending (Req 23.4)
- ✅ Sender initialization
- ✅ Control enable/disable
- ✅ Brake commands (0.0-1.0)
- ✅ Steering commands (±0.5 rad)
- ✅ Command clamping
- ✅ Emergency stop
- ✅ Control release
- ✅ Watchdog timeout

### 5. Performance
- ✅ Send latency < 1ms avg
- ✅ Receive latency < 1ms avg
- ✅ Sustained 100 Hz telemetry

### 6. End-to-End Integration
- ✅ Complete telemetry pipeline
- ✅ Complete command pipeline
- ✅ Bidirectional communication

## Expected Test Output

```
============================= test session starts ============================
collected 32 items

TestCANConnectionAndReconnection
  test_initial_connection_success PASSED                              [  3%]
  test_connection_failure_handling PASSED                             [  6%]
  test_reconnection_after_disconnect PASSED                           [  9%]
  test_reconnection_after_error PASSED                                [ 12%]
  test_multiple_reconnection_attempts PASSED                          [ 15%]

TestMessageDecoding
  test_signal_decode_little_endian_unsigned PASSED                    [ 18%]
  test_signal_decode_big_endian_signed PASSED                         [ 21%]
  test_signal_encode_with_scale_and_offset PASSED                     [ 25%]
  test_message_decode_multiple_signals PASSED                         [ 28%]
  test_message_encode_multiple_signals PASSED                         [ 31%]
  test_dbc_parser_load_and_decode PASSED                              [ 34%]

TestTelemetryReading
  test_telemetry_reader_initialization PASSED                         [ 37%]
  test_telemetry_reader_start_stop PASSED                             [ 40%]
  test_telemetry_update_rate_100hz PASSED                             [ 43%]
  test_telemetry_data_update PASSED                                   [ 46%]
  test_telemetry_all_fields_update PASSED                             [ 50%]

TestCommandSending
  test_command_sender_initialization PASSED                           [ 53%]
  test_send_brake_command_disabled PASSED                             [ 56%]
  test_send_brake_command_enabled PASSED                              [ 59%]
  test_brake_command_clamping PASSED                                  [ 62%]
  test_send_steering_command_enabled PASSED                           [ 65%]
  test_steering_command_clamping PASSED                               [ 68%]
  test_emergency_stop_command PASSED                                  [ 71%]
  test_release_control_command PASSED                                 [ 75%]
  test_watchdog_timeout_detection PASSED                              [ 78%]
  test_enable_disable_control PASSED                                  [ 81%]

TestCANIntegrationPerformance
  test_frame_send_latency PASSED                                      [ 84%]
  test_frame_receive_latency PASSED                                   [ 87%]
  test_telemetry_reading_sustained_100hz PASSED                       [ 90%]

TestCANEndToEndIntegration
  test_complete_telemetry_pipeline PASSED                             [ 93%]
  test_complete_command_pipeline PASSED                               [ 96%]
  test_bidirectional_communication PASSED                             [100%]

============================== 32 passed in 5.58s ============================
```

## Key Validations

### Safety Mechanisms
- Control enable/disable flag
- Brake clamping: 0.0 to 1.0
- Steering clamping: ±0.5 radians
- Watchdog timeout: 0.5 seconds
- Emergency stop: maximum brake
- Control release: zero commands

### Performance Metrics
- Frame send latency: < 1ms average, < 5ms max
- Frame receive latency: < 1ms average, < 5ms max
- Telemetry update rate: 100 Hz ±10%
- Sustained operation: 2+ seconds

### Robustness
- Automatic reconnection on errors
- Graceful connection failure handling
- Thread-safe telemetry access
- Error recovery mechanisms

## Requirements Coverage

| Requirement | Description | Tests | Status |
|-------------|-------------|-------|--------|
| 23.1 | CAN connection and reconnection | 5 | ✅ Pass |
| 23.2 | Telemetry reading at 100 Hz | 6 | ✅ Pass |
| 23.3 | Message decoding | 6 | ✅ Pass |
| 23.4 | Command sending | 10 | ✅ Pass |

## Troubleshooting

### If Tests Fail

1. **Check Python version**: Requires Python 3.10+
   ```bash
   python3 --version
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Run with verbose output**:
   ```bash
   python3 -m pytest tests/unit/test_can_integration.py -vv --tb=short
   ```

4. **Check specific failing test**:
   ```bash
   python3 -m pytest tests/unit/test_can_integration.py::TestName::test_name -vv
   ```

## Related Files

- `src/can/interface.py` - CANInterface implementation
- `src/can/dbc_parser.py` - DBC parser and message codec
- `src/can/telemetry_reader.py` - Telemetry reader (100 Hz)
- `src/can/command_sender.py` - Command sender with safety
- `configs/vehicle.dbc` - Example DBC file
- `.kiro/specs/sentinel-safety-system/TASK_31_5_SUMMARY.md` - Detailed summary

## Next Steps

After validating CAN bus integration:
1. Test with real CAN hardware (vcan0 or physical CAN interface)
2. Validate with actual vehicle DBC file
3. Integrate with SENTINEL main system
4. Test end-to-end with risk assessment and alerts
5. Perform on-vehicle validation

## Notes

- Tests use mocked SocketCAN interface for CI/CD compatibility
- Real CAN hardware testing requires Linux with SocketCAN support
- Virtual CAN (vcan0) can be used for integration testing without hardware
- All tests are thread-safe and can run in parallel
