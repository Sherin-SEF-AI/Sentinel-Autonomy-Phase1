"""
Comprehensive integration tests for CAN bus system.

Tests CAN connection, reconnection, message decoding, telemetry reading at 100 Hz,
and command sending as specified in task 31.5.
"""

import pytest
import time
import struct
import socket
from unittest.mock import Mock, MagicMock, patch, call
from threading import Thread

from src.can.interface import CANInterface
from src.can.dbc_parser import DBCParser, Signal, Message
from src.can.telemetry_reader import TelemetryReader
from src.can.command_sender import CommandSender
from src.core.data_structures import VehicleTelemetry


class TestCANConnectionAndReconnection:
    """Test CAN connection and reconnection functionality (Requirement 23.1)."""
    
    @patch('socket.socket')
    def test_initial_connection_success(self, mock_socket_class):
        """Test successful initial connection to CAN bus."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        result = interface.connect()
        
        assert result is True
        assert interface.is_connected() is True
        mock_socket.bind.assert_called_once_with(('vcan0',))
        mock_socket.settimeout.assert_called_once()
    
    @patch('socket.socket')
    def test_connection_failure_handling(self, mock_socket_class):
        """Test handling of connection failures."""
        mock_socket_class.side_effect = OSError("Interface not found")
        
        interface = CANInterface(channel='vcan0')
        result = interface.connect()
        
        assert result is False
        assert interface.is_connected() is False
    
    @patch('socket.socket')
    @patch('time.sleep')
    def test_reconnection_after_disconnect(self, mock_sleep, mock_socket_class):
        """Test reconnection after intentional disconnect."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0', reconnect_delay=0.5)
        
        # Initial connection
        interface.connect()
        assert interface.is_connected() is True
        
        # Disconnect
        interface.disconnect()
        assert interface.is_connected() is False
        
        # Reconnect
        result = interface.reconnect()
        assert result is True
        assert interface.is_connected() is True
        mock_sleep.assert_called_once_with(0.5)
    
    @patch('socket.socket')
    def test_reconnection_after_error(self, mock_socket_class):
        """Test automatic reconnection after communication error."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        # Simulate send error
        mock_socket.send.side_effect = OSError("Connection lost")
        result = interface.send_frame(0x100, b'\x01\x02')
        
        # Should attempt reconnection
        assert result is False
        # Verify reconnection was attempted (socket closed and recreated)
        assert mock_socket.close.called
    
    @patch('socket.socket')
    def test_multiple_reconnection_attempts(self, mock_socket_class):
        """Test multiple reconnection attempts."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0', reconnect_delay=0.1)
        
        # Simulate multiple connection failures then success
        mock_socket_class.side_effect = [
            OSError("Failed"),
            OSError("Failed"),
            mock_socket
        ]
        
        # First attempt fails
        result1 = interface.connect()
        assert result1 is False
        
        # Second attempt fails
        result2 = interface.reconnect()
        assert result2 is False
        
        # Third attempt succeeds
        result3 = interface.reconnect()
        assert result3 is True
        assert interface.is_connected() is True


class TestMessageDecoding:
    """Test CAN message decoding functionality (Requirement 23.3)."""
    
    def test_signal_decode_little_endian_unsigned(self):
        """Test decoding unsigned little-endian signal."""
        signal = Signal(
            name='Speed',
            start_bit=0,
            length=16,
            byte_order='little',
            signed=False,
            scale=0.01,
            offset=0.0,
            min_value=0.0,
            max_value=655.35,
            unit='m/s'
        )
        
        # Encode value 50.00 m/s = 5000 raw
        data = struct.pack('<H', 5000) + b'\x00' * 6
        decoded = signal.decode(data)
        
        assert abs(decoded - 50.0) < 0.01
    
    def test_signal_decode_big_endian_signed(self):
        """Test decoding signed big-endian signal."""
        signal = Signal(
            name='SteeringAngle',
            start_bit=0,
            length=16,
            byte_order='big',
            signed=True,
            scale=0.001,
            offset=0.0,
            min_value=-32.768,
            max_value=32.767,
            unit='rad'
        )
        
        # Encode value -0.5 rad = -500 raw
        data = struct.pack('>h', -500) + b'\x00' * 6
        decoded = signal.decode(data)
        
        assert abs(decoded - (-0.5)) < 0.001
    
    def test_signal_encode_with_scale_and_offset(self):
        """Test encoding signal with scale and offset."""
        signal = Signal(
            name='Temperature',
            start_bit=0,
            length=8,
            byte_order='little',
            signed=False,
            scale=0.5,
            offset=-40.0,
            min_value=-40.0,
            max_value=87.5,
            unit='C'
        )
        
        # Encode 20.0 C = (20.0 - (-40.0)) / 0.5 = 120 raw
        raw = signal.encode(20.0)
        assert raw == 120
    
    def test_message_decode_multiple_signals(self):
        """Test decoding message with multiple signals."""
        speed_signal = Signal(
            name='Speed',
            start_bit=0,
            length=16,
            byte_order='little',
            signed=False,
            scale=0.01,
            offset=0.0,
            min_value=0.0,
            max_value=655.35,
            unit='m/s'
        )
        
        gear_signal = Signal(
            name='Gear',
            start_bit=16,
            length=8,
            byte_order='little',
            signed=False,
            scale=1.0,
            offset=0.0,
            min_value=0.0,
            max_value=7.0,
            unit=''
        )
        
        message = Message(
            message_id=0x100,
            name='Powertrain',
            dlc=8,
            signals={'Speed': speed_signal, 'Gear': gear_signal}
        )
        
        # Create data: speed=30.0 m/s (3000 raw), gear=3
        data = struct.pack('<HB', 3000, 3) + b'\x00' * 5
        decoded = message.decode(data)
        
        assert 'Speed' in decoded
        assert 'Gear' in decoded
        assert abs(decoded['Speed'] - 30.0) < 0.01
        assert decoded['Gear'] == 3.0
    
    def test_message_encode_multiple_signals(self):
        """Test encoding message with multiple signals."""
        brake_signal = Signal(
            name='BrakeCommand',
            start_bit=0,
            length=8,
            byte_order='little',
            signed=False,
            scale=0.01,
            offset=0.0,
            min_value=0.0,
            max_value=1.0,
            unit=''
        )
        
        message = Message(
            message_id=0x700,
            name='BrakeControl',
            dlc=8,
            signals={'BrakeCommand': brake_signal}
        )
        
        # Encode brake command 0.5 = 50 raw
        data = message.encode({'BrakeCommand': 0.5})
        
        assert len(data) == 8
        assert data[0] == 50
    
    def test_dbc_parser_load_and_decode(self, tmp_path):
        """Test loading DBC file and decoding messages."""
        # Create minimal DBC file
        dbc_content = """VERSION ""
NS_ :
BS_:
BU_:

BO_ 256 Speed: 8 Vector__XXX
 SG_ vehicle_speed : 0|16@1+ (0.01,0) [0|655.35] "m/s" Vector__XXX

BO_ 512 Steering: 8 Vector__XXX
 SG_ Angle : 0|16@1- (0.001,0) [-32.768|32.767] "rad" Vector__XXX
"""
        dbc_file = tmp_path / "test.dbc"
        dbc_file.write_text(dbc_content)
        
        parser = DBCParser(str(dbc_file))
        
        # Test speed message decode
        speed_data = struct.pack('<H', 5000) + b'\x00' * 6
        decoded = parser.decode_message(0x100, speed_data)
        
        assert decoded is not None
        assert 'vehicle_speed' in decoded
        assert abs(decoded['vehicle_speed'] - 50.0) < 0.01


class TestTelemetryReading:
    """Test telemetry reading at 100 Hz (Requirement 23.2)."""
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_telemetry_reader_initialization(self, mock_parser, mock_interface):
        """Test telemetry reader initializes correctly."""
        reader = TelemetryReader(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            update_rate=100.0
        )
        
        assert reader is not None
        assert reader.update_rate == 100.0
        assert reader.running is False
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_telemetry_reader_start_stop(self, mock_parser, mock_interface):
        """Test starting and stopping telemetry reader."""
        mock_interface.receive_frame.return_value = None
        
        reader = TelemetryReader(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            update_rate=100.0
        )
        
        # Start reader
        reader.start()
        assert reader.is_running() is True
        
        # Stop reader
        time.sleep(0.1)  # Let thread start
        reader.stop()
        assert reader.is_running() is False
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_telemetry_update_rate_100hz(self, mock_parser, mock_interface):
        """Test telemetry reading achieves 100 Hz update rate."""
        # Mock CAN frames
        speed_frame = (0x100, struct.pack('<H', 3000) + b'\x00' * 6)
        mock_interface.receive_frame.return_value = speed_frame
        
        # Mock decoder
        mock_parser.decode_message.return_value = {'Speed': 30.0}
        
        reader = TelemetryReader(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            message_ids={'speed': 0x100},
            update_rate=100.0
        )
        
        reader.start()
        time.sleep(0.5)  # Run for 0.5 seconds
        reader.stop()
        
        # Should have called receive_frame approximately 50 times (100 Hz * 0.5s)
        call_count = mock_interface.receive_frame.call_count
        assert 40 <= call_count <= 60, f"Expected ~50 calls, got {call_count}"
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_telemetry_data_update(self, mock_parser, mock_interface):
        """Test telemetry data is updated correctly."""
        # Mock speed message
        speed_frame = (0x100, struct.pack('<H', 5000) + b'\x00' * 6)
        mock_interface.receive_frame.return_value = speed_frame
        mock_parser.decode_message.return_value = {'Speed': 50.0}
        
        reader = TelemetryReader(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            message_ids={'speed': 0x100},
            update_rate=100.0
        )
        
        reader.start()
        time.sleep(0.1)  # Let it process some frames
        reader.stop()
        
        telemetry = reader.get_telemetry()
        assert telemetry.speed == 50.0
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_telemetry_all_fields_update(self, mock_parser, mock_interface):
        """Test all telemetry fields are updated correctly."""
        # Create mock frames for different message types
        frames = [
            (0x100, struct.pack('<H', 3000) + b'\x00' * 6),  # Speed
            (0x200, struct.pack('<h', 500) + b'\x00' * 6),   # Steering
            (0x300, struct.pack('<H', 200) + b'\x00' * 6),   # Brake
            (0x400, struct.pack('<B', 75) + b'\x00' * 7),    # Throttle
            (0x500, struct.pack('<B', 3) + b'\x00' * 7),     # Gear
            (0x600, struct.pack('<B', 1) + b'\x00' * 7),     # Turn signal
        ]
        
        frame_iter = iter(frames * 10)  # Repeat frames
        mock_interface.receive_frame.side_effect = lambda timeout: next(frame_iter, None)
        
        # Mock decoder responses
        def decode_side_effect(msg_id, data):
            decoders = {
                0x100: {'Speed': 30.0},
                0x200: {'Angle': 0.5},
                0x300: {'Pressure': 2.0},
                0x400: {'Position': 0.75},
                0x500: {'Gear': 3.0},
                0x600: {'Signal': 1.0},
            }
            return decoders.get(msg_id, {})
        
        mock_parser.decode_message.side_effect = decode_side_effect
        
        reader = TelemetryReader(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            message_ids={
                'speed': 0x100,
                'steering': 0x200,
                'brake': 0x300,
                'throttle': 0x400,
                'gear': 0x500,
                'turn_signal': 0x600,
            },
            update_rate=100.0
        )
        
        reader.start()
        time.sleep(0.2)  # Let it process frames
        reader.stop()
        
        telemetry = reader.get_telemetry()
        assert telemetry.speed == 30.0
        assert telemetry.steering_angle == 0.5
        assert telemetry.brake_pressure == 2.0
        assert telemetry.throttle_position == 0.75
        assert telemetry.gear == 3
        assert telemetry.turn_signal == 'left'


class TestCommandSending:
    """Test command sending functionality (Requirement 23.4)."""
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_command_sender_initialization(self, mock_parser, mock_interface):
        """Test command sender initializes correctly."""
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=False
        )
        
        assert sender is not None
        assert sender.is_control_enabled() is False
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_send_brake_command_disabled(self, mock_parser, mock_interface):
        """Test brake command is not sent when control disabled."""
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=False
        )
        
        result = sender.send_brake_command(0.5)
        
        assert result is False
        mock_interface.send_frame.assert_not_called()
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_send_brake_command_enabled(self, mock_parser, mock_interface):
        """Test brake command is sent when control enabled."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x50\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            command_ids={'brake': 0x700},
            enable_control=True
        )
        
        result = sender.send_brake_command(0.5)
        
        assert result is True
        mock_parser.encode_message.assert_called_once_with(0x700, {'BrakeCommand': 0.5})
        mock_interface.send_frame.assert_called_once()
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_brake_command_clamping(self, mock_parser, mock_interface):
        """Test brake command is clamped to valid range."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x64\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=True
        )
        
        # Test upper limit
        sender.send_brake_command(1.5)
        call_args = mock_parser.encode_message.call_args[0]
        assert call_args[1]['BrakeCommand'] <= 1.0
        
        # Test lower limit
        sender.send_brake_command(-0.5)
        call_args = mock_parser.encode_message.call_args[0]
        assert call_args[1]['BrakeCommand'] >= 0.0
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_send_steering_command_enabled(self, mock_parser, mock_interface):
        """Test steering command is sent when control enabled."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\xC8\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            command_ids={'steering': 0x800},
            enable_control=True
        )
        
        result = sender.send_steering_command(0.2)
        
        assert result is True
        mock_parser.encode_message.assert_called_once_with(0x800, {'SteeringCommand': 0.2})
        mock_interface.send_frame.assert_called_once()
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_steering_command_clamping(self, mock_parser, mock_interface):
        """Test steering command is clamped to safe limits."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=True
        )
        
        # Test upper limit (max 0.5 rad)
        sender.send_steering_command(1.0)
        call_args = mock_parser.encode_message.call_args[0]
        assert call_args[1]['SteeringCommand'] <= 0.5
        
        # Test lower limit
        sender.send_steering_command(-1.0)
        call_args = mock_parser.encode_message.call_args[0]
        assert call_args[1]['SteeringCommand'] >= -0.5
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_emergency_stop_command(self, mock_parser, mock_interface):
        """Test emergency stop sends maximum brake command."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x64\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=True
        )
        
        result = sender.send_emergency_stop()
        
        assert result is True
        # Verify maximum brake was commanded
        call_args = mock_parser.encode_message.call_args[0]
        assert call_args[1]['BrakeCommand'] == 1.0
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_release_control_command(self, mock_parser, mock_interface):
        """Test release control sends zero commands."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=True
        )
        
        result = sender.release_control()
        
        assert result is True
        # Should have sent both brake and steering zero commands
        assert mock_parser.encode_message.call_count == 2
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_watchdog_timeout_detection(self, mock_parser, mock_interface):
        """Test watchdog timeout is detected correctly."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            enable_control=True
        )
        
        # Send initial command
        sender.send_brake_command(0.5)
        
        # Check watchdog immediately (should be healthy)
        assert sender.check_watchdog() is True
        
        # Wait for timeout to occur
        time.sleep(0.6)  # Exceed 0.5s timeout
        
        # Check watchdog after timeout
        assert sender.check_watchdog() is False
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_enable_disable_control(self, mock_parser, mock_interface):
        """Test enabling and disabling control commands."""
        mock_interface.send_frame.return_value = True
        mock_parser.encode_message.return_value = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        
        sender = CommandSender(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            command_ids={'brake': 0x700, 'steering': 0x800},
            enable_control=False
        )
        
        # Initially disabled
        assert sender.is_control_enabled() is False
        
        # Enable control
        sender.set_enable_control(True)
        assert sender.is_control_enabled() is True
        
        # Send a command while enabled
        sender.send_brake_command(0.5)
        assert mock_parser.encode_message.call_count == 1
        
        # Disable control (should attempt to release control)
        sender.set_enable_control(False)
        assert sender.is_control_enabled() is False
        # Note: release_control is called but commands won't be sent since control is now disabled
        # The important thing is that control state changed correctly


class TestCANIntegrationPerformance:
    """Test CAN bus performance requirements."""
    
    @patch('socket.socket')
    def test_frame_send_latency(self, mock_socket_class):
        """Test CAN frame sending latency is acceptable."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        # Measure send latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            interface.send_frame(0x100, b'\x01\x02\x03\x04')
            latencies.append(time.perf_counter() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Should be very fast (< 1ms average, < 5ms max)
        assert avg_latency < 0.001, f"Average latency {avg_latency*1000:.2f}ms exceeds 1ms"
        assert max_latency < 0.005, f"Max latency {max_latency*1000:.2f}ms exceeds 5ms"
    
    @patch('socket.socket')
    def test_frame_receive_latency(self, mock_socket_class):
        """Test CAN frame receiving latency is acceptable."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        # Mock received frame
        frame = struct.pack("=IB3x8s", 0x100, 4, b'\x01\x02\x03\x04\x00\x00\x00\x00')
        mock_socket.recv.return_value = frame
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        # Measure receive latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            interface.receive_frame()
            latencies.append(time.perf_counter() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Should be very fast (< 1ms average, < 5ms max)
        assert avg_latency < 0.001, f"Average latency {avg_latency*1000:.2f}ms exceeds 1ms"
        assert max_latency < 0.005, f"Max latency {max_latency*1000:.2f}ms exceeds 5ms"
    
    @patch('src.can.interface.CANInterface')
    @patch('src.can.dbc_parser.DBCParser')
    def test_telemetry_reading_sustained_100hz(self, mock_parser, mock_interface):
        """Test telemetry reader sustains 100 Hz for extended period."""
        # Mock CAN frames
        frame = (0x100, b'\x00\x00\x00\x00\x00\x00\x00\x00')
        mock_interface.receive_frame.return_value = frame
        mock_parser.decode_message.return_value = {'Speed': 0.0}
        
        reader = TelemetryReader(
            can_interface=mock_interface,
            dbc_parser=mock_parser,
            update_rate=100.0
        )
        
        reader.start()
        time.sleep(2.0)  # Run for 2 seconds
        reader.stop()
        
        # Should have processed approximately 200 frames (100 Hz * 2s)
        call_count = mock_interface.receive_frame.call_count
        expected = 200
        tolerance = 20  # Allow 10% tolerance
        
        assert expected - tolerance <= call_count <= expected + tolerance, \
            f"Expected ~{expected} calls, got {call_count}"


class TestCANEndToEndIntegration:
    """End-to-end integration tests for complete CAN system."""
    
    @patch('socket.socket')
    def test_complete_telemetry_pipeline(self, mock_socket_class, tmp_path):
        """Test complete pipeline from CAN bus to telemetry data."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        # Create DBC file
        dbc_content = """VERSION ""
NS_ :
BS_:
BU_:

BO_ 256 Speed: 8 Vector__XXX
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "m/s" Vector__XXX
"""
        dbc_file = tmp_path / "test.dbc"
        dbc_file.write_text(dbc_content)
        
        # Setup components
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        parser = DBCParser(str(dbc_file))
        
        # Mock received speed frame (30.0 m/s = 3000 raw)
        speed_frame = struct.pack("=IB3x8s", 0x100, 2, 
                                  struct.pack('<H', 3000).ljust(8, b'\x00'))
        mock_socket.recv.return_value = speed_frame
        
        reader = TelemetryReader(
            can_interface=interface,
            dbc_parser=parser,
            message_ids={'speed': 0x100},
            update_rate=100.0
        )
        
        reader.start()
        time.sleep(0.2)
        reader.stop()
        
        telemetry = reader.get_telemetry()
        assert abs(telemetry.speed - 30.0) < 0.1
    
    @patch('socket.socket')
    def test_complete_command_pipeline(self, mock_socket_class, tmp_path):
        """Test complete pipeline from command to CAN bus."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        # Create DBC file with command message
        dbc_content = """VERSION ""
NS_ :
BS_:
BU_:

BO_ 1792 BrakeControl: 8 Vector__XXX
 SG_ BrakeCommand : 0|8@1+ (0.01,0) [0|1.0] "" Vector__XXX
"""
        dbc_file = tmp_path / "test.dbc"
        dbc_file.write_text(dbc_content)
        
        # Setup components
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        parser = DBCParser(str(dbc_file))
        
        sender = CommandSender(
            can_interface=interface,
            dbc_parser=parser,
            command_ids={'brake': 0x700},
            enable_control=True
        )
        
        # Send brake command
        result = sender.send_brake_command(0.5)
        
        assert result is True
        mock_socket.send.assert_called_once()
        
        # Verify frame format
        call_args = mock_socket.send.call_args[0][0]
        can_id, dlc = struct.unpack("=IB3x", call_args[:8])
        assert can_id == 0x700
        assert dlc == 8
    
    @patch('socket.socket')
    def test_bidirectional_communication(self, mock_socket_class, tmp_path):
        """Test bidirectional CAN communication (telemetry + commands)."""
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        
        # Create comprehensive DBC file
        dbc_content = """VERSION ""
NS_ :
BS_:
BU_:

BO_ 256 Speed: 8 Vector__XXX
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "m/s" Vector__XXX

BO_ 1792 BrakeControl: 8 Vector__XXX
 SG_ BrakeCommand : 0|8@1+ (0.01,0) [0|1.0] "" Vector__XXX
"""
        dbc_file = tmp_path / "test.dbc"
        dbc_file.write_text(dbc_content)
        
        # Setup components
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        parser = DBCParser(str(dbc_file))
        
        # Setup telemetry reader
        speed_frame = struct.pack("=IB3x8s", 0x100, 2,
                                  struct.pack('<H', 5000).ljust(8, b'\x00'))
        mock_socket.recv.return_value = speed_frame
        
        reader = TelemetryReader(
            can_interface=interface,
            dbc_parser=parser,
            message_ids={'speed': 0x100},
            update_rate=100.0
        )
        
        # Setup command sender
        sender = CommandSender(
            can_interface=interface,
            dbc_parser=parser,
            command_ids={'brake': 0x700},
            enable_control=True
        )
        
        # Start telemetry reading
        reader.start()
        time.sleep(0.1)
        
        # Send command
        sender.send_brake_command(0.8)
        
        # Stop telemetry
        time.sleep(0.1)
        reader.stop()
        
        # Verify telemetry was read
        telemetry = reader.get_telemetry()
        assert abs(telemetry.speed - 50.0) < 0.1
        
        # Verify command was sent
        assert mock_socket.send.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
