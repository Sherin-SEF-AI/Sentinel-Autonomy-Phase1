"""Test suite for CAN bus integration module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import socket
import struct

from src.can import CANInterface, DBCParser, TelemetryReader, CommandSender


@pytest.fixture
def mock_socket():
    """Fixture providing mock socket for CAN communication."""
    mock_sock = MagicMock(spec=socket.socket)
    mock_sock.bind = MagicMock()
    mock_sock.settimeout = MagicMock()
    mock_sock.send = MagicMock()
    mock_sock.recv = MagicMock()
    mock_sock.close = MagicMock()
    mock_sock.gettimeout = MagicMock(return_value=0.1)
    return mock_sock


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration for CAN bus testing."""
    config = {
        'can_bus': {
            'enabled': True,
            'interface': 'socketcan',
            'channel': 'vcan0',
            'dbc_file': 'configs/vehicle.dbc',
            'enable_control': False,
            'log_traffic': True,
            'log_file': 'logs/can_traffic.log',
            'message_ids': {
                'speed': 0x100,
                'steering': 0x200,
                'brake': 0x300,
                'throttle': 0x400,
                'gear': 0x500,
                'turn_signal': 0x600
            },
            'command_ids': {
                'brake': 0x700,
                'steering': 0x800
            },
            'telemetry': {
                'update_rate': 100.0
            },
            'commands': {
                'max_brake': 1.0,
                'max_steering_angle': 0.5,
                'watchdog_timeout': 0.5
            }
        }
    }
    return config


class TestCANInterface:
    """Test suite for CANInterface class."""
    
    def test_initialization(self):
        """Test that CANInterface initializes correctly with valid configuration."""
        interface = CANInterface(channel='vcan0', reconnect_delay=1.0)
        
        assert interface is not None
        assert interface.channel == 'vcan0'
        assert interface.reconnect_delay == 1.0
        assert interface.socket is None
        assert interface.connected is False
    
    @patch('socket.socket')
    def test_connect_success(self, mock_socket_class, mock_socket):
        """Test successful connection to CAN bus."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        result = interface.connect()
        
        assert result is True
        assert interface.connected is True
        assert interface.socket is not None
        mock_socket.bind.assert_called_once_with(('vcan0',))
        mock_socket.settimeout.assert_called_once_with(0.1)
    
    @patch('socket.socket')
    def test_connect_failure(self, mock_socket_class):
        """Test connection failure handling."""
        mock_socket_class.side_effect = OSError("CAN interface not found")
        
        interface = CANInterface(channel='vcan0')
        result = interface.connect()
        
        assert result is False
        assert interface.connected is False
        assert interface.socket is None
    
    @patch('socket.socket')
    def test_disconnect(self, mock_socket_class, mock_socket):
        """Test disconnection from CAN bus."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        interface.disconnect()
        
        assert interface.connected is False
        assert interface.socket is None
        mock_socket.close.assert_called_once()
    
    @patch('socket.socket')
    @patch('time.sleep')
    def test_reconnect(self, mock_sleep, mock_socket_class, mock_socket):
        """Test reconnection to CAN bus."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0', reconnect_delay=1.0)
        interface.connect()
        result = interface.reconnect()
        
        assert result is True
        assert interface.connected is True
        mock_sleep.assert_called_once_with(1.0)
    
    @patch('socket.socket')
    def test_send_frame_success(self, mock_socket_class, mock_socket):
        """Test sending CAN frame successfully."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        can_id = 0x100
        data = b'\x01\x02\x03\x04'
        result = interface.send_frame(can_id, data)
        
        assert result is True
        mock_socket.send.assert_called_once()
        
        # Verify frame format
        call_args = mock_socket.send.call_args[0][0]
        unpacked_id, unpacked_dlc = struct.unpack("=IB3x", call_args[:8])
        assert unpacked_id == can_id
        assert unpacked_dlc == len(data)
    
    @patch('socket.socket')
    def test_send_frame_extended_id(self, mock_socket_class, mock_socket):
        """Test sending CAN frame with extended ID."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        can_id = 0x12345678
        data = b'\xFF\xEE'
        result = interface.send_frame(can_id, data, extended=True)
        
        assert result is True
        mock_socket.send.assert_called_once()
        
        # Verify extended flag is set
        call_args = mock_socket.send.call_args[0][0]
        unpacked_id, _ = struct.unpack("=IB3x", call_args[:8])
        assert unpacked_id & interface.CAN_EFF_FLAG
    
    @patch('socket.socket')
    def test_send_frame_not_connected(self, mock_socket_class):
        """Test sending frame when not connected."""
        interface = CANInterface(channel='vcan0')
        
        result = interface.send_frame(0x100, b'\x01\x02')
        
        assert result is False
    
    @patch('socket.socket')
    def test_send_frame_data_too_long(self, mock_socket_class, mock_socket):
        """Test sending frame with data exceeding 8 bytes."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        data = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09'  # 9 bytes
        result = interface.send_frame(0x100, data)
        
        assert result is False
        mock_socket.send.assert_not_called()
    
    @patch('socket.socket')
    def test_receive_frame_success(self, mock_socket_class, mock_socket):
        """Test receiving CAN frame successfully."""
        mock_socket_class.return_value = mock_socket
        
        # Create mock CAN frame
        can_id = 0x200
        data = b'\xAA\xBB\xCC\xDD'
        frame = struct.pack("=IB3x8s", can_id, len(data), data.ljust(8, b'\x00'))
        mock_socket.recv.return_value = frame
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        result = interface.receive_frame()
        
        assert result is not None
        received_id, received_data = result
        assert received_id == can_id
        assert received_data == data
    
    @patch('socket.socket')
    def test_receive_frame_timeout(self, mock_socket_class, mock_socket):
        """Test receiving frame with timeout."""
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = socket.timeout()
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        result = interface.receive_frame(timeout=0.5)
        
        assert result is None
    
    @patch('socket.socket')
    def test_receive_frame_not_connected(self, mock_socket_class):
        """Test receiving frame when not connected."""
        interface = CANInterface(channel='vcan0')
        
        result = interface.receive_frame()
        
        assert result is None
    
    @patch('socket.socket')
    def test_context_manager(self, mock_socket_class, mock_socket):
        """Test CANInterface as context manager."""
        mock_socket_class.return_value = mock_socket
        
        with CANInterface(channel='vcan0') as interface:
            assert interface.connected is True
        
        assert interface.connected is False
        mock_socket.close.assert_called_once()
    
    @patch('socket.socket')
    def test_is_connected(self, mock_socket_class, mock_socket):
        """Test connection status check."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        assert interface.is_connected() is False
        
        interface.connect()
        assert interface.is_connected() is True
        
        interface.disconnect()
        assert interface.is_connected() is False


class TestDBCParser:
    """Test suite for DBCParser class."""
    
    def test_initialization(self, tmp_path):
        """Test that DBCParser initializes correctly."""
        # Create a minimal DBC file
        dbc_file = tmp_path / "test.dbc"
        dbc_file.write_text("""
VERSION ""

NS_ :
    NS_DESC_
    CM_
    BA_DEF_
    BA_
    VAL_
    CAT_DEF_
    CAT_
    FILTER
    BA_DEF_DEF_
    EV_DATA_
    ENVVAR_DATA_
    SGTYPE_
    SGTYPE_VAL_
    BA_DEF_SGTYPE_
    BA_SGTYPE_
    SIG_TYPE_REF_
    VAL_TABLE_
    SIG_GROUP_
    SIG_VALTYPE_
    SIGTYPE_VALTYPE_
    BO_TX_BU_
    BA_DEF_REL_
    BA_REL_
    BA_SGTYPE_REL_
    SG_MUL_VAL_

BS_:

BU_:

BO_ 256 Speed: 8 Vector__XXX
 SG_ vehicle_speed : 0|16@1+ (0.01,0) [0|655.35] "m/s" Vector__XXX
""")
        
        parser = DBCParser(str(dbc_file))
        assert parser is not None
    
    def test_parse_message(self, tmp_path):
        """Test parsing CAN message from DBC definition."""
        dbc_file = tmp_path / "test.dbc"
        dbc_file.write_text("""
VERSION ""
NS_ :
BS_:
BU_:
BO_ 256 Speed: 8 Vector__XXX
 SG_ vehicle_speed : 0|16@1+ (0.01,0) [0|655.35] "m/s" Vector__XXX
""")
        
        parser = DBCParser(str(dbc_file))
        
        # Test decoding a speed message
        data = struct.pack(">H", 5000)  # 50.00 m/s
        decoded = parser.decode_message(0x100, data)
        
        assert decoded is not None
        assert 'vehicle_speed' in decoded


class TestTelemetryReader:
    """Test suite for TelemetryReader class."""
    
    @patch('src.can.interface.CANInterface')
    def test_initialization(self, mock_interface_class, mock_config):
        """Test that TelemetryReader initializes correctly."""
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface
        
        reader = TelemetryReader(mock_config, mock_interface)
        
        assert reader is not None
        assert reader.interface == mock_interface
    
    @patch('src.can.interface.CANInterface')
    def test_start_reading(self, mock_interface_class, mock_config):
        """Test starting telemetry reading thread."""
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface
        
        reader = TelemetryReader(mock_config, mock_interface)
        reader.start()
        
        assert reader.is_running is True
    
    @patch('src.can.interface.CANInterface')
    def test_stop_reading(self, mock_interface_class, mock_config):
        """Test stopping telemetry reading thread."""
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface
        
        reader = TelemetryReader(mock_config, mock_interface)
        reader.start()
        reader.stop()
        
        assert reader.is_running is False
    
    @patch('src.can.interface.CANInterface')
    def test_get_latest_telemetry(self, mock_interface_class, mock_config):
        """Test retrieving latest telemetry data."""
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface
        
        reader = TelemetryReader(mock_config, mock_interface)
        telemetry = reader.get_latest_telemetry()
        
        assert telemetry is not None
        assert hasattr(telemetry, 'speed')
        assert hasattr(telemetry, 'steering_angle')


class TestCommandSender:
    """Test suite for CommandSender class."""
    
    @patch('src.can.interface.CANInterface')
    def test_initialization(self, mock_interface_class, mock_config):
        """Test that CommandSender initializes correctly."""
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface
        
        sender = CommandSender(mock_config, mock_interface)
        
        assert sender is not None
        assert sender.interface == mock_interface
    
    @patch('src.can.interface.CANInterface')
    def test_send_brake_command(self, mock_interface_class, mock_config):
        """Test sending brake command."""
        mock_interface = MagicMock()
        mock_interface.send_frame = MagicMock(return_value=True)
        mock_interface_class.return_value = mock_interface
        
        sender = CommandSender(mock_config, mock_interface)
        result = sender.send_brake_command(0.5)
        
        assert result is True
        mock_interface.send_frame.assert_called_once()
    
    @patch('src.can.interface.CANInterface')
    def test_send_brake_command_exceeds_max(self, mock_interface_class, mock_config):
        """Test brake command clamping to maximum value."""
        mock_interface = MagicMock()
        mock_interface.send_frame = MagicMock(return_value=True)
        mock_interface_class.return_value = mock_interface
        
        sender = CommandSender(mock_config, mock_interface)
        result = sender.send_brake_command(1.5)  # Exceeds max of 1.0
        
        assert result is True
        # Verify command was clamped
        call_args = mock_interface.send_frame.call_args
        assert call_args is not None
    
    @patch('src.can.interface.CANInterface')
    def test_send_steering_command(self, mock_interface_class, mock_config):
        """Test sending steering command."""
        mock_interface = MagicMock()
        mock_interface.send_frame = MagicMock(return_value=True)
        mock_interface_class.return_value = mock_interface
        
        sender = CommandSender(mock_config, mock_interface)
        result = sender.send_steering_command(0.2)
        
        assert result is True
        mock_interface.send_frame.assert_called_once()
    
    @patch('src.can.interface.CANInterface')
    def test_watchdog_timeout(self, mock_interface_class, mock_config):
        """Test watchdog timeout mechanism."""
        mock_interface = MagicMock()
        mock_interface_class.return_value = mock_interface
        
        sender = CommandSender(mock_config, mock_interface)
        
        # Verify watchdog is initialized
        assert hasattr(sender, 'watchdog_timeout')
        assert sender.watchdog_timeout == 0.5


class TestCANModuleIntegration:
    """Integration tests for CAN module components."""
    
    @patch('socket.socket')
    def test_full_telemetry_pipeline(self, mock_socket_class, mock_socket, mock_config):
        """Test complete telemetry reading pipeline."""
        mock_socket_class.return_value = mock_socket
        
        # Create mock CAN frames for speed message
        speed_frame = struct.pack("=IB3x8s", 0x100, 2, struct.pack(">H", 5000).ljust(8, b'\x00'))
        mock_socket.recv.return_value = speed_frame
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        # Receive and verify frame
        result = interface.receive_frame()
        assert result is not None
        can_id, data = result
        assert can_id == 0x100
    
    @patch('socket.socket')
    def test_command_sending_pipeline(self, mock_socket_class, mock_socket, mock_config):
        """Test complete command sending pipeline."""
        mock_socket_class.return_value = mock_socket
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        # Send brake command
        result = interface.send_frame(0x700, b'\x80\x00')
        assert result is True
        mock_socket.send.assert_called_once()
    
    @pytest.mark.performance
    @patch('socket.socket')
    def test_performance_send_receive(self, mock_socket_class, mock_socket):
        """Test that CAN operations complete within performance requirements."""
        import time
        
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.return_value = struct.pack("=IB3x8s", 0x100, 4, b'\x01\x02\x03\x04\x00\x00\x00\x00')
        
        interface = CANInterface(channel='vcan0')
        interface.connect()
        
        # Test send performance
        start_time = time.perf_counter()
        for _ in range(100):
            interface.send_frame(0x100, b'\x01\x02\x03\x04')
        send_time = time.perf_counter() - start_time
        
        # Test receive performance
        start_time = time.perf_counter()
        for _ in range(100):
            interface.receive_frame()
        receive_time = time.perf_counter() - start_time
        
        # Each operation should be fast (< 1ms average)
        assert send_time / 100 < 0.001, f"Send took {send_time/100*1000:.2f}ms avg, expected < 1ms"
        assert receive_time / 100 < 0.001, f"Receive took {receive_time/100*1000:.2f}ms avg, expected < 1ms"
