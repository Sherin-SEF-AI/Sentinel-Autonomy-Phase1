"""
CAN Bus Interface

Provides SocketCAN connection management for Linux systems.
"""

import logging
import socket
import struct
import time
from typing import Optional, Tuple
from threading import Lock

logger = logging.getLogger(__name__)


class CANInterface:
    """
    SocketCAN interface for CAN bus communication.
    
    Handles connection, reconnection, and low-level CAN frame transmission/reception.
    """
    
    # CAN frame format constants
    CAN_EFF_FLAG = 0x80000000  # Extended frame format
    CAN_RTR_FLAG = 0x40000000  # Remote transmission request
    CAN_ERR_FLAG = 0x20000000  # Error frame
    CAN_EFF_MASK = 0x1FFFFFFF  # Extended frame ID mask
    CAN_SFF_MASK = 0x000007FF  # Standard frame ID mask
    
    def __init__(self, channel: str = 'can0', reconnect_delay: float = 1.0):
        """
        Initialize CAN interface.
        
        Args:
            channel: CAN interface name (e.g., 'can0', 'vcan0')
            reconnect_delay: Delay in seconds between reconnection attempts
        """
        self.channel = channel
        self.reconnect_delay = reconnect_delay
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.lock = Lock()
        
        logger.info(f"Initializing CAN interface on {channel}")
    
    def connect(self) -> bool:
        """
        Connect to CAN bus.
        
        Returns:
            True if connection successful, False otherwise
        """
        with self.lock:
            try:
                # Create SocketCAN socket
                self.socket = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                
                # Bind to CAN interface
                self.socket.bind((self.channel,))
                
                # Set socket timeout for non-blocking reads
                self.socket.settimeout(0.1)
                
                self.connected = True
                logger.info(f"Connected to CAN bus on {self.channel}")
                return True
                
            except OSError as e:
                logger.error(f"Failed to connect to CAN bus: {e}")
                self.connected = False
                self.socket = None
                return False
    
    def disconnect(self) -> None:
        """Disconnect from CAN bus."""
        with self.lock:
            if self.socket:
                try:
                    self.socket.close()
                    logger.info(f"Disconnected from CAN bus on {self.channel}")
                except Exception as e:
                    logger.error(f"Error closing CAN socket: {e}")
                finally:
                    self.socket = None
                    self.connected = False
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to CAN bus.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        logger.info(f"Attempting to reconnect to CAN bus on {self.channel}")
        self.disconnect()
        time.sleep(self.reconnect_delay)
        return self.connect()
    
    def is_connected(self) -> bool:
        """
        Check if connected to CAN bus.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected
    
    def send_frame(self, can_id: int, data: bytes, extended: bool = False) -> bool:
        """
        Send a CAN frame.
        
        Args:
            can_id: CAN message ID
            data: Message data (up to 8 bytes)
            extended: Use extended frame format (29-bit ID)
        
        Returns:
            True if frame sent successfully, False otherwise
        """
        if not self.connected or not self.socket:
            logger.warning("Cannot send frame: not connected to CAN bus")
            return False
        
        if len(data) > 8:
            logger.error(f"CAN data too long: {len(data)} bytes (max 8)")
            return False
        
        try:
            # Apply extended frame flag if needed
            if extended:
                can_id |= self.CAN_EFF_FLAG
            
            # Pack CAN frame: ID (4 bytes) + DLC (1 byte) + padding (3 bytes) + data (8 bytes)
            frame = struct.pack("=IB3x8s", can_id, len(data), data.ljust(8, b'\x00'))
            
            with self.lock:
                self.socket.send(frame)
            
            logger.debug(f"Sent CAN frame: ID=0x{can_id:X}, data={data.hex()}")
            return True
            
        except OSError as e:
            logger.error(f"Failed to send CAN frame: {e}")
            # Attempt reconnection on error
            self.reconnect()
            return False
    
    def receive_frame(self, timeout: Optional[float] = None) -> Optional[Tuple[int, bytes]]:
        """
        Receive a CAN frame.
        
        Args:
            timeout: Timeout in seconds (None uses socket default)
        
        Returns:
            Tuple of (can_id, data) if frame received, None otherwise
        """
        if not self.connected or not self.socket:
            return None
        
        try:
            # Set timeout if specified
            if timeout is not None:
                old_timeout = self.socket.gettimeout()
                self.socket.settimeout(timeout)
            
            # Receive CAN frame
            with self.lock:
                frame = self.socket.recv(16)  # CAN frame is 16 bytes
            
            # Restore timeout
            if timeout is not None:
                self.socket.settimeout(old_timeout)
            
            # Unpack frame
            can_id, dlc = struct.unpack("=IB3x", frame[:8])
            data = frame[8:8+dlc]
            
            # Remove extended/RTR/ERR flags from ID
            if can_id & self.CAN_EFF_FLAG:
                can_id &= self.CAN_EFF_MASK
            else:
                can_id &= self.CAN_SFF_MASK
            
            logger.debug(f"Received CAN frame: ID=0x{can_id:X}, data={data.hex()}")
            return (can_id, data)
            
        except socket.timeout:
            return None
        except OSError as e:
            logger.error(f"Failed to receive CAN frame: {e}")
            # Attempt reconnection on error
            self.reconnect()
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
