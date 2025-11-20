"""
DBC File Parser

Parses CAN database (DBC) files to extract message and signal definitions.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """CAN signal definition."""
    name: str
    start_bit: int
    length: int
    byte_order: str  # 'little' or 'big'
    signed: bool
    scale: float
    offset: float
    min_value: float
    max_value: float
    unit: str
    
    def decode(self, data: bytes) -> float:
        """
        Decode signal value from CAN data.
        
        Args:
            data: CAN message data bytes
        
        Returns:
            Decoded physical value
        """
        # Extract raw value from data
        raw_value = 0
        
        if self.byte_order == 'little':
            # Little endian (Intel format)
            byte_pos = self.start_bit // 8
            bit_pos = self.start_bit % 8
            
            for i in range(self.length):
                if byte_pos < len(data):
                    bit = (data[byte_pos] >> bit_pos) & 1
                    raw_value |= (bit << i)
                    bit_pos += 1
                    if bit_pos >= 8:
                        bit_pos = 0
                        byte_pos += 1
        else:
            # Big endian (Motorola format)
            byte_pos = self.start_bit // 8
            bit_pos = 7 - (self.start_bit % 8)
            
            for i in range(self.length):
                if byte_pos < len(data):
                    bit = (data[byte_pos] >> bit_pos) & 1
                    raw_value |= (bit << (self.length - 1 - i))
                    bit_pos -= 1
                    if bit_pos < 0:
                        bit_pos = 7
                        byte_pos += 1
        
        # Handle signed values
        if self.signed and raw_value >= (1 << (self.length - 1)):
            raw_value -= (1 << self.length)
        
        # Apply scale and offset
        physical_value = raw_value * self.scale + self.offset
        
        return physical_value
    
    def encode(self, value: float) -> int:
        """
        Encode physical value to raw integer.
        
        Args:
            value: Physical value to encode
        
        Returns:
            Raw integer value
        """
        # Clamp to min/max
        value = max(self.min_value, min(self.max_value, value))
        
        # Remove offset and scale
        raw_value = int((value - self.offset) / self.scale)
        
        # Handle signed values
        if self.signed and raw_value < 0:
            raw_value += (1 << self.length)
        
        # Mask to length
        raw_value &= ((1 << self.length) - 1)
        
        return raw_value


@dataclass
class Message:
    """CAN message definition."""
    message_id: int
    name: str
    dlc: int  # Data length code (number of bytes)
    signals: Dict[str, Signal]
    
    def decode(self, data: bytes) -> Dict[str, float]:
        """
        Decode all signals in message.
        
        Args:
            data: CAN message data bytes
        
        Returns:
            Dictionary of signal name to decoded value
        """
        result = {}
        for signal_name, signal in self.signals.items():
            try:
                result[signal_name] = signal.decode(data)
            except Exception as e:
                logger.error(f"Failed to decode signal {signal_name}: {e}")
        return result
    
    def encode(self, signal_values: Dict[str, float]) -> bytes:
        """
        Encode signal values into CAN message data.
        
        Args:
            signal_values: Dictionary of signal name to physical value
        
        Returns:
            Encoded CAN message data bytes
        """
        # Initialize data buffer
        data = bytearray(self.dlc)
        
        # Encode each signal
        for signal_name, value in signal_values.items():
            if signal_name not in self.signals:
                logger.warning(f"Unknown signal: {signal_name}")
                continue
            
            signal = self.signals[signal_name]
            raw_value = signal.encode(value)
            
            # Write raw value to data buffer
            if signal.byte_order == 'little':
                byte_pos = signal.start_bit // 8
                bit_pos = signal.start_bit % 8
                
                for i in range(signal.length):
                    if byte_pos < len(data):
                        bit = (raw_value >> i) & 1
                        if bit:
                            data[byte_pos] |= (1 << bit_pos)
                        else:
                            data[byte_pos] &= ~(1 << bit_pos)
                        bit_pos += 1
                        if bit_pos >= 8:
                            bit_pos = 0
                            byte_pos += 1
            else:
                # Big endian
                byte_pos = signal.start_bit // 8
                bit_pos = 7 - (signal.start_bit % 8)
                
                for i in range(signal.length):
                    if byte_pos < len(data):
                        bit = (raw_value >> (signal.length - 1 - i)) & 1
                        if bit:
                            data[byte_pos] |= (1 << bit_pos)
                        else:
                            data[byte_pos] &= ~(1 << bit_pos)
                        bit_pos -= 1
                        if bit_pos < 0:
                            bit_pos = 7
                            byte_pos += 1
        
        return bytes(data)


class DBCParser:
    """
    Parser for CAN database (DBC) files.
    
    Extracts message and signal definitions for encoding/decoding CAN traffic.
    """
    
    def __init__(self, dbc_file: Optional[str] = None):
        """
        Initialize DBC parser.
        
        Args:
            dbc_file: Path to DBC file (optional, can load later)
        """
        self.messages: Dict[int, Message] = {}
        self.message_names: Dict[str, int] = {}
        
        if dbc_file:
            self.load(dbc_file)
    
    def load(self, dbc_file: str) -> None:
        """
        Load and parse DBC file.
        
        Args:
            dbc_file: Path to DBC file
        """
        logger.info(f"Loading DBC file: {dbc_file}")
        
        try:
            with open(dbc_file, 'r') as f:
                content = f.read()
            
            self._parse_dbc(content)
            logger.info(f"Loaded {len(self.messages)} messages from DBC file")
            
        except FileNotFoundError:
            logger.error(f"DBC file not found: {dbc_file}")
            raise
        except Exception as e:
            logger.error(f"Failed to parse DBC file: {e}")
            raise
    
    def _parse_dbc(self, content: str) -> None:
        """
        Parse DBC file content.
        
        Args:
            content: DBC file content as string
        """
        # Parse messages
        message_pattern = r'BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)'
        for match in re.finditer(message_pattern, content):
            message_id = int(match.group(1))
            message_name = match.group(2)
            dlc = int(match.group(3))
            
            message = Message(
                message_id=message_id,
                name=message_name,
                dlc=dlc,
                signals={}
            )
            
            self.messages[message_id] = message
            self.message_names[message_name] = message_id
        
        # Parse signals
        signal_pattern = r'SG_\s+(\w+)\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*\(([^,]+),([^)]+)\)\s*\[([^|]+)\|([^\]]+)\]\s*"([^"]*)"\s*'
        
        current_message_id = None
        for line in content.split('\n'):
            # Check if this is a message line
            message_match = re.match(message_pattern, line)
            if message_match:
                current_message_id = int(message_match.group(1))
                continue
            
            # Check if this is a signal line
            signal_match = re.match(signal_pattern, line.strip())
            if signal_match and current_message_id is not None:
                signal_name = signal_match.group(1)
                start_bit = int(signal_match.group(2))
                length = int(signal_match.group(3))
                byte_order = 'little' if signal_match.group(4) == '1' else 'big'
                signed = signal_match.group(5) == '-'
                scale = float(signal_match.group(6))
                offset = float(signal_match.group(7))
                min_value = float(signal_match.group(8))
                max_value = float(signal_match.group(9))
                unit = signal_match.group(10)
                
                signal = Signal(
                    name=signal_name,
                    start_bit=start_bit,
                    length=length,
                    byte_order=byte_order,
                    signed=signed,
                    scale=scale,
                    offset=offset,
                    min_value=min_value,
                    max_value=max_value,
                    unit=unit
                )
                
                if current_message_id in self.messages:
                    self.messages[current_message_id].signals[signal_name] = signal
    
    def get_message(self, message_id: int) -> Optional[Message]:
        """
        Get message definition by ID.
        
        Args:
            message_id: CAN message ID
        
        Returns:
            Message definition or None if not found
        """
        return self.messages.get(message_id)
    
    def get_message_by_name(self, name: str) -> Optional[Message]:
        """
        Get message definition by name.
        
        Args:
            name: Message name
        
        Returns:
            Message definition or None if not found
        """
        message_id = self.message_names.get(name)
        if message_id is not None:
            return self.messages.get(message_id)
        return None
    
    def decode_message(self, message_id: int, data: bytes) -> Optional[Dict[str, float]]:
        """
        Decode CAN message.
        
        Args:
            message_id: CAN message ID
            data: Message data bytes
        
        Returns:
            Dictionary of signal values or None if message unknown
        """
        message = self.get_message(message_id)
        if message:
            return message.decode(data)
        return None
    
    def encode_message(self, message_id: int, signal_values: Dict[str, float]) -> Optional[bytes]:
        """
        Encode CAN message.
        
        Args:
            message_id: CAN message ID
            signal_values: Dictionary of signal name to value
        
        Returns:
            Encoded message data or None if message unknown
        """
        message = self.get_message(message_id)
        if message:
            return message.encode(signal_values)
        return None
