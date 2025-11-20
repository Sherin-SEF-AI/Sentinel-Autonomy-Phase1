"""
CAN Bus Logger

Logs all CAN messages to file with timestamps for debugging and playback.
"""

import logging
import time
import json
from pathlib import Path
from typing import Optional, List, Tuple
from threading import Thread, Lock

from .interface import CANInterface

logger = logging.getLogger(__name__)


class CANLogger:
    """
    Logs CAN bus traffic to file.
    
    Records all CAN messages with timestamps for debugging and playback.
    """
    
    def __init__(
        self,
        can_interface: CANInterface,
        log_file: str = "logs/can_traffic.log",
        enable_logging: bool = True
    ):
        """
        Initialize CAN logger.
        
        Args:
            can_interface: CAN interface to monitor
            log_file: Path to log file
            enable_logging: Enable logging (default: True)
        """
        self.can_interface = can_interface
        self.log_file = Path(log_file)
        self.enable_logging = enable_logging
        
        # Create log directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Logging state
        self.running = False
        self.thread: Optional[Thread] = None
        self.file_handle = None
        self.message_count = 0
        self.lock = Lock()
        
        logger.info(f"CAN Logger initialized: log_file={log_file}, enabled={enable_logging}")
    
    def start(self) -> None:
        """Start CAN logging thread."""
        if not self.enable_logging:
            logger.info("CAN logging disabled")
            return
        
        if self.running:
            logger.warning("CAN logger already running")
            return
        
        try:
            # Open log file
            self.file_handle = open(self.log_file, 'a')
            
            # Write header
            self.file_handle.write(f"# CAN Bus Log - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.write("# Format: timestamp,message_id,data_hex\n")
            self.file_handle.flush()
            
            # Start logging thread
            self.running = True
            self.thread = Thread(target=self._logging_loop, daemon=True)
            self.thread.start()
            
            logger.info("Started CAN logging thread")
            
        except Exception as e:
            logger.error(f"Failed to start CAN logging: {e}")
            self.running = False
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
    
    def stop(self) -> None:
        """Stop CAN logging thread."""
        if not self.running:
            return
        
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.file_handle:
            self.file_handle.write(f"# CAN Bus Log - Stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file_handle.write(f"# Total messages logged: {self.message_count}\n")
            self.file_handle.close()
            self.file_handle = None
        
        logger.info(f"Stopped CAN logging. Total messages: {self.message_count}")
    
    def _logging_loop(self) -> None:
        """Main CAN logging loop."""
        logger.info("CAN logging loop started")
        
        while self.running:
            try:
                # Read CAN frame
                frame = self.can_interface.receive_frame(timeout=0.1)
                
                if frame:
                    message_id, data = frame
                    self._log_message(message_id, data)
            
            except Exception as e:
                logger.error(f"Error in CAN logging loop: {e}")
        
        logger.info("CAN logging loop stopped")
    
    def _log_message(self, message_id: int, data: bytes) -> None:
        """
        Log a CAN message.
        
        Args:
            message_id: CAN message ID
            data: Message data bytes
        """
        if not self.file_handle:
            return
        
        try:
            timestamp = time.time()
            data_hex = data.hex()
            
            # Write log entry
            with self.lock:
                self.file_handle.write(f"{timestamp:.6f},0x{message_id:X},{data_hex}\n")
                self.file_handle.flush()
                self.message_count += 1
        
        except Exception as e:
            logger.error(f"Failed to log CAN message: {e}")
    
    def is_running(self) -> bool:
        """
        Check if logger is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running
    
    def get_message_count(self) -> int:
        """
        Get total number of messages logged.
        
        Returns:
            Message count
        """
        return self.message_count


class CANPlayback:
    """
    Plays back CAN messages from log file.
    
    Supports frame-by-frame playback and speed control.
    """
    
    def __init__(self, log_file: str):
        """
        Initialize CAN playback.
        
        Args:
            log_file: Path to CAN log file
        """
        self.log_file = Path(log_file)
        self.messages: List[Tuple[float, int, bytes]] = []
        self.current_index = 0
        
        logger.info(f"CAN Playback initialized: log_file={log_file}")
    
    def load(self) -> bool:
        """
        Load CAN log file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse log entry
                    parts = line.split(',')
                    if len(parts) != 3:
                        continue
                    
                    timestamp = float(parts[0])
                    message_id = int(parts[1], 16)
                    data = bytes.fromhex(parts[2])
                    
                    self.messages.append((timestamp, message_id, data))
            
            logger.info(f"Loaded {len(self.messages)} CAN messages from log")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load CAN log: {e}")
            return False
    
    def get_message_count(self) -> int:
        """
        Get total number of messages in log.
        
        Returns:
            Message count
        """
        return len(self.messages)
    
    def get_current_index(self) -> int:
        """
        Get current playback index.
        
        Returns:
            Current index
        """
        return self.current_index
    
    def seek(self, index: int) -> None:
        """
        Seek to specific message index.
        
        Args:
            index: Message index
        """
        self.current_index = max(0, min(index, len(self.messages) - 1))
    
    def get_next_message(self) -> Optional[Tuple[float, int, bytes]]:
        """
        Get next message in playback.
        
        Returns:
            Tuple of (timestamp, message_id, data) or None if at end
        """
        if self.current_index >= len(self.messages):
            return None
        
        message = self.messages[self.current_index]
        self.current_index += 1
        return message
    
    def get_message_at(self, index: int) -> Optional[Tuple[float, int, bytes]]:
        """
        Get message at specific index.
        
        Args:
            index: Message index
        
        Returns:
            Tuple of (timestamp, message_id, data) or None if invalid index
        """
        if 0 <= index < len(self.messages):
            return self.messages[index]
        return None
    
    def reset(self) -> None:
        """Reset playback to beginning."""
        self.current_index = 0
    
    def get_duration(self) -> float:
        """
        Get total duration of log in seconds.
        
        Returns:
            Duration in seconds
        """
        if len(self.messages) < 2:
            return 0.0
        
        start_time = self.messages[0][0]
        end_time = self.messages[-1][0]
        return end_time - start_time
    
    def get_messages_in_range(self, start_time: float, end_time: float) -> List[Tuple[float, int, bytes]]:
        """
        Get all messages within time range.
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            List of messages in range
        """
        result = []
        for timestamp, message_id, data in self.messages:
            if start_time <= timestamp <= end_time:
                result.append((timestamp, message_id, data))
        return result
