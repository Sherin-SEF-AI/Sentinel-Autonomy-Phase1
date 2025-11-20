"""
CAN Telemetry Reader

Reads and decodes vehicle telemetry from CAN bus at 100 Hz.
"""

import logging
import time
from threading import Thread, Lock
from typing import Optional

from src.core.data_structures import VehicleTelemetry
from .interface import CANInterface
from .dbc_parser import DBCParser

logger = logging.getLogger(__name__)


class TelemetryReader:
    """
    Reads vehicle telemetry from CAN bus.
    
    Continuously reads CAN messages and decodes them into VehicleTelemetry dataclass.
    """
    
    # Default message IDs (can be overridden via config)
    DEFAULT_MESSAGE_IDS = {
        'speed': 0x100,
        'steering': 0x200,
        'brake': 0x300,
        'throttle': 0x400,
        'gear': 0x500,
        'turn_signal': 0x600,
    }
    
    def __init__(
        self,
        can_interface: CANInterface,
        dbc_parser: DBCParser,
        message_ids: Optional[dict] = None,
        update_rate: float = 100.0
    ):
        """
        Initialize telemetry reader.
        
        Args:
            can_interface: CAN interface for communication
            dbc_parser: DBC parser for message decoding
            message_ids: Dictionary mapping telemetry type to CAN message ID
            update_rate: Target update rate in Hz
        """
        self.can_interface = can_interface
        self.dbc_parser = dbc_parser
        self.message_ids = message_ids or self.DEFAULT_MESSAGE_IDS
        self.update_rate = update_rate
        
        # Current telemetry state
        self.telemetry = VehicleTelemetry(
            timestamp=time.time(),
            speed=0.0,
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.0,
            gear=0,
            turn_signal='none'
        )
        self.telemetry_lock = Lock()
        
        # Thread control
        self.running = False
        self.thread: Optional[Thread] = None
        
        logger.info(f"Initialized telemetry reader at {update_rate} Hz")
    
    def start(self) -> None:
        """Start telemetry reading thread."""
        if self.running:
            logger.warning("Telemetry reader already running")
            return
        
        self.running = True
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("Started telemetry reader thread")
    
    def stop(self) -> None:
        """Stop telemetry reading thread."""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("Stopped telemetry reader thread")
    
    def get_telemetry(self) -> VehicleTelemetry:
        """
        Get latest telemetry data.
        
        Returns:
            Current VehicleTelemetry dataclass
        """
        with self.telemetry_lock:
            return VehicleTelemetry(
                timestamp=self.telemetry.timestamp,
                speed=self.telemetry.speed,
                steering_angle=self.telemetry.steering_angle,
                brake_pressure=self.telemetry.brake_pressure,
                throttle_position=self.telemetry.throttle_position,
                gear=self.telemetry.gear,
                turn_signal=self.telemetry.turn_signal
            )
    
    def _read_loop(self) -> None:
        """Main telemetry reading loop."""
        loop_period = 1.0 / self.update_rate
        
        logger.info("Telemetry reading loop started")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Read CAN frame with timeout
                frame = self.can_interface.receive_frame(timeout=loop_period)
                
                if frame:
                    message_id, data = frame
                    self._process_frame(message_id, data)
            
            except Exception as e:
                logger.error(f"Error in telemetry reading loop: {e}")
            
            # Sleep to maintain update rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, loop_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.info("Telemetry reading loop stopped")
    
    def _process_frame(self, message_id: int, data: bytes) -> None:
        """
        Process received CAN frame.
        
        Args:
            message_id: CAN message ID
            data: Message data bytes
        """
        # Decode message
        decoded = self.dbc_parser.decode_message(message_id, data)
        if not decoded:
            return
        
        # Update telemetry based on message ID
        with self.telemetry_lock:
            self.telemetry.timestamp = time.time()
            
            if message_id == self.message_ids.get('speed'):
                # Speed message
                if 'Speed' in decoded:
                    self.telemetry.speed = decoded['Speed']
                    logger.debug(f"Updated speed: {self.telemetry.speed:.2f} m/s")
            
            elif message_id == self.message_ids.get('steering'):
                # Steering angle message
                if 'Angle' in decoded:
                    self.telemetry.steering_angle = decoded['Angle']
                    logger.debug(f"Updated steering: {self.telemetry.steering_angle:.3f} rad")
            
            elif message_id == self.message_ids.get('brake'):
                # Brake pressure message
                if 'Pressure' in decoded:
                    self.telemetry.brake_pressure = decoded['Pressure']
                    logger.debug(f"Updated brake: {self.telemetry.brake_pressure:.2f} bar")
            
            elif message_id == self.message_ids.get('throttle'):
                # Throttle position message
                if 'Position' in decoded:
                    self.telemetry.throttle_position = decoded['Position']
                    logger.debug(f"Updated throttle: {self.telemetry.throttle_position:.2f}")
            
            elif message_id == self.message_ids.get('gear'):
                # Gear message
                if 'Gear' in decoded:
                    self.telemetry.gear = int(decoded['Gear'])
                    logger.debug(f"Updated gear: {self.telemetry.gear}")
            
            elif message_id == self.message_ids.get('turn_signal'):
                # Turn signal message
                if 'Signal' in decoded:
                    signal_value = int(decoded['Signal'])
                    if signal_value == 1:
                        self.telemetry.turn_signal = 'left'
                    elif signal_value == 2:
                        self.telemetry.turn_signal = 'right'
                    else:
                        self.telemetry.turn_signal = 'none'
                    logger.debug(f"Updated turn signal: {self.telemetry.turn_signal}")
    
    def is_running(self) -> bool:
        """
        Check if telemetry reader is running.
        
        Returns:
            True if running, False otherwise
        """
        return self.running
