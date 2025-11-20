"""
CAN Command Sender

Sends control commands to vehicle via CAN bus with safety checks.
"""

import logging
import time
from typing import Optional

from .interface import CANInterface
from .dbc_parser import DBCParser

logger = logging.getLogger(__name__)


class CommandSender:
    """
    Sends control commands to vehicle via CAN bus.
    
    Implements safety checks and limits for brake and steering interventions.
    """
    
    # Default command message IDs
    DEFAULT_COMMAND_IDS = {
        'brake': 0x700,
        'steering': 0x800,
    }
    
    # Safety limits
    MAX_BRAKE_COMMAND = 1.0  # Maximum brake command (0-1)
    MAX_STEERING_ANGLE = 0.5  # Maximum steering angle in radians (~28 degrees)
    COMMAND_TIMEOUT = 0.5  # Watchdog timeout in seconds
    
    def __init__(
        self,
        can_interface: CANInterface,
        dbc_parser: DBCParser,
        command_ids: Optional[dict] = None,
        enable_control: bool = False
    ):
        """
        Initialize command sender.
        
        Args:
            can_interface: CAN interface for communication
            dbc_parser: DBC parser for message encoding
            command_ids: Dictionary mapping command type to CAN message ID
            enable_control: Enable sending control commands (default: False for safety)
        """
        self.can_interface = can_interface
        self.dbc_parser = dbc_parser
        self.command_ids = command_ids or self.DEFAULT_COMMAND_IDS
        self.enable_control = enable_control
        
        # Watchdog tracking
        self.last_brake_command_time = 0.0
        self.last_steering_command_time = 0.0
        
        if enable_control:
            logger.warning("Control commands ENABLED - vehicle can be controlled via CAN bus")
        else:
            logger.info("Control commands DISABLED - commands will be logged but not sent")
    
    def send_brake_command(self, brake_value: float) -> bool:
        """
        Send brake intervention command.
        
        Args:
            brake_value: Brake command (0.0 = no braking, 1.0 = full braking)
        
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.enable_control:
            logger.debug(f"Brake command not sent (control disabled): {brake_value:.2f}")
            return False
        
        # Apply safety limits
        brake_value = max(0.0, min(self.MAX_BRAKE_COMMAND, brake_value))
        
        # Check watchdog timeout
        current_time = time.time()
        if current_time - self.last_brake_command_time > self.COMMAND_TIMEOUT:
            logger.warning("Brake command watchdog timeout - resetting")
        
        # Encode message
        message_id = self.command_ids['brake']
        signal_values = {'BrakeCommand': brake_value}
        
        data = self.dbc_parser.encode_message(message_id, signal_values)
        if data is None:
            logger.error(f"Failed to encode brake command message (ID: 0x{message_id:X})")
            return False
        
        # Send command
        success = self.can_interface.send_frame(message_id, data)
        
        if success:
            self.last_brake_command_time = current_time
            logger.info(f"Sent brake command: {brake_value:.2f}")
        else:
            logger.error("Failed to send brake command")
        
        return success
    
    def send_steering_command(self, steering_angle: float) -> bool:
        """
        Send steering intervention command.
        
        Args:
            steering_angle: Steering angle in radians (positive = left, negative = right)
        
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.enable_control:
            logger.debug(f"Steering command not sent (control disabled): {steering_angle:.3f} rad")
            return False
        
        # Apply safety limits
        steering_angle = max(-self.MAX_STEERING_ANGLE, min(self.MAX_STEERING_ANGLE, steering_angle))
        
        # Check watchdog timeout
        current_time = time.time()
        if current_time - self.last_steering_command_time > self.COMMAND_TIMEOUT:
            logger.warning("Steering command watchdog timeout - resetting")
        
        # Encode message
        message_id = self.command_ids['steering']
        signal_values = {'SteeringCommand': steering_angle}
        
        data = self.dbc_parser.encode_message(message_id, signal_values)
        if data is None:
            logger.error(f"Failed to encode steering command message (ID: 0x{message_id:X})")
            return False
        
        # Send command
        success = self.can_interface.send_frame(message_id, data)
        
        if success:
            self.last_steering_command_time = current_time
            logger.info(f"Sent steering command: {steering_angle:.3f} rad")
        else:
            logger.error("Failed to send steering command")
        
        return success
    
    def send_emergency_stop(self) -> bool:
        """
        Send emergency stop command (full braking).
        
        Returns:
            True if command sent successfully, False otherwise
        """
        logger.warning("Sending EMERGENCY STOP command")
        return self.send_brake_command(self.MAX_BRAKE_COMMAND)
    
    def release_control(self) -> bool:
        """
        Release all control commands (return control to driver).
        
        Returns:
            True if commands sent successfully, False otherwise
        """
        logger.info("Releasing vehicle control")
        
        # Send zero commands
        brake_success = self.send_brake_command(0.0)
        steering_success = self.send_steering_command(0.0)
        
        return brake_success and steering_success
    
    def check_watchdog(self) -> bool:
        """
        Check if watchdog timeout has occurred.
        
        Returns:
            True if watchdog is healthy, False if timeout occurred
        """
        current_time = time.time()
        
        brake_timeout = (current_time - self.last_brake_command_time) > self.COMMAND_TIMEOUT
        steering_timeout = (current_time - self.last_steering_command_time) > self.COMMAND_TIMEOUT
        
        if brake_timeout and self.last_brake_command_time > 0:
            logger.warning("Brake command watchdog timeout")
            return False
        
        if steering_timeout and self.last_steering_command_time > 0:
            logger.warning("Steering command watchdog timeout")
            return False
        
        return True
    
    def set_enable_control(self, enable: bool) -> None:
        """
        Enable or disable control commands.
        
        Args:
            enable: True to enable control, False to disable
        """
        if enable and not self.enable_control:
            logger.warning("Enabling vehicle control commands")
            self.enable_control = True
        elif not enable and self.enable_control:
            logger.info("Disabling vehicle control commands")
            self.enable_control = False
            # Release control when disabling
            self.release_control()
    
    def is_control_enabled(self) -> bool:
        """
        Check if control commands are enabled.
        
        Returns:
            True if control enabled, False otherwise
        """
        return self.enable_control
