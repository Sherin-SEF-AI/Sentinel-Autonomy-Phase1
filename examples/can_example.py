"""
CAN Bus Integration Example

Demonstrates how to use the CAN bus integration module to read vehicle
telemetry and send control commands.
"""

import time
import logging
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.can import (
    CANInterface,
    DBCParser,
    TelemetryReader,
    CommandSender,
    CANLogger
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    
    # Configuration
    can_channel = 'vcan0'  # Use virtual CAN for testing
    dbc_file = 'configs/vehicle.dbc'
    log_file = 'logs/can_traffic.log'
    
    logger.info("=== CAN Bus Integration Example ===")
    
    # Initialize CAN interface
    logger.info(f"Connecting to CAN bus on {can_channel}...")
    can_interface = CANInterface(channel=can_channel)
    
    if not can_interface.connect():
        logger.error("Failed to connect to CAN bus")
        logger.info("Make sure virtual CAN is set up:")
        logger.info("  sudo modprobe vcan")
        logger.info("  sudo ip link add dev vcan0 type vcan")
        logger.info("  sudo ip link set vcan0 up")
        return
    
    logger.info("Connected to CAN bus")
    
    # Load DBC file
    logger.info(f"Loading DBC file: {dbc_file}")
    dbc_parser = DBCParser(dbc_file)
    logger.info(f"Loaded {len(dbc_parser.messages)} message definitions")
    
    # Initialize telemetry reader
    logger.info("Starting telemetry reader...")
    telemetry_reader = TelemetryReader(
        can_interface=can_interface,
        dbc_parser=dbc_parser,
        update_rate=100.0  # 100 Hz
    )
    telemetry_reader.start()
    
    # Initialize CAN logger
    logger.info("Starting CAN logger...")
    can_logger = CANLogger(
        can_interface=can_interface,
        log_file=log_file,
        enable_logging=True
    )
    can_logger.start()
    
    # Initialize command sender (control disabled for safety)
    logger.info("Initializing command sender (control disabled)...")
    command_sender = CommandSender(
        can_interface=can_interface,
        dbc_parser=dbc_parser,
        enable_control=False  # Disabled for safety
    )
    
    try:
        # Read telemetry for 10 seconds
        logger.info("\n=== Reading Telemetry ===")
        logger.info("Reading telemetry for 10 seconds...")
        logger.info("(Use 'cansend vcan0 100#0102030405060708' to send test messages)")
        
        start_time = time.time()
        while time.time() - start_time < 10.0:
            # Get latest telemetry
            telemetry = telemetry_reader.get_telemetry()
            
            # Print telemetry every second
            if int(time.time() - start_time) % 1 == 0:
                logger.info(f"\nCurrent Telemetry:")
                logger.info(f"  Speed: {telemetry.speed:.2f} m/s")
                logger.info(f"  Steering: {telemetry.steering_angle:.3f} rad")
                logger.info(f"  Brake: {telemetry.brake_pressure:.2f} bar")
                logger.info(f"  Throttle: {telemetry.throttle_position:.2f}")
                logger.info(f"  Gear: {telemetry.gear}")
                logger.info(f"  Turn Signal: {telemetry.turn_signal}")
                time.sleep(1.0)
        
        # Demonstrate command sending (will be logged but not sent)
        logger.info("\n=== Command Sending Demo ===")
        logger.info("Attempting to send commands (control disabled)...")
        
        # Try to send brake command
        logger.info("Sending brake command: 0.3 (30%)")
        command_sender.send_brake_command(0.3)
        
        # Try to send steering command
        logger.info("Sending steering command: 0.1 rad (left)")
        command_sender.send_steering_command(0.1)
        
        # Show how to enable control (not recommended without proper safety measures)
        logger.info("\nTo enable control commands:")
        logger.info("  command_sender.set_enable_control(True)")
        logger.info("  WARNING: Only enable in controlled test environment!")
        
        # Show logging statistics
        logger.info(f"\n=== Logging Statistics ===")
        logger.info(f"Messages logged: {can_logger.get_message_count()}")
        logger.info(f"Log file: {log_file}")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    finally:
        # Clean up
        logger.info("\n=== Shutting Down ===")
        
        logger.info("Stopping telemetry reader...")
        telemetry_reader.stop()
        
        logger.info("Stopping CAN logger...")
        can_logger.stop()
        
        logger.info("Disconnecting from CAN bus...")
        can_interface.disconnect()
        
        logger.info("Done!")


def playback_example():
    """Example of playing back CAN log."""
    from src.can import CANPlayback
    
    log_file = 'logs/can_traffic.log'
    
    logger.info("=== CAN Playback Example ===")
    logger.info(f"Loading log file: {log_file}")
    
    playback = CANPlayback(log_file)
    
    if not playback.load():
        logger.error("Failed to load log file")
        return
    
    logger.info(f"Loaded {playback.get_message_count()} messages")
    logger.info(f"Duration: {playback.get_duration():.2f} seconds")
    
    # Play back first 10 messages
    logger.info("\nPlaying back first 10 messages:")
    for i in range(min(10, playback.get_message_count())):
        timestamp, message_id, data = playback.get_next_message()
        logger.info(f"  [{timestamp:.6f}] ID=0x{message_id:X} Data={data.hex()}")
    
    logger.info("\nPlayback complete")


if __name__ == '__main__':
    # Run main example
    main()
    
    # Uncomment to run playback example
    # playback_example()
