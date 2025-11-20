#!/usr/bin/env python3
"""
Test script to demonstrate GUI main logging output.

This script simulates the logging that would occur during GUI startup
without actually launching the GUI (which requires PyQt6).
"""

import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.logging import LoggerSetup

# Setup logging
LoggerSetup.setup(log_level="DEBUG", console_output=True)
logger = logging.getLogger('src.gui_main')


def simulate_gui_startup():
    """Simulate GUI startup logging sequence."""
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("SENTINEL GUI Application - Starting")
    logger.info("=" * 60)
    
    # Configuration loading
    logger.debug("Loading configuration from configs/default.yaml...")
    time.sleep(0.015)  # Simulate 15ms
    config_duration = 0.015
    logger.debug(f"Configuration loaded: duration={config_duration*1000:.2f}ms")
    logger.info("Configuration validated successfully")
    
    # High DPI scaling
    logger.debug("Configuring high DPI scaling...")
    logger.debug("High DPI scaling configured: policy=PassThrough")
    
    # QApplication creation
    logger.debug("Creating QApplication instance...")
    time.sleep(0.045)  # Simulate 45ms
    app_duration = 0.045
    logger.debug(f"QApplication created: duration={app_duration*1000:.2f}ms")
    logger.info("Qt application initialized: name=SENTINEL, domain=sentinel-safety.com")
    
    # Theme manager
    logger.debug("Initializing theme manager...")
    time.sleep(0.012)  # Simulate 12ms
    theme_duration = 0.012
    logger.debug(f"Theme manager initialized: duration={theme_duration*1000:.2f}ms")
    logger.info("Theme manager created successfully")
    
    # Main window
    logger.debug("Creating main window...")
    time.sleep(0.235)  # Simulate 235ms
    window_duration = 0.235
    logger.debug(f"Main window created: duration={window_duration*1000:.2f}ms")
    logger.debug("Showing main window...")
    logger.info("Main window displayed successfully")
    
    # Startup complete
    total_duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"SENTINEL GUI Application started successfully: total_duration={total_duration*1000:.2f}ms")
    logger.info("=" * 60)
    
    # Event loop
    logger.debug("Entering Qt event loop...")
    time.sleep(0.1)  # Simulate brief event loop
    logger.info("Qt event loop exited: exit_code=0")


def simulate_error_scenario():
    """Simulate error logging."""
    print("\n" + "=" * 60)
    print("Simulating Error Scenario")
    print("=" * 60 + "\n")
    
    logger.info("=" * 60)
    logger.info("SENTINEL GUI Application - Starting")
    logger.info("=" * 60)
    
    logger.debug("Loading configuration from configs/default.yaml...")
    
    try:
        # Simulate configuration error
        raise FileNotFoundError("configs/default.yaml")
    except FileNotFoundError as e:
        logger.error(f"Configuration loading failed: file not found - {e}")
        return 1
    
    return 0


def main():
    """Run logging demonstrations."""
    print("\n" + "=" * 60)
    print("GUI Main Logging Output Demonstration")
    print("=" * 60)
    print("\nThis demonstrates the logging output from src/gui_main.py")
    print("during normal startup and error scenarios.\n")
    
    print("=" * 60)
    print("Normal Startup Scenario")
    print("=" * 60 + "\n")
    
    simulate_gui_startup()
    
    simulate_error_scenario()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)
    print("\nKey observations:")
    print("  ✓ Module logger uses 'src.gui_main' namespace")
    print("  ✓ DEBUG logs show detailed initialization steps")
    print("  ✓ INFO logs mark major milestones")
    print("  ✓ Performance timing in milliseconds")
    print("  ✓ ERROR logs include context")
    print("  ✓ Visual separators for readability")
    print("\nLogs are written to: logs/sentinel_YYYYMMDD_HHMMSS.log")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
