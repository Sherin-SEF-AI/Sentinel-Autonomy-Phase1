"""
SENTINEL GUI Application Entry Point

Launches the PyQt6 desktop application for SENTINEL system monitoring and control.
"""

import sys
import logging
import time
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Note: PYTHONPATH should include src directory
# sys.path modification removed to allow proper package imports

from core.logging import LoggerSetup
from core.config import ConfigManager
from gui.main_window import SENTINELMainWindow
from gui.themes import ThemeManager

# Module logger
logger = logging.getLogger(__name__)


def main():
    """Main entry point for GUI application"""
    start_time = time.time()
    
    # Setup logging
    logger.debug("Initializing logging system...")
    LoggerSetup.setup(log_level="INFO", console_output=True)
    
    logger.info("=" * 60)
    logger.info("SENTINEL GUI Application - Starting")
    logger.info("=" * 60)
    
    # Load configuration
    logger.debug("Loading configuration from configs/default.yaml...")
    try:
        config_start = time.time()
        config = ConfigManager('configs/default.yaml')
        config_duration = time.time() - config_start
        logger.debug(f"Configuration loaded: duration={config_duration*1000:.2f}ms")
        
        if not config.validate():
            logger.error("Configuration validation failed: invalid configuration file")
            return 1
        
        logger.info("Configuration validated successfully")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration loading failed: file not found - {e}")
        return 1
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}", exc_info=True)
        return 1
    
    # Enable high DPI scaling
    logger.debug("Configuring high DPI scaling...")
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
        logger.debug("High DPI scaling configured: policy=PassThrough")
    except Exception as e:
        logger.warning(f"High DPI scaling configuration failed: {e}")
    
    # Create application
    logger.debug("Creating QApplication instance...")
    try:
        app_start = time.time()
        app = QApplication(sys.argv)
        app.setApplicationName("SENTINEL")
        app.setOrganizationName("SENTINEL")
        app.setOrganizationDomain("sentinel-safety.com")
        app_duration = time.time() - app_start
        logger.debug(f"QApplication created: duration={app_duration*1000:.2f}ms")
        logger.info("Qt application initialized: name=SENTINEL, domain=sentinel-safety.com")
    except Exception as e:
        logger.error(f"QApplication creation failed: {e}", exc_info=True)
        return 1
    
    # Create theme manager
    logger.debug("Initializing theme manager...")
    try:
        theme_start = time.time()
        theme_manager = ThemeManager(app)
        theme_duration = time.time() - theme_start
        logger.debug(f"Theme manager initialized: duration={theme_duration*1000:.2f}ms")
        logger.info("Theme manager created successfully")
    except Exception as e:
        logger.error(f"Theme manager initialization failed: {e}", exc_info=True)
        return 1
    
    # Create and show main window
    logger.debug("Creating main window...")
    try:
        window_start = time.time()
        main_window = SENTINELMainWindow(theme_manager, config)
        window_duration = time.time() - window_start
        logger.debug(f"Main window created: duration={window_duration*1000:.2f}ms")
        
        logger.debug("Showing main window...")
        main_window.show()
        logger.info("Main window displayed successfully")
        
    except Exception as e:
        logger.error(f"Main window creation failed: {e}", exc_info=True)
        return 1
    
    # Log startup summary
    total_duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"SENTINEL GUI Application started successfully: total_duration={total_duration*1000:.2f}ms")
    logger.info("=" * 60)
    
    # Run event loop
    logger.debug("Entering Qt event loop...")
    try:
        exit_code = app.exec()
        logger.info(f"Qt event loop exited: exit_code={exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Qt event loop error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    main()
