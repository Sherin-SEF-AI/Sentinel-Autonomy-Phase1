#!/usr/bin/env python3
"""
Test script for Configuration Dock Widget

Tests the configuration dock widget with all tabs and controls.
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt

# Add src to path
sys.path.insert(0, 'src')

from gui.widgets import ConfigurationDockWidget

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main test function"""
    logger.info("Starting Configuration Dock Widget test")
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Configuration Dock Widget Test")
    window.setGeometry(100, 100, 1200, 800)
    
    # Create configuration dock
    config_dock = ConfigurationDockWidget('configs/default.yaml')
    
    # Connect signals
    config_dock.config_changed.connect(
        lambda config: logger.info(f"Configuration changed: {len(config)} keys")
    )
    config_dock.config_saved.connect(
        lambda path: logger.info(f"Configuration saved to: {path}")
    )
    
    # Add dock to window
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, config_dock)
    
    # Show window
    window.show()
    
    logger.info("Configuration Dock Widget test window displayed")
    logger.info("Test all tabs and controls:")
    logger.info("  1. Cameras tab - adjust FPS and device IDs")
    logger.info("  2. Detection tab - adjust thresholds and tracking parameters")
    logger.info("  3. DMS tab - adjust temporal smoothing")
    logger.info("  4. Risk tab - adjust weights and thresholds")
    logger.info("  5. Alerts tab - adjust suppression and escalation")
    logger.info("  6. Try Save, Reset, Import, and Export buttons")
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
