#!/usr/bin/env python3
"""
Simple test script to verify the SENTINEL GUI works.

This script launches the GUI application to test:
- Main window creation
- Menu bar and toolbar
- Theme system
- Video display widgets
- Multi-monitor support
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer

from gui.main_window import SENTINELMainWindow
from gui.themes import ThemeManager
from core.logging import setup_logging


def test_gui():
    """Test the GUI application"""
    # Setup logging
    setup_logging()
    
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("SENTINEL Test")
    app.setOrganizationName("SENTINEL")
    
    # Create theme manager
    theme_manager = ThemeManager(app)
    
    # Create main window
    main_window = SENTINELMainWindow(theme_manager)
    main_window.show()
    
    # Show welcome message
    QMessageBox.information(
        main_window,
        "SENTINEL GUI Test",
        "GUI initialized successfully!\n\n"
        "Test features:\n"
        "- Menu bar (File, System, View, Tools, Analytics, Help)\n"
        "- Toolbar with quick actions\n"
        "- Theme switching (View > Theme)\n"
        "- Multi-monitor support (View > Move to Monitor)\n"
        "- Window state persistence\n\n"
        "Try starting the system (F5) to see the video displays."
    )
    
    # Optionally add test frames after a delay
    def add_test_frames():
        """Add test frames to video displays"""
        # Create test frames (colored rectangles)
        interior_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        interior_frame[:, :] = [100, 50, 150]  # Purple
        
        front_left_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        front_left_frame[:, :] = [50, 150, 100]  # Teal
        
        front_right_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        front_right_frame[:, :] = [150, 100, 50]  # Orange
        
        bev_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        bev_frame[:, :] = [50, 100, 150]  # Blue
        
        # Update displays
        main_window.live_monitor.update_all_frames({
            'interior': interior_frame,
            'front_left': front_left_frame,
            'front_right': front_right_frame,
            'bev': bev_frame
        })
    
    # Add test frames after 2 seconds
    QTimer.singleShot(2000, add_test_frames)
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    test_gui()
