#!/usr/bin/env python3
"""
Test script for Performance Dock Widget

Tests all performance monitoring functionality including:
- FPS graph with 30 FPS target line
- Latency graph with 100ms threshold and P95 calculation
- Module breakdown stacked bar chart
- Resource usage gauges (GPU/CPU)
- Performance logging with export capabilities

Run this script to visually verify all performance monitoring features.
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QDockWidget, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer

# Add src to path
sys.path.insert(0, 'src')

from gui.widgets import PerformanceDockWidget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TestWindow(QMainWindow):
    """Test window for performance dock widget"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Performance Dock Widget Test - All Features")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget with instructions
        central = QWidget()
        layout = QVBoxLayout(central)
        
        instructions = QLabel("""
        <h2>Performance Monitoring Dock Test</h2>
        <p><b>Features being tested:</b></p>
        <ul>
            <li><b>FPS Tab:</b> Real-time FPS graph with 30 FPS target line (green above, red below)</li>
            <li><b>Latency Tab:</b> End-to-end latency with 100ms threshold, P95 calculation, and violation tracking</li>
            <li><b>Modules Tab:</b> Stacked bar chart showing time breakdown by pipeline stage</li>
            <li><b>Resources Tab:</b> GPU memory (max 8GB) and CPU usage (max 60%) gauges with peak tracking</li>
            <li><b>Logging Tab:</b> Start/stop logging, export reports, and performance summary</li>
        </ul>
        <p><b>Instructions:</b></p>
        <ol>
            <li>Switch between tabs to see different performance metrics</li>
            <li>Mock data updates every second (FPS ~30, latency ~80ms with occasional spikes)</li>
            <li>In Logging tab, click "Start Logging" to save metrics to file</li>
            <li>Use "Export Performance Report" to save summary as text file</li>
            <li>Watch for color coding: green=good, yellow=warning, red=critical</li>
        </ol>
        <p style="color: #00ff00;"><b>Status: Mock data generation active</b></p>
        """)
        instructions.setWordWrap(True)
        instructions.setStyleSheet("background-color: #2a2a2a; padding: 20px; border-radius: 10px;")
        layout.addWidget(instructions)
        
        self.setCentralWidget(central)
        
        # Create performance dock
        self.perf_dock = QDockWidget("Performance Monitor", self)
        self.perf_widget = PerformanceDockWidget()
        self.perf_dock.setWidget(self.perf_widget)
        self.perf_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable | 
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.perf_dock)
        
        # Start monitoring with mock data
        self.perf_widget.start_monitoring()
        
        logger.info("Test window initialized - all 5 tabs should be functional")
        logger.info("Tab 1: FPS graph")
        logger.info("Tab 2: Latency graph with P95")
        logger.info("Tab 3: Module breakdown")
        logger.info("Tab 4: Resource usage (GPU/CPU)")
        logger.info("Tab 5: Performance logging")
    
    def closeEvent(self, event):
        """Handle close event"""
        self.perf_widget.stop_monitoring()
        logger.info("Performance monitoring stopped")
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle('Fusion')
    
    # Dark palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = TestWindow()
    window.show()
    
    print("\n" + "="*70)
    print("PERFORMANCE DOCK WIDGET TEST")
    print("="*70)
    print("\nAll 5 performance monitoring tabs are now active:")
    print("  1. FPS Graph - Real-time frame rate with 30 FPS target")
    print("  2. Latency Graph - End-to-end latency with P95 tracking")
    print("  3. Module Breakdown - Pipeline stage timing breakdown")
    print("  4. Resource Usage - GPU memory and CPU usage gauges")
    print("  5. Performance Logging - Log to file and export reports")
    print("\nMock data updates every second. Close window to exit.")
    print("="*70 + "\n")
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
