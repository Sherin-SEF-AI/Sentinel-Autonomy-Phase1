"""
Example: SENTINEL Worker Thread Integration

Demonstrates how to use the SentinelWorker thread for background processing
with GUI integration.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt

from gui.workers import SentinelWorker
from core.config import ConfigManager
from core.logging import setup_logging


class SimpleWorkerDemo(QMainWindow):
    """
    Simple demonstration of worker thread integration.
    
    Shows how to:
    - Create and start a worker thread
    - Connect signals to update GUI
    - Handle errors
    - Stop worker gracefully
    """
    
    def __init__(self, config: ConfigManager):
        super().__init__()
        
        self.config = config
        self.worker = None
        
        self.setWindowTitle("SENTINEL Worker Thread Demo")
        self.setMinimumSize(800, 600)
        
        # Create UI
        self._create_ui()
        
        logging.info("Worker demo initialized")
    
    def _create_ui(self):
        """Create simple UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Status label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # FPS label
        self.fps_label = QLabel("FPS: 0.00")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fps_label)
        
        # Frame count label
        self.frame_label = QLabel("Frames: 0")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.frame_label)
        
        # Latency label
        self.latency_label = QLabel("Latency: 0.00ms")
        self.latency_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.latency_label)
        
        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: red;")
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.error_label)
        
        # Start button
        self.start_button = QPushButton("Start System")
        self.start_button.clicked.connect(self._start_worker)
        layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop System")
        self.stop_button.clicked.connect(self._stop_worker)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
    
    def _start_worker(self):
        """Start the worker thread."""
        logging.info("Starting worker thread...")
        
        try:
            # Create worker
            self.worker = SentinelWorker(self.config)
            
            # Connect signals
            self.worker.status_changed.connect(self._on_status_changed)
            self.worker.performance_ready.connect(self._on_performance_ready)
            self.worker.error_occurred.connect(self._on_error)
            self.worker.frame_ready.connect(self._on_frame_ready)
            
            # Start worker
            self.worker.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.error_label.setText("")
            
            logging.info("Worker thread started")
            
        except Exception as e:
            logging.error(f"Failed to start worker: {e}")
            self.error_label.setText(f"Error: {str(e)}")
    
    def _stop_worker(self):
        """Stop the worker thread."""
        logging.info("Stopping worker thread...")
        
        if self.worker:
            # Request stop
            self.worker.stop()
            
            # Wait for completion
            if not self.worker.wait(5000):
                logging.warning("Worker did not stop gracefully, terminating...")
                self.worker.terminate()
                self.worker.wait()
            
            self.worker = None
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")
        
        logging.info("Worker thread stopped")
    
    def _on_status_changed(self, status: str):
        """Handle status change signal."""
        self.status_label.setText(f"Status: {status}")
        logging.info(f"Status: {status}")
    
    def _on_performance_ready(self, metrics: dict):
        """Handle performance metrics signal."""
        fps = metrics.get('fps', 0.0)
        frame_count = metrics.get('frame_count', 0)
        total_latency = metrics.get('total_latency_ms', 0.0)
        
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.frame_label.setText(f"Frames: {frame_count}")
        self.latency_label.setText(f"Latency: {total_latency:.2f}ms")
    
    def _on_error(self, error_type: str, error_message: str):
        """Handle error signal."""
        logging.error(f"Worker error ({error_type}): {error_message}")
        self.error_label.setText(f"Error ({error_type}): {error_message}")
        
        # Stop on fatal errors
        if error_type in ['Fatal', 'Initialization']:
            self._stop_worker()
    
    def _on_frame_ready(self, frames: dict):
        """Handle frame ready signal."""
        # Just log frame reception
        logging.debug(f"Received frames: {list(frames.keys())}")
    
    def closeEvent(self, event):
        """Handle window close."""
        if self.worker and self.worker.isRunning():
            logging.info("Stopping worker before closing...")
            self._stop_worker()
        event.accept()


def main():
    """Main entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Worker Thread Demo")
    
    # Load configuration
    try:
        config = ConfigManager('configs/default.yaml')
        if not config.validate():
            logger.error("Invalid configuration")
            return 1
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show demo window
    demo = SimpleWorkerDemo(config)
    demo.show()
    
    logger.info("Demo window shown")
    
    # Run event loop
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
