"""
Live Monitor Widget

Central monitoring widget with 2x2 camera grid layout for displaying
live camera feeds and BEV output.
"""

import logging
import numpy as np
from typing import Dict, Optional
from PyQt6.QtWidgets import QWidget, QGridLayout, QSplitter
from PyQt6.QtCore import Qt, QTimer

from .video_display import VideoDisplayWidget
from .bev_canvas import BEVCanvas

logger = logging.getLogger(__name__)


class LiveMonitorWidget(QWidget):
    """
    Central monitoring widget with 2x2 camera grid.
    
    Layout:
    ┌─────────────┬─────────────┐
    │  Interior   │ Front Left  │
    ├─────────────┼─────────────┤
    │ Front Right │     BEV     │
    └─────────────┴─────────────┘
    
    Features:
    - Real-time video display at 30 FPS
    - Responsive layout with QSplitter
    - Efficient frame updates
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.update_rate = 30  # Hz
        self.video_displays: Dict[str, VideoDisplayWidget] = {}
        
        self._init_ui()
        self._setup_update_timer()
        
        logger.info("LiveMonitorWidget initialized")
    
    def _init_ui(self):
        """Initialize UI with 2x2 grid layout"""
        layout = QGridLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)
        
        # Create video display widgets for each camera
        self.interior_display = VideoDisplayWidget("Interior Camera")
        self.front_left_display = VideoDisplayWidget("Front Left Camera")
        self.front_right_display = VideoDisplayWidget("Front Right Camera")
        
        # Create interactive BEV canvas
        self.bev_canvas = BEVCanvas()
        
        # Store references (keep bev_display for backward compatibility)
        self.video_displays = {
            'interior': self.interior_display,
            'front_left': self.front_left_display,
            'front_right': self.front_right_display,
        }
        
        # BEV canvas is accessed separately
        self.bev_display = self.bev_canvas
        
        # Create splitters for responsive layout
        # Top row splitter
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.interior_display)
        top_splitter.addWidget(self.front_left_display)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        
        # Bottom row splitter
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.front_right_display)
        bottom_splitter.addWidget(self.bev_canvas)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        
        # Main vertical splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_splitter)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        
        # Add to layout
        layout.addWidget(main_splitter, 0, 0)
        
        self.setLayout(layout)
        
        logger.debug("2x2 camera grid layout created")
    
    def _setup_update_timer(self):
        """Setup timer for periodic frame updates"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._on_update_timer)
        
        # Timer will be started when system starts
        # Update interval in milliseconds
        self.update_interval = int(1000 / self.update_rate)
        
        logger.debug(f"Update timer configured for {self.update_rate} Hz")
    
    def start_updates(self):
        """Start periodic frame updates"""
        if not self.update_timer.isActive():
            self.update_timer.start(self.update_interval)
            logger.info(f"Started frame updates at {self.update_rate} Hz")
    
    def stop_updates(self):
        """Stop periodic frame updates"""
        if self.update_timer.isActive():
            self.update_timer.stop()
            logger.info("Stopped frame updates")
    
    def _on_update_timer(self):
        """Handle update timer timeout"""
        # This will be connected to actual data source in future tasks
        # For now, it's a placeholder for the update mechanism
        pass
    
    def update_camera_frame(self, camera_id: str, frame: np.ndarray):
        """
        Update frame for a specific camera.
        
        Args:
            camera_id: Camera identifier ('interior', 'front_left', 'front_right', 'bev')
            frame: Numpy array containing the frame data
        """
        if camera_id == 'bev':
            # Update BEV canvas
            self.bev_canvas.update_bev_image(frame)
        elif camera_id in self.video_displays:
            self.video_displays[camera_id].update_frame(frame)
        else:
            logger.warning(f"Unknown camera ID: {camera_id}")
    
    def update_all_frames(self, frames: Dict[str, np.ndarray]):
        """
        Update all camera frames at once.
        
        Args:
            frames: Dictionary mapping camera IDs to frame arrays
        """
        for camera_id, frame in frames.items():
            self.update_camera_frame(camera_id, frame)
    
    def clear_all_frames(self):
        """Clear all camera displays"""
        for display in self.video_displays.values():
            display.clear_frame()
        self.bev_canvas.clear_bev_image()
        logger.debug("All camera frames cleared")
    
    def set_update_rate(self, rate: int):
        """
        Set the frame update rate.
        
        Args:
            rate: Update rate in Hz (1-60)
        """
        if 1 <= rate <= 60:
            self.update_rate = rate
            self.update_interval = int(1000 / rate)
            
            # Restart timer if active
            if self.update_timer.isActive():
                self.update_timer.stop()
                self.update_timer.start(self.update_interval)
            
            logger.info(f"Update rate changed to {rate} Hz")
        else:
            logger.warning(f"Invalid update rate: {rate}. Must be between 1 and 60 Hz")
