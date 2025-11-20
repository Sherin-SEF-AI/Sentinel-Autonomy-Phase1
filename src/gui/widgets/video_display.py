"""
Video Display Widget

Displays camera feed or processed video frames with efficient rendering.
"""

import logging
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor

logger = logging.getLogger(__name__)


class VideoDisplayWidget(QWidget):
    """
    Widget for displaying video frames from camera feeds or processed outputs.
    
    Features:
    - Efficient numpy array to QPixmap conversion
    - Automatic aspect ratio preservation
    - Label overlay for camera name
    - Smooth scaling
    """
    
    def __init__(self, camera_name: str = "Camera", parent=None):
        super().__init__(parent)
        
        self.camera_name = camera_name
        self.current_frame = None
        
        self._init_ui()
        
        logger.debug(f"VideoDisplayWidget created for {camera_name}")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                color: #888888;
            }
        """)
        self.video_label.setText(f"{self.camera_name}\n(No Signal)")
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Camera name label
        self.name_label = QLabel(self.camera_name)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 4px;
                font-weight: bold;
                border: 1px solid #333333;
            }
        """)
        
        layout.addWidget(self.video_label)
        layout.addWidget(self.name_label)
        
        self.setLayout(layout)
    
    def update_frame(self, frame: np.ndarray):
        """
        Update the displayed frame.
        
        Args:
            frame: Numpy array in BGR or RGB format (H, W, 3)
        """
        if frame is None or frame.size == 0:
            self.clear_frame()
            return
        
        try:
            self.current_frame = frame
            
            # Convert numpy array to QPixmap
            pixmap = self._numpy_to_pixmap(frame)
            
            # Scale to fit label while preserving aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Error updating frame for {self.camera_name}: {e}")
            self.clear_frame()
    
    def clear_frame(self):
        """Clear the current frame and show no signal message"""
        self.current_frame = None
        self.video_label.clear()
        self.video_label.setText(f"{self.camera_name}\n(No Signal)")
    
    def _numpy_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """
        Convert numpy array to QPixmap.
        
        Args:
            frame: Numpy array in BGR or RGB format (H, W, 3)
            
        Returns:
            QPixmap object
        """
        height, width, channels = frame.shape
        
        # Ensure frame is in RGB format (OpenCV uses BGR)
        if channels == 3:
            # Assume BGR from OpenCV, convert to RGB
            rgb_frame = frame[:, :, ::-1].copy()
        else:
            rgb_frame = frame.copy()
        
        # Create QImage from numpy array
        bytes_per_line = channels * width
        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Convert to QPixmap
        return QPixmap.fromImage(q_image)
    
    def resizeEvent(self, event):
        """Handle resize event to rescale current frame"""
        super().resizeEvent(event)
        
        if self.current_frame is not None:
            # Rescale current frame to new size
            pixmap = self._numpy_to_pixmap(self.current_frame)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
