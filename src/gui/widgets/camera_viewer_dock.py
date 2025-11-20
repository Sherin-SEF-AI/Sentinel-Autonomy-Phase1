"""
Camera Feed Viewer Dock Widget

Displays live feeds from all active cameras in a grid layout.
"""

import logging
from typing import Optional, Dict
import numpy as np

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QGroupBox, QPushButton, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)


class CameraFeedWidget(QWidget):
    """Widget displaying a single camera feed."""

    def __init__(self, camera_name: str, parent=None):
        super().__init__(parent)
        self.camera_name = camera_name
        self.logger = logging.getLogger(f"{__name__}.{camera_name}")

        # UI setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Camera name label
        self.name_label = QLabel(f"📷 {camera_name.replace('_', ' ').title()}")
        self.name_label.setStyleSheet(
            "font-weight: bold; font-size: 11pt; padding: 5px;"
        )
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.name_label)

        # Image display
        self.image_label = QLabel("No Feed")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(320, 240)
        self.image_label.setStyleSheet(
            "background-color: #2b2b2b; border: 2px solid #444; "
            "color: #888; font-size: 14pt;"
        )
        self.image_label.setScaledContents(False)
        layout.addWidget(self.image_label)

        # Status label
        self.status_label = QLabel("⚪ Waiting...")
        self.status_label.setStyleSheet("font-size: 9pt; color: #888; padding: 3px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Stats
        self.frame_count = 0
        self.last_update_time = None

        self.logger.debug(f"CameraFeedWidget initialized for {camera_name}")

    def update_frame(self, frame: np.ndarray):
        """
        Update the displayed frame.

        Args:
            frame: NumPy array (H, W, 3) in BGR format
        """
        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1].copy()

            height, width, channels = rgb_frame.shape
            bytes_per_line = channels * width

            # Create QImage
            q_image = QImage(
                rgb_frame.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )

            # Scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.image_label.setPixmap(scaled_pixmap)

            # Update stats
            self.frame_count += 1
            self.status_label.setText(f"🟢 Live - Frame {self.frame_count}")

        except Exception as e:
            self.logger.error(f"Error updating frame: {e}")
            self.status_label.setText(f"🔴 Error: {e}")

    def set_no_feed(self):
        """Set to no feed state."""
        self.image_label.clear()
        self.image_label.setText("No Feed")
        self.status_label.setText("⚪ No Signal")


class CameraViewerDock(QDockWidget):
    """
    Dock widget for viewing live camera feeds.

    Displays all active camera feeds in a grid layout with controls.
    """

    def __init__(self, parent=None):
        super().__init__("📷 Camera Feeds", parent)
        self.logger = logging.getLogger(__name__)

        # Create central widget
        central_widget = QWidget()
        self.setWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Control bar
        control_layout = QHBoxLayout()

        # View mode selector
        control_layout.addWidget(QLabel("View Mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems([
            "Grid View",
            "Single Camera",
            "Picture-in-Picture"
        ])
        self.view_mode_combo.currentTextChanged.connect(self._on_view_mode_changed)
        control_layout.addWidget(self.view_mode_combo)

        control_layout.addStretch()

        # Freeze button
        self.freeze_button = QPushButton("❄️ Freeze")
        self.freeze_button.setCheckable(True)
        self.freeze_button.clicked.connect(self._on_freeze_clicked)
        control_layout.addWidget(self.freeze_button)

        # Screenshot button
        self.screenshot_button = QPushButton("📸 Capture")
        self.screenshot_button.clicked.connect(self._on_screenshot_clicked)
        control_layout.addWidget(self.screenshot_button)

        main_layout.addLayout(control_layout)

        # Camera feed grid
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)

        # Create camera feed widgets
        self.camera_feeds: Dict[str, CameraFeedWidget] = {}
        camera_names = ['interior', 'front_left', 'front_right']

        for idx, name in enumerate(camera_names):
            feed_widget = CameraFeedWidget(name)
            self.camera_feeds[name] = feed_widget
            row = idx // 2
            col = idx % 2
            self.grid_layout.addWidget(feed_widget, row, col)

        main_layout.addWidget(self.grid_widget)

        # State
        self.frozen = False
        self.current_frames: Dict[str, np.ndarray] = {}

        self.logger.info("CameraViewerDock initialized with 3 camera feeds")

    @pyqtSlot(dict, name="updateCameraFrames")
    def update_camera_frames(self, frames: Dict[int, np.ndarray]):
        """
        Update all camera feeds.

        Args:
            frames: Dictionary mapping camera_id to frame array
        """
        if self.frozen:
            return

        # Map camera IDs to names
        id_to_name = {0: 'interior', 1: 'front_left', 2: 'front_right'}

        for cam_id, frame in frames.items():
            cam_name = id_to_name.get(cam_id)
            if cam_name and cam_name in self.camera_feeds:
                self.camera_feeds[cam_name].update_frame(frame)
                self.current_frames[cam_name] = frame

        # Mark missing cameras
        for cam_id, cam_name in id_to_name.items():
            if cam_id not in frames and cam_name in self.camera_feeds:
                self.camera_feeds[cam_name].set_no_feed()

    def _on_view_mode_changed(self, mode: str):
        """Handle view mode change."""
        self.logger.info(f"View mode changed to: {mode}")
        # TODO: Implement different view layouts

    def _on_freeze_clicked(self, checked: bool):
        """Handle freeze button click."""
        self.frozen = checked
        if checked:
            self.freeze_button.setText("▶️ Unfreeze")
            self.logger.info("Camera feeds frozen")
        else:
            self.freeze_button.setText("❄️ Freeze")
            self.logger.info("Camera feeds unfrozen")

    def _on_screenshot_clicked(self):
        """Handle screenshot button click."""
        self.logger.info("Screenshot captured")
        # TODO: Implement screenshot functionality
        import time
        from pathlib import Path

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_dir = Path("screenshots")
        screenshot_dir.mkdir(exist_ok=True)

        # Save current frames
        for cam_name, frame in self.current_frames.items():
            if frame is not None:
                import cv2
                filename = screenshot_dir / f"{cam_name}_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)
                self.logger.info(f"Saved screenshot: {filename}")

    def clear_feeds(self):
        """Clear all camera feeds."""
        for feed in self.camera_feeds.values():
            feed.set_no_feed()
        self.current_frames.clear()
        self.logger.debug("Camera feeds cleared")
