"""Incident review and playback widget."""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QSlider, QTextEdit, QSplitter,
    QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QPixmap, QImage

import numpy as np
import cv2


class IncidentReviewWidget(QWidget):
    """
    Widget for reviewing recorded incident scenarios.

    Features:
    - Browse recorded scenarios by date/severity
    - Video playback with controls
    - Frame-by-frame navigation
    - Metadata display
    - Export capabilities
    """

    def __init__(self, scenarios_dir: str = "scenarios", parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self.scenarios_dir = Path(scenarios_dir)
        self.current_scenario: Optional[Dict] = None
        self.current_frame_idx = 0
        self.frames: List[np.ndarray] = []
        self.is_playing = False

        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._advance_frame)
        self.playback_fps = 10  # Default playback FPS

        self.setup_ui()
        self.load_scenarios()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Scenario list
        left_panel = self._create_scenario_list_panel()
        splitter.addWidget(left_panel)

        # Right panel - Playback and details
        right_panel = self._create_playback_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes
        splitter.setSizes([300, 700])

        layout.addWidget(splitter)

    def _create_scenario_list_panel(self) -> QWidget:
        """Create the scenario list panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title = QLabel("📁 Recorded Scenarios")
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh List")
        self.refresh_btn.clicked.connect(self.load_scenarios)
        layout.addWidget(self.refresh_btn)

        # Scenario list
        self.scenario_list = QListWidget()
        self.scenario_list.itemClicked.connect(self._on_scenario_selected)
        layout.addWidget(self.scenario_list)

        # Statistics
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)

        self.stats_label = QLabel("No scenarios loaded")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_frame)

        return panel

    def _create_playback_panel(self) -> QWidget:
        """Create the playback panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Video display
        video_group = QGroupBox("Video Playback")
        video_layout = QVBoxLayout()

        self.video_label = QLabel("No scenario selected")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #1e1e1e; border: 2px solid #444;")
        self.video_label.setScaledContents(True)
        video_layout.addWidget(self.video_label)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶️ Play")
        self.play_btn.clicked.connect(self._toggle_playback)
        self.play_btn.setEnabled(False)
        controls_layout.addWidget(self.play_btn)

        self.prev_frame_btn = QPushButton("⏮️ Prev")
        self.prev_frame_btn.clicked.connect(self._prev_frame)
        self.prev_frame_btn.setEnabled(False)
        controls_layout.addWidget(self.prev_frame_btn)

        self.next_frame_btn = QPushButton("⏭️ Next")
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.next_frame_btn.setEnabled(False)
        controls_layout.addWidget(self.next_frame_btn)

        self.export_btn = QPushButton("💾 Export")
        self.export_btn.clicked.connect(self._export_scenario)
        self.export_btn.setEnabled(False)
        controls_layout.addWidget(self.export_btn)

        video_layout.addLayout(controls_layout)

        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.frame_slider.setEnabled(False)
        slider_layout.addWidget(self.frame_slider)

        self.frame_label = QLabel("0/0")
        slider_layout.addWidget(self.frame_label)

        video_layout.addLayout(slider_layout)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)

        # Metadata display
        metadata_group = QGroupBox("Scenario Details")
        metadata_layout = QVBoxLayout()

        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        self.metadata_text.setMaximumHeight(150)
        metadata_layout.addWidget(self.metadata_text)

        metadata_group.setLayout(metadata_layout)
        layout.addWidget(metadata_group)

        return panel

    def load_scenarios(self):
        """Load list of recorded scenarios."""
        self.scenario_list.clear()

        if not self.scenarios_dir.exists():
            self.stats_label.setText("Scenarios directory not found")
            return

        # Find all scenario metadata files
        scenario_files = list(self.scenarios_dir.glob("*/metadata.json"))

        if not scenario_files:
            self.stats_label.setText("No scenarios found")
            return

        # Load and display scenarios
        scenarios = []
        for meta_file in scenario_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    scenarios.append((meta_file.parent, metadata))
            except Exception as e:
                self.logger.error(f"Failed to load {meta_file}: {e}")

        # Sort by timestamp (newest first)
        scenarios.sort(key=lambda x: x[1].get('start_time', 0), reverse=True)

        # Add to list
        for scenario_dir, metadata in scenarios:
            timestamp = metadata.get('start_time', 0)
            dt = datetime.fromtimestamp(timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            trigger = metadata.get('trigger_reason', 'Unknown')
            severity = metadata.get('severity', 'unknown')

            # Create list item
            item_text = f"[{severity.upper()}] {date_str} - {trigger}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, scenario_dir)

            # Color code by severity
            if severity == 'critical':
                item.setForeground(Qt.GlobalColor.red)
            elif severity == 'high':
                item.setForeground(Qt.GlobalColor.yellow)

            self.scenario_list.addItem(item)

        # Update statistics
        total = len(scenarios)
        critical = sum(1 for _, m in scenarios if m.get('severity') == 'critical')
        high = sum(1 for _, m in scenarios if m.get('severity') == 'high')

        self.stats_label.setText(
            f"Total Scenarios: {total}\n"
            f"Critical: {critical}\n"
            f"High: {high}"
        )

        self.logger.info(f"Loaded {total} scenarios")

    def _on_scenario_selected(self, item: QListWidgetItem):
        """Handle scenario selection."""
        scenario_dir = item.data(Qt.ItemDataRole.UserRole)
        self.load_scenario(scenario_dir)

    def load_scenario(self, scenario_dir: Path):
        """
        Load a scenario for playback.

        Args:
            scenario_dir: Path to scenario directory
        """
        try:
            # Load metadata
            meta_file = scenario_dir / "metadata.json"
            with open(meta_file, 'r') as f:
                self.current_scenario = json.load(f)

            # Load frames
            self.frames = []
            frames_dir = scenario_dir / "frames"

            if frames_dir.exists():
                frame_files = sorted(frames_dir.glob("frame_*.jpg"))
                for frame_file in frame_files:
                    frame = cv2.imread(str(frame_file))
                    if frame is not None:
                        self.frames.append(frame)

            if not self.frames:
                self.logger.warning(f"No frames found in {scenario_dir}")
                return

            self.logger.info(f"Loaded scenario with {len(self.frames)} frames")

            # Update UI
            self.current_frame_idx = 0
            self.frame_slider.setMaximum(len(self.frames) - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)

            # Enable controls
            self.play_btn.setEnabled(True)
            self.prev_frame_btn.setEnabled(True)
            self.next_frame_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

            # Display first frame
            self._display_frame(0)

            # Display metadata
            self._display_metadata()

        except Exception as e:
            self.logger.error(f"Failed to load scenario: {e}")
            self.video_label.setText(f"Error loading scenario: {e}")

    def _display_frame(self, frame_idx: int):
        """Display a specific frame."""
        if not self.frames or frame_idx < 0 or frame_idx >= len(self.frames):
            return

        frame = self.frames[frame_idx]

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

        # Display
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

        # Update frame counter
        self.frame_label.setText(f"{frame_idx + 1}/{len(self.frames)}")
        self.current_frame_idx = frame_idx

    def _display_metadata(self):
        """Display scenario metadata."""
        if not self.current_scenario:
            return

        metadata_text = []
        metadata_text.append(f"**Trigger:** {self.current_scenario.get('trigger_reason', 'N/A')}")
        metadata_text.append(f"**Severity:** {self.current_scenario.get('severity', 'N/A').upper()}")

        timestamp = self.current_scenario.get('start_time', 0)
        dt = datetime.fromtimestamp(timestamp)
        metadata_text.append(f"**Date/Time:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")

        duration = self.current_scenario.get('duration', 0)
        metadata_text.append(f"**Duration:** {duration:.1f}s")

        metadata_text.append(f"**Frames:** {len(self.frames)}")

        # Risk information
        if 'risk_assessment' in self.current_scenario:
            risk = self.current_scenario['risk_assessment']
            metadata_text.append(f"\n**Risk Assessment:**")
            metadata_text.append(f"- Top Risks: {len(risk.get('top_risks', []))}")

        # Driver state
        if 'driver_state' in self.current_scenario:
            driver = self.current_scenario['driver_state']
            metadata_text.append(f"\n**Driver State:**")
            metadata_text.append(f"- Attention: {driver.get('readiness_score', 0):.0f}")

        self.metadata_text.setMarkdown("\n".join(metadata_text))

    def _toggle_playback(self):
        """Toggle playback."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start playback."""
        self.is_playing = True
        self.play_btn.setText("⏸️ Pause")
        self.playback_timer.start(int(1000 / self.playback_fps))

    def _stop_playback(self):
        """Stop playback."""
        self.is_playing = False
        self.play_btn.setText("▶️ Play")
        self.playback_timer.stop()

    def _advance_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame_idx < len(self.frames) - 1:
            self.frame_slider.setValue(self.current_frame_idx + 1)
        else:
            # Loop back to start
            self.frame_slider.setValue(0)

    def _prev_frame(self):
        """Go to previous frame."""
        if self.current_frame_idx > 0:
            self.frame_slider.setValue(self.current_frame_idx - 1)

    def _next_frame(self):
        """Go to next frame."""
        if self.current_frame_idx < len(self.frames) - 1:
            self.frame_slider.setValue(self.current_frame_idx + 1)

    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        self._display_frame(value)

    def _export_scenario(self):
        """Export current scenario."""
        if not self.current_scenario:
            return

        # TODO: Implement export (video file, report, etc.)
        self.logger.info("Export functionality not yet implemented")
