"""
Scenarios Dock Widget for SENTINEL GUI

Provides scenario management interface with:
- List of recorded scenarios with thumbnails
- Search and filtering
- Scenario replay dialog
- Export and delete actions
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QListWidgetItem, QLabel, QPushButton, QLineEdit, QComboBox,
    QDialog, QMessageBox, QFileDialog, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QIcon
import cv2
import numpy as np


logger = logging.getLogger(__name__)


class ScenarioListItem(QWidget):
    """
    Custom widget for scenario list items.
    
    Displays:
    - Thumbnail image
    - Timestamp
    - Duration
    - Trigger type
    """
    
    def __init__(self, scenario_name: str, metadata: Dict[str, Any], thumbnail: Optional[QPixmap] = None):
        super().__init__()
        
        self.scenario_name = scenario_name
        self.metadata = metadata
        
        # Create layout
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(120, 90)
        self.thumbnail_label.setScaledContents(True)
        self.thumbnail_label.setStyleSheet("border: 1px solid #555; background-color: #222;")
        
        if thumbnail:
            self.thumbnail_label.setPixmap(thumbnail)
        else:
            self.thumbnail_label.setText("No Preview")
            self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(self.thumbnail_label)
        
        # Info section
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        
        # Timestamp
        timestamp_str = metadata.get('timestamp', 'Unknown')
        try:
            dt = datetime.fromisoformat(timestamp_str)
            timestamp_display = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp_display = timestamp_str
        
        timestamp_label = QLabel(f"<b>{timestamp_display}</b>")
        info_layout.addWidget(timestamp_label)
        
        # Duration
        duration = metadata.get('duration', 0)
        duration_label = QLabel(f"Duration: {duration:.1f}s")
        duration_label.setStyleSheet("color: #aaa;")
        info_layout.addWidget(duration_label)
        
        # Trigger type
        trigger = metadata.get('trigger', {})
        trigger_type = trigger.get('type', 'Unknown')
        trigger_reason = trigger.get('reason', '')
        
        trigger_label = QLabel(f"Trigger: {trigger_type}")
        trigger_label.setStyleSheet(self._get_trigger_color(trigger_type))
        info_layout.addWidget(trigger_label)
        
        # Reason (if available)
        if trigger_reason:
            reason_label = QLabel(trigger_reason)
            reason_label.setStyleSheet("color: #888; font-size: 10px;")
            reason_label.setWordWrap(True)
            info_layout.addWidget(reason_label)
        
        info_layout.addStretch()
        
        layout.addLayout(info_layout, 1)
        
        self.setLayout(layout)
    
    def _get_trigger_color(self, trigger_type: str) -> str:
        """Get color styling based on trigger type"""
        colors = {
            'critical': 'color: #ff4444; font-weight: bold;',
            'high_risk': 'color: #ff8800;',
            'near_miss': 'color: #ffaa00;',
            'distracted': 'color: #ffcc00;',
            'manual': 'color: #4488ff;'
        }
        return colors.get(trigger_type, 'color: #aaa;')


class ScenariosDockWidget(QDockWidget):
    """
    Dock widget for scenario management.
    
    Features:
    - List all recorded scenarios
    - Search and filter scenarios
    - Double-click to replay
    - Export and delete actions
    """
    
    # Signals
    scenario_selected = pyqtSignal(str)  # Emits scenario name
    scenario_replay_requested = pyqtSignal(str)  # Emits scenario name
    
    def __init__(self, scenarios_path: str = 'scenarios/'):
        super().__init__("Scenarios")
        
        self.scenarios_path = scenarios_path
        self.scenarios_data: Dict[str, Dict[str, Any]] = {}
        
        # Create main widget
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Search and filter controls
        controls_layout = QHBoxLayout()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search scenarios...")
        self.search_box.textChanged.connect(self._on_search_changed)
        controls_layout.addWidget(self.search_box, 2)
        
        # Filter combo box
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "All",
            "Critical",
            "High Risk",
            "Near Miss",
            "Distracted",
            "Manual"
        ])
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        controls_layout.addWidget(self.filter_combo, 1)
        
        layout.addLayout(controls_layout)
        
        # Scenarios list
        self.scenarios_list = QListWidget()
        self.scenarios_list.setSpacing(2)
        self.scenarios_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.scenarios_list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.scenarios_list)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_scenarios)
        buttons_layout.addWidget(self.refresh_button)
        
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_clicked)
        buttons_layout.addWidget(self.export_button)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self._on_delete_clicked)
        buttons_layout.addWidget(self.delete_button)
        
        layout.addLayout(buttons_layout)
        
        # Status label
        self.status_label = QLabel("No scenarios loaded")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)
        self.setWidget(main_widget)
        
        # Load scenarios
        self.refresh_scenarios()
        
        logger.info("ScenariosDockWidget initialized")
    
    def refresh_scenarios(self):
        """Refresh the scenarios list from disk"""
        logger.info("Scenario refresh started")
        refresh_start = datetime.now()
        
        # Clear current list
        self.scenarios_list.clear()
        self.scenarios_data.clear()
        logger.debug("Scenarios list and data cleared")
        
        # Check if scenarios directory exists
        if not os.path.exists(self.scenarios_path):
            self.status_label.setText("Scenarios directory not found")
            logger.warning(f"Scenarios directory not found: path={self.scenarios_path}")
            return
        
        # List all scenario directories
        try:
            scenario_dirs = [
                d for d in os.listdir(self.scenarios_path)
                if os.path.isdir(os.path.join(self.scenarios_path, d))
            ]
            logger.debug(f"Found {len(scenario_dirs)} scenario directories")
        except Exception as e:
            self.status_label.setText(f"Error reading scenarios: {e}")
            logger.error(f"Failed to read scenarios directory: path={self.scenarios_path}, error={e}")
            return
        
        if not scenario_dirs:
            self.status_label.setText("No scenarios found")
            logger.info("No scenarios found in directory")
            return
        
        # Sort by timestamp (newest first)
        scenario_dirs.sort(reverse=True)
        logger.debug(f"Scenarios sorted by timestamp (newest first)")
        
        # Load each scenario
        loaded_count = 0
        failed_count = 0
        for scenario_name in scenario_dirs:
            if self._load_scenario_metadata(scenario_name):
                loaded_count += 1
            else:
                failed_count += 1
        
        if failed_count > 0:
            logger.warning(f"Failed to load {failed_count} scenario(s)")
        
        # Apply current filter
        self._apply_filter()
        
        refresh_duration = (datetime.now() - refresh_start).total_seconds()
        self.status_label.setText(f"{loaded_count} scenario(s) loaded")
        logger.info(f"Scenario refresh completed: loaded={loaded_count}, failed={failed_count}, duration={refresh_duration:.3f}s")
    
    def _load_scenario_metadata(self, scenario_name: str) -> bool:
        """
        Load metadata for a scenario.
        
        Args:
            scenario_name: Name of scenario directory
            
        Returns:
            True if loaded successfully
        """
        scenario_dir = os.path.join(self.scenarios_path, scenario_name)
        metadata_path = os.path.join(scenario_dir, 'metadata.json')
        
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: scenario={scenario_name}, path={metadata_path}")
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.scenarios_data[scenario_name] = metadata
            
            # Log key metadata
            duration = metadata.get('duration', 0)
            trigger_type = metadata.get('trigger', {}).get('type', 'unknown')
            logger.debug(f"Metadata loaded: scenario={scenario_name}, duration={duration:.1f}s, trigger={trigger_type}")
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata: scenario={scenario_name}, error={e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load metadata: scenario={scenario_name}, error={e}")
            return False
    
    def _load_thumbnail(self, scenario_name: str) -> Optional[QPixmap]:
        """
        Load thumbnail image for a scenario.
        
        Args:
            scenario_name: Name of scenario directory
            
        Returns:
            QPixmap thumbnail or None
        """
        scenario_dir = os.path.join(self.scenarios_path, scenario_name)
        metadata = self.scenarios_data.get(scenario_name)
        
        if not metadata:
            logger.debug(f"No metadata available for thumbnail: scenario={scenario_name}")
            return None
        
        # Try to get first frame from BEV video
        video_files = metadata.get('files', {})
        bev_file = video_files.get('bev') or video_files.get('front_left')
        
        if not bev_file:
            logger.debug(f"No video file found for thumbnail: scenario={scenario_name}")
            return None
        
        video_path = os.path.join(scenario_dir, bev_file)
        
        if not os.path.exists(video_path):
            logger.warning(f"Video file not found for thumbnail: scenario={scenario_name}, path={video_path}")
            return None
        
        try:
            # Open video and read first frame
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                logger.warning(f"Failed to read first frame: scenario={scenario_name}, video={bev_file}")
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to QPixmap
            height, width, channel = frame_rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            logger.debug(f"Thumbnail loaded: scenario={scenario_name}, size={width}x{height}")
            return pixmap
            
        except Exception as e:
            logger.error(f"Failed to load thumbnail: scenario={scenario_name}, error={e}")
            return None
    
    def _apply_filter(self):
        """Apply current search and filter to scenarios list"""
        self.scenarios_list.clear()
        
        search_text = self.search_box.text().lower()
        filter_type = self.filter_combo.currentText().lower()
        
        logger.debug(f"Applying filter: search='{search_text}', type={filter_type}")
        
        # Filter scenarios
        matched_count = 0
        for scenario_name, metadata in self.scenarios_data.items():
            # Apply search filter
            if search_text and search_text not in scenario_name.lower():
                # Also check trigger reason
                trigger_reason = metadata.get('trigger', {}).get('reason', '').lower()
                if search_text not in trigger_reason:
                    continue
            
            # Apply type filter
            if filter_type != "all":
                trigger_type = metadata.get('trigger', {}).get('type', '').lower()
                trigger_type = trigger_type.replace('_', ' ')
                
                if filter_type != trigger_type:
                    continue
            
            # Create list item
            self._add_scenario_to_list(scenario_name, metadata)
            matched_count += 1
        
        # Update status
        visible_count = self.scenarios_list.count()
        total_count = len(self.scenarios_data)
        
        if visible_count < total_count:
            self.status_label.setText(f"Showing {visible_count} of {total_count} scenario(s)")
            logger.debug(f"Filter applied: showing={visible_count}, total={total_count}, filtered_out={total_count - visible_count}")
        else:
            self.status_label.setText(f"{total_count} scenario(s)")
            logger.debug(f"Filter applied: showing all {total_count} scenario(s)")
    
    def _add_scenario_to_list(self, scenario_name: str, metadata: Dict[str, Any]):
        """Add a scenario to the list widget"""
        # Load thumbnail
        thumbnail = self._load_thumbnail(scenario_name)
        
        # Create custom widget
        item_widget = ScenarioListItem(scenario_name, metadata, thumbnail)
        
        # Create list item
        list_item = QListWidgetItem(self.scenarios_list)
        list_item.setSizeHint(item_widget.sizeHint())
        list_item.setData(Qt.ItemDataRole.UserRole, scenario_name)
        
        # Add to list
        self.scenarios_list.addItem(list_item)
        self.scenarios_list.setItemWidget(list_item, item_widget)
    
    def _on_search_changed(self, text: str):
        """Handle search text change"""
        self._apply_filter()
    
    def _on_filter_changed(self, filter_type: str):
        """Handle filter change"""
        self._apply_filter()
    
    def _on_selection_changed(self):
        """Handle selection change"""
        selected_items = self.scenarios_list.selectedItems()
        
        if selected_items:
            self.export_button.setEnabled(True)
            self.delete_button.setEnabled(True)
            
            scenario_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
            logger.debug(f"Scenario selected: {scenario_name}")
            self.scenario_selected.emit(scenario_name)
        else:
            self.export_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            logger.debug("Scenario selection cleared")
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle item double-click"""
        scenario_name = item.data(Qt.ItemDataRole.UserRole)
        logger.info(f"Scenario double-clicked: {scenario_name}")
        self.scenario_replay_requested.emit(scenario_name)
    
    def _on_export_clicked(self):
        """Handle export button click"""
        selected_items = self.scenarios_list.selectedItems()
        
        if not selected_items:
            return
        
        scenario_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
        logger.info(f"Export requested for scenario: {scenario_name}")
        
        # Show file dialog
        export_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if export_dir:
            self._export_scenario(scenario_name, export_dir)
    
    def _export_scenario(self, scenario_name: str, export_dir: str):
        """Export scenario to specified directory"""
        logger.info(f"Scenario export started: scenario={scenario_name}, dest={export_dir}")
        export_start = datetime.now()
        
        try:
            import shutil
            
            source_dir = os.path.join(self.scenarios_path, scenario_name)
            dest_dir = os.path.join(export_dir, scenario_name)
            
            # Check source directory size
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(source_dir)
                for filename in filenames
            )
            logger.debug(f"Export size: {total_size / (1024*1024):.2f} MB")
            
            # Copy scenario directory
            shutil.copytree(source_dir, dest_dir)
            
            export_duration = (datetime.now() - export_start).total_seconds()
            
            QMessageBox.information(
                self,
                "Export Successful",
                f"Scenario exported to:\n{dest_dir}"
            )
            
            logger.info(f"Scenario export completed: scenario={scenario_name}, dest={dest_dir}, size_mb={total_size/(1024*1024):.2f}, duration={export_duration:.3f}s")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export scenario:\n{str(e)}"
            )
            logger.error(f"Scenario export failed: scenario={scenario_name}, dest={export_dir}, error={e}")
    
    def _on_delete_clicked(self):
        """Handle delete button click"""
        selected_items = self.scenarios_list.selectedItems()
        
        if not selected_items:
            return
        
        scenario_name = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete scenario:\n{scenario_name}?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._delete_scenario(scenario_name)
    
    def _delete_scenario(self, scenario_name: str):
        """Delete a scenario from disk"""
        logger.info(f"Scenario deletion started: scenario={scenario_name}")
        
        try:
            import shutil
            
            scenario_dir = os.path.join(self.scenarios_path, scenario_name)
            
            # Get size before deletion for logging
            try:
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(scenario_dir)
                    for filename in filenames
                )
                size_mb = total_size / (1024 * 1024)
            except:
                size_mb = 0
            
            shutil.rmtree(scenario_dir)
            
            logger.info(f"Scenario deleted: scenario={scenario_name}, size_mb={size_mb:.2f}")
            
            # Refresh list
            self.refresh_scenarios()
            
            QMessageBox.information(
                self,
                "Delete Successful",
                f"Scenario deleted:\n{scenario_name}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Delete Failed",
                f"Failed to delete scenario:\n{str(e)}"
            )
            logger.error(f"Scenario deletion failed: scenario={scenario_name}, error={e}")
    
    def get_selected_scenario(self) -> Optional[str]:
        """
        Get currently selected scenario name.
        
        Returns:
            Scenario name or None
        """
        selected_items = self.scenarios_list.selectedItems()
        
        if selected_items:
            return selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        return None
    
    def get_scenario_metadata(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a scenario.
        
        Args:
            scenario_name: Name of scenario
            
        Returns:
            Metadata dictionary or None
        """
        return self.scenarios_data.get(scenario_name)



class ScenarioReplayDialog(QDialog):
    """
    Modal dialog for scenario playback.
    
    Features:
    - Synchronized video players for all cameras
    - Playback controls (play/pause, step, speed)
    - Timeline scrubber
    - Annotations overlay
    """
    
    def __init__(self, scenario_name: str, scenarios_path: str, parent=None):
        super().__init__(parent)
        
        self.scenario_name = scenario_name
        self.scenarios_path = scenarios_path
        self.scenario_dir = os.path.join(scenarios_path, scenario_name)
        
        # Playback state
        self.metadata: Optional[Dict[str, Any]] = None
        self.annotations: Optional[Dict[str, Any]] = None
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
        self.current_frame_index = 0
        self.num_frames = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.show_annotations = True
        
        # Timer for playback
        from PyQt6.QtCore import QTimer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._advance_frame)
        
        # Initialize UI
        self._init_ui()
        
        # Load scenario
        if not self._load_scenario():
            QMessageBox.critical(self, "Error", f"Failed to load scenario: {scenario_name}")
            self.reject()
            return
        
        # Display first frame
        self._display_frame(0)
        
        logger.info(f"ScenarioReplayDialog opened for: {scenario_name}")
    
    def _init_ui(self):
        """Initialize the dialog UI"""
        self.setWindowTitle(f"Scenario Replay - {self.scenario_name}")
        self.setMinimumSize(1200, 800)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Video displays grid
        from PyQt6.QtWidgets import QGridLayout
        video_grid = QGridLayout()
        video_grid.setSpacing(5)
        
        # Create video display labels
        self.video_labels = {}
        
        # Interior camera (top-left)
        self.video_labels['interior'] = self._create_video_label("Interior Camera")
        video_grid.addWidget(self.video_labels['interior'], 0, 0)
        
        # Front left camera (top-right)
        self.video_labels['front_left'] = self._create_video_label("Front Left Camera")
        video_grid.addWidget(self.video_labels['front_left'], 0, 1)
        
        # Front right camera (bottom-left)
        self.video_labels['front_right'] = self._create_video_label("Front Right Camera")
        video_grid.addWidget(self.video_labels['front_right'], 1, 0)
        
        # BEV (bottom-right)
        self.video_labels['bev'] = self._create_video_label("Bird's Eye View")
        video_grid.addWidget(self.video_labels['bev'], 1, 1)
        
        layout.addLayout(video_grid, 1)
        
        # Timeline scrubber
        from PyQt6.QtWidgets import QSlider
        timeline_layout = QHBoxLayout()
        
        self.timeline_label = QLabel("Frame: 0 / 0 | Time: 0.00s")
        timeline_layout.addWidget(self.timeline_label)
        
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.valueChanged.connect(self._on_timeline_changed)
        timeline_layout.addWidget(self.timeline_slider, 1)
        
        layout.addLayout(timeline_layout)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._on_play_pause)
        controls_layout.addWidget(self.play_button)
        
        # Step backward button
        self.step_back_button = QPushButton("◄ Step Back")
        self.step_back_button.clicked.connect(self._on_step_backward)
        controls_layout.addWidget(self.step_back_button)
        
        # Step forward button
        self.step_forward_button = QPushButton("Step Forward ►")
        self.step_forward_button.clicked.connect(self._on_step_forward)
        controls_layout.addWidget(self.step_forward_button)
        
        controls_layout.addStretch()
        
        # Speed control
        speed_label = QLabel("Speed:")
        controls_layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "1.5x", "2x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        controls_layout.addWidget(self.speed_combo)
        
        # Annotations toggle
        self.annotations_checkbox = QPushButton("Annotations: ON")
        self.annotations_checkbox.setCheckable(True)
        self.annotations_checkbox.setChecked(True)
        self.annotations_checkbox.clicked.connect(self._on_annotations_toggled)
        controls_layout.addWidget(self.annotations_checkbox)
        
        layout.addLayout(controls_layout)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _create_video_label(self, title: str) -> QWidget:
        """Create a video display widget with title"""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; background-color: #333; padding: 3px;")
        layout.addWidget(title_label)
        
        # Video display
        video_label = QLabel()
        video_label.setMinimumSize(400, 300)
        video_label.setScaledContents(True)
        video_label.setStyleSheet("border: 1px solid #555; background-color: #000;")
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(video_label)
        
        container.setLayout(layout)
        container.video_label = video_label  # Store reference
        
        return container
    
    def _load_scenario(self) -> bool:
        """Load scenario data from disk"""
        # Load metadata
        metadata_path = os.path.join(self.scenario_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata not found: {metadata_path}")
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return False
        
        # Load annotations
        annotations_path = os.path.join(self.scenario_dir, 'annotations.json')
        if not os.path.exists(annotations_path):
            logger.error(f"Annotations not found: {annotations_path}")
            return False
        
        try:
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        except Exception as e:
            logger.error(f"Error loading annotations: {e}")
            return False
        
        # Open video captures
        video_files = self.metadata.get('files', {})
        
        for camera_name, video_file in video_files.items():
            video_path = os.path.join(self.scenario_dir, video_file)
            
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    self.video_captures[camera_name] = cap
                    logger.debug(f"Opened video: {camera_name}")
                else:
                    logger.warning(f"Failed to open video: {video_path}")
        
        # Get number of frames
        self.num_frames = self.metadata.get('num_frames', len(self.annotations.get('frames', [])))
        
        # Update timeline
        self.timeline_slider.setMaximum(max(0, self.num_frames - 1))
        
        logger.info(f"Scenario loaded: {self.num_frames} frames, duration={self.metadata.get('duration', 0):.2f}s")
        
        return True
    
    def _display_frame(self, frame_index: int):
        """Display a specific frame"""
        if frame_index < 0 or frame_index >= self.num_frames:
            return
        
        self.current_frame_index = frame_index
        
        # Update timeline
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setValue(frame_index)
        self.timeline_slider.blockSignals(False)
        
        # Get frame timestamp
        frames_data = self.annotations.get('frames', [])
        if frame_index < len(frames_data):
            timestamp = frames_data[frame_index].get('timestamp', 0)
        else:
            timestamp = 0
        
        # Update timeline label
        self.timeline_label.setText(f"Frame: {frame_index + 1} / {self.num_frames} | Time: {timestamp:.2f}s")
        
        # Display camera frames
        for camera_name, cap in self.video_captures.items():
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply annotations if enabled
                if self.show_annotations and frame_index < len(frames_data):
                    frame_rgb = self._apply_annotations(frame_rgb, frames_data[frame_index], camera_name)
                
                # Convert to QPixmap
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                
                # Display in label
                if camera_name in self.video_labels:
                    self.video_labels[camera_name].video_label.setPixmap(pixmap)
    
    def _apply_annotations(self, frame: np.ndarray, frame_data: Dict[str, Any], camera_name: str) -> np.ndarray:
        """Apply annotations overlay to frame"""
        # For BEV, draw detections and risk zones
        if camera_name == 'bev':
            # Draw detections (simplified - just bounding boxes)
            detections = frame_data.get('detections_3d', [])
            for detection in detections:
                # This is simplified - in a real implementation, you'd project 3D boxes to BEV
                pass
        
        # Add alert text if any
        alerts = frame_data.get('alerts', [])
        if alerts:
            y_offset = 30
            for alert in alerts[:3]:  # Show top 3 alerts
                message = alert.get('message', '')
                urgency = alert.get('urgency', 'info')
                
                # Choose color based on urgency
                if urgency == 'critical':
                    color = (255, 0, 0)
                elif urgency == 'warning':
                    color = (255, 165, 0)
                else:
                    color = (0, 255, 255)
                
                cv2.putText(frame, message, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2, cv2.LINE_AA)
                y_offset += 25
        
        return frame
    
    def _on_play_pause(self):
        """Handle play/pause button click"""
        if self.is_playing:
            self._pause()
        else:
            self._play()
    
    def _play(self):
        """Start playback"""
        self.is_playing = True
        self.play_button.setText("Pause")
        
        # Calculate timer interval based on speed
        # Assuming 30 FPS
        base_interval = int(1000 / 30)  # ms
        interval = int(base_interval / self.playback_speed)
        
        self.playback_timer.start(interval)
        logger.debug(f"Playback started at {self.playback_speed}x speed")
    
    def _pause(self):
        """Pause playback"""
        self.is_playing = False
        self.play_button.setText("Play")
        self.playback_timer.stop()
        logger.debug("Playback paused")
    
    def _advance_frame(self):
        """Advance to next frame during playback"""
        if self.current_frame_index < self.num_frames - 1:
            self._display_frame(self.current_frame_index + 1)
        else:
            # Reached end, stop playback
            self._pause()
            logger.debug("Playback reached end")
    
    def _on_step_forward(self):
        """Step forward one frame"""
        if self.current_frame_index < self.num_frames - 1:
            self._display_frame(self.current_frame_index + 1)
    
    def _on_step_backward(self):
        """Step backward one frame"""
        if self.current_frame_index > 0:
            self._display_frame(self.current_frame_index - 1)
    
    def _on_timeline_changed(self, value: int):
        """Handle timeline slider change"""
        if not self.is_playing:
            self._display_frame(value)
    
    def _on_speed_changed(self, speed_text: str):
        """Handle playback speed change"""
        speed_map = {
            "0.25x": 0.25,
            "0.5x": 0.5,
            "1x": 1.0,
            "1.5x": 1.5,
            "2x": 2.0
        }
        
        self.playback_speed = speed_map.get(speed_text, 1.0)
        logger.debug(f"Playback speed changed to {self.playback_speed}x")
        
        # If playing, restart timer with new interval
        if self.is_playing:
            self._pause()
            self._play()
    
    def _on_annotations_toggled(self, checked: bool):
        """Handle annotations toggle"""
        self.show_annotations = checked
        
        if checked:
            self.annotations_checkbox.setText("Annotations: ON")
        else:
            self.annotations_checkbox.setText("Annotations: OFF")
        
        # Refresh current frame
        self._display_frame(self.current_frame_index)
        
        logger.debug(f"Annotations {'enabled' if checked else 'disabled'}")
    
    def closeEvent(self, event):
        """Handle dialog close"""
        # Stop playback
        if self.is_playing:
            self._pause()
        
        # Release video captures
        for cap in self.video_captures.values():
            cap.release()
        
        self.video_captures.clear()
        
        logger.info("ScenarioReplayDialog closed")
        
        event.accept()
