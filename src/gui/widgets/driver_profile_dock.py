"""
Driver Profile Management Dock Widget

Displays driver profile information, statistics, and behavior trends.
"""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QGridLayout, QProgressBar, QComboBox,
    QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DriverProfileDock(QDockWidget):
    """
    Dock widget for driver profile management.
    
    Displays:
    - Current driver information
    - Driving style and scores
    - Behavior trends over time
    - Profile management controls
    """
    
    # Signals
    profile_reset_requested = pyqtSignal(str)  # driver_id
    profile_deleted_requested = pyqtSignal(str)  # driver_id
    
    def __init__(self, parent=None):
        """Initialize driver profile dock."""
        super().__init__("Driver Profile", parent)
        
        self.current_driver_id: Optional[str] = None
        self.profile_data: Optional[Dict] = None
        
        self._init_ui()
        
        logger.info("DriverProfileDock initialized")
    
    def _init_ui(self):
        """Initialize UI components."""
        # Main widget
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        
        # Driver info section
        self.driver_info_group = self._create_driver_info_section()
        layout.addWidget(self.driver_info_group)
        
        # Scores section
        self.scores_group = self._create_scores_section()
        layout.addWidget(self.scores_group)
        
        # Trends section
        self.trends_group = self._create_trends_section()
        layout.addWidget(self.trends_group)
        
        # Statistics section
        self.stats_group = self._create_statistics_section()
        layout.addWidget(self.stats_group)
        
        # Controls section
        controls_layout = self._create_controls_section()
        layout.addLayout(controls_layout)
        
        layout.addStretch()
        
        self.setWidget(main_widget)
        
        # Set initial state
        self._set_no_driver_state()
    
    def _create_driver_info_section(self) -> QGroupBox:
        """Create driver information section."""
        group = QGroupBox("Driver Information")
        layout = QGridLayout()
        
        # Driver ID
        layout.addWidget(QLabel("Driver ID:"), 0, 0)
        self.driver_id_label = QLabel("None")
        self.driver_id_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.driver_id_label, 0, 1)
        
        # Driving style
        layout.addWidget(QLabel("Driving Style:"), 1, 0)
        self.driving_style_label = QLabel("Unknown")
        self.driving_style_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.driving_style_label, 1, 1)
        
        # Session count
        layout.addWidget(QLabel("Total Sessions:"), 2, 0)
        self.session_count_label = QLabel("0")
        layout.addWidget(self.session_count_label, 2, 1)
        
        # Total distance
        layout.addWidget(QLabel("Total Distance:"), 3, 0)
        self.total_distance_label = QLabel("0.0 km")
        layout.addWidget(self.total_distance_label, 3, 1)
        
        # Total time
        layout.addWidget(QLabel("Total Time:"), 4, 0)
        self.total_time_label = QLabel("0.0 hours")
        layout.addWidget(self.total_time_label, 4, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_scores_section(self) -> QGroupBox:
        """Create scores section with progress bars."""
        group = QGroupBox("Performance Scores")
        layout = QVBoxLayout()
        
        # Safety score
        safety_layout = QHBoxLayout()
        safety_layout.addWidget(QLabel("Safety:"))
        self.safety_bar = QProgressBar()
        self.safety_bar.setRange(0, 100)
        self.safety_bar.setValue(0)
        self.safety_bar.setFormat("%v/100")
        safety_layout.addWidget(self.safety_bar)
        layout.addLayout(safety_layout)
        
        # Attention score
        attention_layout = QHBoxLayout()
        attention_layout.addWidget(QLabel("Attention:"))
        self.attention_bar = QProgressBar()
        self.attention_bar.setRange(0, 100)
        self.attention_bar.setValue(0)
        self.attention_bar.setFormat("%v/100")
        attention_layout.addWidget(self.attention_bar)
        layout.addLayout(attention_layout)
        
        # Eco-driving score
        eco_layout = QHBoxLayout()
        eco_layout.addWidget(QLabel("Eco-Driving:"))
        self.eco_bar = QProgressBar()
        self.eco_bar.setRange(0, 100)
        self.eco_bar.setValue(0)
        self.eco_bar.setFormat("%v/100")
        eco_layout.addWidget(self.eco_bar)
        layout.addLayout(eco_layout)
        
        # Overall score
        overall_layout = QHBoxLayout()
        overall_label = QLabel("Overall:")
        overall_label.setStyleSheet("font-weight: bold;")
        overall_layout.addWidget(overall_label)
        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(0)
        self.overall_bar.setFormat("%v/100")
        self.overall_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: #00aaff;
            }
        """)
        overall_layout.addWidget(self.overall_bar)
        layout.addLayout(overall_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_trends_section(self) -> QGroupBox:
        """Create trends visualization section."""
        group = QGroupBox("Score Trends")
        layout = QVBoxLayout()
        
        # Create plot widget
        self.trends_plot = pg.PlotWidget()
        self.trends_plot.setBackground('w')
        self.trends_plot.setLabel('left', 'Score', units='')
        self.trends_plot.setLabel('bottom', 'Session', units='')
        self.trends_plot.setYRange(0, 100)
        self.trends_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trends_plot.setMinimumHeight(200)
        
        # Add legend
        self.trends_plot.addLegend()
        
        # Initialize plot lines
        self.safety_line = self.trends_plot.plot(
            [], [], pen=pg.mkPen(color='r', width=2), name='Safety'
        )
        self.attention_line = self.trends_plot.plot(
            [], [], pen=pg.mkPen(color='b', width=2), name='Attention'
        )
        self.eco_line = self.trends_plot.plot(
            [], [], pen=pg.mkPen(color='g', width=2), name='Eco-Driving'
        )
        
        layout.addWidget(self.trends_plot)
        
        group.setLayout(layout)
        return group
    
    def _create_statistics_section(self) -> QGroupBox:
        """Create detailed statistics section."""
        group = QGroupBox("Behavior Statistics")
        layout = QGridLayout()
        
        # Reaction time
        layout.addWidget(QLabel("Avg Reaction Time:"), 0, 0)
        self.reaction_time_label = QLabel("0.00 s")
        layout.addWidget(self.reaction_time_label, 0, 1)
        
        # Following distance
        layout.addWidget(QLabel("Avg Following Distance:"), 1, 0)
        self.following_distance_label = QLabel("0.0 m")
        layout.addWidget(self.following_distance_label, 1, 1)
        
        # Lane change frequency
        layout.addWidget(QLabel("Lane Changes/Hour:"), 2, 0)
        self.lane_change_freq_label = QLabel("0.0")
        layout.addWidget(self.lane_change_freq_label, 2, 1)
        
        # Average speed
        layout.addWidget(QLabel("Avg Speed:"), 3, 0)
        self.avg_speed_label = QLabel("0.0 km/h")
        layout.addWidget(self.avg_speed_label, 3, 1)
        
        # Risk tolerance
        layout.addWidget(QLabel("Risk Tolerance:"), 4, 0)
        self.risk_tolerance_label = QLabel("0.50")
        layout.addWidget(self.risk_tolerance_label, 4, 1)
        
        group.setLayout(layout)
        return group
    
    def _create_controls_section(self) -> QHBoxLayout:
        """Create control buttons section."""
        layout = QHBoxLayout()
        
        # Reset profile button
        self.reset_button = QPushButton("Reset Profile")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)
        
        # Delete profile button
        self.delete_button = QPushButton("Delete Profile")
        self.delete_button.clicked.connect(self._on_delete_clicked)
        self.delete_button.setEnabled(False)
        self.delete_button.setStyleSheet("background-color: #ff4444;")
        layout.addWidget(self.delete_button)
        
        return layout
    
    @pyqtSlot(dict)
    def update_profile(self, profile_data: Dict):
        """
        Update display with profile data.
        
        Args:
            profile_data: Dictionary with profile information
        """
        self.profile_data = profile_data
        self.current_driver_id = profile_data.get('driver_id')
        
        # Update driver info
        self.driver_id_label.setText(self.current_driver_id or "Unknown")
        
        driving_style = profile_data.get('driving_style', 'unknown').capitalize()
        self.driving_style_label.setText(driving_style)
        self._set_driving_style_color(driving_style.lower())
        
        self.session_count_label.setText(str(profile_data.get('session_count', 0)))
        
        total_distance_km = profile_data.get('total_distance', 0.0) / 1000.0
        self.total_distance_label.setText(f"{total_distance_km:.2f} km")
        
        total_time_hours = profile_data.get('total_time', 0.0) / 3600.0
        self.total_time_label.setText(f"{total_time_hours:.2f} hours")
        
        # Update scores
        safety_score = int(profile_data.get('safety_score', 0))
        attention_score = int(profile_data.get('attention_score', 0))
        eco_score = int(profile_data.get('eco_score', 0))
        overall_score = int((safety_score + attention_score + eco_score) / 3)
        
        self.safety_bar.setValue(safety_score)
        self._set_score_color(self.safety_bar, safety_score)
        
        self.attention_bar.setValue(attention_score)
        self._set_score_color(self.attention_bar, attention_score)
        
        self.eco_bar.setValue(eco_score)
        self._set_score_color(self.eco_bar, eco_score)
        
        self.overall_bar.setValue(overall_score)
        
        # Update statistics
        self.reaction_time_label.setText(
            f"{profile_data.get('avg_reaction_time', 0.0):.2f} s"
        )
        self.following_distance_label.setText(
            f"{profile_data.get('avg_following_distance', 0.0):.1f} m"
        )
        self.lane_change_freq_label.setText(
            f"{profile_data.get('avg_lane_change_freq', 0.0):.1f}"
        )
        
        avg_speed_kmh = profile_data.get('avg_speed', 0.0) * 3.6
        self.avg_speed_label.setText(f"{avg_speed_kmh:.1f} km/h")
        
        self.risk_tolerance_label.setText(
            f"{profile_data.get('risk_tolerance', 0.5):.2f}"
        )
        
        # Enable controls
        self.reset_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        
        logger.debug(f"Profile display updated for {self.current_driver_id}")
    
    @pyqtSlot(list)
    def update_trends(self, history: list):
        """
        Update trends plot with historical data.
        
        Args:
            history: List of historical profile snapshots
        """
        if not history:
            return
        
        # Extract scores from history
        sessions = list(range(len(history)))
        safety_scores = [h.get('safety_score', 0) for h in history]
        attention_scores = [h.get('attention_score', 0) for h in history]
        eco_scores = [h.get('eco_score', 0) for h in history]
        
        # Update plot lines
        self.safety_line.setData(sessions, safety_scores)
        self.attention_line.setData(sessions, attention_scores)
        self.eco_line.setData(sessions, eco_scores)
        
        logger.debug(f"Trends updated with {len(history)} data points")
    
    def _set_driving_style_color(self, style: str):
        """Set color for driving style label."""
        colors = {
            'aggressive': '#ff4444',
            'normal': '#44ff44',
            'cautious': '#4444ff',
            'unknown': '#888888'
        }
        color = colors.get(style, '#888888')
        self.driving_style_label.setStyleSheet(f"font-weight: bold; color: {color};")
    
    def _set_score_color(self, progress_bar: QProgressBar, score: int):
        """Set color for score progress bar based on value."""
        if score >= 80:
            color = '#44ff44'  # Green
        elif score >= 60:
            color = '#ffaa00'  # Orange
        else:
            color = '#ff4444'  # Red
        
        progress_bar.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)
    
    def _set_no_driver_state(self):
        """Set UI to no driver state."""
        self.driver_id_label.setText("No driver identified")
        self.driving_style_label.setText("Unknown")
        self.session_count_label.setText("0")
        self.total_distance_label.setText("0.0 km")
        self.total_time_label.setText("0.0 hours")
        
        self.safety_bar.setValue(0)
        self.attention_bar.setValue(0)
        self.eco_bar.setValue(0)
        self.overall_bar.setValue(0)
        
        self.reaction_time_label.setText("0.00 s")
        self.following_distance_label.setText("0.0 m")
        self.lane_change_freq_label.setText("0.0")
        self.avg_speed_label.setText("0.0 km/h")
        self.risk_tolerance_label.setText("0.50")
        
        # Clear trends
        self.safety_line.setData([], [])
        self.attention_line.setData([], [])
        self.eco_line.setData([], [])
        
        # Disable controls
        self.reset_button.setEnabled(False)
        self.delete_button.setEnabled(False)
    
    def _on_reset_clicked(self):
        """Handle reset button click."""
        if self.current_driver_id is None:
            return
        
        reply = QMessageBox.question(
            self,
            "Reset Profile",
            f"Are you sure you want to reset the profile for {self.current_driver_id}?\n"
            "This will clear all statistics but keep the driver identity.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.profile_reset_requested.emit(self.current_driver_id)
            logger.info(f"Profile reset requested for {self.current_driver_id}")
    
    def _on_delete_clicked(self):
        """Handle delete button click."""
        if self.current_driver_id is None:
            return
        
        reply = QMessageBox.warning(
            self,
            "Delete Profile",
            f"Are you sure you want to permanently delete the profile for {self.current_driver_id}?\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.profile_deleted_requested.emit(self.current_driver_id)
            self._set_no_driver_state()
            logger.info(f"Profile deletion requested for {self.current_driver_id}")
    
    def clear(self):
        """Clear all displayed data."""
        self._set_no_driver_state()
        self.current_driver_id = None
        self.profile_data = None
