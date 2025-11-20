"""Trip statistics display widget."""

import logging
from datetime import timedelta
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont

from src.core.data_structures import TripStats


class TripStatsWidget(QWidget):
    """
    Widget displaying current trip statistics.
    Shows duration, distance, speeds, events, and safety score.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("🚗 Trip Statistics")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Main stats
        main_frame = QFrame()
        main_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        main_layout = QGridLayout(main_frame)

        # Duration
        main_layout.addWidget(QLabel("Duration:"), 0, 0)
        self.duration_label = QLabel("--")
        self.duration_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.duration_label, 0, 1)

        # Distance
        main_layout.addWidget(QLabel("Distance:"), 1, 0)
        self.distance_label = QLabel("--")
        self.distance_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.distance_label, 1, 1)

        # Average speed
        main_layout.addWidget(QLabel("Avg Speed:"), 2, 0)
        self.avg_speed_label = QLabel("--")
        self.avg_speed_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.avg_speed_label, 2, 1)

        # Max speed
        main_layout.addWidget(QLabel("Max Speed:"), 3, 0)
        self.max_speed_label = QLabel("--")
        self.max_speed_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.max_speed_label, 3, 1)

        layout.addWidget(main_frame)

        # Safety events
        events_frame = QFrame()
        events_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        events_layout = QVBoxLayout(events_frame)

        events_title = QLabel("Safety Events")
        events_title.setStyleSheet("font-weight: bold;")
        events_layout.addWidget(events_title)

        events_grid = QGridLayout()

        # Hard brakes
        events_grid.addWidget(QLabel("🛑 Hard Brakes:"), 0, 0)
        self.hard_brakes_label = QLabel("0")
        self.hard_brakes_label.setStyleSheet("font-weight: bold;")
        events_grid.addWidget(self.hard_brakes_label, 0, 1)

        # Rapid accelerations
        events_grid.addWidget(QLabel("⚡ Rapid Accel:"), 1, 0)
        self.rapid_accel_label = QLabel("0")
        self.rapid_accel_label.setStyleSheet("font-weight: bold;")
        events_grid.addWidget(self.rapid_accel_label, 1, 1)

        # Lane departures
        events_grid.addWidget(QLabel("🛣️ Lane Depart:"), 2, 0)
        self.lane_depart_label = QLabel("0")
        self.lane_depart_label.setStyleSheet("font-weight: bold;")
        events_grid.addWidget(self.lane_depart_label, 2, 1)

        # Collision warnings
        events_grid.addWidget(QLabel("⚠️ Collisions:"), 3, 0)
        self.collision_warn_label = QLabel("0")
        self.collision_warn_label.setStyleSheet("font-weight: bold;")
        events_grid.addWidget(self.collision_warn_label, 3, 1)

        # Blind spot warnings
        events_grid.addWidget(QLabel("👁️ Blind Spot:"), 4, 0)
        self.blind_spot_warn_label = QLabel("0")
        self.blind_spot_warn_label.setStyleSheet("font-weight: bold;")
        events_grid.addWidget(self.blind_spot_warn_label, 4, 1)

        events_layout.addLayout(events_grid)
        layout.addWidget(events_frame)

        # Safety score
        score_frame = QFrame()
        score_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        score_layout = QVBoxLayout(score_frame)

        score_title = QLabel("Trip Safety Score")
        score_title.setStyleSheet("font-weight: bold;")
        score_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_layout.addWidget(score_title)

        self.safety_score_label = QLabel("--")
        self.safety_score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_font = QFont()
        score_font.setPointSize(28)
        score_font.setBold(True)
        self.safety_score_label.setFont(score_font)
        score_layout.addWidget(self.safety_score_label)

        layout.addWidget(score_frame)

        # Driver attention
        attention_frame = QFrame()
        attention_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        attention_layout = QHBoxLayout(attention_frame)

        attention_layout.addWidget(QLabel("Avg Attention:"))
        self.attention_label = QLabel("--")
        self.attention_label.setStyleSheet("font-weight: bold;")
        attention_layout.addWidget(self.attention_label)
        attention_layout.addStretch()

        layout.addWidget(attention_frame)

        layout.addStretch()

    @pyqtSlot(object, name="updateTripStats")
    def update_trip_stats(self, stats: TripStats):
        """Update trip statistics display."""
        if stats is None:
            return

        # Update duration
        duration = int(stats.duration)
        hours = duration // 3600
        minutes = (duration % 3600) // 60
        seconds = duration % 60

        if hours > 0:
            duration_text = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_text = f"{minutes}m {seconds}s"
        else:
            duration_text = f"{seconds}s"

        self.duration_label.setText(duration_text)

        # Update distance
        distance_km = stats.distance / 1000
        self.distance_label.setText(f"{distance_km:.2f} km")

        # Update speeds
        avg_speed_kmh = stats.average_speed * 3.6
        max_speed_kmh = stats.max_speed * 3.6
        self.avg_speed_label.setText(f"{avg_speed_kmh:.1f} km/h")
        self.max_speed_label.setText(f"{max_speed_kmh:.1f} km/h")

        # Update safety events
        self.hard_brakes_label.setText(str(stats.num_hard_brakes))
        self.rapid_accel_label.setText(str(stats.num_rapid_accelerations))
        self.lane_depart_label.setText(str(stats.num_lane_departures))
        self.collision_warn_label.setText(str(stats.num_collision_warnings))
        self.blind_spot_warn_label.setText(str(stats.num_blind_spot_warnings))

        # Color code event counts
        self._color_code_label(self.hard_brakes_label, stats.num_hard_brakes)
        self._color_code_label(self.rapid_accel_label, stats.num_rapid_accelerations)
        self._color_code_label(self.lane_depart_label, stats.num_lane_departures)
        self._color_code_label(self.collision_warn_label, stats.num_collision_warnings)
        self._color_code_label(self.blind_spot_warn_label, stats.num_blind_spot_warnings)

        # Update safety score
        safety_score = int(stats.safety_score)
        self.safety_score_label.setText(f"{safety_score}")

        # Color based on score
        if safety_score >= 90:
            color = "#4ade80"  # Green
        elif safety_score >= 75:
            color = "#fbbf24"  # Yellow
        elif safety_score >= 60:
            color = "#fb923c"  # Orange
        else:
            color = "#f87171"  # Red

        self.safety_score_label.setStyleSheet(f"color: {color};")

        # Update attention score
        attention_score = int(stats.average_attention_score)
        self.attention_label.setText(f"{attention_score}")

        if attention_score >= 80:
            color = "#4ade80"
        elif attention_score >= 60:
            color = "#fbbf24"
        else:
            color = "#f87171"

        self.attention_label.setStyleSheet(f"font-weight: bold; color: {color};")

    def _color_code_label(self, label: QLabel, count: int):
        """Color code event count labels."""
        if count == 0:
            color = "#4ade80"  # Green
        elif count <= 2:
            color = "#fbbf24"  # Yellow
        elif count <= 5:
            color = "#fb923c"  # Orange
        else:
            color = "#f87171"  # Red

        label.setStyleSheet(f"font-weight: bold; color: {color};")
