"""Advanced features dock widget combining all new features."""

import logging
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTabWidget, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont

from src.core.data_structures import (
    LaneState, BlindSpotWarning, CollisionWarning, TrafficSign,
    RoadCondition, ParkingSpace, DriverScore, TripStats
)
from .safety_indicators import SafetyIndicatorsWidget
from .driver_score_widget import DriverScoreWidget
from .trip_stats_widget import TripStatsWidget
from .gps_widget import GPSWidget


class AdvancedFeaturesDock(QDockWidget):
    """
    Dock widget displaying all advanced safety and analytics features.
    Uses tabs to organize different feature categories.
    """

    def __init__(self, parent=None):
        super().__init__("🚀 Advanced Features", parent)
        self.logger = logging.getLogger(__name__)

        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface."""
        # Main widget
        main_widget = QWidget()
        layout = QVBoxLayout(main_widget)
        layout.setContentsMargins(5, 5, 5, 5)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)

        # Safety tab
        self.safety_widget = SafetyIndicatorsWidget()
        scroll_safety = QScrollArea()
        scroll_safety.setWidget(self.safety_widget)
        scroll_safety.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_safety, "🛡️ Safety")

        # Driver Score tab
        self.driver_score_widget = DriverScoreWidget()
        scroll_score = QScrollArea()
        scroll_score.setWidget(self.driver_score_widget)
        scroll_score.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_score, "📊 Score")

        # Trip Stats tab
        self.trip_stats_widget = TripStatsWidget()
        scroll_trip = QScrollArea()
        scroll_trip.setWidget(self.trip_stats_widget)
        scroll_trip.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_trip, "🚗 Trip")

        # Road Conditions tab
        self.road_widget = self._create_road_conditions_widget()
        scroll_road = QScrollArea()
        scroll_road.setWidget(self.road_widget)
        scroll_road.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_road, "🌦️ Road")

        # Traffic Signs tab
        self.signs_widget = self._create_traffic_signs_widget()
        scroll_signs = QScrollArea()
        scroll_signs.setWidget(self.signs_widget)
        scroll_signs.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_signs, "🚦 Signs")

        # GPS tab
        self.gps_widget = GPSWidget()
        scroll_gps = QScrollArea()
        scroll_gps.setWidget(self.gps_widget)
        scroll_gps.setWidgetResizable(True)
        self.tab_widget.addTab(scroll_gps, "🌐 GPS")

        layout.addWidget(self.tab_widget)

        self.setWidget(main_widget)

    def _create_road_conditions_widget(self) -> QWidget:
        """Create road conditions display widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Title
        title = QLabel("🌦️ Road Conditions")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Surface condition
        surface_frame = QFrame()
        surface_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        surface_layout = QVBoxLayout(surface_frame)

        surface_title = QLabel("Surface Type")
        surface_title.setStyleSheet("font-weight: bold;")
        surface_layout.addWidget(surface_title)

        self.surface_type_label = QLabel("Unknown")
        self.surface_type_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        surface_font = QFont()
        surface_font.setPointSize(16)
        surface_font.setBold(True)
        self.surface_type_label.setFont(surface_font)
        surface_layout.addWidget(self.surface_type_label)

        self.friction_label = QLabel("Friction: --")
        self.friction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        surface_layout.addWidget(self.friction_label)

        layout.addWidget(surface_frame)

        # Visibility
        visibility_frame = QFrame()
        visibility_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        visibility_layout = QVBoxLayout(visibility_frame)

        visibility_title = QLabel("Visibility")
        visibility_title.setStyleSheet("font-weight: bold;")
        visibility_layout.addWidget(visibility_title)

        self.visibility_label = QLabel("Clear")
        self.visibility_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vis_font = QFont()
        vis_font.setPointSize(14)
        vis_font.setBold(True)
        self.visibility_label.setFont(vis_font)
        visibility_layout.addWidget(self.visibility_label)

        layout.addWidget(visibility_frame)

        # Hazards
        hazards_frame = QFrame()
        hazards_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        hazards_layout = QVBoxLayout(hazards_frame)

        hazards_title = QLabel("Detected Hazards")
        hazards_title.setStyleSheet("font-weight: bold;")
        hazards_layout.addWidget(hazards_title)

        self.hazards_label = QLabel("None")
        self.hazards_label.setWordWrap(True)
        self.hazards_label.setStyleSheet("color: #888;")
        hazards_layout.addWidget(self.hazards_label)

        layout.addWidget(hazards_frame)

        layout.addStretch()

        return widget

    def _create_traffic_signs_widget(self) -> QWidget:
        """Create traffic signs display widget."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        # Title
        title = QLabel("🚦 Traffic Signs")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Current speed limit
        speed_frame = QFrame()
        speed_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        speed_layout = QVBoxLayout(speed_frame)

        speed_title = QLabel("Current Speed Limit")
        speed_title.setStyleSheet("font-weight: bold;")
        speed_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_layout.addWidget(speed_title)

        self.speed_limit_label = QLabel("--")
        self.speed_limit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_font = QFont()
        speed_font.setPointSize(36)
        speed_font.setBold(True)
        self.speed_limit_label.setFont(speed_font)
        self.speed_limit_label.setStyleSheet("color: #fbbf24;")
        speed_layout.addWidget(self.speed_limit_label)

        unit_label = QLabel("km/h")
        unit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_layout.addWidget(unit_label)

        layout.addWidget(speed_frame)

        # Detected signs
        signs_frame = QFrame()
        signs_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        signs_layout = QVBoxLayout(signs_frame)

        signs_title = QLabel("Recent Signs")
        signs_title.setStyleSheet("font-weight: bold;")
        signs_layout.addWidget(signs_title)

        self.signs_list_label = QLabel("No signs detected")
        self.signs_list_label.setWordWrap(True)
        self.signs_list_label.setStyleSheet("color: #888;")
        signs_layout.addWidget(self.signs_list_label)

        layout.addWidget(signs_frame)

        layout.addStretch()

        return widget

    # Signal handlers for safety indicators
    @pyqtSlot(object, name="updateLaneState")
    def update_lane_state(self, lane_state: LaneState):
        """Update lane state in safety tab."""
        self.safety_widget.update_lane_state(lane_state)

    @pyqtSlot(object, name="updateBlindSpotWarning")
    def update_blind_spot_warning(self, warning: BlindSpotWarning):
        """Update blind spot warning in safety tab."""
        self.safety_widget.update_blind_spot(warning)

    @pyqtSlot(object, name="updateCollisionWarning")
    def update_collision_warning(self, warning: CollisionWarning):
        """Update collision warning in safety tab."""
        self.safety_widget.update_collision_warning(warning)

    # Signal handler for driver score
    @pyqtSlot(object, name="updateDriverScore")
    def update_driver_score(self, score: DriverScore):
        """Update driver score in score tab."""
        self.driver_score_widget.update_driver_score(score)

    # Signal handler for trip stats
    @pyqtSlot(object, name="updateTripStats")
    def update_trip_stats(self, stats: TripStats):
        """Update trip statistics in trip tab."""
        self.trip_stats_widget.update_trip_stats(stats)

    # Signal handler for road conditions
    @pyqtSlot(object, name="updateRoadCondition")
    def update_road_condition(self, condition: RoadCondition):
        """Update road conditions in road tab."""
        if condition is None:
            return

        # Update surface type
        surface = condition.surface_type.upper()
        self.surface_type_label.setText(surface)

        # Color based on surface
        if surface == "DRY":
            color = "#4ade80"
        elif surface == "WET":
            color = "#60a5fa"
        elif surface == "SNOW":
            color = "#e0e7ff"
        elif surface == "ICE":
            color = "#fca5a5"
        else:
            color = "#888"

        self.surface_type_label.setStyleSheet(f"color: {color};")

        # Update friction
        friction = int(condition.friction_estimate * 100)
        self.friction_label.setText(f"Friction: {friction}%")

        # Update visibility
        self.visibility_label.setText(condition.visibility.title())

        # Update hazards
        if condition.hazards:
            hazards_text = "\n".join([f"⚠️ {h.title()}" for h in condition.hazards])
            self.hazards_label.setText(hazards_text)
            self.hazards_label.setStyleSheet("color: #fb923c;")
        else:
            self.hazards_label.setText("None")
            self.hazards_label.setStyleSheet("color: #888;")

    # Signal handler for traffic signs
    @pyqtSlot(list, name="updateTrafficSigns")
    def update_traffic_signs(self, signs: list):
        """Update traffic signs in signs tab."""
        if not signs:
            self.speed_limit_label.setText("--")
            self.signs_list_label.setText("No signs detected")
            self.signs_list_label.setStyleSheet("color: #888;")
            return

        # Extract speed limits
        speed_limits = [s for s in signs if s.sign_type == 'speed_limit' and s.value]

        if speed_limits:
            # Show the most recent/confident speed limit
            latest = max(speed_limits, key=lambda s: s.confidence)
            self.speed_limit_label.setText(str(latest.value))
        else:
            self.speed_limit_label.setText("--")

        # List all detected signs
        signs_text = []
        for sign in signs[-10:]:  # Show last 10
            sign_name = sign.sign_class.replace('_', ' ').title()
            confidence = int(sign.confidence * 100)
            signs_text.append(f"• {sign_name} ({confidence}%)")

        self.signs_list_label.setText("\n".join(signs_text))
        self.signs_list_label.setStyleSheet("color: #fff;")

    # Signal handlers for GPS
    @pyqtSlot(object, name="updateGPSData")
    def update_gps_data(self, location_info: dict):
        """Update GPS data in GPS tab."""
        self.gps_widget.update_gps_data(location_info)

    @pyqtSlot(object, name="updateSpeedViolation")
    def update_speed_violation(self, violation: dict):
        """Update speed violation warning in GPS tab."""
        self.gps_widget.update_speed_violation(violation)
