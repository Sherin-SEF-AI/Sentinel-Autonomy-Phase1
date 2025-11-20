"""Safety indicators widget for blind spot, collision warning, and lane departure."""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor

from src.core.data_structures import BlindSpotWarning, CollisionWarning, LaneState


class SafetyIndicatorsWidget(QWidget):
    """
    Widget displaying real-time safety indicators:
    - Blind spot warnings (left/right)
    - Forward collision warning level
    - Lane departure warning
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self.setup_ui()

        # Blink timer for critical warnings
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self._toggle_blink)
        self.blink_timer.setInterval(500)  # 500ms blink rate
        self.blink_state = False

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("🛡️ Safety Indicators")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Blind Spot Indicators
        blind_spot_frame = self._create_blind_spot_section()
        layout.addWidget(blind_spot_frame)

        # Collision Warning Indicator
        collision_frame = self._create_collision_section()
        layout.addWidget(collision_frame)

        # Lane Departure Indicator
        lane_frame = self._create_lane_section()
        layout.addWidget(lane_frame)

        layout.addStretch()

    def _create_blind_spot_section(self) -> QFrame:
        """Create blind spot monitoring section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Blind Spot Monitor")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        # Left and Right indicators
        indicators_layout = QHBoxLayout()

        # Left indicator
        self.left_blind_spot = QLabel("◀ LEFT")
        self.left_blind_spot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_blind_spot.setMinimumHeight(60)
        self.left_blind_spot.setStyleSheet("""
            background-color: #2e2e2e;
            color: #888;
            border: 2px solid #444;
            border-radius: 5px;
            font-size: 14pt;
            font-weight: bold;
        """)
        indicators_layout.addWidget(self.left_blind_spot)

        # Right indicator
        self.right_blind_spot = QLabel("RIGHT ▶")
        self.right_blind_spot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_blind_spot.setMinimumHeight(60)
        self.right_blind_spot.setStyleSheet("""
            background-color: #2e2e2e;
            color: #888;
            border: 2px solid #444;
            border-radius: 5px;
            font-size: 14pt;
            font-weight: bold;
        """)
        indicators_layout.addWidget(self.right_blind_spot)

        layout.addLayout(indicators_layout)

        # Status text
        self.blind_spot_status = QLabel("Clear")
        self.blind_spot_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.blind_spot_status)

        return frame

    def _create_collision_section(self) -> QFrame:
        """Create forward collision warning section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Forward Collision Warning")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        # Warning level indicator
        self.collision_indicator = QLabel("NONE")
        self.collision_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.collision_indicator.setMinimumHeight(60)
        self.collision_indicator.setStyleSheet("""
            background-color: #1e4d2b;
            color: #4ade80;
            border: 2px solid #22c55e;
            border-radius: 5px;
            font-size: 16pt;
            font-weight: bold;
        """)
        layout.addWidget(self.collision_indicator)

        # Details
        self.collision_details = QLabel("")
        self.collision_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.collision_details.setWordWrap(True)
        layout.addWidget(self.collision_details)

        return frame

    def _create_lane_section(self) -> QFrame:
        """Create lane departure warning section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Lane Departure Warning")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        # Lane status indicator
        self.lane_indicator = QLabel("CENTERED")
        self.lane_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lane_indicator.setMinimumHeight(50)
        self.lane_indicator.setStyleSheet("""
            background-color: #1e4d2b;
            color: #4ade80;
            border: 2px solid #22c55e;
            border-radius: 5px;
            font-size: 14pt;
            font-weight: bold;
        """)
        layout.addWidget(self.lane_indicator)

        # Details
        self.lane_details = QLabel("")
        self.lane_details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lane_details)

        return frame

    @pyqtSlot(object, name="updateBlindSpotWarning")
    def update_blind_spot(self, warning: BlindSpotWarning):
        """Update blind spot indicators."""
        if warning is None:
            return

        # Update left indicator
        if warning.left_blind_spot:
            self.left_blind_spot.setStyleSheet("""
                background-color: #991b1b;
                color: #fca5a5;
                border: 2px solid #dc2626;
                border-radius: 5px;
                font-size: 14pt;
                font-weight: bold;
            """)
        else:
            self.left_blind_spot.setStyleSheet("""
                background-color: #2e2e2e;
                color: #888;
                border: 2px solid #444;
                border-radius: 5px;
                font-size: 14pt;
                font-weight: bold;
            """)

        # Update right indicator
        if warning.right_blind_spot:
            self.right_blind_spot.setStyleSheet("""
                background-color: #991b1b;
                color: #fca5a5;
                border: 2px solid #dc2626;
                border-radius: 5px;
                font-size: 14pt;
                font-weight: bold;
            """)
        else:
            self.right_blind_spot.setStyleSheet("""
                background-color: #2e2e2e;
                color: #888;
                border: 2px solid #444;
                border-radius: 5px;
                font-size: 14pt;
                font-weight: bold;
            """)

        # Update status
        if warning.warning_active:
            status = f"⚠️ WARNING: {warning.warning_side.upper()}"
            if not self.blink_timer.isActive():
                self.blink_timer.start()
        else:
            if warning.left_blind_spot and warning.right_blind_spot:
                status = "Vehicles in both blind spots"
            elif warning.left_blind_spot:
                status = "Vehicle in left blind spot"
            elif warning.right_blind_spot:
                status = "Vehicle in right blind spot"
            else:
                status = "Clear"
                if self.blink_timer.isActive():
                    self.blink_timer.stop()

        self.blind_spot_status.setText(status)

    @pyqtSlot(object, name="updateCollisionWarning")
    def update_collision_warning(self, warning: CollisionWarning):
        """Update collision warning display."""
        if warning is None:
            return

        level = warning.warning_level.upper()

        # Update indicator based on warning level
        if level == "NONE":
            self.collision_indicator.setText("CLEAR")
            self.collision_indicator.setStyleSheet("""
                background-color: #1e4d2b;
                color: #4ade80;
                border: 2px solid #22c55e;
                border-radius: 5px;
                font-size: 16pt;
                font-weight: bold;
            """)
            self.collision_details.setText("")
            if self.blink_timer.isActive():
                self.blink_timer.stop()

        elif level == "CAUTION":
            self.collision_indicator.setText("⚠️ CAUTION")
            self.collision_indicator.setStyleSheet("""
                background-color: #713f12;
                color: #fbbf24;
                border: 2px solid #f59e0b;
                border-radius: 5px;
                font-size: 16pt;
                font-weight: bold;
            """)
            if warning.time_to_collision:
                self.collision_details.setText(f"TTC: {warning.time_to_collision:.1f}s")

        elif level == "WARNING":
            self.collision_indicator.setText("⚠️ WARNING")
            self.collision_indicator.setStyleSheet("""
                background-color: #9a3412;
                color: #fb923c;
                border: 2px solid #ea580c;
                border-radius: 5px;
                font-size: 16pt;
                font-weight: bold;
            """)
            if warning.time_to_collision:
                self.collision_details.setText(
                    f"TTC: {warning.time_to_collision:.1f}s\n"
                    f"Brake: {warning.recommended_deceleration:.1f} m/s²"
                )

        elif level == "CRITICAL":
            self.collision_indicator.setText("🚨 CRITICAL")
            self.collision_indicator.setStyleSheet("""
                background-color: #7f1d1d;
                color: #fca5a5;
                border: 2px solid #dc2626;
                border-radius: 5px;
                font-size: 16pt;
                font-weight: bold;
            """)
            if warning.time_to_collision:
                self.collision_details.setText(
                    f"TTC: {warning.time_to_collision:.1f}s\n"
                    f"BRAKE NOW: {warning.recommended_deceleration:.1f} m/s²"
                )
            if not self.blink_timer.isActive():
                self.blink_timer.start()

    @pyqtSlot(object, name="updateLaneState")
    def update_lane_state(self, lane_state: LaneState):
        """Update lane departure warning display."""
        if lane_state is None:
            return

        if lane_state.departure_warning:
            side = lane_state.departure_side.upper()
            self.lane_indicator.setText(f"⚠️ DEPARTING {side}")
            self.lane_indicator.setStyleSheet("""
                background-color: #9a3412;
                color: #fb923c;
                border: 2px solid #ea580c;
                border-radius: 5px;
                font-size: 14pt;
                font-weight: bold;
            """)

            ttc_text = f"TTC: {lane_state.time_to_lane_crossing:.1f}s" if lane_state.time_to_lane_crossing else ""
            offset_text = f"Offset: {lane_state.lateral_offset:.2f}m"
            self.lane_details.setText(f"{ttc_text}\n{offset_text}")

            if not self.blink_timer.isActive():
                self.blink_timer.start()
        else:
            self.lane_indicator.setText("CENTERED")
            self.lane_indicator.setStyleSheet("""
                background-color: #1e4d2b;
                color: #4ade80;
                border: 2px solid #22c55e;
                border-radius: 5px;
                font-size: 14pt;
                font-weight: bold;
            """)

            if abs(lane_state.lateral_offset) > 0.1:
                self.lane_details.setText(f"Offset: {lane_state.lateral_offset:.2f}m")
            else:
                self.lane_details.setText("")

            if self.blink_timer.isActive():
                self.blink_timer.stop()

    def _toggle_blink(self):
        """Toggle blink state for critical warnings."""
        self.blink_state = not self.blink_state
        # Blinking effect is handled by timer - could add opacity changes here
