"""GPS information display widget."""

import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont


class GPSWidget(QWidget):
    """
    Widget displaying GPS and location information:
    - Current coordinates (lat/lon)
    - Altitude
    - GPS speed
    - Heading
    - Satellite count and fix quality
    - Speed limit for current location
    - Speed violation warnings
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
        title = QLabel("🌐 GPS & Location")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # GPS Status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        status_layout = QVBoxLayout(status_frame)

        status_title = QLabel("GPS Status")
        status_title.setStyleSheet("font-weight: bold;")
        status_layout.addWidget(status_title)

        self.gps_status_label = QLabel("No GPS data")
        self.gps_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.gps_status_label)

        layout.addWidget(status_frame)

        # Position
        position_frame = QFrame()
        position_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        position_layout = QGridLayout(position_frame)

        position_layout.addWidget(QLabel("Latitude:"), 0, 0)
        self.latitude_label = QLabel("--")
        self.latitude_label.setStyleSheet("font-weight: bold;")
        position_layout.addWidget(self.latitude_label, 0, 1)

        position_layout.addWidget(QLabel("Longitude:"), 1, 0)
        self.longitude_label = QLabel("--")
        self.longitude_label.setStyleSheet("font-weight: bold;")
        position_layout.addWidget(self.longitude_label, 1, 1)

        position_layout.addWidget(QLabel("Altitude:"), 2, 0)
        self.altitude_label = QLabel("--")
        self.altitude_label.setStyleSheet("font-weight: bold;")
        position_layout.addWidget(self.altitude_label, 2, 1)

        layout.addWidget(position_frame)

        # Motion
        motion_frame = QFrame()
        motion_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        motion_layout = QGridLayout(motion_frame)

        motion_layout.addWidget(QLabel("GPS Speed:"), 0, 0)
        self.gps_speed_label = QLabel("--")
        self.gps_speed_label.setStyleSheet("font-weight: bold;")
        motion_layout.addWidget(self.gps_speed_label, 0, 1)

        motion_layout.addWidget(QLabel("Heading:"), 1, 0)
        self.heading_label = QLabel("--")
        self.heading_label.setStyleSheet("font-weight: bold;")
        motion_layout.addWidget(self.heading_label, 1, 1)

        layout.addWidget(motion_frame)

        # GPS Quality
        quality_frame = QFrame()
        quality_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        quality_layout = QGridLayout(quality_frame)

        quality_layout.addWidget(QLabel("Satellites:"), 0, 0)
        self.satellites_label = QLabel("--")
        self.satellites_label.setStyleSheet("font-weight: bold;")
        quality_layout.addWidget(self.satellites_label, 0, 1)

        quality_layout.addWidget(QLabel("Fix Quality:"), 1, 0)
        self.fix_quality_label = QLabel("--")
        self.fix_quality_label.setStyleSheet("font-weight: bold;")
        quality_layout.addWidget(self.fix_quality_label, 1, 1)

        quality_layout.addWidget(QLabel("HDOP:"), 2, 0)
        self.hdop_label = QLabel("--")
        self.hdop_label.setStyleSheet("font-weight: bold;")
        quality_layout.addWidget(self.hdop_label, 2, 1)

        layout.addWidget(quality_frame)

        # Speed Limit
        speed_limit_frame = QFrame()
        speed_limit_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        speed_limit_layout = QVBoxLayout(speed_limit_frame)

        speed_limit_title = QLabel("Speed Limit")
        speed_limit_title.setStyleSheet("font-weight: bold;")
        speed_limit_layout.addWidget(speed_limit_title)

        self.speed_limit_label = QLabel("--")
        self.speed_limit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_limit_font = QFont()
        speed_limit_font.setPointSize(24)
        speed_limit_font.setBold(True)
        self.speed_limit_label.setFont(speed_limit_font)
        speed_limit_layout.addWidget(self.speed_limit_label)

        self.road_info_label = QLabel("")
        self.road_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.road_info_label.setWordWrap(True)
        speed_limit_layout.addWidget(self.road_info_label)

        layout.addWidget(speed_limit_frame)

        # Speed Violation Warning
        self.violation_frame = QFrame()
        self.violation_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        violation_layout = QVBoxLayout(self.violation_frame)

        self.violation_label = QLabel("")
        self.violation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.violation_label.setWordWrap(True)
        violation_font = QFont()
        violation_font.setPointSize(14)
        violation_font.setBold(True)
        self.violation_label.setFont(violation_font)
        violation_layout.addWidget(self.violation_label)

        self.violation_frame.hide()  # Hidden by default
        layout.addWidget(self.violation_frame)

        layout.addStretch()

    @pyqtSlot(object, name="updateGPSData")
    def update_gps_data(self, location_info: dict):
        """
        Update GPS display.

        Args:
            location_info: Dictionary with GPS and location data
        """
        if location_info is None:
            self.gps_status_label.setText("No GPS data")
            self.gps_status_label.setStyleSheet("color: #888;")
            return

        # Update status
        fix_quality = location_info.get('fix_quality', 0)
        satellites = location_info.get('satellites', 0)

        if fix_quality == 0:
            status = "❌ No Fix"
            status_color = "#f87171"
        elif fix_quality == 1:
            status = f"✅ GPS Fix ({satellites} sats)"
            status_color = "#4ade80"
        elif fix_quality == 2:
            status = f"✅ DGPS Fix ({satellites} sats)"
            status_color = "#4ade80"
        else:
            status = f"GPS ({satellites} sats)"
            status_color = "#fbbf24"

        self.gps_status_label.setText(status)
        self.gps_status_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

        # Update position
        lat = location_info.get('latitude', 0.0)
        lon = location_info.get('longitude', 0.0)
        alt = location_info.get('altitude', 0.0)

        self.latitude_label.setText(f"{lat:.6f}°")
        self.longitude_label.setText(f"{lon:.6f}°")
        self.altitude_label.setText(f"{alt:.1f} m")

        # Update motion
        gps_speed = location_info.get('speed_gps', 0.0)
        heading = location_info.get('heading', 0.0)

        self.gps_speed_label.setText(f"{gps_speed:.1f} km/h")
        self.heading_label.setText(f"{heading:.1f}° {self._heading_to_cardinal(heading)}")

        # Update quality metrics
        self.satellites_label.setText(str(satellites))

        fix_text = ["No Fix", "GPS", "DGPS"][min(fix_quality, 2)]
        self.fix_quality_label.setText(fix_text)

        hdop = location_info.get('hdop', 99.9)
        self.hdop_label.setText(f"{hdop:.1f}")

        # Color code HDOP (lower is better)
        if hdop < 2.0:
            hdop_color = "#4ade80"  # Excellent
        elif hdop < 5.0:
            hdop_color = "#fbbf24"  # Good
        else:
            hdop_color = "#f87171"  # Poor

        self.hdop_label.setStyleSheet(f"font-weight: bold; color: {hdop_color};")

        # Update speed limit
        speed_limit = location_info.get('speed_limit')
        if speed_limit is not None:
            self.speed_limit_label.setText(f"{int(speed_limit)}")
            self.speed_limit_label.setStyleSheet("color: #fbbf24;")
        else:
            self.speed_limit_label.setText("--")
            self.speed_limit_label.setStyleSheet("color: #888;")

        # Update road info
        road_name = location_info.get('road_name')
        road_type = location_info.get('road_type')

        if road_name or road_type:
            road_info = []
            if road_name:
                road_info.append(road_name)
            if road_type:
                road_info.append(f"({road_type})")
            self.road_info_label.setText(" ".join(road_info))
        else:
            self.road_info_label.setText("")

    @pyqtSlot(object, name="updateSpeedViolation")
    def update_speed_violation(self, violation: dict):
        """
        Update speed violation warning.

        Args:
            violation: Speed violation info or None
        """
        if violation is None:
            self.violation_frame.hide()
            return

        # Show violation warning
        self.violation_frame.show()

        current_speed = violation.get('current_speed', 0)
        speed_limit = violation.get('speed_limit', 0)
        excess = violation.get('excess', 0)
        severity = violation.get('severity', 'low')

        warning_text = f"⚠️ SPEEDING\n{current_speed:.0f} km/h in {speed_limit:.0f} km/h zone\n(+{excess:.0f} km/h)"

        self.violation_label.setText(warning_text)

        # Color code by severity
        if severity == 'critical':
            bg_color = "#7f1d1d"
            text_color = "#fca5a5"
            border_color = "#dc2626"
        elif severity == 'high':
            bg_color = "#9a3412"
            text_color = "#fb923c"
            border_color = "#ea580c"
        elif severity == 'medium':
            bg_color = "#713f12"
            text_color = "#fbbf24"
            border_color = "#f59e0b"
        else:
            bg_color = "#365314"
            text_color = "#a3e635"
            border_color = "#84cc16"

        self.violation_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 5px;
            }}
        """)
        self.violation_label.setStyleSheet(f"color: {text_color};")

    def _heading_to_cardinal(self, heading: float) -> str:
        """Convert heading in degrees to cardinal direction."""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((heading + 22.5) / 45.0) % 8
        return directions[index]
