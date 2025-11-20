"""
Vehicle Telemetry Dock Widget

Displays real-time vehicle telemetry from CAN bus.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush

from core.data_structures import VehicleTelemetry
from .circular_gauge import CircularGaugeWidget

logger = logging.getLogger(__name__)


class SteeringIndicator(QWidget):
    """Visual indicator for steering angle."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.SteeringIndicator")
        self.steering_angle = 0.0  # radians
        self.setMinimumSize(200, 100)
        self.logger.debug("SteeringIndicator initialized")
    
    def set_steering_angle(self, angle: float):
        """
        Set steering angle.
        
        Args:
            angle: Steering angle in radians (positive = left)
        """
        self.logger.debug(f"Steering angle updated: {angle:.3f} rad ({angle * 57.2958:.1f}°)")
        self.steering_angle = angle
        self.update()
    
    def paintEvent(self, event):
        """Paint steering indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        
        # Draw steering wheel
        wheel_radius = min(width, height) // 3
        painter.setPen(QPen(QColor(200, 200, 200), 3))
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawEllipse(center_x - wheel_radius, center_y - wheel_radius,
                          wheel_radius * 2, wheel_radius * 2)
        
        # Draw center mark
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(center_x, center_y - 10, center_x, center_y + 10)
        
        # Draw steering angle indicator
        import math
        angle_deg = math.degrees(self.steering_angle)
        
        # Rotate painter
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(-angle_deg)  # Negative for correct direction
        
        # Draw indicator line
        painter.setPen(QPen(QColor(0, 170, 255), 4))
        painter.drawLine(0, 0, 0, -wheel_radius + 10)
        
        painter.restore()
        
        # Draw angle text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont('Arial', 12, QFont.Weight.Bold))
        angle_text = f"{angle_deg:.1f}°"
        painter.drawText(0, height - 20, width, 20, Qt.AlignmentFlag.AlignCenter, angle_text)


class BarIndicator(QWidget):
    """Horizontal bar indicator for brake/throttle."""
    
    def __init__(self, label: str, color: QColor, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.BarIndicator")
        self.label = label
        self.color = color
        self.value = 0.0  # 0-1 range
        self.setMinimumHeight(40)
        self.logger.debug(f"BarIndicator initialized: label={label}")
    
    def set_value(self, value: float):
        """
        Set indicator value.
        
        Args:
            value: Value in 0-1 range
        """
        clamped_value = max(0.0, min(1.0, value))
        if abs(clamped_value - value) > 0.01:
            self.logger.debug(f"{self.label} value clamped: {value:.3f} -> {clamped_value:.3f}")
        self.value = clamped_value
        self.update()
    
    def paintEvent(self, event):
        """Paint bar indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw label
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont('Arial', 10))
        painter.drawText(0, 0, 80, height, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.label)
        
        # Draw background bar
        bar_x = 90
        bar_width = width - bar_x - 60
        bar_height = 20
        bar_y = (height - bar_height) // 2
        
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.drawRect(bar_x, bar_y, bar_width, bar_height)
        
        # Draw value bar
        value_width = int(bar_width * self.value)
        painter.setBrush(QBrush(self.color))
        painter.drawRect(bar_x, bar_y, value_width, bar_height)
        
        # Draw value text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(QFont('Arial', 10, QFont.Weight.Bold))
        value_text = f"{self.value * 100:.0f}%"
        painter.drawText(width - 55, 0, 50, height, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, value_text)


class GearIndicator(QWidget):
    """Gear position indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.GearIndicator")
        self.gear = 0
        self.setMinimumSize(80, 80)
        self.logger.debug("GearIndicator initialized")
    
    def set_gear(self, gear: int):
        """
        Set gear position.
        
        Args:
            gear: Gear number (0=N, -1=R, 1-6=forward gears)
        """
        if gear != self.gear:
            gear_str = "N" if gear == 0 else ("R" if gear == -1 else str(gear))
            self.logger.debug(f"Gear changed: {self.gear} -> {gear} ({gear_str})")
        self.gear = gear
        self.update()
    
    def paintEvent(self, event):
        """Paint gear indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.drawRoundedRect(10, 10, width - 20, height - 20, 10, 10)
        
        # Draw gear text
        if self.gear == 0:
            gear_text = "N"
            color = QColor(200, 200, 200)
        elif self.gear == -1:
            gear_text = "R"
            color = QColor(255, 100, 100)
        else:
            gear_text = str(self.gear)
            color = QColor(100, 255, 100)
        
        painter.setPen(QPen(color))
        painter.setFont(QFont('Arial', 32, QFont.Weight.Bold))
        painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter, gear_text)


class TurnSignalIndicator(QWidget):
    """Turn signal indicator."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.TurnSignalIndicator")
        self.turn_signal = 'none'
        self.setMinimumSize(200, 60)
        self.logger.debug("TurnSignalIndicator initialized")
    
    def set_turn_signal(self, signal: str):
        """
        Set turn signal state.
        
        Args:
            signal: 'left', 'right', or 'none'
        """
        if signal != self.turn_signal:
            self.logger.debug(f"Turn signal changed: {self.turn_signal} -> {signal}")
        self.turn_signal = signal
        self.update()
    
    def paintEvent(self, event):
        """Paint turn signal indicator."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw left arrow
        left_color = QColor(255, 200, 0) if self.turn_signal == 'left' else QColor(80, 80, 80)
        painter.setBrush(QBrush(left_color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Left arrow triangle
        from PyQt6.QtCore import QPointF
        left_arrow = [
            QPointF(40, height // 2),
            QPointF(60, height // 2 - 15),
            QPointF(60, height // 2 + 15)
        ]
        painter.drawPolygon(left_arrow)
        
        # Right arrow
        right_color = QColor(255, 200, 0) if self.turn_signal == 'right' else QColor(80, 80, 80)
        painter.setBrush(QBrush(right_color))
        
        right_arrow = [
            QPointF(width - 40, height // 2),
            QPointF(width - 60, height // 2 - 15),
            QPointF(width - 60, height // 2 + 15)
        ]
        painter.drawPolygon(right_arrow)
        
        # Draw center text
        painter.setPen(QPen(QColor(200, 200, 200)))
        painter.setFont(QFont('Arial', 10))
        painter.drawText(0, 0, width, height, Qt.AlignmentFlag.AlignCenter, "TURN SIGNAL")


class VehicleTelemetryDock(QWidget):
    """
    Dock widget for displaying vehicle telemetry from CAN bus.
    
    Shows speedometer, steering angle, brake/throttle, gear, and turn signals.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.setup_ui()
        self.logger.info("Vehicle Telemetry Dock initialized")
    
    def setup_ui(self):
        """Set up the user interface."""
        self.logger.debug("Setting up Vehicle Telemetry Dock UI")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Vehicle Telemetry")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00aaff;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Speedometer (circular gauge)
        speed_frame = QFrame()
        speed_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        speed_layout = QVBoxLayout(speed_frame)
        
        speed_label = QLabel("Speed")
        speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        speed_layout.addWidget(speed_label)
        
        self.speedometer = CircularGaugeWidget(
            min_value=0,
            max_value=50,  # 50 m/s = 180 km/h
            unit="m/s",
            zones=[(0, 15, QColor(100, 255, 100)),
                   (15, 30, QColor(255, 200, 0)),
                   (30, 50, QColor(255, 100, 100))]
        )
        self.speedometer.setMinimumSize(200, 200)
        speed_layout.addWidget(self.speedometer)
        
        layout.addWidget(speed_frame)
        
        # Steering angle indicator
        steering_frame = QFrame()
        steering_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        steering_layout = QVBoxLayout(steering_frame)
        
        steering_label = QLabel("Steering Angle")
        steering_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        steering_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        steering_layout.addWidget(steering_label)
        
        self.steering_indicator = SteeringIndicator()
        steering_layout.addWidget(self.steering_indicator)
        
        layout.addWidget(steering_frame)
        
        # Brake and throttle bars
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_frame)
        
        self.brake_bar = BarIndicator("Brake", QColor(255, 100, 100))
        controls_layout.addWidget(self.brake_bar)
        
        self.throttle_bar = BarIndicator("Throttle", QColor(100, 255, 100))
        controls_layout.addWidget(self.throttle_bar)
        
        layout.addWidget(controls_frame)
        
        # Gear and turn signal
        status_layout = QHBoxLayout()
        
        # Gear indicator
        gear_frame = QFrame()
        gear_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        gear_layout = QVBoxLayout(gear_frame)
        
        gear_label = QLabel("Gear")
        gear_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gear_label.setStyleSheet("font-size: 12px; font-weight: bold;")
        gear_layout.addWidget(gear_label)
        
        self.gear_indicator = GearIndicator()
        gear_layout.addWidget(self.gear_indicator)
        
        status_layout.addWidget(gear_frame)
        
        # Turn signal indicator
        signal_frame = QFrame()
        signal_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        signal_layout = QVBoxLayout(signal_frame)
        
        self.turn_signal_indicator = TurnSignalIndicator()
        signal_layout.addWidget(self.turn_signal_indicator)
        
        status_layout.addWidget(signal_frame)
        
        layout.addLayout(status_layout)
        
        layout.addStretch()
        
        self.logger.debug("Vehicle Telemetry Dock UI setup completed")
    
    @pyqtSlot(object)
    def update_telemetry(self, telemetry: VehicleTelemetry):
        """
        Update telemetry display.
        
        Args:
            telemetry: VehicleTelemetry dataclass with current values
        """
        try:
            self.logger.debug(
                f"Telemetry update received: speed={telemetry.speed:.2f}m/s, "
                f"steering={telemetry.steering_angle:.3f}rad, "
                f"brake={telemetry.brake_pressure:.2f}bar, "
                f"throttle={telemetry.throttle_position:.2f}, "
                f"gear={telemetry.gear}, signal={telemetry.turn_signal}"
            )
            
            # Update speedometer
            self.speedometer.set_value(telemetry.speed)
            
            # Update steering
            self.steering_indicator.set_steering_angle(telemetry.steering_angle)
            
            # Update brake (normalize pressure to 0-1, assuming max 10 bar)
            brake_normalized = min(1.0, telemetry.brake_pressure / 10.0)
            self.brake_bar.set_value(brake_normalized)
            
            # Update throttle
            self.throttle_bar.set_value(telemetry.throttle_position)
            
            # Update gear
            self.gear_indicator.set_gear(telemetry.gear)
            
            # Update turn signal
            self.turn_signal_indicator.set_turn_signal(telemetry.turn_signal)
            
            self.logger.debug("Telemetry display updated successfully")
            
        except Exception as e:
            self.logger.error(f"Telemetry display update failed: {e}", exc_info=True)
