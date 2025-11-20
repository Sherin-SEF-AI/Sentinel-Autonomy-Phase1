"""
Gaze Direction Widget

Custom widget for visualizing driver gaze direction with 3D head model.
"""

import logging
import math
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath, QPolygonF

logger = logging.getLogger(__name__)


class GazeDirectionWidget(QWidget):
    """
    Gaze direction visualization widget.
    
    Features:
    - 3D head visualization with pitch and yaw
    - Gaze vector as arrow
    - Attention zone highlighting
    - Color-coded zones
    """
    
    # Attention zones (8 zones around vehicle)
    ZONES = {
        'front': (0, -30, 30),           # zone_id, yaw_min, yaw_max
        'front_left': (1, 30, 75),
        'left': (2, 75, 105),
        'rear_left': (3, 105, 150),
        'rear': (4, 150, -150),
        'rear_right': (5, -150, -105),
        'right': (6, -105, -75),
        'front_right': (7, -75, -30)
    }
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Gaze state
        self._pitch = 0.0  # degrees, positive = looking up
        self._yaw = 0.0    # degrees, positive = looking left
        self._attention_zone = 'front'
        self._face_detected = False
        
        # Widget settings
        self.setMinimumSize(200, 200)
        
        logger.debug("GazeDirectionWidget created")
    
    def set_gaze(self, pitch: float, yaw: float, attention_zone: str = None):
        """
        Set gaze direction.
        
        Args:
            pitch: Pitch angle in degrees (positive = up)
            yaw: Yaw angle in degrees (positive = left)
            attention_zone: Name of attention zone (optional, will be calculated if not provided)
        """
        self._pitch = pitch
        self._yaw = yaw
        self._face_detected = True
        
        # Calculate attention zone if not provided
        if attention_zone:
            self._attention_zone = attention_zone
        else:
            self._attention_zone = self._calculate_attention_zone(yaw)
        
        self.update()
    
    def clear_gaze(self):
        """Clear gaze data (no face detected)"""
        self._face_detected = False
        self._pitch = 0.0
        self._yaw = 0.0
        self._attention_zone = 'front'
        self.update()
    
    def _calculate_attention_zone(self, yaw: float) -> str:
        """Calculate attention zone from yaw angle"""
        # Normalize yaw to -180 to 180
        yaw = ((yaw + 180) % 360) - 180
        
        for zone_name, (zone_id, yaw_min, yaw_max) in self.ZONES.items():
            if zone_name == 'rear':
                # Special case for rear (wraps around)
                if yaw >= yaw_min or yaw <= yaw_max:
                    return zone_name
            else:
                if yaw_min <= yaw <= yaw_max:
                    return zone_name
        
        return 'front'  # Default
    
    def paintEvent(self, event):
        """Paint the gaze visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        size = min(width, height)
        
        # Center
        center_x = width / 2
        center_y = height / 2
        
        if not self._face_detected:
            # Draw "No Face Detected" message
            self._draw_no_face_message(painter, center_x, center_y)
            return
        
        # Draw attention zones (background)
        self._draw_attention_zones(painter, center_x, center_y, size)
        
        # Draw head outline
        self._draw_head_outline(painter, center_x, center_y, size)
        
        # Draw pitch and yaw indicators
        self._draw_pitch_indicator(painter, center_x, center_y, size)
        self._draw_yaw_indicator(painter, center_x, center_y, size)
        
        # Draw gaze vector
        self._draw_gaze_vector(painter, center_x, center_y, size)
        
        # Draw zone label
        self._draw_zone_label(painter, center_x, center_y, size)
    
    def _draw_no_face_message(self, painter: QPainter, center_x: float, center_y: float):
        """Draw no face detected message"""
        font = QFont()
        font.setPointSize(12)
        painter.setFont(font)
        painter.setPen(QPen(QColor(150, 150, 150)))
        
        text_rect = QRectF(center_x - 100, center_y - 20, 200, 40)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, "No Face Detected")
    
    def _draw_attention_zones(self, painter: QPainter, center_x: float, center_y: float, size: float):
        """Draw attention zones as sectors"""
        radius = size * 0.45
        
        # Draw 8 zones as pie slices
        for zone_name, (zone_id, yaw_min, yaw_max) in self.ZONES.items():
            # Calculate angles for Qt (0 = 3 o'clock, positive = counter-clockwise)
            # Convert from our yaw system (0 = front, positive = left)
            
            if zone_name == 'rear':
                # Special handling for rear zone
                qt_start_angle = 90 - yaw_min
                qt_span_angle = 60  # 30 degrees on each side
            else:
                qt_start_angle = 90 - yaw_max  # Qt angles
                qt_span_angle = yaw_max - yaw_min
            
            # Highlight current zone
            if zone_name == self._attention_zone:
                color = QColor(76, 175, 80, 100)  # Green, semi-transparent
            else:
                color = QColor(60, 60, 60, 50)  # Gray, very transparent
            
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setBrush(QBrush(color))
            
            # Draw pie slice
            rect = QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
            painter.drawPie(rect, int(qt_start_angle * 16), int(qt_span_angle * 16))
    
    def _draw_head_outline(self, painter: QPainter, center_x: float, center_y: float, size: float):
        """Draw head outline (simplified 3D representation)"""
        head_radius = size * 0.15
        
        # Apply yaw rotation to head position
        yaw_rad = math.radians(self._yaw)
        
        # Calculate head center with yaw offset
        head_x = center_x + head_radius * 0.3 * math.sin(yaw_rad)
        
        # Apply pitch to vertical position
        pitch_rad = math.radians(self._pitch)
        head_y = center_y - head_radius * 0.3 * math.sin(pitch_rad)
        
        # Draw head circle
        painter.setPen(QPen(QColor(200, 200, 200), 3))
        painter.setBrush(QBrush(QColor(80, 80, 80)))
        painter.drawEllipse(
            QPointF(head_x, head_y),
            head_radius,
            head_radius
        )
        
        # Draw facial features for orientation
        self._draw_facial_features(painter, head_x, head_y, head_radius, yaw_rad, pitch_rad)
    
    def _draw_facial_features(
        self,
        painter: QPainter,
        head_x: float,
        head_y: float,
        head_radius: float,
        yaw_rad: float,
        pitch_rad: float
    ):
        """Draw simplified facial features"""
        # Eyes
        eye_y_offset = -head_radius * 0.2 + head_radius * 0.3 * math.sin(pitch_rad)
        eye_x_offset = head_radius * 0.3
        
        # Left eye (from viewer perspective)
        left_eye_x = head_x - eye_x_offset * math.cos(yaw_rad)
        left_eye_y = head_y + eye_y_offset
        
        # Right eye
        right_eye_x = head_x + eye_x_offset * math.cos(yaw_rad)
        right_eye_y = head_y + eye_y_offset
        
        eye_radius = head_radius * 0.1
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        
        # Only draw eyes if they're visible (not turned too far)
        if abs(self._yaw) < 80:
            if self._yaw < 60:  # Right eye visible
                painter.drawEllipse(QPointF(right_eye_x, right_eye_y), eye_radius, eye_radius)
            if self._yaw > -60:  # Left eye visible
                painter.drawEllipse(QPointF(left_eye_x, left_eye_y), eye_radius, eye_radius)
        
        # Nose (simple line)
        nose_length = head_radius * 0.4
        nose_x = head_x + nose_length * math.sin(yaw_rad)
        nose_y = head_y + nose_length * 0.3 * math.sin(pitch_rad)
        
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.drawLine(QPointF(head_x, head_y), QPointF(nose_x, nose_y))
    
    def _draw_pitch_indicator(self, painter: QPainter, center_x: float, center_y: float, size: float):
        """Draw pitch angle indicator"""
        # Draw pitch arc on the side
        indicator_x = center_x + size * 0.35
        indicator_y = center_y
        arc_radius = size * 0.08
        
        # Background arc
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(indicator_x - arc_radius, indicator_y - arc_radius, arc_radius * 2, arc_radius * 2)
        painter.drawArc(rect, 0, 180 * 16)  # Semi-circle
        
        # Pitch indicator line
        pitch_angle = -self._pitch  # Invert for display
        pitch_rad = math.radians(pitch_angle)
        
        line_length = arc_radius * 1.2
        line_x = indicator_x + line_length * math.cos(pitch_rad)
        line_y = indicator_y + line_length * math.sin(pitch_rad)
        
        painter.setPen(QPen(QColor(76, 175, 80), 3))
        painter.drawLine(QPointF(indicator_x, indicator_y), QPointF(line_x, line_y))
        
        # Label
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(200, 200, 200)))
        text_rect = QRectF(indicator_x - 30, indicator_y + arc_radius + 5, 60, 20)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, f"Pitch: {self._pitch:.0f}°")
    
    def _draw_yaw_indicator(self, painter: QPainter, center_x: float, center_y: float, size: float):
        """Draw yaw angle indicator"""
        # Draw yaw arc at the bottom
        indicator_x = center_x
        indicator_y = center_y + size * 0.35
        arc_radius = size * 0.08
        
        # Background arc
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rect = QRectF(indicator_x - arc_radius, indicator_y - arc_radius, arc_radius * 2, arc_radius * 2)
        painter.drawArc(rect, -45 * 16, 90 * 16)  # 90 degree arc
        
        # Yaw indicator line
        yaw_angle = 90 - self._yaw  # Convert to Qt angle system
        yaw_rad = math.radians(yaw_angle)
        
        line_length = arc_radius * 1.2
        line_x = indicator_x + line_length * math.cos(yaw_rad)
        line_y = indicator_y + line_length * math.sin(yaw_rad)
        
        painter.setPen(QPen(QColor(33, 150, 243), 3))
        painter.drawLine(QPointF(indicator_x, indicator_y), QPointF(line_x, line_y))
        
        # Label
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(200, 200, 200)))
        text_rect = QRectF(indicator_x - 30, indicator_y + arc_radius + 5, 60, 20)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, f"Yaw: {self._yaw:.0f}°")
    
    def _draw_gaze_vector(self, painter: QPainter, center_x: float, center_y: float, size: float):
        """Draw gaze vector as arrow"""
        # Calculate gaze direction
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)
        
        # Start from head center
        head_radius = size * 0.15
        start_x = center_x + head_radius * 0.3 * math.sin(yaw_rad)
        start_y = center_y - head_radius * 0.3 * math.sin(pitch_rad)
        
        # Gaze vector length
        gaze_length = size * 0.25
        
        # Calculate end point (simplified 2D projection)
        end_x = start_x + gaze_length * math.sin(yaw_rad)
        end_y = start_y - gaze_length * math.sin(pitch_rad)
        
        # Draw arrow
        painter.setPen(QPen(QColor(255, 193, 7), 3))
        painter.drawLine(QPointF(start_x, start_y), QPointF(end_x, end_y))
        
        # Draw arrowhead
        arrow_size = size * 0.03
        arrow_angle = math.atan2(end_y - start_y, end_x - start_x)
        
        arrow_point1_x = end_x - arrow_size * math.cos(arrow_angle - math.pi / 6)
        arrow_point1_y = end_y - arrow_size * math.sin(arrow_angle - math.pi / 6)
        arrow_point2_x = end_x - arrow_size * math.cos(arrow_angle + math.pi / 6)
        arrow_point2_y = end_y - arrow_size * math.sin(arrow_angle + math.pi / 6)
        
        arrow_polygon = QPolygonF([
            QPointF(end_x, end_y),
            QPointF(arrow_point1_x, arrow_point1_y),
            QPointF(arrow_point2_x, arrow_point2_y)
        ])
        
        painter.setBrush(QBrush(QColor(255, 193, 7)))
        painter.drawPolygon(arrow_polygon)
    
    def _draw_zone_label(self, painter: QPainter, center_x: float, center_y: float, size: float):
        """Draw current attention zone label"""
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        
        # Format zone name
        zone_display = self._attention_zone.replace('_', ' ').title()
        
        painter.setPen(QPen(QColor(76, 175, 80)))
        text_rect = QRectF(center_x - 80, center_y - size * 0.45, 160, 30)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, f"Zone: {zone_display}")
