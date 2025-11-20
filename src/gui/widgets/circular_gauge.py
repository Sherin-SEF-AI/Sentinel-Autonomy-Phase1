"""
Circular Gauge Widget

Custom widget for displaying values as circular gauges with color zones.
"""

import logging
import time
from typing import Optional, Tuple
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF, QPointF, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QConicalGradient, QPainterPath

logger = logging.getLogger(__name__)


class CircularGaugeWidget(QWidget):
    """
    Circular gauge widget with custom painting.
    
    Features:
    - Background arc with gradient
    - Value arc with color zones (green/yellow/red)
    - Needle pointer
    - Center value text
    - Smooth value animations
    """
    
    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 100.0,
        value: float = 0.0,
        title: str = "",
        unit: str = "",
        green_zone: Tuple[float, float] = (70.0, 100.0),
        yellow_zone: Tuple[float, float] = (50.0, 70.0),
        red_zone: Tuple[float, float] = (0.0, 50.0),
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._min_value = min_value
        self._max_value = max_value
        self._value = value
        self._animated_value = value  # For smooth animations
        self._title = title
        self._unit = unit
        
        # Color zones (min, max)
        self._green_zone = green_zone
        self._yellow_zone = yellow_zone
        self._red_zone = red_zone
        
        # Gauge parameters
        self._start_angle = 135  # Start at bottom-left (degrees)
        self._span_angle = 270   # 270 degrees arc
        
        # Animation
        self._animation = QPropertyAnimation(self, b"animatedValue")
        self._animation.setDuration(500)  # 500ms animation
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        
        # Widget settings
        self.setMinimumSize(150, 150)
        
        # Performance tracking
        self._last_log_time = 0.0
        self._log_throttle_interval = 1.0  # Log value changes at most once per second
        self._paint_count = 0
        self._last_paint_time = 0.0
        
        logger.info(
            f"CircularGaugeWidget initialized: title='{title}', "
            f"range=[{min_value}, {max_value}], value={value}, unit='{unit}'"
        )
    
    @pyqtProperty(float)
    def animatedValue(self) -> float:
        """Get animated value (for animation)"""
        return self._animated_value
    
    @animatedValue.setter
    def animatedValue(self, value: float):
        """Set animated value and trigger repaint"""
        self._animated_value = value
        self.update()
    
    def set_value(self, value: float, animate: bool = True):
        """
        Set gauge value with optional animation.
        
        Args:
            value: New value to display
            animate: Whether to animate the transition
        """
        # Clamp value to range
        original_value = value
        value = max(self._min_value, min(self._max_value, value))
        
        if original_value != value:
            logger.debug(
                f"Value clamped: original={original_value:.2f}, "
                f"clamped={value:.2f}, range=[{self._min_value}, {self._max_value}]"
            )
        
        if value != self._value:
            old_value = self._value
            self._value = value
            
            # Throttled logging to avoid spam during rapid updates
            current_time = time.time()
            if current_time - self._last_log_time >= self._log_throttle_interval:
                color_zone = self._get_zone_name(value)
                logger.debug(
                    f"Gauge value changed: title='{self._title}', "
                    f"old={old_value:.2f}, new={value:.2f}, "
                    f"zone={color_zone}, animate={animate}"
                )
                self._last_log_time = current_time
            
            if animate:
                # Animate from current animated value to new value
                self._animation.stop()
                self._animation.setStartValue(self._animated_value)
                self._animation.setEndValue(value)
                self._animation.start()
            else:
                # Set immediately
                self._animated_value = value
                self.update()
    
    def _get_zone_name(self, value: float) -> str:
        """Get color zone name for a value"""
        if self._green_zone[0] <= value <= self._green_zone[1]:
            return "green"
        elif self._yellow_zone[0] <= value <= self._yellow_zone[1]:
            return "yellow"
        elif self._red_zone[0] <= value <= self._red_zone[1]:
            return "red"
        else:
            return "unknown"
    
    def get_value(self) -> float:
        """Get current gauge value"""
        return self._value
    
    def set_title(self, title: str):
        """Set gauge title"""
        if title != self._title:
            logger.debug(f"Gauge title changed: old='{self._title}', new='{title}'")
            self._title = title
            self.update()
    
    def set_unit(self, unit: str):
        """Set value unit"""
        if unit != self._unit:
            logger.debug(f"Gauge unit changed: old='{self._unit}', new='{unit}'")
            self._unit = unit
            self.update()
    
    def set_color_zones(
        self,
        green_zone: Tuple[float, float],
        yellow_zone: Tuple[float, float],
        red_zone: Tuple[float, float]
    ):
        """Set color zone ranges"""
        logger.debug(
            f"Color zones updated: green={green_zone}, "
            f"yellow={yellow_zone}, red={red_zone}"
        )
        self._green_zone = green_zone
        self._yellow_zone = yellow_zone
        self._red_zone = red_zone
        self.update()
    
    def _get_color_for_value(self, value: float) -> QColor:
        """Get color based on value and zones"""
        if self._green_zone[0] <= value <= self._green_zone[1]:
            return QColor(76, 175, 80)  # Green
        elif self._yellow_zone[0] <= value <= self._yellow_zone[1]:
            return QColor(255, 193, 7)  # Yellow/Amber
        elif self._red_zone[0] <= value <= self._red_zone[1]:
            return QColor(244, 67, 54)  # Red
        else:
            # Default to gray if outside all zones
            return QColor(158, 158, 158)
    
    def _value_to_angle(self, value: float) -> float:
        """Convert value to angle in degrees"""
        # Normalize value to 0-1 range
        normalized = (value - self._min_value) / (self._max_value - self._min_value)
        # Convert to angle
        angle = self._start_angle + (normalized * self._span_angle)
        return angle
    
    def paintEvent(self, event):
        """Paint the gauge"""
        paint_start = time.time()
        
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Get widget dimensions
            width = self.width()
            height = self.height()
            size = min(width, height)
            
            # Center the gauge
            center_x = width / 2
            center_y = height / 2
            
            # Calculate gauge dimensions
            margin = size * 0.1
            gauge_size = size - 2 * margin
            gauge_rect = QRectF(
                center_x - gauge_size / 2,
                center_y - gauge_size / 2,
                gauge_size,
                gauge_size
            )
            
            # Draw background arc with gradient
            self._draw_background_arc(painter, gauge_rect)
            
            # Draw color zone arcs
            self._draw_zone_arcs(painter, gauge_rect)
            
            # Draw value arc
            self._draw_value_arc(painter, gauge_rect)
            
            # Draw tick marks
            self._draw_tick_marks(painter, gauge_rect, center_x, center_y)
            
            # Draw needle
            self._draw_needle(painter, center_x, center_y, gauge_size / 2)
            
            # Draw center circle
            self._draw_center_circle(painter, center_x, center_y, gauge_size)
            
            # Draw value text
            self._draw_value_text(painter, center_x, center_y, gauge_size)
            
            # Draw title
            self._draw_title(painter, center_x, center_y, gauge_size)
            
            # Performance tracking (log every 100 paints)
            self._paint_count += 1
            paint_duration = time.time() - paint_start
            
            if self._paint_count % 100 == 0:
                logger.debug(
                    f"Paint performance: title='{self._title}', "
                    f"count={self._paint_count}, duration={paint_duration*1000:.2f}ms"
                )
            
            # Warn if paint is slow (>16ms for 60 FPS)
            if paint_duration > 0.016:
                logger.warning(
                    f"Slow paint detected: title='{self._title}', "
                    f"duration={paint_duration*1000:.2f}ms (target: <16ms)"
                )
        
        except Exception as e:
            logger.error(
                f"Paint error in CircularGaugeWidget: title='{self._title}', error={e}",
                exc_info=True
            )
    
    def _draw_background_arc(self, painter: QPainter, rect: QRectF):
        """Draw background arc with gradient"""
        pen = QPen()
        pen.setWidth(int(rect.width() * 0.08))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setColor(QColor(60, 60, 60, 100))
        
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Draw full arc
        painter.drawArc(rect, self._start_angle * 16, self._span_angle * 16)
    
    def _draw_zone_arcs(self, painter: QPainter, rect: QRectF):
        """Draw color zone indicators on outer edge"""
        pen = QPen()
        pen.setWidth(int(rect.width() * 0.02))
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Expand rect slightly for zone arcs
        zone_rect = rect.adjusted(-5, -5, 5, 5)
        
        # Draw each zone
        zones = [
            (self._red_zone, QColor(244, 67, 54)),
            (self._yellow_zone, QColor(255, 193, 7)),
            (self._green_zone, QColor(76, 175, 80))
        ]
        
        for (zone_min, zone_max), color in zones:
            if zone_min >= self._min_value and zone_max <= self._max_value:
                start_angle = self._value_to_angle(zone_min)
                end_angle = self._value_to_angle(zone_max)
                span = end_angle - start_angle
                
                pen.setColor(color)
                painter.setPen(pen)
                painter.drawArc(zone_rect, int(start_angle * 16), int(span * 16))
    
    def _draw_value_arc(self, painter: QPainter, rect: QRectF):
        """Draw value arc"""
        pen = QPen()
        pen.setWidth(int(rect.width() * 0.08))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        
        # Get color for current value
        color = self._get_color_for_value(self._animated_value)
        pen.setColor(color)
        
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        # Calculate span from start to current value
        value_angle = self._value_to_angle(self._animated_value)
        span = value_angle - self._start_angle
        
        # Draw arc from start to current value
        painter.drawArc(rect, self._start_angle * 16, int(span * 16))
    
    def _draw_tick_marks(self, painter: QPainter, rect: QRectF, center_x: float, center_y: float):
        """Draw tick marks around the gauge"""
        import math
        
        pen = QPen()
        pen.setColor(QColor(200, 200, 200))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Draw major ticks at 0%, 25%, 50%, 75%, 100%
        num_ticks = 5
        radius_outer = rect.width() / 2 - rect.width() * 0.08
        radius_inner = radius_outer - rect.width() * 0.05
        
        for i in range(num_ticks):
            value = self._min_value + (i / (num_ticks - 1)) * (self._max_value - self._min_value)
            angle_deg = self._value_to_angle(value)
            angle_rad = math.radians(angle_deg)
            
            # Calculate tick endpoints
            x_outer = center_x + radius_outer * math.cos(angle_rad)
            y_outer = center_y + radius_outer * math.sin(angle_rad)
            x_inner = center_x + radius_inner * math.cos(angle_rad)
            y_inner = center_y + radius_inner * math.sin(angle_rad)
            
            painter.drawLine(QPointF(x_inner, y_inner), QPointF(x_outer, y_outer))
    
    def _draw_needle(self, painter: QPainter, center_x: float, center_y: float, radius: float):
        """Draw needle pointer"""
        import math
        
        # Calculate needle angle
        angle_deg = self._value_to_angle(self._animated_value)
        angle_rad = math.radians(angle_deg)
        
        # Needle dimensions
        needle_length = radius * 0.7
        needle_width = radius * 0.03
        
        # Calculate needle tip
        tip_x = center_x + needle_length * math.cos(angle_rad)
        tip_y = center_y + needle_length * math.sin(angle_rad)
        
        # Calculate needle base points (perpendicular to needle)
        base_angle1 = angle_rad + math.pi / 2
        base_angle2 = angle_rad - math.pi / 2
        
        base1_x = center_x + needle_width * math.cos(base_angle1)
        base1_y = center_y + needle_width * math.sin(base_angle1)
        base2_x = center_x + needle_width * math.cos(base_angle2)
        base2_y = center_y + needle_width * math.sin(base_angle2)
        
        # Draw needle as triangle
        path = QPainterPath()
        path.moveTo(tip_x, tip_y)
        path.lineTo(base1_x, base1_y)
        path.lineTo(base2_x, base2_y)
        path.closeSubpath()
        
        # Get color for needle
        color = self._get_color_for_value(self._animated_value)
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawPath(path)
    
    def _draw_center_circle(self, painter: QPainter, center_x: float, center_y: float, gauge_size: float):
        """Draw center circle"""
        circle_radius = gauge_size * 0.15
        
        # Outer circle (border)
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setBrush(QBrush(QColor(40, 40, 40)))
        painter.drawEllipse(
            QPointF(center_x, center_y),
            circle_radius,
            circle_radius
        )
    
    def _draw_value_text(self, painter: QPainter, center_x: float, center_y: float, gauge_size: float):
        """Draw value text in center"""
        # Format value
        if isinstance(self._animated_value, float):
            value_text = f"{self._animated_value:.1f}"
        else:
            value_text = str(int(self._animated_value))
        
        if self._unit:
            value_text += f" {self._unit}"
        
        # Set font
        font = QFont()
        font.setPointSize(int(gauge_size * 0.08))
        font.setBold(True)
        painter.setFont(font)
        
        # Get color for value
        color = self._get_color_for_value(self._animated_value)
        painter.setPen(QPen(color))
        
        # Draw text
        text_rect = QRectF(
            center_x - gauge_size * 0.3,
            center_y - gauge_size * 0.05,
            gauge_size * 0.6,
            gauge_size * 0.1
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, value_text)
    
    def _draw_title(self, painter: QPainter, center_x: float, center_y: float, gauge_size: float):
        """Draw title above value"""
        if not self._title:
            return
        
        # Set font
        font = QFont()
        font.setPointSize(int(gauge_size * 0.05))
        painter.setFont(font)
        
        painter.setPen(QPen(QColor(200, 200, 200)))
        
        # Draw text
        text_rect = QRectF(
            center_x - gauge_size * 0.4,
            center_y - gauge_size * 0.15,
            gauge_size * 0.8,
            gauge_size * 0.1
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, self._title)
