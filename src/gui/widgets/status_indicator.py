"""
Status Indicator Widget

Custom widget for displaying status with icon, label, and color coding.
"""

import logging
from typing import Optional
from PyQt6.QtWidgets import QWidget, QLabel, QHBoxLayout, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush

logger = logging.getLogger(__name__)


class StatusIndicatorWidget(QWidget):
    """
    Status indicator with icon and label.
    
    Features:
    - Color-coded states (green OK, yellow warning, red critical)
    - Icon indicator (circle)
    - Text label
    - Pulsing animation for warnings
    """
    
    # Status states
    STATE_OK = 'ok'
    STATE_WARNING = 'warning'
    STATE_CRITICAL = 'critical'
    STATE_UNKNOWN = 'unknown'
    
    # Colors for each state
    COLORS = {
        STATE_OK: QColor(76, 175, 80),        # Green
        STATE_WARNING: QColor(255, 193, 7),   # Yellow
        STATE_CRITICAL: QColor(244, 67, 54),  # Red
        STATE_UNKNOWN: QColor(158, 158, 158)  # Gray
    }
    
    def __init__(
        self,
        label: str,
        initial_state: str = STATE_UNKNOWN,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._label_text = label
        self._state = initial_state
        self._pulse_opacity = 1.0
        
        # Pulsing animation
        self._pulse_animation = QPropertyAnimation(self, b"pulseOpacity")
        self._pulse_animation.setDuration(1000)  # 1 second
        self._pulse_animation.setStartValue(1.0)
        self._pulse_animation.setEndValue(0.3)
        self._pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_animation.setLoopCount(-1)  # Infinite loop
        
        self._init_ui()
        
        logger.debug(f"StatusIndicatorWidget created: {label}")
    
    @pyqtProperty(float)
    def pulseOpacity(self) -> float:
        """Get pulse opacity (for animation)"""
        return self._pulse_opacity
    
    @pulseOpacity.setter
    def pulseOpacity(self, opacity: float):
        """Set pulse opacity and trigger repaint"""
        self._pulse_opacity = opacity
        self.update()
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Icon area (will be custom painted)
        self.icon_size = 20
        self.setMinimumHeight(self.icon_size + 10)
        
        # Label
        self.label = QLabel(self._label_text)
        font = QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        layout.addWidget(self.label)
        
        layout.addStretch()
        
        self.setLayout(layout)
        self.setMinimumWidth(150)
    
    def set_state(self, state: str):
        """
        Set indicator state.
        
        Args:
            state: One of STATE_OK, STATE_WARNING, STATE_CRITICAL, STATE_UNKNOWN
        """
        if state not in self.COLORS:
            logger.warning(f"Invalid state: {state}, using UNKNOWN")
            state = self.STATE_UNKNOWN
        
        self._state = state
        
        # Start/stop pulsing animation based on state
        if state == self.STATE_WARNING or state == self.STATE_CRITICAL:
            if not self._pulse_animation.state() == QPropertyAnimation.State.Running:
                self._pulse_animation.start()
        else:
            if self._pulse_animation.state() == QPropertyAnimation.State.Running:
                self._pulse_animation.stop()
            self._pulse_opacity = 1.0
        
        self.update()
    
    def get_state(self) -> str:
        """Get current state"""
        return self._state
    
    def paintEvent(self, event):
        """Paint the status indicator"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get color for current state
        color = self.COLORS[self._state]
        
        # Apply pulse opacity if animating
        if self._pulse_animation.state() == QPropertyAnimation.State.Running:
            color.setAlphaF(self._pulse_opacity)
        
        # Draw circle indicator
        circle_x = 10
        circle_y = self.height() / 2
        circle_radius = self.icon_size / 2
        
        # Outer glow for warning/critical
        if self._state in [self.STATE_WARNING, self.STATE_CRITICAL]:
            glow_color = QColor(color)
            glow_color.setAlpha(50)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow_color))
            painter.drawEllipse(
                int(circle_x - circle_radius * 1.5),
                int(circle_y - circle_radius * 1.5),
                int(circle_radius * 3),
                int(circle_radius * 3)
            )
        
        # Main circle
        painter.setPen(QPen(color.darker(120), 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(
            int(circle_x - circle_radius),
            int(circle_y - circle_radius),
            int(circle_radius * 2),
            int(circle_radius * 2)
        )


class DriverStatusPanel(QWidget):
    """
    Panel with multiple driver status indicators.
    
    Displays:
    - Drowsiness status
    - Distraction status
    - Eyes-on-road status
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._init_ui()
        
        logger.debug("DriverStatusPanel created")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Driver Status")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Status indicators
        self.drowsiness_indicator = StatusIndicatorWidget("Drowsiness")
        self.distraction_indicator = StatusIndicatorWidget("Distraction")
        self.eyes_on_road_indicator = StatusIndicatorWidget("Eyes on Road")
        
        layout.addWidget(self.drowsiness_indicator)
        layout.addWidget(self.distraction_indicator)
        layout.addWidget(self.eyes_on_road_indicator)
        
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_status(self, driver_state: dict):
        """
        Update status indicators from driver state.
        
        Args:
            driver_state: Dictionary containing driver state data
        """
        # Drowsiness status
        if 'drowsiness' in driver_state:
            drowsiness = driver_state['drowsiness']
            drowsiness_score = drowsiness.get('score', 0.0)
            
            if drowsiness_score < 30:
                self.drowsiness_indicator.set_state(StatusIndicatorWidget.STATE_OK)
            elif drowsiness_score < 60:
                self.drowsiness_indicator.set_state(StatusIndicatorWidget.STATE_WARNING)
            else:
                self.drowsiness_indicator.set_state(StatusIndicatorWidget.STATE_CRITICAL)
        
        # Distraction status
        if 'distraction' in driver_state:
            distraction = driver_state['distraction']
            distraction_type = distraction.get('type', 'none')
            distraction_duration = distraction.get('duration', 0.0)
            
            if distraction_type == 'none' or distraction_duration < 1.0:
                self.distraction_indicator.set_state(StatusIndicatorWidget.STATE_OK)
            elif distraction_duration < 2.0:
                self.distraction_indicator.set_state(StatusIndicatorWidget.STATE_WARNING)
            else:
                self.distraction_indicator.set_state(StatusIndicatorWidget.STATE_CRITICAL)
        
        # Eyes on road status
        if 'gaze' in driver_state:
            gaze = driver_state['gaze']
            attention_zone = gaze.get('attention_zone', 'front')
            
            # Eyes on road if looking at front zones
            if attention_zone in ['front', 'front_left', 'front_right']:
                self.eyes_on_road_indicator.set_state(StatusIndicatorWidget.STATE_OK)
            elif attention_zone in ['left', 'right']:
                self.eyes_on_road_indicator.set_state(StatusIndicatorWidget.STATE_WARNING)
            else:
                self.eyes_on_road_indicator.set_state(StatusIndicatorWidget.STATE_CRITICAL)
    
    def clear_status(self):
        """Clear all status indicators"""
        self.drowsiness_indicator.set_state(StatusIndicatorWidget.STATE_UNKNOWN)
        self.distraction_indicator.set_state(StatusIndicatorWidget.STATE_UNKNOWN)
        self.eyes_on_road_indicator.set_state(StatusIndicatorWidget.STATE_UNKNOWN)
