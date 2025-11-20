"""
Risk Assessment Panel

Complete panel for displaying risk assessment with all components.
"""

import logging
from typing import Optional, Dict, Any, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QSplitter, QListWidget, QListWidgetItem, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QFont, QPainter, QPen, QBrush, QColor, QPolygonF
import pyqtgraph as pg
from collections import deque
import time

from .circular_gauge import CircularGaugeWidget

logger = logging.getLogger(__name__)


class RiskAssessmentPanel(QWidget):
    """
    Complete risk assessment panel.
    
    Integrates:
    - Overall risk gauge
    - Hazards list
    - Zone risk radar chart
    - TTC display
    - Risk timeline
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        logger.debug("RiskAssessmentPanel initialization started")
        
        # Risk history for timeline (last 5 minutes at 1 Hz = 300 points)
        self._risk_history = deque(maxlen=300)
        self._time_history = deque(maxlen=300)
        self._alert_events = []  # List of (timestamp, urgency) tuples
        
        # Current data
        self._current_risk = 0.0
        self._current_hazards = []
        self._zone_risks = [0.0] * 8  # 8 zones
        self._min_ttc = float('inf')
        
        self._init_ui()
        
        logger.info("RiskAssessmentPanel created successfully: history_capacity=300, zones=8")
    
    def _init_ui(self):
        """Initialize UI components"""
        logger.debug("Initializing RiskAssessmentPanel UI components")
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title = QLabel("Risk Assessment")
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(15)
        
        # Top row: Risk gauge and TTC display
        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        
        # Overall risk gauge
        self._risk_gauge = CircularGaugeWidget(
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            title="Overall Risk",
            unit="",
            green_zone=(0.0, 0.5),
            yellow_zone=(0.5, 0.7),
            red_zone=(0.7, 1.0)
        )
        self._risk_gauge.setMinimumSize(200, 200)
        top_row.addWidget(self._risk_gauge)
        
        # TTC display
        self._ttc_widget = TTCDisplayWidget()
        self._ttc_widget.setMinimumSize(200, 200)
        top_row.addWidget(self._ttc_widget)
        
        content_layout.addLayout(top_row)
        
        # Middle row: Hazards list and Zone radar
        middle_row = QHBoxLayout()
        middle_row.setSpacing(10)
        
        # Hazards list
        hazards_container = QWidget()
        hazards_layout = QVBoxLayout()
        hazards_layout.setContentsMargins(0, 0, 0, 0)
        
        hazards_label = QLabel("Active Hazards (Top 3)")
        hazards_label_font = QFont()
        hazards_label_font.setPointSize(10)
        hazards_label_font.setBold(True)
        hazards_label.setFont(hazards_label_font)
        hazards_layout.addWidget(hazards_label)
        
        self._hazards_list = QListWidget()
        self._hazards_list.setMinimumHeight(150)
        hazards_layout.addWidget(self._hazards_list)
        
        hazards_container.setLayout(hazards_layout)
        middle_row.addWidget(hazards_container, 1)
        
        # Zone risk radar chart
        radar_container = QWidget()
        radar_layout = QVBoxLayout()
        radar_layout.setContentsMargins(0, 0, 0, 0)
        
        radar_label = QLabel("Zone Risk Distribution")
        radar_label_font = QFont()
        radar_label_font.setPointSize(10)
        radar_label_font.setBold(True)
        radar_label.setFont(radar_label_font)
        radar_layout.addWidget(radar_label)
        
        self._zone_radar = ZoneRiskRadarChart()
        self._zone_radar.setMinimumSize(250, 250)
        radar_layout.addWidget(self._zone_radar)
        
        radar_container.setLayout(radar_layout)
        middle_row.addWidget(radar_container, 1)
        
        content_layout.addLayout(middle_row)
        
        # Bottom: Risk timeline
        timeline_label = QLabel("Risk Timeline (Last 5 Minutes)")
        timeline_label_font = QFont()
        timeline_label_font.setPointSize(10)
        timeline_label_font.setBold(True)
        timeline_label.setFont(timeline_label_font)
        content_layout.addWidget(timeline_label)
        
        self._risk_timeline = pg.PlotWidget()
        self._risk_timeline.setMinimumHeight(150)
        self._risk_timeline.setBackground('#2b2b2b')
        self._risk_timeline.setLabel('left', 'Risk Score', units='')
        self._risk_timeline.setLabel('bottom', 'Time', units='s')
        self._risk_timeline.setYRange(0, 1.0)
        self._risk_timeline.showGrid(x=True, y=True, alpha=0.3)
        
        # Add threshold lines
        self._risk_timeline.addLine(y=0.7, pen=pg.mkPen(color='r', width=2, style=Qt.PenStyle.DashLine))
        self._risk_timeline.addLine(y=0.5, pen=pg.mkPen(color='y', width=2, style=Qt.PenStyle.DashLine))
        
        # Risk curve
        self._risk_curve = self._risk_timeline.plot(
            pen=pg.mkPen(color='#00aaff', width=2),
            name='Risk Score'
        )
        
        # Alert markers
        self._alert_scatter = pg.ScatterPlotItem(
            size=10,
            brush=pg.mkBrush(255, 0, 0, 200)
        )
        self._risk_timeline.addItem(self._alert_scatter)
        
        content_layout.addWidget(self._risk_timeline)
        
        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        logger.debug("RiskAssessmentPanel UI components initialized: gauge, ttc_display, hazards_list, zone_radar, timeline")
    
    def update_risk_score(self, risk_score: float):
        """
        Update overall risk score.
        
        Args:
            risk_score: Risk score (0.0 to 1.0)
        """
        prev_risk = self._current_risk
        self._current_risk = risk_score
        self._risk_gauge.set_value(risk_score, animate=True)
        
        # Add to history
        current_time = time.time()
        self._risk_history.append(risk_score)
        self._time_history.append(current_time)
        
        # Update timeline
        self._update_timeline()
        
        # Log state transitions
        if prev_risk <= 0.5 and risk_score > 0.5:
            logger.info(f"Risk level increased to MEDIUM: {risk_score:.3f}")
        elif prev_risk <= 0.7 and risk_score > 0.7:
            logger.warning(f"Risk level increased to HIGH: {risk_score:.3f}")
        elif prev_risk <= 0.9 and risk_score > 0.9:
            logger.warning(f"Risk level increased to CRITICAL: {risk_score:.3f}")
        elif prev_risk > 0.7 and risk_score <= 0.7:
            logger.info(f"Risk level decreased from HIGH: {risk_score:.3f}")
        
        logger.debug(f"Risk score updated: prev={prev_risk:.3f}, current={risk_score:.3f}, history_size={len(self._risk_history)}")
    
    def update_hazards(self, hazards: List[Dict[str, Any]]):
        """
        Update hazards list.
        
        Args:
            hazards: List of hazard dictionaries with keys:
                - type: str (vehicle, pedestrian, cyclist, etc.)
                - zone: str (front, front-left, etc.)
                - ttc: float (time to collision in seconds)
                - risk_score: float (0.0 to 1.0)
                - attended: bool (whether driver is looking at it)
        """
        prev_count = len(self._current_hazards)
        self._current_hazards = hazards[:3]  # Top 3
        
        # Clear and repopulate list
        self._hazards_list.clear()
        
        for hazard in self._current_hazards:
            item_widget = HazardListItem(hazard)
            item = QListWidgetItem(self._hazards_list)
            item.setSizeHint(item_widget.sizeHint())
            self._hazards_list.addItem(item)
            self._hazards_list.setItemWidget(item, item_widget)
        
        # Log significant changes
        unattended_count = sum(1 for h in self._current_hazards if not h.get('attended', False))
        if unattended_count > 0:
            logger.warning(f"Hazards updated: {len(hazards)} total, {unattended_count} unattended in top 3")
        else:
            logger.debug(f"Hazards updated: {len(hazards)} total, showing top 3, all attended")
        
        # Log new hazards
        if len(self._current_hazards) > prev_count:
            logger.info(f"New hazards detected: count increased from {prev_count} to {len(self._current_hazards)}")
    
    def update_zone_risks(self, zone_risks: List[float]):
        """
        Update zone risk values.
        
        Args:
            zone_risks: List of 8 risk values (0.0 to 1.0) for each zone
        """
        if len(zone_risks) != 8:
            logger.error(f"Invalid zone risks count: expected=8, got={len(zone_risks)}")
            return
        
        self._zone_risks = zone_risks
        self._zone_radar.set_zone_risks(zone_risks)
        
        # Log high-risk zones
        high_risk_zones = [(i, risk) for i, risk in enumerate(zone_risks) if risk > 0.7]
        if high_risk_zones:
            zone_names = ["Front", "Front-Right", "Right", "Rear-Right", "Rear", "Rear-Left", "Left", "Front-Left"]
            high_risk_str = ", ".join([f"{zone_names[i]}={risk:.2f}" for i, risk in high_risk_zones])
            logger.warning(f"High-risk zones detected: {high_risk_str}")
        
        max_risk = max(zone_risks)
        logger.debug(f"Zone risks updated: max={max_risk:.3f}, values={[f'{r:.2f}' for r in zone_risks]}")
    
    def update_ttc(self, min_ttc: float):
        """
        Update minimum time-to-collision.
        
        Args:
            min_ttc: Minimum TTC in seconds
        """
        prev_ttc = self._min_ttc
        self._min_ttc = min_ttc
        self._ttc_widget.set_ttc(min_ttc)
        
        # Log critical TTC changes
        if min_ttc < 1.5 and prev_ttc >= 1.5:
            logger.warning(f"TTC entered CRITICAL range: {min_ttc:.2f}s")
        elif min_ttc < 3.0 and prev_ttc >= 3.0:
            logger.info(f"TTC entered WARNING range: {min_ttc:.2f}s")
        elif min_ttc >= 3.0 and prev_ttc < 3.0:
            logger.info(f"TTC returned to SAFE range: {min_ttc:.2f}s")
        
        logger.debug(f"TTC updated: prev={prev_ttc:.2f}s, current={min_ttc:.2f}s")
    
    def add_alert_event(self, urgency: str):
        """
        Add alert event marker to timeline.
        
        Args:
            urgency: Alert urgency ('info', 'warning', 'critical')
        """
        current_time = time.time()
        self._alert_events.append((current_time, urgency))
        
        # Keep only last 5 minutes
        cutoff_time = current_time - 300
        prev_count = len(self._alert_events)
        self._alert_events = [
            (t, u) for t, u in self._alert_events if t > cutoff_time
        ]
        
        self._update_timeline()
        
        # Log alert events with appropriate level
        if urgency == 'critical':
            logger.warning(f"CRITICAL alert event added to timeline: total_events={len(self._alert_events)}")
        elif urgency == 'warning':
            logger.info(f"WARNING alert event added to timeline: total_events={len(self._alert_events)}")
        else:
            logger.debug(f"Alert event added: urgency={urgency}, total_events={len(self._alert_events)}")
        
        if prev_count > len(self._alert_events):
            logger.debug(f"Alert events pruned: removed={prev_count - len(self._alert_events)} old events")
    
    def _update_timeline(self):
        """Update risk timeline plot"""
        if len(self._time_history) == 0:
            logger.debug("Timeline update skipped: no data available")
            return
        
        # Convert to relative time (seconds ago)
        current_time = time.time()
        relative_times = [-(current_time - t) for t in self._time_history]
        
        # Update curve
        self._risk_curve.setData(relative_times, list(self._risk_history))
        
        # Update alert markers
        if self._alert_events:
            alert_times = [-(current_time - t) for t, _ in self._alert_events]
            # Get risk score at alert time (approximate)
            alert_risks = []
            for alert_time in alert_times:
                # Find closest risk value
                if relative_times:
                    idx = min(range(len(relative_times)), 
                             key=lambda i: abs(relative_times[i] - alert_time))
                    alert_risks.append(self._risk_history[idx])
                else:
                    alert_risks.append(0.5)
            
            self._alert_scatter.setData(alert_times, alert_risks)
        
        # Set x-axis range to last 5 minutes
        self._risk_timeline.setXRange(-300, 0)
        
        logger.debug(f"Timeline updated: data_points={len(self._risk_history)}, alert_markers={len(self._alert_events)}")


class TTCDisplayWidget(QWidget):
    """
    Time-to-collision display widget with countdown timer.
    
    Color coded by urgency:
    - Green: >3s
    - Yellow: 1.5-3s
    - Red: <1.5s
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._ttc = float('inf')
        self._animation_phase = 0.0
        
        # Animation timer for countdown effect
        self._animation_timer = QTimer()
        self._animation_timer.timeout.connect(self._animate)
        self._animation_timer.start(50)  # 20 Hz animation
        
        self.setMinimumSize(150, 150)
        
        logger.info("TTCDisplayWidget created: animation_rate=20Hz, min_size=150x150")
    
    def set_ttc(self, ttc: float):
        """Set time-to-collision value"""
        prev_ttc = self._ttc
        self._ttc = ttc
        self.update()
        
        # Log significant TTC changes
        if ttc < 1.5 and prev_ttc >= 1.5:
            logger.warning(f"TTCDisplayWidget: TTC critical: {ttc:.2f}s")
        
        logger.debug(f"TTCDisplayWidget: TTC set: prev={prev_ttc:.2f}s, current={ttc:.2f}s")
    
    def _animate(self):
        """Animation tick"""
        self._animation_phase += 0.1
        if self._animation_phase > 1.0:
            self._animation_phase = 0.0
        self.update()
    
    def _get_color_for_ttc(self, ttc: float) -> QColor:
        """Get color based on TTC value"""
        if ttc > 3.0:
            return QColor(76, 175, 80)  # Green
        elif ttc > 1.5:
            return QColor(255, 193, 7)  # Yellow
        else:
            return QColor(244, 67, 54)  # Red
    
    def paintEvent(self, event):
        """Paint the TTC display"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        size = min(width, height)
        
        center_x = width / 2
        center_y = height / 2
        
        # Draw background circle
        radius = size * 0.4
        color = self._get_color_for_ttc(self._ttc)
        
        # Pulsing effect for critical TTC
        if self._ttc < 1.5:
            pulse = 1.0 + 0.1 * self._animation_phase
            radius *= pulse
            alpha = int(200 - 50 * self._animation_phase)
            color.setAlpha(alpha)
        
        painter.setPen(QPen(color, 3))
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 50)))
        painter.drawEllipse(QPointF(center_x, center_y), radius, radius)
        
        # Draw TTC value
        font = QFont()
        font.setPointSize(int(size * 0.12))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(color))
        
        if self._ttc == float('inf'):
            ttc_text = "∞"
        else:
            ttc_text = f"{self._ttc:.1f}s"
        
        text_rect = QRectF(
            center_x - size * 0.3,
            center_y - size * 0.1,
            size * 0.6,
            size * 0.2
        )
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, ttc_text)
        
        # Draw label
        font.setPointSize(int(size * 0.06))
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QPen(QColor(200, 200, 200)))
        
        label_rect = QRectF(
            center_x - size * 0.3,
            center_y + size * 0.05,
            size * 0.6,
            size * 0.1
        )
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "Min TTC")


class HazardListItem(QWidget):
    """
    Custom widget for displaying a hazard in the list.
    
    Shows:
    - Hazard icon and type
    - Zone
    - TTC
    - Attention status
    - Risk score progress bar
    """
    
    def __init__(self, hazard: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._hazard = hazard
        self._init_ui()
        
        logger.debug(f"HazardListItem created: type={hazard.get('type')}, zone={hazard.get('zone')}, "
                    f"ttc={hazard.get('ttc', float('inf')):.2f}s, risk={hazard.get('risk_score', 0.0):.2f}")
    
    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Top row: Type and attention status
        top_row = QHBoxLayout()
        
        # Hazard type with icon
        type_label = QLabel(self._get_icon() + " " + self._hazard.get('type', 'Unknown'))
        type_font = QFont()
        type_font.setBold(True)
        type_label.setFont(type_font)
        top_row.addWidget(type_label)
        
        top_row.addStretch()
        
        # Attention status
        attended = self._hazard.get('attended', False)
        attention_label = QLabel("👁️ Attended" if attended else "⚠️ Unattended")
        attention_label.setStyleSheet(
            f"color: {'#4caf50' if attended else '#ff5722'}; font-weight: bold;"
        )
        top_row.addWidget(attention_label)
        
        layout.addLayout(top_row)
        
        # Middle row: Zone and TTC
        middle_row = QHBoxLayout()
        
        zone = self._hazard.get('zone', 'unknown')
        zone_label = QLabel(f"Zone: {zone}")
        middle_row.addWidget(zone_label)
        
        middle_row.addStretch()
        
        ttc = self._hazard.get('ttc', float('inf'))
        if ttc == float('inf'):
            ttc_text = "TTC: ∞"
        else:
            ttc_text = f"TTC: {ttc:.1f}s"
        ttc_label = QLabel(ttc_text)
        middle_row.addWidget(ttc_label)
        
        layout.addLayout(middle_row)
        
        # Bottom row: Risk score progress bar
        risk_score = self._hazard.get('risk_score', 0.0)
        risk_bar = QProgressBar()
        risk_bar.setMinimum(0)
        risk_bar.setMaximum(100)
        risk_bar.setValue(int(risk_score * 100))
        risk_bar.setFormat(f"Risk: {risk_score:.2f}")
        
        # Color based on risk
        if risk_score > 0.7:
            risk_bar.setStyleSheet("""
                QProgressBar::chunk { background-color: #f44336; }
            """)
        elif risk_score > 0.5:
            risk_bar.setStyleSheet("""
                QProgressBar::chunk { background-color: #ffc107; }
            """)
        else:
            risk_bar.setStyleSheet("""
                QProgressBar::chunk { background-color: #4caf50; }
            """)
        
        layout.addWidget(risk_bar)
        
        self.setLayout(layout)
    
    def _get_icon(self) -> str:
        """Get emoji icon for hazard type"""
        hazard_type = self._hazard.get('type', '').lower()
        icons = {
            'vehicle': '🚗',
            'pedestrian': '🚶',
            'cyclist': '🚴',
            'motorcycle': '🏍️',
            'truck': '🚚',
            'bus': '🚌'
        }
        return icons.get(hazard_type, '⚠️')


class ZoneRiskRadarChart(QWidget):
    """
    Octagonal radar chart for displaying risk in 8 zones around vehicle.
    
    Zones (clockwise from front):
    0: Front
    1: Front-Right
    2: Right
    3: Rear-Right
    4: Rear
    5: Rear-Left
    6: Left
    7: Front-Left
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._zone_risks = [0.0] * 8
        self._zone_labels = [
            "Front", "Front-Right", "Right", "Rear-Right",
            "Rear", "Rear-Left", "Left", "Front-Left"
        ]
        
        self.setMinimumSize(200, 200)
        
        logger.info("ZoneRiskRadarChart created: zones=8, min_size=200x200")
    
    def set_zone_risks(self, zone_risks: List[float]):
        """Set risk values for all zones"""
        if len(zone_risks) != 8:
            logger.error(f"ZoneRiskRadarChart: Invalid zone count: expected=8, got={len(zone_risks)}")
            return
        
        prev_max = max(self._zone_risks) if self._zone_risks else 0.0
        self._zone_risks = zone_risks
        current_max = max(zone_risks)
        
        self.update()
        
        # Log significant changes
        if current_max > 0.7 and prev_max <= 0.7:
            logger.warning(f"ZoneRiskRadarChart: Maximum zone risk entered HIGH range: {current_max:.2f}")
        
        logger.debug(f"ZoneRiskRadarChart: Zone risks set: max={current_max:.2f}, avg={sum(zone_risks)/8:.2f}")
    
    def paintEvent(self, event):
        """Paint the radar chart"""
        import math
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        size = min(width, height)
        
        center_x = width / 2
        center_y = height / 2
        radius = size * 0.35
        
        # Draw background grid (3 levels: 0.33, 0.66, 1.0)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        for level in [0.33, 0.66, 1.0]:
            points = []
            for i in range(8):
                angle = math.radians(i * 45 - 90)  # Start from top (front)
                x = center_x + radius * level * math.cos(angle)
                y = center_y + radius * level * math.sin(angle)
                points.append(QPointF(x, y))
            
            polygon = QPolygonF(points)
            painter.drawPolygon(polygon)
        
        # Draw radial lines
        for i in range(8):
            angle = math.radians(i * 45 - 90)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            painter.drawLine(QPointF(center_x, center_y), QPointF(x, y))
        
        # Draw risk polygon
        risk_points = []
        for i in range(8):
            angle = math.radians(i * 45 - 90)
            risk_value = self._zone_risks[i]
            x = center_x + radius * risk_value * math.cos(angle)
            y = center_y + radius * risk_value * math.sin(angle)
            risk_points.append(QPointF(x, y))
        
        risk_polygon = QPolygonF(risk_points)
        
        # Fill with semi-transparent color
        max_risk = max(self._zone_risks) if self._zone_risks else 0.0
        if max_risk > 0.7:
            fill_color = QColor(244, 67, 54, 100)  # Red
        elif max_risk > 0.5:
            fill_color = QColor(255, 193, 7, 100)  # Yellow
        else:
            fill_color = QColor(76, 175, 80, 100)  # Green
        
        painter.setPen(QPen(fill_color, 2))
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(risk_polygon)
        
        # Draw zone labels
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(200, 200, 200)))
        
        label_radius = radius * 1.15
        for i in range(8):
            angle = math.radians(i * 45 - 90)
            x = center_x + label_radius * math.cos(angle)
            y = center_y + label_radius * math.sin(angle)
            
            # Draw label with background
            label = self._zone_labels[i]
            text_rect = QRectF(x - 40, y - 10, 80, 20)
            
            # Highlight high-risk zones
            if self._zone_risks[i] > 0.7:
                painter.setPen(QPen(QColor(244, 67, 54)))
                font.setBold(True)
                painter.setFont(font)
            else:
                painter.setPen(QPen(QColor(200, 200, 200)))
                font.setBold(False)
                painter.setFont(font)
            
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
