"""
Driver State Panel

Complete panel for displaying driver state with all components.
"""

import logging
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QSplitter
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from .circular_gauge import CircularGaugeWidget
from .gaze_direction import GazeDirectionWidget
from .metric_display import DriverMetricsPanel
from .status_indicator import DriverStatusPanel
from .trend_graph import DriverTrendGraphsPanel
from .warning_animations import WarningAnimationManager, ThresholdMonitor

logger = logging.getLogger(__name__)


class DriverStatePanel(QWidget):
    """
    Complete driver state panel.
    
    Integrates:
    - Readiness gauge
    - Gaze direction visualization
    - Metrics grid
    - Status indicators
    - Trend graphs
    - Warning animations
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Animation manager
        self._animation_manager = WarningAnimationManager()
        self._threshold_monitor = ThresholdMonitor(self._animation_manager)
        
        # Setup threshold monitoring
        self._setup_threshold_monitoring()
        
        self._init_ui()
        
        logger.info("DriverStatePanel created")
    
    def _setup_threshold_monitoring(self):
        """Setup threshold monitoring for metrics"""
        # Alertness: warning < 50, critical < 30
        self._threshold_monitor.register_metric(
            'alertness',
            warning_threshold=50.0,
            critical_threshold=30.0,
            reverse=False  # Higher is better
        )
        
        # Attention: warning < 50, critical < 30
        self._threshold_monitor.register_metric(
            'attention',
            warning_threshold=50.0,
            critical_threshold=30.0,
            reverse=False
        )
        
        # Readiness: warning < 50, critical < 30
        self._threshold_monitor.register_metric(
            'readiness',
            warning_threshold=50.0,
            critical_threshold=30.0,
            reverse=False
        )
    
    def _init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title = QLabel("Driver State Monitor")
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
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(15)
        
        # Top section: Readiness gauge and gaze direction
        top_section = QHBoxLayout()
        top_section.setSpacing(10)
        
        # Readiness gauge
        gauge_container = QWidget()
        gauge_layout = QVBoxLayout()
        gauge_layout.setContentsMargins(5, 5, 5, 5)
        
        self.readiness_gauge = CircularGaugeWidget(
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            title="Driver Readiness",
            unit="%",
            green_zone=(70.0, 100.0),
            yellow_zone=(50.0, 70.0),
            red_zone=(0.0, 50.0)
        )
        self.readiness_gauge.setMinimumSize(200, 200)
        gauge_layout.addWidget(self.readiness_gauge)
        gauge_container.setLayout(gauge_layout)
        
        top_section.addWidget(gauge_container)
        
        # Gaze direction
        gaze_container = QWidget()
        gaze_layout = QVBoxLayout()
        gaze_layout.setContentsMargins(5, 5, 5, 5)
        
        self.gaze_widget = GazeDirectionWidget()
        self.gaze_widget.setMinimumSize(250, 250)
        gaze_layout.addWidget(self.gaze_widget)
        gaze_container.setLayout(gaze_layout)
        
        top_section.addWidget(gaze_container)
        
        content_layout.addLayout(top_section)
        
        # Separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        content_layout.addWidget(separator1)
        
        # Middle section: Metrics and status
        middle_section = QHBoxLayout()
        middle_section.setSpacing(10)
        
        # Metrics panel
        self.metrics_panel = DriverMetricsPanel()
        middle_section.addWidget(self.metrics_panel)
        
        # Status panel
        self.status_panel = DriverStatusPanel()
        middle_section.addWidget(self.status_panel)
        
        content_layout.addLayout(middle_section)
        
        # Separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        content_layout.addWidget(separator2)
        
        # Bottom section: Trend graphs
        self.trend_graphs = DriverTrendGraphsPanel()
        content_layout.addWidget(self.trend_graphs)
        
        content_widget.setLayout(content_layout)
        scroll_area.setWidget(content_widget)
        
        main_layout.addWidget(scroll_area)
        
        self.setLayout(main_layout)
    
    def update_driver_state(self, driver_state: Dict[str, Any]):
        """
        Update all components with new driver state.
        
        Args:
            driver_state: Dictionary containing driver state data
        """
        # Update readiness gauge
        if 'readiness_score' in driver_state:
            readiness = driver_state['readiness_score']
            self.readiness_gauge.set_value(readiness, animate=True)
            
            # Monitor threshold crossing
            self._threshold_monitor.update_metric(
                'readiness',
                readiness,
                widget=self.readiness_gauge
            )
        
        # Update gaze direction
        if 'gaze' in driver_state and 'face_detected' in driver_state:
            if driver_state['face_detected']:
                gaze = driver_state['gaze']
                pitch = gaze.get('pitch', 0.0)
                yaw = gaze.get('yaw', 0.0)
                attention_zone = gaze.get('attention_zone', 'front')
                
                self.gaze_widget.set_gaze(pitch, yaw, attention_zone)
            else:
                self.gaze_widget.clear_gaze()
        
        # Update metrics panel
        self.metrics_panel.update_metrics(driver_state)
        
        # Update status panel
        self.status_panel.update_status(driver_state)
        
        # Update trend graphs
        self.trend_graphs.update_graphs(driver_state)
        
        # Monitor alertness and attention thresholds
        if 'drowsiness' in driver_state and 'score' in driver_state['drowsiness']:
            alertness = 100 - driver_state['drowsiness']['score']
            self._threshold_monitor.update_metric('alertness', alertness)
        
        if 'readiness_score' in driver_state:
            self._threshold_monitor.update_metric('attention', driver_state['readiness_score'])
    
    def clear_driver_state(self):
        """Clear all driver state displays"""
        self.readiness_gauge.set_value(0.0, animate=False)
        self.gaze_widget.clear_gaze()
        self.metrics_panel.clear_metrics()
        self.status_panel.clear_status()
        self.trend_graphs.clear_graphs()
        self._threshold_monitor.reset_all()
        
        logger.debug("Driver state cleared")
    
    def set_sounds_enabled(self, enabled: bool):
        """Enable or disable warning sounds"""
        self._animation_manager.set_sounds_enabled(enabled)
    
    def load_warning_sounds(self, warning_sound_path: str, critical_sound_path: str):
        """
        Load warning sound files.
        
        Args:
            warning_sound_path: Path to warning sound file
            critical_sound_path: Path to critical alert sound file
        """
        self._animation_manager.load_sound('warning_alert', warning_sound_path)
        self._animation_manager.load_sound('critical_alert', critical_sound_path)
        
        logger.info("Warning sounds loaded")
