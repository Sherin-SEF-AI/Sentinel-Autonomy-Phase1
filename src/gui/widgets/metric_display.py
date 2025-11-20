"""
Metric Display Widget

Custom widget for displaying labeled metrics with color coding.
"""

import logging
from typing import Optional, Tuple
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPalette

logger = logging.getLogger(__name__)


class MetricDisplayWidget(QWidget):
    """
    Single metric display with label, value, and unit.
    
    Features:
    - Labeled metric display
    - Value formatting with units
    - Color coding based on thresholds
    - Real-time updates
    """
    
    def __init__(
        self,
        label: str,
        unit: str = "",
        decimal_places: int = 1,
        good_threshold: Optional[float] = None,
        warning_threshold: Optional[float] = None,
        reverse_thresholds: bool = False,  # True if lower is better
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._label_text = label
        self._unit = unit
        self._decimal_places = decimal_places
        self._value = 0.0
        self._good_threshold = good_threshold
        self._warning_threshold = warning_threshold
        self._reverse_thresholds = reverse_thresholds
        
        self._init_ui()
        
        logger.debug(f"MetricDisplayWidget created: {label}")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Label
        self.label = QLabel(self._label_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(9)
        self.label.setFont(font)
        self.label.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(self.label)
        
        # Value
        self.value_label = QLabel("--")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.value_label.setFont(font)
        layout.addWidget(self.value_label)
        
        self.setLayout(layout)
        self.setMinimumWidth(100)
    
    def set_value(self, value: float):
        """
        Set metric value and update display.
        
        Args:
            value: New metric value
        """
        self._value = value
        
        # Format value
        if self._decimal_places == 0:
            value_text = f"{int(value)}"
        else:
            value_text = f"{value:.{self._decimal_places}f}"
        
        if self._unit:
            value_text += f" {self._unit}"
        
        self.value_label.setText(value_text)
        
        # Update color based on thresholds
        self._update_color()
    
    def _update_color(self):
        """Update value color based on thresholds"""
        if self._good_threshold is None or self._warning_threshold is None:
            # No thresholds, use default color
            self.value_label.setStyleSheet("color: #ffffff;")
            return
        
        # Determine color based on value and thresholds
        if self._reverse_thresholds:
            # Lower is better (e.g., blink rate)
            if self._value <= self._good_threshold:
                color = "#4caf50"  # Green
            elif self._value <= self._warning_threshold:
                color = "#ffc107"  # Yellow
            else:
                color = "#f44336"  # Red
        else:
            # Higher is better (e.g., alertness)
            if self._value >= self._good_threshold:
                color = "#4caf50"  # Green
            elif self._value >= self._warning_threshold:
                color = "#ffc107"  # Yellow
            else:
                color = "#f44336"  # Red
        
        self.value_label.setStyleSheet(f"color: {color};")
    
    def clear_value(self):
        """Clear the displayed value"""
        self.value_label.setText("--")
        self.value_label.setStyleSheet("color: #aaaaaa;")


class MetricsGridWidget(QWidget):
    """
    Grid of metric displays.
    
    Displays multiple metrics in a grid layout with automatic color coding.
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._metrics = {}
        self._init_ui()
        
        logger.debug("MetricsGridWidget created")
    
    def _init_ui(self):
        """Initialize UI components"""
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        self.setLayout(self.layout)
    
    def add_metric(
        self,
        key: str,
        label: str,
        unit: str = "",
        decimal_places: int = 1,
        good_threshold: Optional[float] = None,
        warning_threshold: Optional[float] = None,
        reverse_thresholds: bool = False
    ):
        """
        Add a metric to the grid.
        
        Args:
            key: Unique identifier for the metric
            label: Display label
            unit: Unit of measurement
            decimal_places: Number of decimal places to display
            good_threshold: Threshold for good values
            warning_threshold: Threshold for warning values
            reverse_thresholds: True if lower values are better
        """
        metric_widget = MetricDisplayWidget(
            label=label,
            unit=unit,
            decimal_places=decimal_places,
            good_threshold=good_threshold,
            warning_threshold=warning_threshold,
            reverse_thresholds=reverse_thresholds
        )
        
        self._metrics[key] = metric_widget
        self.layout.addWidget(metric_widget)
        
        logger.debug(f"Added metric: {key}")
    
    def set_metric_value(self, key: str, value: float):
        """
        Set value for a specific metric.
        
        Args:
            key: Metric identifier
            value: New value
        """
        if key in self._metrics:
            self._metrics[key].set_value(value)
        else:
            logger.warning(f"Metric not found: {key}")
    
    def clear_metric(self, key: str):
        """Clear a specific metric"""
        if key in self._metrics:
            self._metrics[key].clear_value()
    
    def clear_all_metrics(self):
        """Clear all metrics"""
        for metric in self._metrics.values():
            metric.clear_value()


class DriverMetricsPanel(QWidget):
    """
    Complete driver metrics panel with predefined metrics.
    
    Displays:
    - Alertness score
    - Attention score
    - Blink rate
    - Head pose angles
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._init_ui()
        
        logger.debug("DriverMetricsPanel created")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        title = QLabel("Driver Metrics")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Create metrics grid
        self.metrics_grid = MetricsGridWidget()
        
        # Add metrics
        self.metrics_grid.add_metric(
            key='alertness',
            label='Alertness',
            unit='%',
            decimal_places=0,
            good_threshold=70.0,
            warning_threshold=50.0,
            reverse_thresholds=False
        )
        
        self.metrics_grid.add_metric(
            key='attention',
            label='Attention',
            unit='%',
            decimal_places=0,
            good_threshold=70.0,
            warning_threshold=50.0,
            reverse_thresholds=False
        )
        
        self.metrics_grid.add_metric(
            key='blink_rate',
            label='Blink Rate',
            unit='bpm',
            decimal_places=1,
            good_threshold=20.0,
            warning_threshold=30.0,
            reverse_thresholds=True  # Lower is better
        )
        
        self.metrics_grid.add_metric(
            key='head_pitch',
            label='Head Pitch',
            unit='°',
            decimal_places=1,
            good_threshold=None,  # No color coding
            warning_threshold=None
        )
        
        self.metrics_grid.add_metric(
            key='head_yaw',
            label='Head Yaw',
            unit='°',
            decimal_places=1,
            good_threshold=None,  # No color coding
            warning_threshold=None
        )
        
        self.metrics_grid.add_metric(
            key='head_roll',
            label='Head Roll',
            unit='°',
            decimal_places=1,
            good_threshold=None,  # No color coding
            warning_threshold=None
        )
        
        layout.addWidget(self.metrics_grid)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_metrics(self, driver_state: dict):
        """
        Update all metrics from driver state.
        
        Args:
            driver_state: Dictionary containing driver state data
        """
        # Alertness (derived from drowsiness score)
        if 'drowsiness' in driver_state and 'score' in driver_state['drowsiness']:
            alertness = 100 - driver_state['drowsiness']['score']
            self.metrics_grid.set_metric_value('alertness', alertness)
        
        # Attention (from readiness score or gaze)
        if 'readiness_score' in driver_state:
            self.metrics_grid.set_metric_value('attention', driver_state['readiness_score'])
        
        # Blink rate (calculate from eye state)
        if 'eye_state' in driver_state:
            # This would need to be calculated from frame rate and blink count
            # For now, use a placeholder
            blink_rate = driver_state['eye_state'].get('blink_rate', 0.0)
            self.metrics_grid.set_metric_value('blink_rate', blink_rate)
        
        # Head pose
        if 'head_pose' in driver_state:
            head_pose = driver_state['head_pose']
            self.metrics_grid.set_metric_value('head_pitch', head_pose.get('pitch', 0.0))
            self.metrics_grid.set_metric_value('head_yaw', head_pose.get('yaw', 0.0))
            self.metrics_grid.set_metric_value('head_roll', head_pose.get('roll', 0.0))
    
    def clear_metrics(self):
        """Clear all metrics"""
        self.metrics_grid.clear_all_metrics()
