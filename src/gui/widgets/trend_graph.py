"""
Trend Graph Widget

Custom widget for displaying real-time trend graphs using PyQtGraph.
"""

import logging
from typing import Optional, List, Tuple
from collections import deque
import time
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg

logger = logging.getLogger(__name__)


class TrendGraphWidget(QWidget):
    """
    Real-time trend graph widget.
    
    Features:
    - Scrolling time axis
    - Multiple data series
    - Threshold lines
    - Auto-scaling
    - 60 second history
    """
    
    def __init__(
        self,
        title: str,
        y_label: str = "",
        y_range: Optional[Tuple[float, float]] = None,
        history_seconds: int = 60,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self._title = title
        self._y_label = y_label
        self._y_range = y_range
        self._history_seconds = history_seconds
        
        # Data storage
        self._max_points = history_seconds * 30  # Assuming 30 FPS
        self._time_data = deque(maxlen=self._max_points)
        self._value_data = deque(maxlen=self._max_points)
        self._start_time = time.time()
        
        # Threshold lines
        self._threshold_lines = []
        
        self._init_ui()
        
        logger.debug(f"TrendGraphWidget created: {title}")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel(self._title)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Configure axes
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        if self._y_label:
            self.plot_widget.setLabel('left', self._y_label)
        
        # Set Y range if specified
        if self._y_range:
            self.plot_widget.setYRange(self._y_range[0], self._y_range[1])
        
        # Create data curve
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#2196f3', width=2),
            name='Value'
        )
        
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
        self.setMinimumHeight(150)
    
    def add_threshold_line(self, value: float, label: str = "", color: str = '#ff9800'):
        """
        Add a horizontal threshold line.
        
        Args:
            value: Y value for the threshold
            label: Optional label for the threshold
            color: Line color (hex string)
        """
        line = pg.InfiniteLine(
            pos=value,
            angle=0,
            pen=pg.mkPen(color=color, width=2, style=Qt.PenStyle.DashLine),
            label=label,
            labelOpts={'position': 0.95, 'color': color}
        )
        self.plot_widget.addItem(line)
        self._threshold_lines.append(line)
        
        logger.debug(f"Added threshold line at {value}: {label}")
    
    def add_data_point(self, value: float):
        """
        Add a new data point.
        
        Args:
            value: Y value to add
        """
        current_time = time.time() - self._start_time
        
        self._time_data.append(current_time)
        self._value_data.append(value)
        
        # Update plot
        self._update_plot()
    
    def _update_plot(self):
        """Update the plot with current data"""
        if len(self._time_data) > 0:
            # Convert deques to lists for plotting
            time_list = list(self._time_data)
            value_list = list(self._value_data)
            
            # Update curve
            self.curve.setData(time_list, value_list)
            
            # Auto-scroll X axis to show last N seconds
            if time_list[-1] > self._history_seconds:
                self.plot_widget.setXRange(
                    time_list[-1] - self._history_seconds,
                    time_list[-1]
                )
    
    def clear_data(self):
        """Clear all data points"""
        self._time_data.clear()
        self._value_data.clear()
        self._start_time = time.time()
        self.curve.setData([], [])
        
        logger.debug(f"Cleared data for {self._title}")


class DriverTrendGraphsPanel(QWidget):
    """
    Panel with multiple trend graphs for driver metrics.
    
    Displays:
    - Alertness over time
    - Attention over time
    - Readiness score over time
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._init_ui()
        
        logger.debug("DriverTrendGraphsPanel created")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Driver Metrics Trends")
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Alertness graph
        self.alertness_graph = TrendGraphWidget(
            title="Alertness",
            y_label="Score",
            y_range=(0, 100),
            history_seconds=60
        )
        self.alertness_graph.add_threshold_line(70, "Good", '#4caf50')
        self.alertness_graph.add_threshold_line(50, "Warning", '#ffc107')
        layout.addWidget(self.alertness_graph)
        
        # Attention graph
        self.attention_graph = TrendGraphWidget(
            title="Attention",
            y_label="Score",
            y_range=(0, 100),
            history_seconds=60
        )
        self.attention_graph.add_threshold_line(70, "Good", '#4caf50')
        self.attention_graph.add_threshold_line(50, "Warning", '#ffc107')
        layout.addWidget(self.attention_graph)
        
        # Readiness graph
        self.readiness_graph = TrendGraphWidget(
            title="Driver Readiness",
            y_label="Score",
            y_range=(0, 100),
            history_seconds=60
        )
        self.readiness_graph.add_threshold_line(70, "Good", '#4caf50')
        self.readiness_graph.add_threshold_line(50, "Warning", '#ffc107')
        layout.addWidget(self.readiness_graph)
        
        self.setLayout(layout)
    
    def update_graphs(self, driver_state: dict):
        """
        Update all graphs with new data.
        
        Args:
            driver_state: Dictionary containing driver state data
        """
        # Alertness (derived from drowsiness)
        if 'drowsiness' in driver_state and 'score' in driver_state['drowsiness']:
            alertness = 100 - driver_state['drowsiness']['score']
            self.alertness_graph.add_data_point(alertness)
        
        # Attention (could be derived from gaze or distraction)
        if 'distraction' in driver_state:
            # Calculate attention as inverse of distraction
            distraction_conf = driver_state['distraction'].get('confidence', 0.0)
            attention = 100 - (distraction_conf * 100)
            self.attention_graph.add_data_point(attention)
        
        # Readiness score
        if 'readiness_score' in driver_state:
            self.readiness_graph.add_data_point(driver_state['readiness_score'])
    
    def clear_graphs(self):
        """Clear all graph data"""
        self.alertness_graph.clear_data()
        self.attention_graph.clear_data()
        self.readiness_graph.clear_data()
