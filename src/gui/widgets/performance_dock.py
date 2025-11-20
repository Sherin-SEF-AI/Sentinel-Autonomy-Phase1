"""
Performance Monitoring Dock Widget

Provides real-time performance monitoring including FPS, latency, module breakdown,
and resource usage displays.
"""

import logging
from collections import deque
from typing import Dict, List, Optional
import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QTabWidget, QDockWidget
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor
import pyqtgraph as pg
import numpy as np

logger = logging.getLogger(__name__)


class FPSGraphWidget(QWidget):
    """
    Widget displaying FPS over time with target line.
    
    Features:
    - Real-time FPS plotting over last 60 seconds
    - 30 FPS target line
    - Color coding below threshold (red when < 30 FPS)
    - Auto-scaling Y-axis
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        logger.debug("Initializing FPS Graph Widget")
        
        # Data storage (60 seconds at 1 Hz = 60 points)
        self.max_points = 60
        self.timestamps = deque(maxlen=self.max_points)
        self.fps_values = deque(maxlen=self.max_points)
        
        # Target FPS
        self.target_fps = 30.0
        
        # Initialize UI
        self._init_ui()
        
        logger.info(f"FPS Graph Widget initialized: max_points={self.max_points}, target_fps={self.target_fps}")
    
    def _init_ui(self):
        """Initialize the UI components"""
        logger.debug("Initializing FPS Graph UI components")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setTitle("Frames Per Second", color='w', size='12pt')
        self.plot_widget.setLabel('left', 'FPS', color='w')
        self.plot_widget.setLabel('bottom', 'Time (seconds ago)', color='w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 40, padding=0.1)
        
        logger.debug("FPS plot widget configured")
        
        # Create FPS line plot
        self.fps_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#00ff00', width=2),
            name='FPS'
        )
        
        # Create target line (30 FPS)
        self.target_line = pg.InfiniteLine(
            pos=self.target_fps,
            angle=0,
            pen=pg.mkPen(color='#ffaa00', width=2, style=Qt.PenStyle.DashLine),
            label='Target (30 FPS)',
            labelOpts={'position': 0.95, 'color': '#ffaa00'}
        )
        self.plot_widget.addItem(self.target_line)
        
        # Add legend
        self.plot_widget.addLegend()
        
        layout.addWidget(self.plot_widget)
        
        # Current FPS label
        self.fps_label = QLabel("Current FPS: --")
        self.fps_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00ff00;")
        self.fps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.fps_label)
    
    def update_fps(self, fps: float):
        """
        Update FPS graph with new value.
        
        Args:
            fps: Current frames per second
        """
        current_time = time.time()
        
        # Add new data point
        self.timestamps.append(current_time)
        self.fps_values.append(fps)
        
        # Update plot
        if len(self.timestamps) > 1:
            # Convert timestamps to relative time (seconds ago)
            latest_time = self.timestamps[-1]
            relative_times = [latest_time - t for t in self.timestamps]
            relative_times.reverse()  # Reverse so 0 is now
            fps_array = list(self.fps_values)
            fps_array.reverse()
            
            # Update curve
            self.fps_curve.setData(relative_times, fps_array)
        
        # Update label with color coding
        if fps < self.target_fps:
            color = '#ff0000'  # Red below target
            logger.warning(f"FPS below target: current={fps:.1f}, target={self.target_fps}")
        else:
            color = '#00ff00'  # Green at or above target
        
        self.fps_label.setText(f"Current FPS: {fps:.1f}")
        self.fps_label.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {color};")
        
        logger.debug(f"FPS updated: value={fps:.1f}, data_points={len(self.fps_values)}")
    
    def clear(self):
        """Clear all data"""
        self.timestamps.clear()
        self.fps_values.clear()
        self.fps_curve.setData([], [])
        self.fps_label.setText("Current FPS: --")
        self.fps_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #00ff00;")
        logger.debug("FPS graph cleared")


class LatencyGraphWidget(QWidget):
    """
    Widget displaying end-to-end latency over time with threshold line.
    
    Features:
    - Real-time latency plotting over last 60 seconds
    - 100ms threshold line
    - Show p95 latency value
    - Color coding for violations (red when > 100ms)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        logger.debug("Initializing Latency Graph Widget")
        
        # Data storage (60 seconds at 1 Hz = 60 points)
        self.max_points = 60
        self.timestamps = deque(maxlen=self.max_points)
        self.latency_values = deque(maxlen=self.max_points)
        
        # Threshold
        self.threshold_ms = 100.0
        
        # Initialize UI
        self._init_ui()
        
        logger.info(f"Latency Graph Widget initialized: max_points={self.max_points}, threshold={self.threshold_ms}ms")
    
    def _init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setTitle("End-to-End Latency", color='w', size='12pt')
        self.plot_widget.setLabel('left', 'Latency (ms)', color='w')
        self.plot_widget.setLabel('bottom', 'Time (seconds ago)', color='w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(0, 150, padding=0.1)
        
        # Create latency line plot
        self.latency_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#00aaff', width=2),
            name='Latency'
        )
        
        # Create threshold line (100ms)
        self.threshold_line = pg.InfiniteLine(
            pos=self.threshold_ms,
            angle=0,
            pen=pg.mkPen(color='#ff0000', width=2, style=Qt.PenStyle.DashLine),
            label='Threshold (100ms)',
            labelOpts={'position': 0.95, 'color': '#ff0000'}
        )
        self.plot_widget.addItem(self.threshold_line)
        
        # Add legend
        self.plot_widget.addLegend()
        
        layout.addWidget(self.plot_widget)
        
        # Statistics labels
        stats_layout = QHBoxLayout()
        
        self.current_label = QLabel("Current: --")
        self.current_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #00aaff;")
        stats_layout.addWidget(self.current_label)
        
        self.p95_label = QLabel("P95: --")
        self.p95_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffaa00;")
        stats_layout.addWidget(self.p95_label)
        
        self.violations_label = QLabel("Violations: 0")
        self.violations_label.setStyleSheet("font-size: 12pt; color: #ff0000;")
        stats_layout.addWidget(self.violations_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
    
    def update_latency(self, latency_ms: float):
        """
        Update latency graph with new value.
        
        Args:
            latency_ms: Current end-to-end latency in milliseconds
        """
        current_time = time.time()
        
        # Add new data point
        self.timestamps.append(current_time)
        self.latency_values.append(latency_ms)
        
        # Update plot
        if len(self.timestamps) > 1:
            # Convert timestamps to relative time (seconds ago)
            latest_time = self.timestamps[-1]
            relative_times = [latest_time - t for t in self.timestamps]
            relative_times.reverse()  # Reverse so 0 is now
            latency_array = list(self.latency_values)
            latency_array.reverse()
            
            # Update curve with color coding
            if latency_ms > self.threshold_ms:
                # Red for violations
                pen = pg.mkPen(color='#ff0000', width=2)
                logger.warning(f"Latency threshold exceeded: current={latency_ms:.1f}ms, threshold={self.threshold_ms}ms")
            else:
                # Blue for normal
                pen = pg.mkPen(color='#00aaff', width=2)
            
            self.latency_curve.setData(relative_times, latency_array)
            self.latency_curve.setPen(pen)
        
        # Calculate statistics
        if len(self.latency_values) > 0:
            latency_list = list(self.latency_values)
            p95_latency = np.percentile(latency_list, 95)
            violations = sum(1 for lat in latency_list if lat > self.threshold_ms)
            
            # Update labels with color coding
            if latency_ms > self.threshold_ms:
                current_color = '#ff0000'  # Red for violation
            else:
                current_color = '#00aaff'  # Blue for normal
            
            self.current_label.setText(f"Current: {latency_ms:.1f}ms")
            self.current_label.setStyleSheet(f"font-size: 12pt; font-weight: bold; color: {current_color};")
            
            if p95_latency > self.threshold_ms:
                p95_color = '#ff0000'  # Red if p95 exceeds threshold
                logger.warning(f"P95 latency exceeds threshold: p95={p95_latency:.1f}ms, threshold={self.threshold_ms}ms")
            else:
                p95_color = '#ffaa00'  # Orange for normal
            
            self.p95_label.setText(f"P95: {p95_latency:.1f}ms")
            self.p95_label.setStyleSheet(f"font-size: 12pt; font-weight: bold; color: {p95_color};")
            
            self.violations_label.setText(f"Violations: {violations}/{len(latency_list)}")
            
            logger.debug(f"Latency updated: current={latency_ms:.1f}ms, p95={p95_latency:.1f}ms, violations={violations}, data_points={len(latency_list)}")
        else:
            logger.debug(f"Latency updated: {latency_ms:.1f}ms")
    
    def clear(self):
        """Clear all data"""
        self.timestamps.clear()
        self.latency_values.clear()
        self.latency_curve.setData([], [])
        self.current_label.setText("Current: --")
        self.current_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #00aaff;")
        self.p95_label.setText("P95: --")
        self.p95_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffaa00;")
        self.violations_label.setText("Violations: 0")
        logger.debug("Latency graph cleared")


class ModuleBreakdownWidget(QWidget):
    """
    Widget displaying module timing breakdown as stacked bar chart.
    
    Features:
    - Stacked bar chart showing time spent in each pipeline stage
    - Updates at 1 Hz
    - Tooltips with exact values
    - Color-coded modules
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        logger.debug("Initializing Module Breakdown Widget")
        
        # Module colors
        self.module_colors = {
            'Camera': '#ff6b6b',
            'BEV': '#4ecdc4',
            'Segmentation': '#45b7d1',
            'Detection': '#96ceb4',
            'DMS': '#ffeaa7',
            'Intelligence': '#dfe6e9',
            'Alerts': '#fd79a8',
            'Other': '#636e72'
        }
        
        # Data storage
        self.module_timings = {}
        
        # Initialize UI
        self._init_ui()
        
        logger.info(f"Module Breakdown Widget initialized: modules={len(self.module_colors)}")
    
    def _init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')
        self.plot_widget.setTitle("Module Timing Breakdown", color='w', size='12pt')
        self.plot_widget.setLabel('left', 'Time (ms)', color='w')
        self.plot_widget.setLabel('bottom', 'Pipeline Stage', color='w')
        self.plot_widget.showGrid(y=True, alpha=0.3)
        
        # Configure x-axis
        self.plot_widget.getAxis('bottom').setTicks([])
        
        layout.addWidget(self.plot_widget)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Modules:"))
        
        for module, color in self.module_colors.items():
            label = QLabel(f"■ {module}")
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            legend_layout.addWidget(label)
        
        legend_layout.addStretch()
        layout.addLayout(legend_layout)
        
        # Total time label
        self.total_label = QLabel("Total: --")
        self.total_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffffff;")
        self.total_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.total_label)
    
    def update_timings(self, timings: Dict[str, float]):
        """
        Update module breakdown with new timings.
        
        Args:
            timings: Dictionary mapping module names to execution times (ms)
        """
        self.module_timings = timings
        
        # Clear previous plot
        self.plot_widget.clear()
        
        if not timings:
            logger.debug("No module timings to display")
            return
        
        # Calculate total time
        total_time = sum(timings.values())
        
        # Create stacked bar chart
        y_offset = 0
        bar_width = 0.6
        x_pos = 0
        
        for module, time_ms in timings.items():
            color = self.module_colors.get(module, self.module_colors['Other'])
            
            # Create bar segment
            bar = pg.BarGraphItem(
                x=[x_pos],
                height=[time_ms],
                width=bar_width,
                y0=[y_offset],
                brush=color,
                pen=pg.mkPen(color='w', width=1)
            )
            
            # Add tooltip (module name and time)
            bar.setToolTip(f"{module}: {time_ms:.2f}ms")
            
            self.plot_widget.addItem(bar)
            
            # Add text label
            text = pg.TextItem(
                f"{module}\n{time_ms:.1f}ms",
                anchor=(0.5, 0.5),
                color='w'
            )
            text.setPos(x_pos, y_offset + time_ms / 2)
            self.plot_widget.addItem(text)
            
            y_offset += time_ms
        
        # Update total label with color coding
        if total_time > 100:
            color = '#ff0000'  # Red if exceeds 100ms
            logger.warning(f"Total pipeline latency exceeds 100ms: total={total_time:.1f}ms")
        elif total_time > 80:
            color = '#ffaa00'  # Orange if approaching limit
            logger.info(f"Total pipeline latency approaching limit: total={total_time:.1f}ms")
        else:
            color = '#00ff00'  # Green if good
        
        self.total_label.setText(f"Total: {total_time:.1f}ms")
        self.total_label.setStyleSheet(f"font-size: 12pt; font-weight: bold; color: {color};")
        
        # Adjust y-axis range
        self.plot_widget.setYRange(0, max(total_time * 1.1, 100), padding=0)
        
        # Log detailed breakdown
        module_breakdown = ", ".join([f"{m}={t:.1f}ms" for m, t in timings.items()])
        logger.debug(f"Module timings updated: total={total_time:.1f}ms, breakdown=[{module_breakdown}]")
    
    def clear(self):
        """Clear all data"""
        self.module_timings.clear()
        self.plot_widget.clear()
        self.total_label.setText("Total: --")
        self.total_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #ffffff;")
        logger.debug("Module breakdown cleared")


class ResourceUsageWidget(QWidget):
    """
    Widget displaying GPU and CPU resource usage.
    
    Features:
    - GPU memory gauge (max 8GB)
    - CPU usage gauge (max 60%)
    - Show current and peak values
    - Color coding based on thresholds
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        logger.debug("Initializing Resource Usage Widget")
        
        # Thresholds
        self.gpu_max_mb = 8192  # 8GB
        self.cpu_max_percent = 60.0
        
        # Peak tracking
        self.gpu_peak_mb = 0.0
        self.cpu_peak_percent = 0.0
        
        # Initialize UI
        self._init_ui()
        
        logger.info(f"Resource Usage Widget initialized: gpu_max={self.gpu_max_mb}MB, cpu_max={self.cpu_max_percent}%")
    
    def _init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # GPU Memory Section
        gpu_group = QGroupBox("GPU Memory")
        gpu_layout = QVBoxLayout()
        
        # GPU gauge (using circular gauge from existing widgets)
        from .circular_gauge import CircularGaugeWidget
        self.gpu_gauge = CircularGaugeWidget(
            min_value=0,
            max_value=self.gpu_max_mb,
            title="GPU Memory (MB)"
        )
        self.gpu_gauge.set_color_zones(
            green_zone=(0, self.gpu_max_mb * 0.7),      # Green: 0-70%
            yellow_zone=(self.gpu_max_mb * 0.7, self.gpu_max_mb * 0.85),  # Yellow: 70-85%
            red_zone=(self.gpu_max_mb * 0.85, self.gpu_max_mb)  # Red: 85-100%
        )
        gpu_layout.addWidget(self.gpu_gauge)
        
        # GPU stats
        gpu_stats_layout = QHBoxLayout()
        self.gpu_current_label = QLabel("Current: --")
        self.gpu_current_label.setStyleSheet("font-size: 11pt;")
        gpu_stats_layout.addWidget(self.gpu_current_label)
        
        self.gpu_peak_label = QLabel("Peak: --")
        self.gpu_peak_label.setStyleSheet("font-size: 11pt; color: #ffaa00;")
        gpu_stats_layout.addWidget(self.gpu_peak_label)
        
        gpu_stats_layout.addStretch()
        gpu_layout.addLayout(gpu_stats_layout)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # CPU Usage Section
        cpu_group = QGroupBox("CPU Usage")
        cpu_layout = QVBoxLayout()
        
        # CPU gauge
        self.cpu_gauge = CircularGaugeWidget(
            min_value=0,
            max_value=100,
            title="CPU Usage (%)"
        )
        self.cpu_gauge.set_color_zones(
            green_zone=(0, self.cpu_max_percent * 0.8),      # Green: 0-48%
            yellow_zone=(self.cpu_max_percent * 0.8, self.cpu_max_percent),  # Yellow: 48-60%
            red_zone=(self.cpu_max_percent, 100)  # Red: 60-100%
        )
        cpu_layout.addWidget(self.cpu_gauge)
        
        # CPU stats
        cpu_stats_layout = QHBoxLayout()
        self.cpu_current_label = QLabel("Current: --")
        self.cpu_current_label.setStyleSheet("font-size: 11pt;")
        cpu_stats_layout.addWidget(self.cpu_current_label)
        
        self.cpu_peak_label = QLabel("Peak: --")
        self.cpu_peak_label.setStyleSheet("font-size: 11pt; color: #ffaa00;")
        cpu_stats_layout.addWidget(self.cpu_peak_label)
        
        cpu_stats_layout.addStretch()
        cpu_layout.addLayout(cpu_stats_layout)
        
        cpu_group.setLayout(cpu_layout)
        layout.addWidget(cpu_group)
        
        layout.addStretch()
    
    def update_resources(self, gpu_memory_mb: float, cpu_percent: float):
        """
        Update resource usage displays.
        
        Args:
            gpu_memory_mb: GPU memory usage in MB
            cpu_percent: CPU usage percentage
        """
        # Update GPU
        self.gpu_gauge.set_value(gpu_memory_mb)
        
        # Track peak
        peak_updated = False
        if gpu_memory_mb > self.gpu_peak_mb:
            self.gpu_peak_mb = gpu_memory_mb
            peak_updated = True
            logger.info(f"New GPU memory peak: {self.gpu_peak_mb:.0f}MB")
        
        # Update labels with color coding
        gpu_percent = (gpu_memory_mb / self.gpu_max_mb) * 100
        if gpu_percent > 85:
            gpu_color = '#ff0000'
            logger.warning(f"GPU memory usage critical: {gpu_memory_mb:.0f}MB ({gpu_percent:.1f}%), threshold=85%")
        elif gpu_percent > 70:
            gpu_color = '#ffaa00'
            logger.info(f"GPU memory usage high: {gpu_memory_mb:.0f}MB ({gpu_percent:.1f}%)")
        else:
            gpu_color = '#00ff00'
        
        self.gpu_current_label.setText(f"Current: {gpu_memory_mb:.0f} MB ({gpu_percent:.1f}%)")
        self.gpu_current_label.setStyleSheet(f"font-size: 11pt; color: {gpu_color};")
        
        self.gpu_peak_label.setText(f"Peak: {self.gpu_peak_mb:.0f} MB")
        
        # Update CPU
        self.cpu_gauge.set_value(cpu_percent)
        
        # Track peak
        if cpu_percent > self.cpu_peak_percent:
            self.cpu_peak_percent = cpu_percent
            peak_updated = True
            logger.info(f"New CPU usage peak: {self.cpu_peak_percent:.1f}%")
        
        # Update labels with color coding
        if cpu_percent > self.cpu_max_percent:
            cpu_color = '#ff0000'
            logger.warning(f"CPU usage exceeds target: {cpu_percent:.1f}%, target={self.cpu_max_percent}%")
        elif cpu_percent > self.cpu_max_percent * 0.8:
            cpu_color = '#ffaa00'
            logger.info(f"CPU usage approaching limit: {cpu_percent:.1f}%")
        else:
            cpu_color = '#00ff00'
        
        self.cpu_current_label.setText(f"Current: {cpu_percent:.1f}%")
        self.cpu_current_label.setStyleSheet(f"font-size: 11pt; color: {cpu_color};")
        
        self.cpu_peak_label.setText(f"Peak: {self.cpu_peak_percent:.1f}%")
        
        logger.debug(f"Resources updated: GPU={gpu_memory_mb:.0f}MB ({gpu_percent:.1f}%), CPU={cpu_percent:.1f}%, peak_updated={peak_updated}")
    
    def clear(self):
        """Clear all data"""
        self.gpu_gauge.set_value(0)
        self.cpu_gauge.set_value(0)
        self.gpu_peak_mb = 0.0
        self.cpu_peak_percent = 0.0
        self.gpu_current_label.setText("Current: --")
        self.gpu_current_label.setStyleSheet("font-size: 11pt;")
        self.gpu_peak_label.setText("Peak: --")
        self.cpu_current_label.setText("Current: --")
        self.cpu_current_label.setStyleSheet("font-size: 11pt;")
        self.cpu_peak_label.setText("Peak: --")
        logger.debug("Resource usage cleared")


class PerformanceLoggingWidget(QWidget):
    """
    Widget for performance logging controls and summary.
    
    Features:
    - Log performance metrics to file
    - Export performance reports
    - Generate performance summary
    - Control logging on/off
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        logger.debug("Initializing Performance Logging Widget")
        
        # Logging state
        self.logging_enabled = False
        self.log_file = None
        self.log_count = 0
        
        # Performance data accumulation
        self.fps_history = []
        self.latency_history = []
        
        # Initialize UI
        self._init_ui()
        
        logger.info("Performance Logging Widget initialized")
    
    def _init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Logging controls
        controls_group = QGroupBox("Logging Controls")
        controls_layout = QVBoxLayout()
        
        # Start/Stop logging button
        from PyQt6.QtWidgets import QPushButton
        self.toggle_logging_btn = QPushButton("Start Logging")
        self.toggle_logging_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
        """)
        self.toggle_logging_btn.clicked.connect(self._toggle_logging)
        controls_layout.addWidget(self.toggle_logging_btn)
        
        # Log file path
        self.log_path_label = QLabel("Log file: Not logging")
        self.log_path_label.setStyleSheet("font-size: 10pt; color: #888888;")
        self.log_path_label.setWordWrap(True)
        controls_layout.addWidget(self.log_path_label)
        
        # Log count
        self.log_count_label = QLabel("Entries logged: 0")
        self.log_count_label.setStyleSheet("font-size: 10pt;")
        controls_layout.addWidget(self.log_count_label)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        self.export_report_btn = QPushButton("Export Performance Report")
        self.export_report_btn.clicked.connect(self._export_report)
        export_layout.addWidget(self.export_report_btn)
        
        self.export_csv_btn = QPushButton("Export Raw Data (CSV)")
        self.export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(self.export_csv_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Performance summary
        summary_group = QGroupBox("Performance Summary")
        summary_layout = QVBoxLayout()
        
        from PyQt6.QtWidgets import QTextEdit
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMaximumHeight(200)
        self.summary_text.setStyleSheet("font-family: monospace; font-size: 10pt;")
        self._update_summary()
        summary_layout.addWidget(self.summary_text)
        
        self.refresh_summary_btn = QPushButton("Refresh Summary")
        self.refresh_summary_btn.clicked.connect(self._update_summary)
        summary_layout.addWidget(self.refresh_summary_btn)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        layout.addStretch()
    
    def _toggle_logging(self):
        """Toggle performance logging on/off"""
        if self.logging_enabled:
            self._stop_logging()
        else:
            self._start_logging()
    
    def _start_logging(self):
        """Start logging performance metrics"""
        import os
        from datetime import datetime
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs/performance', exist_ok=True)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'logs/performance/perf_{timestamp}.log'
        
        try:
            self.log_file = open(log_filename, 'w')
            # Write header
            self.log_file.write("timestamp,fps,latency_ms,gpu_memory_mb,cpu_percent\n")
            self.log_file.flush()
            
            self.logging_enabled = True
            self.log_count = 0
            
            # Update UI
            self.toggle_logging_btn.setText("Stop Logging")
            self.toggle_logging_btn.setStyleSheet("""
                QPushButton {
                    background-color: #aa0000;
                    color: white;
                    font-size: 14pt;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #cc0000;
                }
            """)
            self.log_path_label.setText(f"Log file: {log_filename}")
            self.log_path_label.setStyleSheet("font-size: 10pt; color: #00ff00;")
            
            logger.info(f"Performance logging started: {log_filename}")
            
        except Exception as e:
            logger.error(f"Failed to start logging: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Logging Error", f"Failed to start logging: {e}")
    
    def _stop_logging(self):
        """Stop logging performance metrics"""
        if self.log_file:
            self.log_file.close()
            self.log_file = None
        
        self.logging_enabled = False
        
        # Update UI
        self.toggle_logging_btn.setText("Start Logging")
        self.toggle_logging_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
        """)
        self.log_path_label.setText("Log file: Not logging")
        self.log_path_label.setStyleSheet("font-size: 10pt; color: #888888;")
        
        logger.info("Performance logging stopped")
    
    def log_metrics(self, fps: float, latency_ms: float, gpu_memory_mb: float, cpu_percent: float):
        """
        Log performance metrics to file.
        
        Args:
            fps: Current FPS
            latency_ms: Current latency in ms
            gpu_memory_mb: GPU memory usage in MB
            cpu_percent: CPU usage percentage
        """
        # Store in history for summary
        self.fps_history.append(fps)
        self.latency_history.append(latency_ms)
        
        # Keep only last 1000 entries in memory
        if len(self.fps_history) > 1000:
            self.fps_history.pop(0)
            self.latency_history.pop(0)
        
        # Write to log file if logging enabled
        if self.logging_enabled and self.log_file:
            try:
                from datetime import datetime
                timestamp = datetime.now().isoformat()
                self.log_file.write(f"{timestamp},{fps:.2f},{latency_ms:.2f},{gpu_memory_mb:.1f},{cpu_percent:.1f}\n")
                self.log_file.flush()
                
                self.log_count += 1
                self.log_count_label.setText(f"Entries logged: {self.log_count}")
                
                # Log milestone every 100 entries
                if self.log_count % 100 == 0:
                    logger.info(f"Performance logging milestone: {self.log_count} entries logged")
            except Exception as e:
                logger.error(f"Failed to write performance log entry: {e}")
                self._stop_logging()
    
    def _export_report(self):
        """Export performance report as text file"""
        from PyQt6.QtWidgets import QFileDialog
        from datetime import datetime
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Performance Report",
            f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("SENTINEL Performance Report\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(self.summary_text.toPlainText())
                
                logger.info(f"Performance report exported: {filename}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Export Success", f"Report exported to:\n{filename}")
                
            except Exception as e:
                logger.error(f"Failed to export report: {e}")
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Export Error", f"Failed to export report: {e}")
    
    def _export_csv(self):
        """Export raw performance data as CSV"""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from datetime import datetime
        
        if not self.fps_history:
            QMessageBox.warning(self, "No Data", "No performance data to export")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Performance Data",
            f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("index,fps,latency_ms\n")
                    for i, (fps, latency) in enumerate(zip(self.fps_history, self.latency_history)):
                        f.write(f"{i},{fps:.2f},{latency:.2f}\n")
                
                logger.info(f"Performance data exported: {filename}")
                QMessageBox.information(self, "Export Success", f"Data exported to:\n{filename}")
                
            except Exception as e:
                logger.error(f"Failed to export data: {e}")
                QMessageBox.warning(self, "Export Error", f"Failed to export data: {e}")
    
    def _update_summary(self):
        """Update performance summary"""
        if not self.fps_history or not self.latency_history:
            self.summary_text.setPlainText("No performance data available yet.\nStart the system to collect metrics.")
            return
        
        # Calculate statistics
        fps_array = np.array(self.fps_history)
        latency_array = np.array(self.latency_history)
        
        summary = []
        summary.append("FPS Statistics:")
        summary.append(f"  Mean:   {np.mean(fps_array):.2f}")
        summary.append(f"  Median: {np.median(fps_array):.2f}")
        summary.append(f"  Min:    {np.min(fps_array):.2f}")
        summary.append(f"  Max:    {np.max(fps_array):.2f}")
        summary.append(f"  Std:    {np.std(fps_array):.2f}")
        summary.append("")
        
        summary.append("Latency Statistics (ms):")
        summary.append(f"  Mean:   {np.mean(latency_array):.2f}")
        summary.append(f"  Median: {np.median(latency_array):.2f}")
        summary.append(f"  P95:    {np.percentile(latency_array, 95):.2f}")
        summary.append(f"  P99:    {np.percentile(latency_array, 99):.2f}")
        summary.append(f"  Min:    {np.min(latency_array):.2f}")
        summary.append(f"  Max:    {np.max(latency_array):.2f}")
        summary.append("")
        
        # Performance assessment
        mean_fps = np.mean(fps_array)
        p95_latency = np.percentile(latency_array, 95)
        
        summary.append("Performance Assessment:")
        if mean_fps >= 30:
            summary.append(f"  FPS: ✓ PASS (target: ≥30 FPS)")
        else:
            summary.append(f"  FPS: ✗ FAIL (target: ≥30 FPS)")
        
        if p95_latency <= 100:
            summary.append(f"  Latency: ✓ PASS (target: ≤100ms at P95)")
        else:
            summary.append(f"  Latency: ✗ FAIL (target: ≤100ms at P95)")
        
        summary.append("")
        summary.append(f"Total samples: {len(self.fps_history)}")
        
        self.summary_text.setPlainText("\n".join(summary))
    
    def clear(self):
        """Clear all data"""
        if self.logging_enabled:
            self._stop_logging()
        
        self.fps_history.clear()
        self.latency_history.clear()
        self.log_count = 0
        self.log_count_label.setText("Entries logged: 0")
        self._update_summary()
        logger.debug("Performance logging cleared")


class PerformanceDockWidget(QDockWidget):
    """
    Performance monitoring dock widget.

    Displays:
    - FPS graph over last 60 seconds
    - Latency graph with threshold
    - Module breakdown (stacked bar chart)
    - Resource usage (GPU/CPU gauges)
    - Performance logging controls
    """

    def __init__(self, parent=None):
        super().__init__("Performance Monitor", parent)

        logger.info("Initializing Performance Dock Widget")

        # Create central widget for the dock
        central_widget = QWidget()
        self.setWidget(central_widget)

        # Initialize UI
        self._init_ui_in_widget(central_widget)
        
        # Update timer (1 Hz for performance metrics)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._generate_mock_data)
        
        logger.info("Performance Dock Widget initialized: tabs=5, update_rate=1Hz")

    def _init_ui_in_widget(self, widget):
        """Initialize the UI components in the given widget"""
        layout = QVBoxLayout(widget)

        # Create tab widget for different performance views
        self.tab_widget = QTabWidget()
        
        # FPS Tab
        self.fps_widget = FPSGraphWidget()
        self.tab_widget.addTab(self.fps_widget, "FPS")
        
        # Latency Tab
        self.latency_widget = LatencyGraphWidget()
        self.tab_widget.addTab(self.latency_widget, "Latency")
        
        # Module Breakdown Tab
        self.module_widget = ModuleBreakdownWidget()
        self.tab_widget.addTab(self.module_widget, "Modules")
        
        # Resource Usage Tab
        self.resource_widget = ResourceUsageWidget()
        self.tab_widget.addTab(self.resource_widget, "Resources")
        
        # Performance Logging Tab
        self.logging_widget = PerformanceLoggingWidget()
        self.tab_widget.addTab(self.logging_widget, "Logging")
        
        layout.addWidget(self.tab_widget)
    
    def start_monitoring(self):
        """Start performance monitoring"""
        logger.info("Starting performance monitoring: update_rate=1Hz")
        self.update_timer.start(1000)  # Update at 1 Hz
        logger.info("Performance monitoring started successfully")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        logger.info("Stopping performance monitoring")
        self.update_timer.stop()
        logger.info("Performance monitoring stopped successfully")
    
    def clear_all(self):
        """Clear all performance data"""
        logger.info("Clearing all performance data")
        self.fps_widget.clear()
        self.latency_widget.clear()
        self.module_widget.clear()
        self.resource_widget.clear()
        self.logging_widget.clear()
        logger.info("All performance data cleared successfully")
    
    def _generate_mock_data(self):
        """Generate mock data for testing (will be replaced with real data)"""
        import random
        
        # Simulate FPS varying around 30
        fps = 30 + random.uniform(-5, 5)
        self.fps_widget.update_fps(fps)
        
        # Simulate latency varying around 80ms with occasional spikes
        if random.random() < 0.1:  # 10% chance of spike
            latency = random.uniform(100, 130)
        else:
            latency = random.uniform(60, 95)
        self.latency_widget.update_latency(latency)
        
        # Simulate module timings
        timings = {
            'Camera': random.uniform(3, 7),
            'BEV': random.uniform(12, 18),
            'Segmentation': random.uniform(12, 18),
            'Detection': random.uniform(15, 25),
            'DMS': random.uniform(20, 30),
            'Intelligence': random.uniform(8, 12),
            'Alerts': random.uniform(3, 7)
        }
        self.module_widget.update_timings(timings)
        
        # Simulate resource usage
        gpu_memory = random.uniform(3000, 7500)  # 3-7.5 GB
        cpu_usage = random.uniform(30, 70)  # 30-70%
        self.resource_widget.update_resources(gpu_memory, cpu_usage)
        
        # Log metrics
        self.logging_widget.log_metrics(fps, latency, gpu_memory, cpu_usage)
    
    def update_fps(self, fps: float):
        """
        Update FPS display.
        
        Args:
            fps: Current frames per second
        """
        self.fps_widget.update_fps(fps)
    
    def update_latency(self, latency_ms: float):
        """
        Update latency display.
        
        Args:
            latency_ms: End-to-end latency in milliseconds
        """
        self.latency_widget.update_latency(latency_ms)
    
    def update_module_timings(self, timings: Dict[str, float]):
        """
        Update module breakdown display.
        
        Args:
            timings: Dictionary mapping module names to execution times (ms)
        """
        self.module_widget.update_timings(timings)
    
    def update_resources(self, gpu_memory_mb: float, cpu_percent: float):
        """
        Update resource usage displays.
        
        Args:
            gpu_memory_mb: GPU memory usage in MB
            cpu_percent: CPU usage percentage
        """
        self.resource_widget.update_resources(gpu_memory_mb, cpu_percent)
    
    def update_all_metrics(self, fps: float, latency_ms: float, 
                          module_timings: Dict[str, float],
                          gpu_memory_mb: float, cpu_percent: float):
        """
        Update all performance metrics at once.
        
        Args:
            fps: Current frames per second
            latency_ms: End-to-end latency in milliseconds
            module_timings: Dictionary mapping module names to execution times (ms)
            gpu_memory_mb: GPU memory usage in MB
            cpu_percent: CPU usage percentage
        """
        logger.debug(f"Updating all metrics: fps={fps:.1f}, latency={latency_ms:.1f}ms, gpu={gpu_memory_mb:.0f}MB, cpu={cpu_percent:.1f}%")
        
        self.update_fps(fps)
        self.update_latency(latency_ms)
        self.update_module_timings(module_timings)
        self.update_resources(gpu_memory_mb, cpu_percent)
        self.logging_widget.log_metrics(fps, latency_ms, gpu_memory_mb, cpu_percent)
