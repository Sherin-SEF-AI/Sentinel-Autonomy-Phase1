"""Advanced analytics dashboard with charts and trends."""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QGroupBox, QComboBox, QPushButton, QFrame, QGridLayout,
    QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QColor, QPen
from PyQt6.QtCharts import (
    QChart, QChartView, QLineSeries, QBarSeries, QBarSet,
    QValueAxis, QDateTimeAxis, QBarCategoryAxis
)


class AnalyticsDashboard(QWidget):
    """
    Advanced analytics dashboard displaying:
    - Trip history and trends
    - Driver performance over time
    - Safety event statistics
    - Comparative analysis
    """

    def __init__(self, trips_dir: str = "data/trips", parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)

        self.trips_dir = Path(trips_dir)
        self.trips_data: List[Dict] = []

        self.setup_ui()
        self.load_trip_data()

    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("📈 Analytics Dashboard")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Time Period:"))

        self.period_combo = QComboBox()
        self.period_combo.addItems(["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"])
        self.period_combo.currentIndexChanged.connect(self._update_charts)
        controls_layout.addWidget(self.period_combo)

        self.refresh_btn = QPushButton("🔄 Refresh Data")
        self.refresh_btn.clicked.connect(self.load_trip_data)
        controls_layout.addWidget(self.refresh_btn)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Summary statistics
        summary_frame = self._create_summary_section()
        layout.addWidget(summary_frame)

        # Tab widget for different chart views
        self.tab_widget = QTabWidget()

        # Safety trends tab
        safety_tab = self._create_safety_trends_tab()
        self.tab_widget.addTab(safety_tab, "🛡️ Safety Trends")

        # Performance tab
        performance_tab = self._create_performance_tab()
        self.tab_widget.addTab(performance_tab, "📊 Performance")

        # Comparison tab
        comparison_tab = self._create_comparison_tab()
        self.tab_widget.addTab(comparison_tab, "📉 Comparison")

        layout.addWidget(self.tab_widget)

    def _create_summary_section(self) -> QFrame:
        """Create summary statistics section."""
        frame = QFrame()
        frame.setFrameStyle(QFrame.Shape.StyledPanel)

        layout = QGridLayout(frame)

        # Summary metrics
        self.total_trips_label = self._create_metric_label("Total Trips", "0")
        layout.addWidget(self.total_trips_label, 0, 0)

        self.total_distance_label = self._create_metric_label("Total Distance", "0 km")
        layout.addWidget(self.total_distance_label, 0, 1)

        self.total_duration_label = self._create_metric_label("Total Time", "0h")
        layout.addWidget(self.total_duration_label, 0, 2)

        self.avg_safety_label = self._create_metric_label("Avg Safety Score", "0")
        layout.addWidget(self.avg_safety_label, 0, 3)

        self.total_events_label = self._create_metric_label("Total Events", "0")
        layout.addWidget(self.total_events_label, 1, 0)

        self.avg_attention_label = self._create_metric_label("Avg Attention", "0")
        layout.addWidget(self.avg_attention_label, 1, 1)

        self.best_score_label = self._create_metric_label("Best Score", "0")
        layout.addWidget(self.best_score_label, 1, 2)

        self.worst_score_label = self._create_metric_label("Worst Score", "0")
        layout.addWidget(self.worst_score_label, 1, 3)

        return frame

    def _create_metric_label(self, title: str, value: str) -> QGroupBox:
        """Create a metric display box."""
        box = QGroupBox(title)
        layout = QVBoxLayout()

        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_font = QFont()
        value_font.setPointSize(18)
        value_font.setBold(True)
        value_label.setFont(value_font)

        layout.addWidget(value_label)
        box.setLayout(layout)

        # Store reference
        setattr(box, 'value_label', value_label)

        return box

    def _create_safety_trends_tab(self) -> QWidget:
        """Create safety trends chart tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Safety score over time chart
        safety_chart_view = QChartView()
        self.safety_chart = QChart()
        self.safety_chart.setTitle("Safety Score Over Time")
        self.safety_chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        safety_chart_view.setChart(self.safety_chart)
        safety_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addWidget(safety_chart_view)

        # Events bar chart
        events_chart_view = QChartView()
        self.events_chart = QChart()
        self.events_chart.setTitle("Safety Events Distribution")
        events_chart_view.setChart(self.events_chart)
        events_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addWidget(events_chart_view)

        return widget

    def _create_performance_tab(self) -> QWidget:
        """Create performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Attention score chart
        attention_chart_view = QChartView()
        self.attention_chart = QChart()
        self.attention_chart.setTitle("Attention Score Trends")
        attention_chart_view.setChart(self.attention_chart)
        attention_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addWidget(attention_chart_view)

        # Speed distribution chart
        speed_chart_view = QChartView()
        self.speed_chart = QChart()
        self.speed_chart.setTitle("Average Speed per Trip")
        speed_chart_view.setChart(self.speed_chart)
        speed_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout.addWidget(speed_chart_view)

        return widget

    def _create_comparison_tab(self) -> QWidget:
        """Create comparison analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Best vs worst trips
        comparison_group = QGroupBox("Best vs Worst Trips Comparison")
        comparison_layout = QVBoxLayout()

        self.comparison_text = QLabel("No data available")
        self.comparison_text.setWordWrap(True)
        comparison_layout.addWidget(self.comparison_text)

        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)

        # Trend analysis
        trend_group = QGroupBox("Trend Analysis")
        trend_layout = QVBoxLayout()

        self.trend_text = QLabel("No data available")
        self.trend_text.setWordWrap(True)
        trend_layout.addWidget(self.trend_text)

        trend_group.setLayout(trend_layout)
        layout.addWidget(trend_group)

        layout.addStretch()

        return widget

    def load_trip_data(self):
        """Load trip data from JSON files."""
        self.trips_data = []

        if not self.trips_dir.exists():
            self.logger.warning(f"Trips directory not found: {self.trips_dir}")
            return

        # Load all trip files
        trip_files = list(self.trips_dir.glob("trip_*.json"))

        for trip_file in trip_files:
            try:
                with open(trip_file, 'r') as f:
                    trip_data = json.load(f)
                    self.trips_data.append(trip_data)
            except Exception as e:
                self.logger.error(f"Failed to load {trip_file}: {e}")

        # Sort by start time
        self.trips_data.sort(key=lambda x: x.get('start_time', 0))

        self.logger.info(f"Loaded {len(self.trips_data)} trips")

        # Update UI
        self._update_summary()
        self._update_charts()

    def _update_summary(self):
        """Update summary statistics."""
        if not self.trips_data:
            return

        # Calculate aggregates
        total_trips = len(self.trips_data)
        total_distance = sum(t.get('distance', 0) for t in self.trips_data) / 1000  # Convert to km
        total_duration = sum(t.get('duration', 0) for t in self.trips_data) / 3600  # Convert to hours

        avg_safety = sum(t.get('safety_score', 0) for t in self.trips_data) / total_trips
        avg_attention = sum(t.get('average_attention_score', 0) for t in self.trips_data) / total_trips

        total_events = sum(
            t.get('num_hard_brakes', 0) +
            t.get('num_rapid_accelerations', 0) +
            t.get('num_lane_departures', 0) +
            t.get('num_collision_warnings', 0) +
            t.get('num_blind_spot_warnings', 0)
            for t in self.trips_data
        )

        safety_scores = [t.get('safety_score', 0) for t in self.trips_data]
        best_score = max(safety_scores) if safety_scores else 0
        worst_score = min(safety_scores) if safety_scores else 0

        # Update labels
        self.total_trips_label.value_label.setText(str(total_trips))
        self.total_distance_label.value_label.setText(f"{total_distance:.1f} km")
        self.total_duration_label.value_label.setText(f"{total_duration:.1f}h")
        self.avg_safety_label.value_label.setText(f"{avg_safety:.0f}")
        self.avg_attention_label.value_label.setText(f"{avg_attention:.0f}")
        self.total_events_label.value_label.setText(str(total_events))
        self.best_score_label.value_label.setText(f"{best_score:.0f}")
        self.worst_score_label.value_label.setText(f"{worst_score:.0f}")

    def _update_charts(self):
        """Update all charts with current data."""
        if not self.trips_data:
            return

        # Filter data based on selected period
        filtered_trips = self._filter_by_period()

        # Update safety chart
        self._update_safety_chart(filtered_trips)

        # Update events chart
        self._update_events_chart(filtered_trips)

        # Update attention chart
        self._update_attention_chart(filtered_trips)

        # Update speed chart
        self._update_speed_chart(filtered_trips)

        # Update comparison
        self._update_comparison(filtered_trips)

    def _filter_by_period(self) -> List[Dict]:
        """Filter trips by selected time period."""
        period = self.period_combo.currentText()

        if period == "All Time":
            return self.trips_data

        now = datetime.now()

        if period == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        elif period == "Last 30 Days":
            cutoff = now - timedelta(days=30)
        elif period == "Last 90 Days":
            cutoff = now - timedelta(days=90)
        else:
            return self.trips_data

        cutoff_timestamp = cutoff.timestamp()

        return [t for t in self.trips_data if t.get('start_time', 0) >= cutoff_timestamp]

    def _update_safety_chart(self, trips: List[Dict]):
        """Update safety score line chart."""
        self.safety_chart.removeAllSeries()

        series = QLineSeries()
        series.setName("Safety Score")

        for i, trip in enumerate(trips):
            score = trip.get('safety_score', 0)
            series.append(i, score)

        pen = QPen(QColor("#4ade80"))
        pen.setWidth(3)
        series.setPen(pen)

        self.safety_chart.addSeries(series)
        self.safety_chart.createDefaultAxes()

        # Set Y axis range
        axis_y = self.safety_chart.axes(Qt.Orientation.Vertical)[0]
        axis_y.setRange(0, 100)
        axis_y.setTitleText("Score")

        axis_x = self.safety_chart.axes(Qt.Orientation.Horizontal)[0]
        axis_x.setTitleText("Trip Number")

    def _update_events_chart(self, trips: List[Dict]):
        """Update safety events bar chart."""
        self.events_chart.removeAllSeries()

        # Aggregate events
        hard_brakes = sum(t.get('num_hard_brakes', 0) for t in trips)
        rapid_accel = sum(t.get('num_rapid_accelerations', 0) for t in trips)
        lane_depart = sum(t.get('num_lane_departures', 0) for t in trips)
        collisions = sum(t.get('num_collision_warnings', 0) for t in trips)
        blind_spot = sum(t.get('num_blind_spot_warnings', 0) for t in trips)

        # Create bar sets
        set0 = QBarSet("Events")
        set0.append([hard_brakes, rapid_accel, lane_depart, collisions, blind_spot])

        series = QBarSeries()
        series.append(set0)

        self.events_chart.addSeries(series)

        # Create axis
        categories = ["Hard Brakes", "Rapid Accel", "Lane Depart", "Collisions", "Blind Spot"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)

        self.events_chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, max(hard_brakes, rapid_accel, lane_depart, collisions, blind_spot) * 1.2)
        self.events_chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

    def _update_attention_chart(self, trips: List[Dict]):
        """Update attention score chart."""
        self.attention_chart.removeAllSeries()

        series = QLineSeries()
        series.setName("Attention Score")

        for i, trip in enumerate(trips):
            score = trip.get('average_attention_score', 0)
            series.append(i, score)

        pen = QPen(QColor("#60a5fa"))
        pen.setWidth(3)
        series.setPen(pen)

        self.attention_chart.addSeries(series)
        self.attention_chart.createDefaultAxes()

        axis_y = self.attention_chart.axes(Qt.Orientation.Vertical)[0]
        axis_y.setRange(0, 100)

    def _update_speed_chart(self, trips: List[Dict]):
        """Update speed chart."""
        self.speed_chart.removeAllSeries()

        series = QLineSeries()
        series.setName("Avg Speed (km/h)")

        for i, trip in enumerate(trips):
            speed = trip.get('average_speed', 0) * 3.6  # m/s to km/h
            series.append(i, speed)

        pen = QPen(QColor("#fbbf24"))
        pen.setWidth(3)
        series.setPen(pen)

        self.speed_chart.addSeries(series)
        self.speed_chart.createDefaultAxes()

    def _update_comparison(self, trips: List[Dict]):
        """Update comparison analysis."""
        if not trips:
            return

        # Find best and worst trips
        best_trip = max(trips, key=lambda t: t.get('safety_score', 0))
        worst_trip = min(trips, key=lambda t: t.get('safety_score', 0))

        comparison_html = f"""
        <h3>Best Trip:</h3>
        <ul>
        <li>Safety Score: {best_trip.get('safety_score', 0):.0f}</li>
        <li>Distance: {best_trip.get('distance', 0)/1000:.1f} km</li>
        <li>Events: {best_trip.get('num_hard_brakes', 0) + best_trip.get('num_rapid_accelerations', 0)}</li>
        </ul>

        <h3>Worst Trip:</h3>
        <ul>
        <li>Safety Score: {worst_trip.get('safety_score', 0):.0f}</li>
        <li>Distance: {worst_trip.get('distance', 0)/1000:.1f} km</li>
        <li>Events: {worst_trip.get('num_hard_brakes', 0) + worst_trip.get('num_rapid_accelerations', 0)}</li>
        </ul>
        """

        self.comparison_text.setText(comparison_html)
