"""
Analytics Dashboard

PyQt6 widget for displaying trip statistics, driver performance metrics,
trends over time, and fleet comparisons.
"""

import numpy as np
from typing import Dict, List, Optional
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QTabWidget, QTableWidget, QTableWidgetItem,
                             QPushButton, QComboBox, QGroupBox, QGridLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)


class AnalyticsDashboard(QWidget):
    """
    Analytics dashboard widget.
    
    Displays:
    - Trip statistics
    - Driver performance metrics
    - Trends over time
    - Fleet comparisons
    """
    
    # Signals
    export_requested = pyqtSignal(str)  # Export type
    
    def __init__(self, config: dict, parent=None):
        """
        Initialize analytics dashboard.
        
        Args:
            config: Configuration dictionary
            parent: Parent widget
        """
        super().__init__(parent)
        self.config = config
        
        # Data
        self.trip_summaries: List[Dict] = []
        self.driver_profiles: Dict[str, Dict] = {}
        self.fleet_stats: Optional[Dict] = None
        
        # Setup UI
        self._setup_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(5000)  # Update every 5 seconds
        
        logger.info("AnalyticsDashboard initialized")
    
    def _setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Analytics Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Driver selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Driver:"))
        self.driver_combo = QComboBox()
        self.driver_combo.addItem("All Drivers")
        self.driver_combo.currentTextChanged.connect(self._on_driver_changed)
        selector_layout.addWidget(self.driver_combo)
        selector_layout.addStretch()
        layout.addLayout(selector_layout)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Overview tab
        self.overview_tab = self._create_overview_tab()
        self.tabs.addTab(self.overview_tab, "Overview")
        
        # Trips tab
        self.trips_tab = self._create_trips_tab()
        self.tabs.addTab(self.trips_tab, "Trip History")
        
        # Performance tab
        self.performance_tab = self._create_performance_tab()
        self.tabs.addTab(self.performance_tab, "Performance")
        
        # Fleet comparison tab
        self.fleet_tab = self._create_fleet_tab()
        self.tabs.addTab(self.fleet_tab, "Fleet Comparison")
        
        layout.addWidget(self.tabs)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.clicked.connect(lambda: self.export_requested.emit('csv'))
        export_layout.addWidget(btn_export_csv)
        
        btn_export_pdf = QPushButton("Export PDF")
        btn_export_pdf.clicked.connect(lambda: self.export_requested.emit('pdf'))
        export_layout.addWidget(btn_export_pdf)
        
        layout.addLayout(export_layout)
    
    def _create_overview_tab(self) -> QWidget:
        """Create overview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics grid
        stats_group = QGroupBox("Overall Statistics")
        stats_layout = QGridLayout()
        
        # Create labels
        self.lbl_total_trips = QLabel("0")
        self.lbl_total_distance = QLabel("0.0 km")
        self.lbl_total_time = QLabel("0.0 hours")
        self.lbl_avg_safety = QLabel("0.0")
        self.lbl_total_alerts = QLabel("0")
        
        # Add to grid
        stats_layout.addWidget(QLabel("Total Trips:"), 0, 0)
        stats_layout.addWidget(self.lbl_total_trips, 0, 1)
        stats_layout.addWidget(QLabel("Total Distance:"), 1, 0)
        stats_layout.addWidget(self.lbl_total_distance, 1, 1)
        stats_layout.addWidget(QLabel("Total Time:"), 2, 0)
        stats_layout.addWidget(self.lbl_total_time, 2, 1)
        stats_layout.addWidget(QLabel("Avg Safety Score:"), 3, 0)
        stats_layout.addWidget(self.lbl_avg_safety, 3, 1)
        stats_layout.addWidget(QLabel("Total Alerts:"), 4, 0)
        stats_layout.addWidget(self.lbl_total_alerts, 4, 1)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Recent activity
        recent_group = QGroupBox("Recent Activity")
        recent_layout = QVBoxLayout()
        self.lbl_recent_activity = QLabel("No recent activity")
        self.lbl_recent_activity.setWordWrap(True)
        recent_layout.addWidget(self.lbl_recent_activity)
        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)
        
        layout.addStretch()
        return widget
    
    def _create_trips_tab(self) -> QWidget:
        """Create trips history tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Trip table
        self.trip_table = QTableWidget()
        self.trip_table.setColumnCount(7)
        self.trip_table.setHorizontalHeaderLabels([
            'Date', 'Duration', 'Distance', 'Avg Speed', 
            'Safety Score', 'Alerts', 'Driver'
        ])
        self.trip_table.setAlternatingRowColors(True)
        layout.addWidget(self.trip_table)
        
        return widget
    
    def _create_performance_tab(self) -> QWidget:
        """Create performance metrics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QGridLayout()
        
        self.lbl_safety_score = QLabel("0.0")
        self.lbl_attention_score = QLabel("0.0")
        self.lbl_eco_score = QLabel("0.0")
        self.lbl_reaction_time = QLabel("0.0 s")
        self.lbl_following_distance = QLabel("0.0 m")
        
        metrics_layout.addWidget(QLabel("Safety Score:"), 0, 0)
        metrics_layout.addWidget(self.lbl_safety_score, 0, 1)
        metrics_layout.addWidget(QLabel("Attention Score:"), 1, 0)
        metrics_layout.addWidget(self.lbl_attention_score, 1, 1)
        metrics_layout.addWidget(QLabel("Eco Score:"), 2, 0)
        metrics_layout.addWidget(self.lbl_eco_score, 2, 1)
        metrics_layout.addWidget(QLabel("Avg Reaction Time:"), 3, 0)
        metrics_layout.addWidget(self.lbl_reaction_time, 3, 1)
        metrics_layout.addWidget(QLabel("Avg Following Distance:"), 4, 0)
        metrics_layout.addWidget(self.lbl_following_distance, 4, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Trends (placeholder for charts)
        trends_group = QGroupBox("Performance Trends")
        trends_layout = QVBoxLayout()
        self.lbl_trends = QLabel("Trend charts require matplotlib")
        self.lbl_trends.setAlignment(Qt.AlignmentFlag.AlignCenter)
        trends_layout.addWidget(self.lbl_trends)
        trends_group.setLayout(trends_layout)
        layout.addWidget(trends_group)
        
        layout.addStretch()
        return widget
    
    def _create_fleet_tab(self) -> QWidget:
        """Create fleet comparison tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Fleet statistics
        fleet_group = QGroupBox("Fleet Statistics")
        fleet_layout = QGridLayout()
        
        self.lbl_fleet_vehicles = QLabel("0")
        self.lbl_fleet_avg_safety = QLabel("0.0")
        self.lbl_fleet_total_distance = QLabel("0.0 km")
        self.lbl_fleet_rank = QLabel("N/A")
        
        fleet_layout.addWidget(QLabel("Total Vehicles:"), 0, 0)
        fleet_layout.addWidget(self.lbl_fleet_vehicles, 0, 1)
        fleet_layout.addWidget(QLabel("Fleet Avg Safety:"), 1, 0)
        fleet_layout.addWidget(self.lbl_fleet_avg_safety, 1, 1)
        fleet_layout.addWidget(QLabel("Fleet Total Distance:"), 2, 0)
        fleet_layout.addWidget(self.lbl_fleet_total_distance, 2, 1)
        fleet_layout.addWidget(QLabel("Your Rank:"), 3, 0)
        fleet_layout.addWidget(self.lbl_fleet_rank, 3, 1)
        
        fleet_group.setLayout(fleet_layout)
        layout.addWidget(fleet_group)
        
        # Comparison message
        self.lbl_fleet_message = QLabel("Fleet data not available")
        self.lbl_fleet_message.setWordWrap(True)
        layout.addWidget(self.lbl_fleet_message)
        
        layout.addStretch()
        return widget
    
    def update_trip_data(self, trip_summaries: List[Dict]):
        """
        Update with new trip data.
        
        Args:
            trip_summaries: List of trip summary dictionaries
        """
        self.trip_summaries = trip_summaries
        self._update_displays()
    
    def update_driver_profile(self, driver_id: str, profile: Dict):
        """
        Update driver profile data.
        
        Args:
            driver_id: Driver identifier
            profile: Driver profile dictionary
        """
        self.driver_profiles[driver_id] = profile
        
        # Update driver combo
        if driver_id not in [self.driver_combo.itemText(i) 
                            for i in range(1, self.driver_combo.count())]:
            self.driver_combo.addItem(driver_id)
        
        self._update_displays()
    
    def update_fleet_stats(self, fleet_stats: Dict):
        """
        Update fleet statistics.
        
        Args:
            fleet_stats: Fleet statistics dictionary
        """
        self.fleet_stats = fleet_stats
        self._update_fleet_display()
    
    def _update_displays(self):
        """Update all displays with current data."""
        self._update_overview()
        self._update_trips_table()
        self._update_performance()
    
    def _update_overview(self):
        """Update overview tab."""
        if not self.trip_summaries:
            return
        
        # Calculate statistics
        total_trips = len(self.trip_summaries)
        total_distance = sum(t.get('distance', 0) for t in self.trip_summaries)
        total_time = sum(t.get('duration', 0) for t in self.trip_summaries)
        
        safety_scores = [t.get('safety_score', 0) for t in self.trip_summaries]
        avg_safety = np.mean(safety_scores) if safety_scores else 0.0
        
        total_alerts = sum(
            sum(t.get('alert_counts', {}).values()) 
            for t in self.trip_summaries
        )
        
        # Update labels
        self.lbl_total_trips.setText(str(total_trips))
        self.lbl_total_distance.setText(f"{total_distance/1000:.2f} km")
        self.lbl_total_time.setText(f"{total_time/3600:.2f} hours")
        self.lbl_avg_safety.setText(f"{avg_safety:.1f}")
        self.lbl_total_alerts.setText(str(total_alerts))
        
        # Recent activity
        if self.trip_summaries:
            last_trip = self.trip_summaries[-1]
            activity_text = (
                f"Last trip: {last_trip.get('start_time', 'Unknown')}\n"
                f"Distance: {last_trip.get('distance', 0)/1000:.2f} km\n"
                f"Safety Score: {last_trip.get('safety_score', 0):.1f}"
            )
            self.lbl_recent_activity.setText(activity_text)
    
    def _update_trips_table(self):
        """Update trips table."""
        self.trip_table.setRowCount(len(self.trip_summaries))
        
        for i, trip in enumerate(self.trip_summaries):
            # Date
            date_str = trip.get('start_time', 'Unknown')
            if hasattr(date_str, 'strftime'):
                date_str = date_str.strftime('%Y-%m-%d %H:%M')
            self.trip_table.setItem(i, 0, QTableWidgetItem(str(date_str)))
            
            # Duration
            duration = trip.get('duration', 0) / 60  # minutes
            self.trip_table.setItem(i, 1, QTableWidgetItem(f"{duration:.1f} min"))
            
            # Distance
            distance = trip.get('distance', 0) / 1000  # km
            self.trip_table.setItem(i, 2, QTableWidgetItem(f"{distance:.2f} km"))
            
            # Avg Speed
            avg_speed = trip.get('avg_speed', 0) * 3.6  # km/h
            self.trip_table.setItem(i, 3, QTableWidgetItem(f"{avg_speed:.1f} km/h"))
            
            # Safety Score
            safety = trip.get('safety_score', 0)
            self.trip_table.setItem(i, 4, QTableWidgetItem(f"{safety:.1f}"))
            
            # Alerts
            alerts = sum(trip.get('alert_counts', {}).values())
            self.trip_table.setItem(i, 5, QTableWidgetItem(str(alerts)))
            
            # Driver
            driver = trip.get('driver_id', 'Unknown')
            self.trip_table.setItem(i, 6, QTableWidgetItem(driver))
        
        self.trip_table.resizeColumnsToContents()
    
    def _update_performance(self):
        """Update performance tab."""
        current_driver = self.driver_combo.currentText()
        
        if current_driver == "All Drivers":
            # Aggregate across all drivers
            if self.driver_profiles:
                profiles = list(self.driver_profiles.values())
                safety = np.mean([p.get('safety_score', 0) for p in profiles])
                attention = np.mean([p.get('attention_score', 0) for p in profiles])
                eco = np.mean([p.get('eco_score', 0) for p in profiles])
            else:
                safety = attention = eco = 0.0
            
            reaction_time = 0.0
            following_distance = 0.0
        else:
            # Specific driver
            profile = self.driver_profiles.get(current_driver, {})
            safety = profile.get('safety_score', 0)
            attention = profile.get('attention_score', 0)
            eco = profile.get('eco_score', 0)
            
            metrics = profile.get('metrics', {})
            reaction_time = metrics.get('reaction_time', {}).get('mean', 0.0)
            following_distance = metrics.get('following_distance', {}).get('mean', 0.0)
        
        # Update labels
        self.lbl_safety_score.setText(f"{safety:.1f}")
        self.lbl_attention_score.setText(f"{attention:.1f}")
        self.lbl_eco_score.setText(f"{eco:.1f}")
        self.lbl_reaction_time.setText(f"{reaction_time:.2f} s")
        self.lbl_following_distance.setText(f"{following_distance:.1f} m")
    
    def _update_fleet_display(self):
        """Update fleet comparison tab."""
        if not self.fleet_stats:
            return
        
        self.lbl_fleet_vehicles.setText(str(self.fleet_stats.get('total_vehicles', 0)))
        self.lbl_fleet_avg_safety.setText(f"{self.fleet_stats.get('avg_safety_score', 0):.1f}")
        self.lbl_fleet_total_distance.setText(
            f"{self.fleet_stats.get('total_distance', 0)/1000:.2f} km"
        )
        
        rank = self.fleet_stats.get('vehicle_rank', 'N/A')
        self.lbl_fleet_rank.setText(str(rank))
        
        # Comparison message
        current_safety = self.driver_profiles.get(
            self.driver_combo.currentText(), {}
        ).get('safety_score', 0)
        fleet_avg = self.fleet_stats.get('avg_safety_score', 0)
        
        if current_safety > fleet_avg:
            message = f"Your safety score ({current_safety:.1f}) is above fleet average ({fleet_avg:.1f}). Great job!"
        elif current_safety < fleet_avg:
            message = f"Your safety score ({current_safety:.1f}) is below fleet average ({fleet_avg:.1f}). Room for improvement."
        else:
            message = f"Your safety score matches the fleet average ({fleet_avg:.1f})."
        
        self.lbl_fleet_message.setText(message)
    
    def _on_driver_changed(self, driver_id: str):
        """Handle driver selection change."""
        self._update_performance()
        self._update_fleet_display()
