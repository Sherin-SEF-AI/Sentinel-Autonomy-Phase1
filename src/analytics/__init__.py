"""Analytics module for SENTINEL system.

Provides trip analytics, behavior reporting, risk heatmaps, and dashboard visualization.
"""

import logging

# Initialize module logger
logger = logging.getLogger(__name__)

# Auto-generated exports
from .analytics_dashboard import AnalyticsDashboard
from .behavior_report import BehaviorReportGenerator
from .report_exporter import ReportExporter
from .risk_heatmap import RiskHeatmap
from .trip_analytics import TripAnalytics, TripSegment, TripSummary

__all__ = [
    'AnalyticsDashboard',
    'BehaviorReportGenerator',
    'ReportExporter',
    'RiskHeatmap',
    'TripAnalytics',
    'TripSegment',
    'TripSummary',
]

logger.debug("Analytics module initialized: all components loaded")
