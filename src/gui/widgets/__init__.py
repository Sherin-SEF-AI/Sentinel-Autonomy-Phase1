"""
Custom widgets for SENTINEL GUI

This module contains all custom widgets used in the SENTINEL application.
"""

# Auto-generated exports
from .alerts_panel import AlertsPanel
from .bev_canvas import BEVCanvas, BEVGraphicsView
from .circular_gauge import CircularGaugeWidget
from .cloud_settings_dock import CloudSettingsDock
from .configuration_dock import ConfigurationDockWidget, LabeledSlider
from .driver_profile_dock import DriverProfileDock
from .driver_state_panel import DriverStatePanel
from .gaze_direction import GazeDirectionWidget
from .live_monitor import LiveMonitorWidget
from .map_view_dock import MapGraphicsView, MapViewDock
from .metric_display import DriverMetricsPanel, MetricDisplayWidget, MetricsGridWidget
from .performance_dock import (
    FPSGraphWidget, LatencyGraphWidget, ModuleBreakdownWidget,
    PerformanceDockWidget, PerformanceLoggingWidget, ResourceUsageWidget
)
from .risk_panel import HazardListItem, RiskAssessmentPanel, TTCDisplayWidget, ZoneRiskRadarChart
from .scenarios_dock import ScenarioListItem, ScenarioReplayDialog, ScenariosDockWidget
from .status_indicator import DriverStatusPanel, StatusIndicatorWidget
from .trend_graph import DriverTrendGraphsPanel, TrendGraphWidget
from .vehicle_telemetry_dock import BarIndicator, GearIndicator, SteeringIndicator, TurnSignalIndicator, VehicleTelemetryDock
from .video_display import VideoDisplayWidget
from .warning_animations import ThresholdMonitor, WarningAnimationManager

__all__ = [
    'AlertsPanel',
    'BarIndicator',
    'BEVCanvas',
    'BEVGraphicsView',
    'CircularGaugeWidget',
    'CloudSettingsDock',
    'ConfigurationDockWidget',
    'DriverMetricsPanel',
    'DriverProfileDock',
    'DriverStatePanel',
    'DriverStatusPanel',
    'DriverTrendGraphsPanel',
    'FPSGraphWidget',
    'GazeDirectionWidget',
    'GearIndicator',
    'HazardListItem',
    'LabeledSlider',
    'LatencyGraphWidget',
    'LiveMonitorWidget',
    'MapGraphicsView',
    'MapViewDock',
    'MetricDisplayWidget',
    'MetricsGridWidget',
    'ModuleBreakdownWidget',
    'PerformanceDockWidget',
    'PerformanceLoggingWidget',
    'ResourceUsageWidget',
    'RiskAssessmentPanel',
    'ScenarioListItem',
    'ScenarioReplayDialog',
    'ScenariosDockWidget',
    'StatusIndicatorWidget',
    'SteeringIndicator',
    'ThresholdMonitor',
    'TrendGraphWidget',
    'TTCDisplayWidget',
    'TurnSignalIndicator',
    'VehicleTelemetryDock',
    'VideoDisplayWidget',
    'WarningAnimationManager',
    'ZoneRiskRadarChart',
]
