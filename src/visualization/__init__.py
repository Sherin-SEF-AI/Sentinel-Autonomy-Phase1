"""
Visualization module for SENTINEL.

Provides real-time dashboard and scenario playback interfaces.
"""

from .backend import (
    VisualizationServer,
    create_server,
    serialize_frame_data,
    StreamingManager,
    StreamingIntegration,
    PerformanceMonitor,
    create_streaming_manager
)
from .camera_overlay import CameraOverlayRenderer

__all__ = [
    'CameraOverlayRenderer',
    'VisualizationServer',
    'create_server',
    'serialize_frame_data',
    'StreamingManager',
    'StreamingIntegration',
    'PerformanceMonitor',
    'create_streaming_manager'
]
