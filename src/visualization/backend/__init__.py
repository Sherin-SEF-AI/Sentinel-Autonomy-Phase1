"""Visualization backend module."""

# Auto-generated exports
from .data_serializer import (
    encode_image,
    serialize_alert,
    serialize_bev_output,
    serialize_detection_3d,
    serialize_driver_state,
    serialize_frame_data,
    serialize_hazard,
    serialize_risk,
    serialize_risk_assessment,
    serialize_segmentation_output
)
from .server import ConnectionManager, VisualizationServer, create_server
from .streaming import (
    PerformanceMonitor,
    StreamingIntegration,
    StreamingManager,
    create_streaming_manager
)

__all__ = [
    'ConnectionManager',
    'PerformanceMonitor',
    'StreamingIntegration',
    'StreamingManager',
    'VisualizationServer',
    'create_server',
    'create_streaming_manager',
    'encode_image',
    'serialize_alert',
    'serialize_bev_output',
    'serialize_detection_3d',
    'serialize_driver_state',
    'serialize_frame_data',
    'serialize_hazard',
    'serialize_risk',
    'serialize_risk_assessment',
    'serialize_segmentation_output'
]
