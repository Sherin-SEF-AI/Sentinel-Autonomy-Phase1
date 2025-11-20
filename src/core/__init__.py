"""Core infrastructure for SENTINEL system."""

# Auto-generated exports
from .config import ConfigManager
from .data_structures import (
    Alert,
    BEVOutput,
    CameraBundle,
    Detection2D,
    Detection3D,
    DriverState,
    Hazard,
    Lane,
    MapFeature,
    Risk,
    RiskAssessment,
    SegmentationOutput,
    VehicleTelemetry,
)
from .interfaces import (
    IAlertSystem,
    IBEVGenerator,
    ICameraManager,
    IContextualIntelligence,
    IDMS,
    IObjectDetector,
    ISemanticSegmentor,
)
from .logging import LoggerSetup

__all__ = [
    'Alert',
    'BEVOutput',
    'CameraBundle',
    'ConfigManager',
    'Detection2D',
    'Detection3D',
    'DriverState',
    'Hazard',
    'IAlertSystem',
    'IBEVGenerator',
    'ICameraManager',
    'IContextualIntelligence',
    'IDMS',
    'IObjectDetector',
    'ISemanticSegmentor',
    'Lane',
    'LoggerSetup',
    'MapFeature',
    'Risk',
    'RiskAssessment',
    'SegmentationOutput',
    'VehicleTelemetry',
]
