"""Core data structures for SENTINEL system."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraBundle:
    """Synchronized frame bundle from all cameras."""
    timestamp: float
    interior: np.ndarray  # (480, 640, 3)
    front_left: np.ndarray  # (720, 1280, 3)
    front_right: np.ndarray  # (720, 1280, 3)


@dataclass
class BEVOutput:
    """Bird's eye view transformation result."""
    timestamp: float
    image: np.ndarray  # (640, 640, 3)
    mask: np.ndarray  # (640, 640) bool - valid regions


@dataclass
class SegmentationOutput:
    """Semantic segmentation result."""
    timestamp: float
    class_map: np.ndarray  # (640, 640) int8
    confidence: np.ndarray  # (640, 640) float32


@dataclass
class Detection2D:
    """2D detection in camera view."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float
    camera_id: int


@dataclass
class Detection3D:
    """3D detection in vehicle frame."""
    bbox_3d: Tuple[float, float, float, float, float, float, float]  # x, y, z, w, h, l, θ
    class_name: str
    confidence: float
    velocity: Tuple[float, float, float]  # vx, vy, vz
    track_id: int


@dataclass
class DriverState:
    """Complete driver state."""
    face_detected: bool
    landmarks: np.ndarray  # (68, 2)
    head_pose: Dict[str, float]  # roll, pitch, yaw
    gaze: Dict[str, Any]  # pitch, yaw, attention_zone
    eye_state: Dict[str, float]  # left_ear, right_ear, perclos
    drowsiness: Dict[str, Any]  # score, yawn_detected, micro_sleep, head_nod
    distraction: Dict[str, Any]  # type, confidence, duration
    readiness_score: float  # 0-100


@dataclass
class Hazard:
    """Identified hazard."""
    object_id: int
    type: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    trajectory: List[Tuple[float, float, float]]
    ttc: float
    zone: str
    base_risk: float  # 0-1


@dataclass
class Risk:
    """Contextual risk assessment."""
    hazard: Hazard
    contextual_score: float  # 0-1
    driver_aware: bool
    urgency: str  # 'low', 'medium', 'high', 'critical'
    intervention_needed: bool


@dataclass
class RiskAssessment:
    """Complete risk assessment output."""
    scene_graph: Dict[str, Any]
    hazards: List[Hazard]
    attention_map: Dict[str, Any]
    top_risks: List[Risk]


@dataclass
class Alert:
    """Generated alert."""
    timestamp: float
    urgency: str  # 'info', 'warning', 'critical'
    modalities: List[str]  # ['visual', 'audio', 'haptic']
    message: str
    hazard_id: int
    dismissed: bool


@dataclass
class MapFeature:
    """HD map feature."""
    feature_id: str
    type: str  # 'lane', 'sign', 'light', 'crosswalk', 'boundary'
    position: Tuple[float, float, float]  # x, y, z in vehicle frame
    attributes: Dict[str, Any]  # type-specific attributes
    geometry: List[Tuple[float, float]]  # Polyline points (x, y)


@dataclass
class Lane:
    """Lane representation from HD map."""
    lane_id: str
    centerline: List[Tuple[float, float, float]]  # (x, y, z) points
    left_boundary: List[Tuple[float, float, float]]
    right_boundary: List[Tuple[float, float, float]]
    width: float
    speed_limit: Optional[float]
    lane_type: str  # 'driving', 'parking', 'shoulder', etc.
    predecessors: List[str]  # Lane IDs
    successors: List[str]  # Lane IDs


@dataclass
class VehicleTelemetry:
    """CAN bus telemetry (placeholder for future CAN integration)."""
    timestamp: float
    speed: float  # m/s
    steering_angle: float  # radians
    brake_pressure: float  # bar
    throttle_position: float  # 0-1
    gear: int
    turn_signal: str  # 'left', 'right', 'none'
