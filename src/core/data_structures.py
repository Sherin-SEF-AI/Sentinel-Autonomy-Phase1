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


@dataclass
class DetectedLane:
    """Lane detected from vision."""
    lane_id: int  # 0=left, 1=ego_left, 2=ego_right, 3=right
    points: np.ndarray  # (N, 2) pixel coordinates
    coefficients: np.ndarray  # Polynomial coefficients (e.g., 2nd order: [a, b, c] for y = ax^2 + bx + c)
    confidence: float  # 0-1
    lane_type: str  # 'dashed', 'solid', 'double'
    color: str  # 'white', 'yellow'


@dataclass
class LaneState:
    """Current lane state and departure warning."""
    timestamp: float
    lanes_detected: List[DetectedLane]
    ego_lane_center: Optional[Tuple[float, float]]  # (x, y) offset from vehicle center
    lateral_offset: float  # meters, negative=left, positive=right
    heading_angle: float  # radians, relative to lane direction
    departure_warning: bool
    departure_side: str  # 'left', 'right', 'none'
    time_to_lane_crossing: Optional[float]  # seconds


@dataclass
class TrafficSign:
    """Detected traffic sign."""
    sign_type: str  # 'stop', 'yield', 'speed_limit', 'warning', etc.
    sign_class: str  # Specific classification (e.g., 'speed_limit_50')
    value: Optional[int]  # For speed limits, etc.
    position: Tuple[float, float, float]  # 3D position in vehicle frame
    bbox_2d: Tuple[float, float, float, float]  # 2D bbox in camera frame
    confidence: float
    camera_id: int


@dataclass
class BlindSpotWarning:
    """Blind spot detection warning."""
    timestamp: float
    left_blind_spot: bool
    right_blind_spot: bool
    left_objects: List[Detection3D]
    right_objects: List[Detection3D]
    warning_active: bool
    warning_side: str  # 'left', 'right', 'both', 'none'


@dataclass
class CollisionWarning:
    """Forward collision warning."""
    timestamp: float
    warning_level: str  # 'none', 'caution', 'warning', 'critical'
    time_to_collision: Optional[float]  # seconds
    target_object: Optional[Detection3D]
    recommended_deceleration: float  # m/s^2
    automatic_braking_required: bool


@dataclass
class RoadCondition:
    """Road surface condition analysis."""
    timestamp: float
    surface_type: str  # 'dry', 'wet', 'snow', 'ice'
    confidence: float
    visibility: str  # 'clear', 'rain', 'fog', 'snow'
    hazards: List[str]  # 'puddle', 'pothole', 'debris', 'construction'
    friction_estimate: float  # 0-1, 1=optimal


@dataclass
class ParkingSpace:
    """Detected parking space."""
    space_id: int
    corners: np.ndarray  # (4, 2) corners in BEV coordinates
    width: float  # meters
    length: float  # meters
    occupied: bool
    confidence: float
    space_type: str  # 'parallel', 'perpendicular', 'angled'


@dataclass
class TripStats:
    """Trip statistics and analytics."""
    start_time: float
    end_time: Optional[float]
    distance: float  # meters
    duration: float  # seconds
    average_speed: float  # m/s
    max_speed: float  # m/s
    num_hard_brakes: int
    num_rapid_accelerations: int
    num_lane_departures: int
    num_collision_warnings: int
    num_blind_spot_warnings: int
    average_attention_score: float  # 0-100
    safety_score: float  # 0-100


@dataclass
class DriverScore:
    """Real-time driver behavior scoring."""
    timestamp: float
    overall_score: float  # 0-100
    attention_score: float  # 0-100
    smoothness_score: float  # 0-100
    safety_score: float  # 0-100
    hazard_response_score: float  # 0-100
    recent_events: List[Dict[str, Any]]  # Recent incidents affecting score

