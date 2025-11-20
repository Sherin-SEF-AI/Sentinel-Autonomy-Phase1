"""Centralized features manager for all SENTINEL capabilities."""

import logging
from typing import Dict, Optional, List
import numpy as np

from src.core.data_structures import (
    Detection3D, DriverState, VehicleTelemetry, SegmentationOutput,
    LaneState, BlindSpotWarning, CollisionWarning, TrafficSign,
    RoadCondition, ParkingSpace, DriverScore, TripStats
)

# Import all feature modules
from src.perception.lanes.detector import LaneDetector
from src.perception.lanes.departure_warning import LaneDepartureWarning
from src.safety.blind_spot import BlindSpotMonitor
from src.safety.collision_warning import ForwardCollisionWarning
from src.perception.signs.detector import TrafficSignDetector
from src.perception.road_analysis import RoadSurfaceAnalyzer
from src.perception.parking import ParkingAssistant
from src.analytics.trip_tracker import TripTracker
from src.analytics.driver_scoring import DriverScoringSystem


class FeaturesManager:
    """
    Centralized manager for all advanced features.

    Coordinates:
    - Lane detection and departure warnings
    - Blind spot monitoring
    - Forward collision warnings
    - Traffic sign recognition
    - Road surface analysis
    - Parking assistance
    - Trip analytics
    - Driver behavior scoring
    """

    def __init__(self, config: Dict):
        """
        Initialize features manager.

        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Extract feature configs
        features_config = config.get('features', {})

        # Initialize perception features
        lane_config = features_config.get('lane_detection', {})
        self.lane_detector = LaneDetector(lane_config.get('detector', {}))
        self.lane_departure_warning = LaneDepartureWarning(lane_config.get('departure_warning', {}))

        self.traffic_sign_detector = TrafficSignDetector(
            features_config.get('traffic_signs', {})
        )

        self.road_analyzer = RoadSurfaceAnalyzer(
            features_config.get('road_analysis', {})
        )

        self.parking_assistant = ParkingAssistant(
            features_config.get('parking', {})
        )

        # Initialize safety features
        self.blind_spot_monitor = BlindSpotMonitor(
            features_config.get('blind_spot', {})
        )

        self.collision_warning = ForwardCollisionWarning(
            features_config.get('collision_warning', {})
        )

        # Initialize analytics
        self.trip_tracker = TripTracker(
            features_config.get('trip_analytics', {})
        )

        self.driver_scoring = DriverScoringSystem(
            features_config.get('driver_scoring', {})
        )

        # Feature enable flags
        self.enable_lane_detection = lane_config.get('enabled', True)
        self.enable_blind_spot = features_config.get('blind_spot', {}).get('enabled', True)
        self.enable_collision_warning = features_config.get('collision_warning', {}).get('enabled', True)
        self.enable_traffic_signs = features_config.get('traffic_signs', {}).get('enabled', True)
        self.enable_road_analysis = features_config.get('road_analysis', {}).get('enabled', True)
        self.enable_parking = features_config.get('parking', {}).get('enabled', False)  # Off by default
        self.enable_trip_tracking = features_config.get('trip_analytics', {}).get('enabled', True)
        self.enable_driver_scoring = features_config.get('driver_scoring', {}).get('enabled', True)

        self.logger.info("Features manager initialized")
        self.logger.info(f"  Lane detection: {self.enable_lane_detection}")
        self.logger.info(f"  Blind spot monitoring: {self.enable_blind_spot}")
        self.logger.info(f"  Collision warning: {self.enable_collision_warning}")
        self.logger.info(f"  Traffic signs: {self.enable_traffic_signs}")
        self.logger.info(f"  Road analysis: {self.enable_road_analysis}")
        self.logger.info(f"  Parking assist: {self.enable_parking}")
        self.logger.info(f"  Trip tracking: {self.enable_trip_tracking}")
        self.logger.info(f"  Driver scoring: {self.enable_driver_scoring}")

    def process_frame(
        self,
        timestamp: float,
        camera_frames: Dict[int, np.ndarray],
        detections_3d: List[Detection3D],
        driver_state: DriverState,
        bev_seg: SegmentationOutput,
        vehicle_telemetry: Optional[VehicleTelemetry],
        top_risks: List
    ) -> Dict:
        """
        Process frame through all enabled features.

        Args:
            timestamp: Current timestamp
            camera_frames: Dictionary of camera frames
            detections_3d: 3D object detections
            driver_state: Driver monitoring state
            bev_seg: BEV segmentation
            vehicle_telemetry: Vehicle telemetry
            top_risks: Top risk assessments

        Returns:
            Dictionary with all feature outputs
        """
        outputs = {}

        # Get front camera (assuming camera_id=0 is front_left or similar)
        front_camera = camera_frames.get(0)  # Front camera

        # Lane detection and departure warning
        if self.enable_lane_detection and front_camera is not None:
            lanes = self.lane_detector.detect(front_camera)
            lane_state = self.lane_departure_warning.assess(
                lanes,
                vehicle_telemetry,
                front_camera.shape[1],  # image width
                timestamp
            )
            outputs['lane_state'] = lane_state
        else:
            outputs['lane_state'] = None

        # Blind spot monitoring
        if self.enable_blind_spot:
            blind_spot_warning = self.blind_spot_monitor.assess(
                detections_3d,
                vehicle_telemetry,
                timestamp
            )
            outputs['blind_spot_warning'] = blind_spot_warning
        else:
            outputs['blind_spot_warning'] = None

        # Forward collision warning
        if self.enable_collision_warning:
            collision_warning = self.collision_warning.assess(
                detections_3d,
                vehicle_telemetry,
                timestamp
            )
            outputs['collision_warning'] = collision_warning
        else:
            outputs['collision_warning'] = None

        # Traffic sign detection
        if self.enable_traffic_signs and front_camera is not None:
            traffic_signs = self.traffic_sign_detector.detect(front_camera, camera_id=0)
            outputs['traffic_signs'] = traffic_signs
        else:
            outputs['traffic_signs'] = []

        # Road surface analysis
        if self.enable_road_analysis and front_camera is not None:
            road_condition = self.road_analyzer.analyze(front_camera, bev_seg, timestamp)
            outputs['road_condition'] = road_condition
        else:
            outputs['road_condition'] = None

        # Parking assistance
        if self.enable_parking:
            parking_spaces = self.parking_assistant.detect_spaces(bev_seg, timestamp)
            outputs['parking_spaces'] = parking_spaces
        else:
            outputs['parking_spaces'] = []

        # Driver scoring
        if self.enable_driver_scoring:
            driver_score = self.driver_scoring.calculate_score(
                driver_state,
                vehicle_telemetry,
                top_risks,
                timestamp
            )
            outputs['driver_score'] = driver_score
        else:
            outputs['driver_score'] = None

        # Trip analytics update
        if self.enable_trip_tracking and self.trip_tracker.is_active:
            # Collect events for trip tracking
            events = {}
            if outputs.get('lane_state') and outputs['lane_state'].departure_warning:
                events['lane_departure'] = True
            if outputs.get('collision_warning') and outputs['collision_warning'].warning_level != 'none':
                events['collision_warning'] = True
            if outputs.get('blind_spot_warning') and outputs['blind_spot_warning'].warning_active:
                events['blind_spot_warning'] = True

            self.trip_tracker.update(vehicle_telemetry, driver_state, events)
            outputs['trip_stats'] = self.trip_tracker.get_current_stats()

            # Log events to driver scoring
            for event_type in events:
                severity = 'high' if event_type == 'collision_warning' else 'medium'
                self.driver_scoring.log_event(event_type, severity)
        else:
            outputs['trip_stats'] = None

        return outputs

    def start_trip(self):
        """Start trip tracking."""
        if self.enable_trip_tracking:
            self.trip_tracker.start_trip()
            self.logger.info("Trip tracking started")

    def end_trip(self) -> Optional[TripStats]:
        """
        End trip tracking.

        Returns:
            Completed trip statistics
        """
        if self.enable_trip_tracking:
            return self.trip_tracker.end_trip()
        return None
