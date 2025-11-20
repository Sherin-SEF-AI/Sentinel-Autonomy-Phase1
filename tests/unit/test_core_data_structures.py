"""Test suite for core data structures module."""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from src.core.data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput,
    Detection2D, Detection3D, DriverState,
    Hazard, Risk, RiskAssessment, Alert,
    MapFeature, Lane, VehicleTelemetry
)


class TestCameraBundle:
    """Test suite for CameraBundle dataclass."""
    
    def test_initialization(self):
        """Test that CameraBundle initializes correctly with valid data."""
        timestamp = 1234567890.123
        interior = np.zeros((480, 640, 3), dtype=np.uint8)
        front_left = np.zeros((720, 1280, 3), dtype=np.uint8)
        front_right = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        bundle = CameraBundle(
            timestamp=timestamp,
            interior=interior,
            front_left=front_left,
            front_right=front_right
        )
        
        assert bundle.timestamp == timestamp
        assert bundle.interior.shape == (480, 640, 3)
        assert bundle.front_left.shape == (720, 1280, 3)
        assert bundle.front_right.shape == (720, 1280, 3)
    
    def test_frame_shapes(self):
        """Test that frame arrays have correct shapes."""
        bundle = CameraBundle(
            timestamp=0.0,
            interior=np.zeros((480, 640, 3), dtype=np.uint8),
            front_left=np.zeros((720, 1280, 3), dtype=np.uint8),
            front_right=np.zeros((720, 1280, 3), dtype=np.uint8)
        )
        
        assert bundle.interior.dtype == np.uint8
        assert bundle.front_left.dtype == np.uint8
        assert bundle.front_right.dtype == np.uint8


class TestBEVOutput:
    """Test suite for BEVOutput dataclass."""
    
    def test_initialization(self):
        """Test that BEVOutput initializes correctly."""
        timestamp = 1234567890.123
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        mask = np.ones((640, 640), dtype=bool)
        
        output = BEVOutput(timestamp=timestamp, image=image, mask=mask)
        
        assert output.timestamp == timestamp
        assert output.image.shape == (640, 640, 3)
        assert output.mask.shape == (640, 640)
        assert output.mask.dtype == bool
    
    def test_mask_validity(self):
        """Test that mask contains valid boolean values."""
        output = BEVOutput(
            timestamp=0.0,
            image=np.zeros((640, 640, 3), dtype=np.uint8),
            mask=np.random.rand(640, 640) > 0.5
        )
        
        assert np.all((output.mask == True) | (output.mask == False))


class TestSegmentationOutput:
    """Test suite for SegmentationOutput dataclass."""
    
    def test_initialization(self):
        """Test that SegmentationOutput initializes correctly."""
        timestamp = 1234567890.123
        class_map = np.random.randint(0, 9, (640, 640), dtype=np.int8)
        confidence = np.random.rand(640, 640).astype(np.float32)
        
        output = SegmentationOutput(
            timestamp=timestamp,
            class_map=class_map,
            confidence=confidence
        )
        
        assert output.timestamp == timestamp
        assert output.class_map.shape == (640, 640)
        assert output.class_map.dtype == np.int8
        assert output.confidence.shape == (640, 640)
        assert output.confidence.dtype == np.float32

    def test_confidence_range(self):
        """Test that confidence values are in valid range [0, 1]."""
        confidence = np.random.rand(640, 640).astype(np.float32)
        output = SegmentationOutput(
            timestamp=0.0,
            class_map=np.zeros((640, 640), dtype=np.int8),
            confidence=confidence
        )
        
        assert np.all(output.confidence >= 0.0)
        assert np.all(output.confidence <= 1.0)


class TestDetection2D:
    """Test suite for Detection2D dataclass."""
    
    def test_initialization(self):
        """Test that Detection2D initializes correctly."""
        detection = Detection2D(
            bbox=(100.0, 200.0, 300.0, 400.0),
            class_name="vehicle",
            confidence=0.95,
            camera_id=1
        )
        
        assert detection.bbox == (100.0, 200.0, 300.0, 400.0)
        assert detection.class_name == "vehicle"
        assert detection.confidence == 0.95
        assert detection.camera_id == 1
    
    def test_bbox_format(self):
        """Test that bbox has correct format (x1, y1, x2, y2)."""
        detection = Detection2D(
            bbox=(10.0, 20.0, 100.0, 200.0),
            class_name="pedestrian",
            confidence=0.85,
            camera_id=0
        )
        
        x1, y1, x2, y2 = detection.bbox
        assert x2 > x1  # x2 should be greater than x1
        assert y2 > y1  # y2 should be greater than y1


class TestDetection3D:
    """Test suite for Detection3D dataclass."""
    
    def test_initialization(self):
        """Test that Detection3D initializes correctly."""
        detection = Detection3D(
            bbox_3d=(5.0, 2.0, 0.0, 2.0, 1.5, 4.5, 0.1),
            class_name="vehicle",
            confidence=0.92,
            velocity=(10.0, 0.5, 0.0),
            track_id=42
        )
        
        assert len(detection.bbox_3d) == 7
        assert detection.class_name == "vehicle"
        assert detection.confidence == 0.92
        assert len(detection.velocity) == 3
        assert detection.track_id == 42

    def test_velocity_components(self):
        """Test that velocity has correct 3D components."""
        detection = Detection3D(
            bbox_3d=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
            class_name="vehicle",
            confidence=0.9,
            velocity=(15.0, -2.0, 0.5),
            track_id=1
        )
        
        vx, vy, vz = detection.velocity
        assert isinstance(vx, (int, float))
        assert isinstance(vy, (int, float))
        assert isinstance(vz, (int, float))


class TestDriverState:
    """Test suite for DriverState dataclass."""
    
    def test_initialization(self):
        """Test that DriverState initializes correctly."""
        state = DriverState(
            face_detected=True,
            landmarks=np.random.rand(68, 2),
            head_pose={'roll': 0.1, 'pitch': -0.2, 'yaw': 0.05},
            gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'forward'},
            eye_state={'left_ear': 0.3, 'right_ear': 0.32, 'perclos': 0.15},
            drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
            distraction={'type': 'none', 'confidence': 0.95, 'duration': 0.0},
            readiness_score=85.0
        )
        
        assert state.face_detected is True
        assert state.landmarks.shape == (68, 2)
        assert 'roll' in state.head_pose
        assert 'attention_zone' in state.gaze
        assert 0.0 <= state.readiness_score <= 100.0
    
    def test_readiness_score_range(self):
        """Test that readiness score is in valid range [0, 100]."""
        state = DriverState(
            face_detected=True,
            landmarks=np.zeros((68, 2)),
            head_pose={},
            gaze={},
            eye_state={},
            drowsiness={},
            distraction={},
            readiness_score=75.5
        )
        
        assert 0.0 <= state.readiness_score <= 100.0


class TestHazard:
    """Test suite for Hazard dataclass."""
    
    def test_initialization(self):
        """Test that Hazard initializes correctly."""
        hazard = Hazard(
            object_id=123,
            type="vehicle",
            position=(10.0, 2.0, 0.0),
            velocity=(15.0, 0.0, 0.0),
            trajectory=[(10.0, 2.0, 0.0), (12.0, 2.0, 0.0), (14.0, 2.0, 0.0)],
            ttc=2.5,
            zone="front",
            base_risk=0.7
        )
        
        assert hazard.object_id == 123
        assert hazard.type == "vehicle"
        assert len(hazard.position) == 3
        assert len(hazard.velocity) == 3
        assert len(hazard.trajectory) == 3
        assert hazard.ttc > 0
        assert 0.0 <= hazard.base_risk <= 1.0
    
    def test_base_risk_range(self):
        """Test that base_risk is in valid range [0, 1]."""
        hazard = Hazard(
            object_id=1,
            type="pedestrian",
            position=(5.0, 1.0, 0.0),
            velocity=(1.0, 0.0, 0.0),
            trajectory=[],
            ttc=3.0,
            zone="front_left",
            base_risk=0.85
        )
        
        assert 0.0 <= hazard.base_risk <= 1.0


class TestRisk:
    """Test suite for Risk dataclass."""
    
    def test_initialization(self):
        """Test that Risk initializes correctly."""
        hazard = Hazard(
            object_id=1,
            type="vehicle",
            position=(10.0, 0.0, 0.0),
            velocity=(10.0, 0.0, 0.0),
            trajectory=[],
            ttc=1.5,
            zone="front",
            base_risk=0.8
        )
        
        risk = Risk(
            hazard=hazard,
            contextual_score=0.9,
            driver_aware=False,
            urgency="critical",
            intervention_needed=True
        )
        
        assert risk.hazard == hazard
        assert 0.0 <= risk.contextual_score <= 1.0
        assert risk.driver_aware is False
        assert risk.urgency in ['low', 'medium', 'high', 'critical']
        assert risk.intervention_needed is True
    
    def test_urgency_levels(self):
        """Test that urgency has valid values."""
        hazard = Hazard(1, "vehicle", (0, 0, 0), (0, 0, 0), [], 1.0, "front", 0.5)
        
        for urgency in ['low', 'medium', 'high', 'critical']:
            risk = Risk(
                hazard=hazard,
                contextual_score=0.5,
                driver_aware=True,
                urgency=urgency,
                intervention_needed=False
            )
            assert risk.urgency == urgency


class TestRiskAssessment:
    """Test suite for RiskAssessment dataclass."""
    
    def test_initialization(self):
        """Test that RiskAssessment initializes correctly."""
        hazard = Hazard(1, "vehicle", (10, 0, 0), (10, 0, 0), [], 2.0, "front", 0.6)
        risk = Risk(hazard, 0.7, False, "high", True)
        
        assessment = RiskAssessment(
            scene_graph={'objects': [], 'relationships': []},
            hazards=[hazard],
            attention_map={'zones': {}},
            top_risks=[risk]
        )
        
        assert isinstance(assessment.scene_graph, dict)
        assert isinstance(assessment.hazards, list)
        assert isinstance(assessment.attention_map, dict)
        assert isinstance(assessment.top_risks, list)
        assert len(assessment.hazards) == 1
        assert len(assessment.top_risks) == 1


class TestAlert:
    """Test suite for Alert dataclass."""
    
    def test_initialization(self):
        """Test that Alert initializes correctly."""
        alert = Alert(
            timestamp=1234567890.123,
            urgency="warning",
            modalities=["visual", "audio"],
            message="Vehicle approaching from left",
            hazard_id=42,
            dismissed=False
        )
        
        assert alert.timestamp > 0
        assert alert.urgency in ["info", "warning", "critical"]
        assert isinstance(alert.modalities, list)
        assert len(alert.message) > 0
        assert alert.hazard_id >= 0
        assert alert.dismissed is False
    
    def test_modalities(self):
        """Test that modalities contain valid values."""
        alert = Alert(
            timestamp=0.0,
            urgency="critical",
            modalities=["visual", "audio", "haptic"],
            message="Emergency brake required",
            hazard_id=1,
            dismissed=False
        )
        
        valid_modalities = {"visual", "audio", "haptic"}
        for modality in alert.modalities:
            assert modality in valid_modalities


class TestMapFeature:
    """Test suite for MapFeature dataclass (newly added)."""
    
    def test_initialization(self):
        """Test that MapFeature initializes correctly with valid data."""
        feature = MapFeature(
            feature_id="sign_001",
            type="sign",
            position=(15.0, 3.0, 1.5),
            attributes={"sign_type": "stop", "text": "STOP"},
            geometry=[(15.0, 3.0), (15.5, 3.0)]
        )
        
        assert feature.feature_id == "sign_001"
        assert feature.type == "sign"
        assert len(feature.position) == 3
        assert isinstance(feature.attributes, dict)
        assert isinstance(feature.geometry, list)
    
    def test_feature_types(self):
        """Test that MapFeature supports all valid feature types."""
        valid_types = ['lane', 'sign', 'light', 'crosswalk', 'boundary']
        
        for feature_type in valid_types:
            feature = MapFeature(
                feature_id=f"{feature_type}_001",
                type=feature_type,
                position=(10.0, 0.0, 0.0),
                attributes={},
                geometry=[]
            )
            assert feature.type == feature_type
    
    def test_position_coordinates(self):
        """Test that position has correct 3D coordinates in vehicle frame."""
        feature = MapFeature(
            feature_id="light_001",
            type="light",
            position=(20.0, -2.5, 3.0),
            attributes={"state": "red"},
            geometry=[]
        )
        
        x, y, z = feature.position
        assert isinstance(x, (int, float))
        assert isinstance(y, (int, float))
        assert isinstance(z, (int, float))
    
    def test_geometry_polyline(self):
        """Test that geometry contains valid polyline points."""
        geometry = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)]
        feature = MapFeature(
            feature_id="boundary_001",
            type="boundary",
            position=(1.5, 0.75, 0.0),
            attributes={"boundary_type": "curb"},
            geometry=geometry
        )
        
        assert len(feature.geometry) == 4
        for point in feature.geometry:
            assert len(point) == 2
            assert isinstance(point[0], (int, float))
            assert isinstance(point[1], (int, float))
    
    def test_attributes_flexibility(self):
        """Test that attributes can store various feature-specific data."""
        # Traffic sign attributes
        sign_feature = MapFeature(
            feature_id="sign_002",
            type="sign",
            position=(10.0, 2.0, 1.0),
            attributes={
                "sign_type": "speed_limit",
                "speed_limit": 50,
                "units": "km/h"
            },
            geometry=[]
        )
        assert sign_feature.attributes["speed_limit"] == 50
        
        # Traffic light attributes
        light_feature = MapFeature(
            feature_id="light_002",
            type="light",
            position=(25.0, 0.0, 3.5),
            attributes={
                "state": "green",
                "time_remaining": 15.0
            },
            geometry=[]
        )
        assert light_feature.attributes["state"] == "green"


class TestLane:
    """Test suite for Lane dataclass (newly added)."""
    
    def test_initialization(self):
        """Test that Lane initializes correctly with valid data."""
        lane = Lane(
            lane_id="lane_001",
            centerline=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)],
            left_boundary=[(0.0, 1.75, 0.0), (10.0, 1.75, 0.0), (20.0, 1.75, 0.0)],
            right_boundary=[(0.0, -1.75, 0.0), (10.0, -1.75, 0.0), (20.0, -1.75, 0.0)],
            width=3.5,
            speed_limit=50.0,
            lane_type="driving",
            predecessors=["lane_000"],
            successors=["lane_002", "lane_003"]
        )
        
        assert lane.lane_id == "lane_001"
        assert len(lane.centerline) == 3
        assert len(lane.left_boundary) == 3
        assert len(lane.right_boundary) == 3
        assert lane.width > 0
        assert lane.speed_limit == 50.0
        assert lane.lane_type == "driving"
        assert len(lane.predecessors) == 1
        assert len(lane.successors) == 2
    
    def test_lane_types(self):
        """Test that Lane supports various lane types."""
        lane_types = ['driving', 'parking', 'shoulder', 'bike', 'bus']
        
        for lane_type in lane_types:
            lane = Lane(
                lane_id=f"lane_{lane_type}",
                centerline=[],
                left_boundary=[],
                right_boundary=[],
                width=3.0,
                speed_limit=None,
                lane_type=lane_type,
                predecessors=[],
                successors=[]
            )
            assert lane.lane_type == lane_type
    
    def test_optional_speed_limit(self):
        """Test that speed_limit can be None for lanes without limits."""
        lane = Lane(
            lane_id="lane_parking",
            centerline=[],
            left_boundary=[],
            right_boundary=[],
            width=2.5,
            speed_limit=None,
            lane_type="parking",
            predecessors=[],
            successors=[]
        )
        
        assert lane.speed_limit is None
    
    def test_centerline_points(self):
        """Test that centerline contains valid 3D points."""
        centerline = [
            (0.0, 0.0, 0.0),
            (5.0, 0.5, 0.0),
            (10.0, 1.0, 0.0),
            (15.0, 1.5, 0.0)
        ]
        
        lane = Lane(
            lane_id="lane_curved",
            centerline=centerline,
            left_boundary=[],
            right_boundary=[],
            width=3.5,
            speed_limit=60.0,
            lane_type="driving",
            predecessors=[],
            successors=[]
        )
        
        assert len(lane.centerline) == 4
        for point in lane.centerline:
            assert len(point) == 3
            x, y, z = point
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert isinstance(z, (int, float))
    
    def test_lane_connectivity(self):
        """Test that lane connectivity (predecessors/successors) works correctly."""
        lane = Lane(
            lane_id="lane_002",
            centerline=[],
            left_boundary=[],
            right_boundary=[],
            width=3.5,
            speed_limit=50.0,
            lane_type="driving",
            predecessors=["lane_001"],
            successors=["lane_003", "lane_004"]
        )
        
        assert "lane_001" in lane.predecessors
        assert "lane_003" in lane.successors
        assert "lane_004" in lane.successors
        assert len(lane.predecessors) == 1
        assert len(lane.successors) == 2
    
    def test_lane_width_positive(self):
        """Test that lane width is always positive."""
        lane = Lane(
            lane_id="lane_narrow",
            centerline=[],
            left_boundary=[],
            right_boundary=[],
            width=2.8,
            speed_limit=30.0,
            lane_type="driving",
            predecessors=[],
            successors=[]
        )
        
        assert lane.width > 0


class TestVehicleTelemetry:
    """Test suite for VehicleTelemetry dataclass (newly added)."""
    
    def test_initialization(self):
        """Test that VehicleTelemetry initializes correctly with valid data."""
        telemetry = VehicleTelemetry(
            timestamp=1234567890.123,
            speed=15.5,
            steering_angle=0.15,
            brake_pressure=0.0,
            throttle_position=0.4,
            gear=3,
            turn_signal="left"
        )
        
        assert telemetry.timestamp > 0
        assert telemetry.speed >= 0
        assert isinstance(telemetry.steering_angle, (int, float))
        assert telemetry.brake_pressure >= 0
        assert 0.0 <= telemetry.throttle_position <= 1.0
        assert isinstance(telemetry.gear, int)
        assert telemetry.turn_signal in ["left", "right", "none"]
    
    def test_speed_units(self):
        """Test that speed is in m/s (meters per second)."""
        # 100 km/h = 27.78 m/s
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=27.78,
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.5,
            gear=4,
            turn_signal="none"
        )
        
        assert telemetry.speed > 0
        # Reasonable highway speed in m/s
        assert 0 <= telemetry.speed <= 50  # ~180 km/h max
    
    def test_steering_angle_radians(self):
        """Test that steering angle is in radians."""
        # Typical steering range: -π/4 to π/4 radians
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=10.0,
            steering_angle=0.785,  # ~45 degrees
            brake_pressure=0.0,
            throttle_position=0.3,
            gear=2,
            turn_signal="right"
        )
        
        assert isinstance(telemetry.steering_angle, (int, float))
        # Reasonable steering angle range
        assert -1.57 <= telemetry.steering_angle <= 1.57  # ±90 degrees
    
    def test_brake_pressure_units(self):
        """Test that brake pressure is in bar."""
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=20.0,
            steering_angle=0.0,
            brake_pressure=50.0,  # bar
            throttle_position=0.0,
            gear=3,
            turn_signal="none"
        )
        
        assert telemetry.brake_pressure >= 0
        # Typical brake pressure range: 0-150 bar
        assert telemetry.brake_pressure <= 200
    
    def test_throttle_position_range(self):
        """Test that throttle position is in valid range [0, 1]."""
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=25.0,
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.75,
            gear=4,
            turn_signal="none"
        )
        
        assert 0.0 <= telemetry.throttle_position <= 1.0
    
    def test_gear_values(self):
        """Test that gear has valid integer values."""
        # Test various gear positions
        for gear in [-1, 0, 1, 2, 3, 4, 5, 6]:
            telemetry = VehicleTelemetry(
                timestamp=0.0,
                speed=0.0 if gear <= 0 else 10.0,
                steering_angle=0.0,
                brake_pressure=0.0,
                throttle_position=0.0,
                gear=gear,
                turn_signal="none"
            )
            assert telemetry.gear == gear
            assert isinstance(telemetry.gear, int)
    
    def test_turn_signal_values(self):
        """Test that turn_signal has valid values."""
        valid_signals = ["left", "right", "none"]
        
        for signal in valid_signals:
            telemetry = VehicleTelemetry(
                timestamp=0.0,
                speed=15.0,
                steering_angle=0.0,
                brake_pressure=0.0,
                throttle_position=0.4,
                gear=3,
                turn_signal=signal
            )
            assert telemetry.turn_signal == signal
            assert telemetry.turn_signal in valid_signals
    
    def test_stationary_vehicle(self):
        """Test telemetry for a stationary vehicle."""
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=0.0,
            steering_angle=0.0,
            brake_pressure=80.0,
            throttle_position=0.0,
            gear=0,  # Neutral
            turn_signal="none"
        )
        
        assert telemetry.speed == 0.0
        assert telemetry.throttle_position == 0.0
        assert telemetry.brake_pressure > 0
    
    def test_accelerating_vehicle(self):
        """Test telemetry for an accelerating vehicle."""
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=12.5,
            steering_angle=0.0,
            brake_pressure=0.0,
            throttle_position=0.8,
            gear=2,
            turn_signal="none"
        )
        
        assert telemetry.speed > 0
        assert telemetry.throttle_position > 0
        assert telemetry.brake_pressure == 0.0


@pytest.mark.performance
class TestDataStructuresPerformance:
    """Performance tests for data structure creation."""
    
    def test_camera_bundle_creation_performance(self):
        """Test that CameraBundle creation is fast."""
        import time
        
        start_time = time.perf_counter()
        for _ in range(1000):
            bundle = CameraBundle(
                timestamp=time.time(),
                interior=np.zeros((480, 640, 3), dtype=np.uint8),
                front_left=np.zeros((720, 1280, 3), dtype=np.uint8),
                front_right=np.zeros((720, 1280, 3), dtype=np.uint8)
            )
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        avg_time_ms = execution_time_ms / 1000
        assert avg_time_ms < 1.0, f"Average creation took {avg_time_ms:.3f}ms, expected < 1ms"
    
    def test_detection3d_creation_performance(self):
        """Test that Detection3D creation is fast."""
        import time
        
        start_time = time.perf_counter()
        for i in range(10000):
            detection = Detection3D(
                bbox_3d=(5.0, 2.0, 0.0, 2.0, 1.5, 4.5, 0.1),
                class_name="vehicle",
                confidence=0.92,
                velocity=(10.0, 0.5, 0.0),
                track_id=i
            )
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        avg_time_us = (execution_time_ms * 1000) / 10000
        assert avg_time_us < 10.0, f"Average creation took {avg_time_us:.2f}μs, expected < 10μs"
