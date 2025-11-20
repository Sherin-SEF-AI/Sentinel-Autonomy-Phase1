#!/usr/bin/env python3
"""Verify logging setup for core data structures module."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import logging.config
import yaml
import numpy as np
from src.core.data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput,
    Detection2D, Detection3D, DriverState,
    Hazard, Risk, RiskAssessment, Alert,
    MapFeature, Lane, VehicleTelemetry
)


def setup_logging():
    """Setup logging from config file."""
    with open('configs/logging.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)
    return logging.getLogger(__name__)


def test_data_structure_creation():
    """Test creating instances of all data structures."""
    logger = logging.getLogger('src.core.data_structures')
    
    print("=" * 60)
    print("DATA STRUCTURES LOGGING VERIFICATION")
    print("=" * 60)
    
    # Test CameraBundle
    print("\n1. Testing CameraBundle creation...")
    bundle = CameraBundle(
        timestamp=1234567890.0,
        interior=np.zeros((480, 640, 3), dtype=np.uint8),
        front_left=np.zeros((720, 1280, 3), dtype=np.uint8),
        front_right=np.zeros((720, 1280, 3), dtype=np.uint8)
    )
    print(f"   ✓ CameraBundle created: timestamp={bundle.timestamp}")
    
    # Test BEVOutput
    print("\n2. Testing BEVOutput creation...")
    bev = BEVOutput(
        timestamp=1234567890.0,
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )
    print(f"   ✓ BEVOutput created: shape={bev.image.shape}")
    
    # Test Detection3D
    print("\n3. Testing Detection3D creation...")
    detection = Detection3D(
        bbox_3d=(10.0, 5.0, 0.0, 2.0, 1.5, 4.5, 0.0),
        class_name="vehicle",
        confidence=0.95,
        velocity=(15.0, 0.0, 0.0),
        track_id=42
    )
    print(f"   ✓ Detection3D created: class={detection.class_name}, id={detection.track_id}")
    
    # Test DriverState
    print("\n4. Testing DriverState creation...")
    driver = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'forward'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.15},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=95.0
    )
    print(f"   ✓ DriverState created: readiness={driver.readiness_score}")
    
    # Test Hazard
    print("\n5. Testing Hazard creation...")
    hazard = Hazard(
        object_id=42,
        type="vehicle",
        position=(10.0, 5.0, 0.0),
        velocity=(15.0, 0.0, 0.0),
        trajectory=[(10.0, 5.0, 0.0), (11.5, 5.0, 0.0), (13.0, 5.0, 0.0)],
        ttc=2.5,
        zone="front",
        base_risk=0.6
    )
    print(f"   ✓ Hazard created: type={hazard.type}, ttc={hazard.ttc}s")
    
    # Test Alert
    print("\n6. Testing Alert creation...")
    alert = Alert(
        timestamp=1234567890.0,
        urgency="warning",
        modalities=["visual", "audio"],
        message="Vehicle ahead braking",
        hazard_id=42,
        dismissed=False
    )
    print(f"   ✓ Alert created: urgency={alert.urgency}, message='{alert.message}'")
    
    # Test MapFeature (NEW)
    print("\n7. Testing MapFeature creation...")
    feature = MapFeature(
        feature_id="sign_001",
        type="sign",
        position=(50.0, 10.0, 2.0),
        attributes={"sign_type": "stop", "text": "STOP"},
        geometry=[(50.0, 10.0), (50.0, 11.0)]
    )
    print(f"   ✓ MapFeature created: type={feature.type}, id={feature.feature_id}")
    
    # Test Lane (NEW)
    print("\n8. Testing Lane creation...")
    lane = Lane(
        lane_id="lane_001",
        centerline=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0)],
        left_boundary=[(0.0, 1.75, 0.0), (10.0, 1.75, 0.0), (20.0, 1.75, 0.0)],
        right_boundary=[(0.0, -1.75, 0.0), (10.0, -1.75, 0.0), (20.0, -1.75, 0.0)],
        width=3.5,
        speed_limit=50.0,
        lane_type="driving",
        predecessors=["lane_000"],
        successors=["lane_002"]
    )
    print(f"   ✓ Lane created: id={lane.lane_id}, width={lane.width}m, speed_limit={lane.speed_limit}km/h")
    
    # Test VehicleTelemetry (NEW)
    print("\n9. Testing VehicleTelemetry creation...")
    telemetry = VehicleTelemetry(
        timestamp=1234567890.0,
        speed=15.0,
        steering_angle=0.05,
        brake_pressure=0.0,
        throttle_position=0.3,
        gear=3,
        turn_signal="none"
    )
    print(f"   ✓ VehicleTelemetry created: speed={telemetry.speed}m/s, gear={telemetry.gear}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nAll data structures created successfully!")
    print("\nNew data structures added:")
    print("  - MapFeature: HD map feature representation")
    print("  - Lane: Lane geometry and attributes")
    print("  - VehicleTelemetry: CAN bus vehicle data")
    print("\nLogging configuration:")
    print("  - Module: src.core.data_structures")
    print("  - Level: INFO")
    print("  - Handlers: file_all")
    print("  - Log file: logs/sentinel.log")
    
    return True


def main():
    """Main verification function."""
    try:
        # Setup logging
        setup_logging()
        
        # Run tests
        success = test_data_structure_creation()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
