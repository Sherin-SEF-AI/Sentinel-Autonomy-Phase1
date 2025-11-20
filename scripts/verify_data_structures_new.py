#!/usr/bin/env python3
"""Verification script for new data structures (MapFeature, Lane, VehicleTelemetry)."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module file to avoid cv2 dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "data_structures",
    Path(__file__).parent.parent / "src" / "core" / "data_structures.py"
)
data_structures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_structures)

MapFeature = data_structures.MapFeature
Lane = data_structures.Lane
VehicleTelemetry = data_structures.VehicleTelemetry

def test_map_feature():
    """Test MapFeature dataclass."""
    print("Testing MapFeature...")
    
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
    assert feature.attributes["sign_type"] == "stop"
    assert len(feature.geometry) == 2
    
    print("  ✓ MapFeature initialization")
    print("  ✓ MapFeature attributes")
    print("  ✓ MapFeature geometry")

def test_lane():
    """Test Lane dataclass."""
    print("\nTesting Lane...")
    
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
    assert lane.width == 3.5
    assert lane.speed_limit == 50.0
    assert lane.lane_type == "driving"
    assert len(lane.predecessors) == 1
    assert len(lane.successors) == 2
    
    print("  ✓ Lane initialization")
    print("  ✓ Lane geometry (centerline, boundaries)")
    print("  ✓ Lane connectivity (predecessors/successors)")
    
    # Test optional speed limit
    lane_no_limit = Lane(
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
    assert lane_no_limit.speed_limit is None
    print("  ✓ Optional speed_limit")

def test_vehicle_telemetry():
    """Test VehicleTelemetry dataclass."""
    print("\nTesting VehicleTelemetry...")
    
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
    assert telemetry.speed == 15.5
    assert telemetry.steering_angle == 0.15
    assert telemetry.brake_pressure == 0.0
    assert 0.0 <= telemetry.throttle_position <= 1.0
    assert telemetry.gear == 3
    assert telemetry.turn_signal in ["left", "right", "none"]
    
    print("  ✓ VehicleTelemetry initialization")
    print("  ✓ Speed (m/s)")
    print("  ✓ Steering angle (radians)")
    print("  ✓ Brake pressure (bar)")
    print("  ✓ Throttle position (0-1)")
    print("  ✓ Gear")
    print("  ✓ Turn signal")
    
    # Test all turn signal values
    for signal in ["left", "right", "none"]:
        t = VehicleTelemetry(0.0, 10.0, 0.0, 0.0, 0.5, 2, signal)
        assert t.turn_signal == signal
    print("  ✓ All turn signal values")

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("DATA STRUCTURES VERIFICATION")
    print("=" * 60)
    
    try:
        test_map_feature()
        test_lane()
        test_vehicle_telemetry()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
