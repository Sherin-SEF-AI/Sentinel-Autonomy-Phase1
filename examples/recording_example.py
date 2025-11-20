"""Example demonstrating the Data Recording Module."""

import time
import numpy as np
from src.recording import ScenarioRecorder
from src.core.data_structures import (
    CameraBundle, BEVOutput, Detection3D, DriverState, 
    RiskAssessment, Hazard, Risk, Alert
)


def create_mock_camera_bundle(timestamp: float) -> CameraBundle:
    """Create mock camera bundle."""
    return CameraBundle(
        timestamp=timestamp,
        interior=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        front_left=np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        front_right=np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    )


def create_mock_bev_output(timestamp: float) -> BEVOutput:
    """Create mock BEV output."""
    return BEVOutput(
        timestamp=timestamp,
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )


def create_mock_detections() -> list:
    """Create mock detections."""
    return [
        Detection3D(
            bbox_3d=(10.0, 2.0, 0.0, 4.5, 1.8, 1.5, 0.0),
            class_name='vehicle',
            confidence=0.95,
            velocity=(5.0, 0.0, 0.0),
            track_id=1
        )
    ]


def create_mock_driver_state() -> DriverState:
    """Create mock driver state."""
    return DriverState(
        face_detected=True,
        landmarks=np.random.rand(68, 2),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=85.0
    )


def create_mock_risk_assessment(high_risk: bool = False) -> RiskAssessment:
    """Create mock risk assessment."""
    hazard = Hazard(
        object_id=1,
        type='vehicle',
        position=(10.0, 2.0, 0.0),
        velocity=(5.0, 0.0, 0.0),
        trajectory=[(10.0, 2.0, 0.0), (15.0, 2.0, 0.0)],
        ttc=2.0 if not high_risk else 1.2,
        zone='front',
        base_risk=0.5 if not high_risk else 0.8
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=0.6 if not high_risk else 0.85,
        driver_aware=True,
        urgency='medium' if not high_risk else 'critical',
        intervention_needed=high_risk
    )
    
    return RiskAssessment(
        scene_graph={'objects': [1], 'relationships': []},
        hazards=[hazard],
        attention_map={'current_zone': 'front', 'zones': {}},
        top_risks=[risk]
    )


def create_mock_alerts(high_risk: bool = False) -> list:
    """Create mock alerts."""
    if not high_risk:
        return []
    
    return [
        Alert(
            timestamp=time.time(),
            urgency='critical',
            modalities=['visual', 'audio'],
            message='Vehicle ahead - low TTC',
            hazard_id=1,
            dismissed=False
        )
    ]


def main():
    """Demonstrate recording module functionality."""
    print("=" * 60)
    print("SENTINEL Data Recording Module Example")
    print("=" * 60)
    
    # Initialize recorder
    config = {
        'enabled': True,
        'triggers': {
            'risk_threshold': 0.7,
            'ttc_threshold': 1.5
        },
        'storage_path': 'scenarios/',
        'max_duration': 30.0
    }
    
    recorder = ScenarioRecorder(config)
    print("\n✓ ScenarioRecorder initialized")
    
    # Simulate normal driving (no triggers)
    print("\n1. Simulating normal driving (30 frames)...")
    start_time = time.time()
    
    for i in range(30):
        timestamp = start_time + i * 0.033  # 30 FPS
        
        recorder.process_frame(
            timestamp=timestamp,
            camera_bundle=create_mock_camera_bundle(timestamp),
            bev_output=create_mock_bev_output(timestamp),
            detections_3d=create_mock_detections(),
            driver_state=create_mock_driver_state(),
            risk_assessment=create_mock_risk_assessment(high_risk=False),
            alerts=create_mock_alerts(high_risk=False)
        )
    
    print(f"   Recording active: {recorder.is_recording()}")
    
    # Simulate high-risk scenario (triggers recording)
    print("\n2. Simulating high-risk scenario (60 frames)...")
    
    for i in range(60):
        timestamp = start_time + (30 + i) * 0.033
        
        recorder.process_frame(
            timestamp=timestamp,
            camera_bundle=create_mock_camera_bundle(timestamp),
            bev_output=create_mock_bev_output(timestamp),
            detections_3d=create_mock_detections(),
            driver_state=create_mock_driver_state(),
            risk_assessment=create_mock_risk_assessment(high_risk=True),
            alerts=create_mock_alerts(high_risk=True)
        )
        
        if i == 0:
            print(f"   Recording triggered!")
            print(f"   Recording active: {recorder.is_recording()}")
    
    # Stop and export
    print("\n3. Stopping recording and exporting scenario...")
    recorder.stop_recording()
    
    print(f"   Recorded frames: {recorder.get_num_recorded_frames()}")
    print(f"   Duration: {recorder.get_recording_duration():.2f}s")
    
    scenario_path = recorder.export_scenario(location="37.7749,-122.4194")
    
    if scenario_path:
        print(f"   ✓ Scenario exported to: {scenario_path}")
    
    # List scenarios
    print("\n4. Listing available scenarios...")
    scenarios = recorder.list_scenarios()
    print(f"   Found {len(scenarios)} scenario(s):")
    for scenario in scenarios[:5]:  # Show first 5
        print(f"   - {scenario}")
    
    # Playback
    if scenarios:
        print("\n5. Loading and playing back scenario...")
        scenario_name = scenarios[0]
        
        if recorder.load_scenario(scenario_name):
            print(f"   ✓ Loaded scenario: {scenario_name}")
            
            metadata = recorder.get_scenario_metadata()
            print(f"   Duration: {metadata['duration']:.2f}s")
            print(f"   Frames: {metadata['num_frames']}")
            print(f"   Trigger: {metadata['trigger']['type']}")
            
            # Get first frame
            frame = recorder.get_playback_frame(0)
            if frame:
                print(f"\n   Frame 0:")
                print(f"   - Cameras: {list(frame['camera_frames'].keys())}")
                print(f"   - Timestamp: {frame['annotations']['timestamp']:.3f}")
                print(f"   - Detections: {len(frame['annotations']['detections_3d'])}")
                print(f"   - Alerts: {len(frame['annotations']['alerts'])}")
            
            # Navigate frames
            print("\n   Navigating frames...")
            next_frame = recorder.next_playback_frame()
            if next_frame:
                print(f"   - Next frame: {next_frame['frame_index']}")
            
            prev_frame = recorder.previous_playback_frame()
            if prev_frame:
                print(f"   - Previous frame: {prev_frame['frame_index']}")
            
            # Seek
            if recorder.seek_playback(10):
                print(f"   - Seeked to frame 10")
            
            recorder.close_playback()
            print("   ✓ Playback closed")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
