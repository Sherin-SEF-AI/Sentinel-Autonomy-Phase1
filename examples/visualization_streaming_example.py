"""
Example demonstrating real-time data streaming for SENTINEL visualization.

This example shows how to:
1. Set up streaming manager
2. Push data at 30 Hz
3. Monitor performance metrics
4. Integrate with SENTINEL system
"""

import asyncio
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.backend import (
    create_server,
    create_streaming_manager,
    StreamingIntegration
)
from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert, Hazard, Risk
)
from src.core.config import ConfigManager


def create_mock_bev(timestamp: float) -> BEVOutput:
    """Create mock BEV output."""
    # Create a simple pattern that changes over time
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Draw a moving circle
    center_x = int(320 + 100 * np.sin(timestamp))
    center_y = int(320 + 100 * np.cos(timestamp))
    
    import cv2
    cv2.circle(image, (center_x, center_y), 50, (0, 255, 0), -1)
    
    return BEVOutput(
        timestamp=timestamp,
        image=image,
        mask=np.ones((640, 640), dtype=bool)
    )


def create_mock_segmentation(timestamp: float) -> SegmentationOutput:
    """Create mock segmentation output."""
    # Create segmentation with some variation
    class_map = np.random.randint(0, 9, (640, 640), dtype=np.int8)
    confidence = np.random.rand(640, 640).astype(np.float32) * 0.5 + 0.5
    
    return SegmentationOutput(
        timestamp=timestamp,
        class_map=class_map,
        confidence=confidence
    )


def create_mock_detections(timestamp: float) -> list:
    """Create mock detections."""
    detections = []
    
    # Moving vehicle
    x = 10.0 + timestamp * 2.0
    detections.append(Detection3D(
        bbox_3d=(x, 2.0, 0.0, 4.5, 1.8, 2.0, 0.1),
        class_name='vehicle',
        confidence=0.95,
        velocity=(2.0, 0.0, 0.0),
        track_id=1
    ))
    
    # Stationary pedestrian
    detections.append(Detection3D(
        bbox_3d=(15.0, -1.0, 0.0, 0.5, 1.7, 0.5, 0.0),
        class_name='pedestrian',
        confidence=0.88,
        velocity=(0.0, 0.0, 0.0),
        track_id=2
    ))
    
    return detections


def create_mock_driver_state(timestamp: float) -> DriverState:
    """Create mock driver state."""
    # Simulate varying readiness
    readiness = 85.0 + 10.0 * np.sin(timestamp * 0.5)
    
    return DriverState(
        face_detected=True,
        landmarks=np.random.rand(68, 2).astype(np.float32),
        head_pose={'roll': 0.1, 'pitch': -0.05, 'yaw': 0.02},
        gaze={'pitch': -0.1, 'yaw': 0.05, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.15},
        drowsiness={
            'score': 0.2,
            'yawn_detected': False,
            'micro_sleep': False,
            'head_nod': False
        },
        distraction={
            'type': 'none',
            'confidence': 0.95,
            'duration': 0.0
        },
        readiness_score=float(readiness)
    )


def create_mock_risk_assessment(timestamp: float) -> RiskAssessment:
    """Create mock risk assessment."""
    # Create hazard with varying risk
    risk_score = 0.5 + 0.3 * np.sin(timestamp * 0.3)
    
    hazard = Hazard(
        object_id=1,
        type='vehicle',
        position=(10.0 + timestamp * 2.0, 2.0, 0.0),
        velocity=(2.0, 0.0, 0.0),
        trajectory=[
            (10.0 + timestamp * 2.0, 2.0, 0.0),
            (15.0 + timestamp * 2.0, 2.0, 0.0),
            (20.0 + timestamp * 2.0, 2.0, 0.0)
        ],
        ttc=2.5,
        zone='front',
        base_risk=float(risk_score)
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=float(risk_score),
        driver_aware=True,
        urgency='medium' if risk_score > 0.6 else 'low',
        intervention_needed=False
    )
    
    return RiskAssessment(
        scene_graph={'objects': [1, 2], 'relationships': []},
        hazards=[hazard],
        attention_map={'front': True, 'left': False, 'right': False},
        top_risks=[risk]
    )


async def simulate_sentinel_system(integration: StreamingIntegration, duration: float = 30.0):
    """
    Simulate SENTINEL system pushing data at 30 Hz.
    
    Args:
        integration: StreamingIntegration instance
        duration: Simulation duration in seconds
    """
    print(f"\nSimulating SENTINEL system for {duration} seconds at 30 Hz...")
    print("Connect WebSocket client to: ws://localhost:8080/ws/stream")
    print("Press Ctrl+C to stop\n")
    
    target_fps = 30
    frame_interval = 1.0 / target_fps
    frame_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            loop_start = time.time()
            
            # Generate timestamp
            timestamp = frame_count / target_fps
            
            # Simulate module processing with latencies
            latency_start = time.time()
            bev = create_mock_bev(timestamp)
            bev_latency = (time.time() - latency_start) * 1000
            
            latency_start = time.time()
            segmentation = create_mock_segmentation(timestamp)
            seg_latency = (time.time() - latency_start) * 1000
            
            latency_start = time.time()
            detections = create_mock_detections(timestamp)
            det_latency = (time.time() - latency_start) * 1000
            
            latency_start = time.time()
            driver_state = create_mock_driver_state(timestamp)
            dms_latency = (time.time() - latency_start) * 1000
            
            latency_start = time.time()
            risk_assessment = create_mock_risk_assessment(timestamp)
            intel_latency = (time.time() - latency_start) * 1000
            
            # Push data to streaming manager
            integration.push_frame_data(
                timestamp=timestamp,
                bev=bev,
                segmentation=segmentation,
                detections=detections,
                driver_state=driver_state,
                risk_assessment=risk_assessment,
                alerts=[],
                latencies={
                    'camera': 5.0,
                    'bev': bev_latency,
                    'segmentation': seg_latency,
                    'detection': det_latency,
                    'dms': dms_latency,
                    'intelligence': intel_latency,
                    'total': bev_latency + seg_latency + det_latency + dms_latency + intel_latency
                }
            )
            
            # Stream current frame
            await integration.stream_current_frame(timestamp)
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                elapsed_total = time.time() - start_time
                actual_fps = frame_count / elapsed_total
                print(f"  Frame {frame_count}: {actual_fps:.1f} FPS, "
                      f"{len(integration.streaming.server.connection_manager.active_connections)} clients")
    
    except KeyboardInterrupt:
        print("\n\nStopping simulation...")
    
    print(f"\nSimulation complete: {frame_count} frames in {time.time() - start_time:.1f}s")


async def main():
    """Run the streaming example."""
    print("=" * 70)
    print("SENTINEL Real-Time Streaming Example")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config_manager = ConfigManager(str(config_path))
    config = config_manager.config
    
    # Create server
    print("\n1. Creating visualization server...")
    server = create_server(config)
    
    # Create streaming manager
    print("2. Creating streaming manager (30 Hz)...")
    streaming_manager = create_streaming_manager(server, target_fps=30)
    
    # Create integration helper
    print("3. Creating streaming integration...")
    integration = StreamingIntegration(streaming_manager)
    
    print("\n4. Server endpoints:")
    print("   - WebSocket stream: ws://localhost:8080/ws/stream")
    print("   - REST API:         http://localhost:8080/api/")
    
    print("\n5. WebSocket client example (JavaScript):")
    print("""
    const ws = new WebSocket('ws://localhost:8080/ws/stream');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`Frame ${data.timestamp}: FPS=${data.performance.fps}`);
    };
    """)
    
    print("\n6. Starting simulation...")
    
    # Run simulation
    await simulate_sentinel_system(integration, duration=30.0)
    
    print("\n" + "=" * 70)
    print("Streaming example complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
