"""
Complete visualization system example for SENTINEL.

This example demonstrates:
1. Setting up the complete visualization system
2. Integrating with SENTINEL processing pipeline
3. Real-time streaming to dashboard
4. Scenario recording and playback
"""

import asyncio
import numpy as np
import time
from pathlib import Path
import sys
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    create_server,
    create_streaming_manager,
    StreamingIntegration
)
from src.core.config import ConfigManager
from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert, Hazard, Risk
)


class MockSentinelSystem:
    """Mock SENTINEL system for demonstration."""
    
    def __init__(self, config):
        self.config = config
        
        # Create visualization server
        print("Creating visualization server...")
        self.viz_server = create_server(config)
        
        # Create streaming manager
        print("Creating streaming manager...")
        self.streaming_manager = create_streaming_manager(
            self.viz_server,
            target_fps=30
        )
        
        # Create integration helper
        self.viz_integration = StreamingIntegration(self.streaming_manager)
        
        # Start server in background thread
        print("Starting server in background...")
        self.server_thread = threading.Thread(
            target=self.viz_server.run,
            kwargs={'host': '0.0.0.0', 'port': 8080},
            daemon=True
        )
        self.server_thread.start()
        
        # Give server time to start
        time.sleep(2)
        
        print("\n" + "=" * 70)
        print("SENTINEL Visualization System Ready!")
        print("=" * 70)
        print("\nAccess points:")
        print("  - Live Dashboard:  http://localhost:8080/")
        print("  - Playback:        http://localhost:8080/playback.html")
        print("  - API Docs:        http://localhost:8080/docs")
        print("  - WebSocket:       ws://localhost:8080/ws/stream")
        print("\n" + "=" * 70 + "\n")
    
    def create_mock_frame_data(self, timestamp: float):
        """Create mock frame data."""
        
        # Mock BEV
        bev = BEVOutput(
            timestamp=timestamp,
            image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
            mask=np.ones((640, 640), dtype=bool)
        )
        
        # Mock segmentation
        segmentation = SegmentationOutput(
            timestamp=timestamp,
            class_map=np.random.randint(0, 9, (640, 640), dtype=np.int8),
            confidence=np.random.rand(640, 640).astype(np.float32)
        )
        
        # Mock detections
        detections = [
            Detection3D(
                bbox_3d=(10.0 + timestamp, 2.0, 0.0, 4.5, 1.8, 2.0, 0.1),
                class_name='vehicle',
                confidence=0.95,
                velocity=(2.0, 0.0, 0.0),
                track_id=1
            ),
            Detection3D(
                bbox_3d=(15.0, -1.0, 0.0, 0.5, 1.7, 0.5, 0.0),
                class_name='pedestrian',
                confidence=0.88,
                velocity=(0.0, 0.0, 0.0),
                track_id=2
            )
        ]
        
        # Mock driver state
        readiness = 85.0 + 10.0 * np.sin(timestamp * 0.5)
        driver_state = DriverState(
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
            distraction={'type': 'none', 'confidence': 0.95, 'duration': 0.0},
            readiness_score=float(readiness)
        )
        
        # Mock risk assessment
        risk_score = 0.5 + 0.3 * np.sin(timestamp * 0.3)
        hazard = Hazard(
            object_id=1,
            type='vehicle',
            position=(10.0 + timestamp, 2.0, 0.0),
            velocity=(2.0, 0.0, 0.0),
            trajectory=[
                (10.0 + timestamp, 2.0, 0.0),
                (15.0 + timestamp, 2.0, 0.0),
                (20.0 + timestamp, 2.0, 0.0)
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
        
        risk_assessment = RiskAssessment(
            scene_graph={'objects': [1, 2], 'relationships': []},
            hazards=[hazard],
            attention_map={'front': True, 'left': False, 'right': False},
            top_risks=[risk]
        )
        
        # Mock alerts (occasionally)
        alerts = []
        if risk_score > 0.8:
            alerts.append(Alert(
                timestamp=timestamp,
                urgency='warning',
                modalities=['visual', 'audio'],
                message='High risk detected ahead',
                hazard_id=1,
                dismissed=False
            ))
        
        return bev, segmentation, detections, driver_state, risk_assessment, alerts
    
    async def run(self, duration: float = 60.0):
        """Run mock SENTINEL system."""
        
        print(f"Running SENTINEL for {duration} seconds...")
        print("Streaming data at 30 Hz to connected clients\n")
        
        target_fps = 30
        frame_interval = 1.0 / target_fps
        frame_count = 0
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                loop_start = time.time()
                
                # Generate timestamp
                timestamp = frame_count / target_fps
                
                # Simulate processing with latencies
                process_start = time.time()
                
                bev, segmentation, detections, driver_state, risk_assessment, alerts = \
                    self.create_mock_frame_data(timestamp)
                
                # Record module latencies
                latencies = {
                    'camera': 5.0,
                    'bev': 14.5,
                    'segmentation': 13.2,
                    'detection': 18.7,
                    'dms': 22.3,
                    'intelligence': 8.9,
                    'total': 82.6
                }
                
                # Push data to visualization
                self.viz_integration.push_frame_data(
                    timestamp=timestamp,
                    bev=bev,
                    segmentation=segmentation,
                    detections=detections,
                    driver_state=driver_state,
                    risk_assessment=risk_assessment,
                    alerts=alerts,
                    latencies=latencies
                )
                
                # Stream to clients
                await self.viz_integration.stream_current_frame(timestamp)
                
                # Progress update
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    clients = len(self.viz_server.connection_manager.active_connections)
                    print(f"  {frame_count} frames | {actual_fps:.1f} FPS | {clients} clients")
                
                # Maintain frame rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\nStopping SENTINEL...")
        
        print(f"\nCompleted: {frame_count} frames in {time.time() - start_time:.1f}s")


async def main():
    """Main entry point."""
    
    print("=" * 70)
    print("SENTINEL Complete Visualization Example")
    print("=" * 70)
    print()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config_manager = ConfigManager(str(config_path))
    config = config_manager.config
    
    # Create and run mock SENTINEL system
    system = MockSentinelSystem(config)
    
    print("Instructions:")
    print("1. Open http://localhost:8080/ in your browser for live view")
    print("2. Open http://localhost:8080/playback.html for scenario playback")
    print("3. Watch real-time data streaming at 30 Hz")
    print("4. Press Ctrl+C to stop\n")
    
    # Run for 60 seconds (or until Ctrl+C)
    await system.run(duration=60.0)
    
    print("\n" + "=" * 70)
    print("Visualization example complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
