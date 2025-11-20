"""
Example demonstrating the FastAPI backend for SENTINEL visualization.

This example shows how to:
1. Create and start the visualization server
2. Stream data to connected WebSocket clients
3. Use REST endpoints for configuration and scenario management
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.backend import create_server, serialize_frame_data
from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert, Hazard, Risk
)
from src.core.config import ConfigManager


def create_mock_data(timestamp: float):
    """Create mock data for demonstration."""
    
    # Mock BEV output
    bev = BEVOutput(
        timestamp=timestamp,
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )
    
    # Mock segmentation output
    segmentation = SegmentationOutput(
        timestamp=timestamp,
        class_map=np.random.randint(0, 9, (640, 640), dtype=np.int8),
        confidence=np.random.rand(640, 640).astype(np.float32)
    )
    
    # Mock detections
    detections = [
        Detection3D(
            bbox_3d=(10.0, 2.0, 0.0, 4.5, 1.8, 2.0, 0.1),
            class_name='vehicle',
            confidence=0.95,
            velocity=(5.0, 0.0, 0.0),
            track_id=1
        ),
        Detection3D(
            bbox_3d=(15.0, -1.0, 0.0, 0.5, 1.7, 0.5, 0.0),
            class_name='pedestrian',
            confidence=0.88,
            velocity=(1.0, 0.5, 0.0),
            track_id=2
        )
    ]
    
    # Mock driver state
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
        distraction={
            'type': 'none',
            'confidence': 0.95,
            'duration': 0.0
        },
        readiness_score=85.0
    )
    
    # Mock risk assessment
    hazard = Hazard(
        object_id=1,
        type='vehicle',
        position=(10.0, 2.0, 0.0),
        velocity=(5.0, 0.0, 0.0),
        trajectory=[(10.0, 2.0, 0.0), (15.0, 2.0, 0.0), (20.0, 2.0, 0.0)],
        ttc=2.5,
        zone='front',
        base_risk=0.6
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=0.65,
        driver_aware=True,
        urgency='medium',
        intervention_needed=False
    )
    
    risk_assessment = RiskAssessment(
        scene_graph={'objects': [1, 2], 'relationships': []},
        hazards=[hazard],
        attention_map={'front': True, 'left': False, 'right': False},
        top_risks=[risk]
    )
    
    # Mock alerts
    alerts = []
    
    # Mock performance metrics
    performance = {
        'fps': 30.5,
        'latency': {
            'camera': 5.2,
            'bev': 14.8,
            'segmentation': 13.5,
            'detection': 18.3,
            'dms': 22.1,
            'intelligence': 8.7,
            'total': 82.6
        },
        'gpu_memory_mb': 3456,
        'cpu_percent': 45.2
    }
    
    return serialize_frame_data(
        timestamp=timestamp,
        bev=bev,
        segmentation=segmentation,
        detections=detections,
        driver_state=driver_state,
        risk_assessment=risk_assessment,
        alerts=alerts,
        performance=performance
    )


async def stream_mock_data(server):
    """Stream mock data at 30 Hz."""
    print("Starting data streaming at 30 Hz...")
    print("Connect WebSocket client to: ws://localhost:8080/ws/stream")
    
    frame_count = 0
    while True:
        timestamp = frame_count / 30.0
        data = create_mock_data(timestamp)
        
        await server.stream_data(data)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Streamed {frame_count} frames ({frame_count // 30} seconds)")
        
        await asyncio.sleep(1.0 / 30.0)  # 30 Hz


def main():
    """Run the visualization backend example."""
    print("=" * 70)
    print("SENTINEL Visualization Backend Example")
    print("=" * 70)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    config_manager = ConfigManager(str(config_path))
    config = config_manager.config
    
    # Create server
    print("\n1. Creating visualization server...")
    server = create_server(config)
    
    print("\n2. Server endpoints:")
    print("   - Health check:     GET  http://localhost:8080/")
    print("   - Configuration:    GET  http://localhost:8080/api/config")
    print("   - Update config:    POST http://localhost:8080/api/config")
    print("   - List scenarios:   GET  http://localhost:8080/api/scenarios")
    print("   - Get scenario:     GET  http://localhost:8080/api/scenarios/{id}")
    print("   - Delete scenario:  DEL  http://localhost:8080/api/scenarios/{id}")
    print("   - System status:    GET  http://localhost:8080/api/status")
    print("   - WebSocket stream: WS   ws://localhost:8080/ws/stream")
    
    print("\n3. Starting server on http://localhost:8080")
    print("   Press Ctrl+C to stop")
    
    # Start streaming task
    async def run_with_streaming():
        # Start streaming in background
        streaming_task = asyncio.create_task(stream_mock_data(server))
        
        # This would normally run the server, but we'll just keep streaming
        try:
            await streaming_task
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            streaming_task.cancel()
    
    # Note: In production, you would run server.run() which starts uvicorn
    # For this example, we demonstrate the streaming functionality
    print("\n4. To run the actual server, use:")
    print("   server.run(host='0.0.0.0', port=8080)")
    
    print("\n5. Example WebSocket client (JavaScript):")
    print("""
    const ws = new WebSocket('ws://localhost:8080/ws/stream');
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received frame:', data.timestamp);
        // Update visualization with data.bev, data.detections, etc.
    };
    """)
    
    print("\n" + "=" * 70)
    print("Backend server ready!")
    print("=" * 70)


if __name__ == "__main__":
    main()
