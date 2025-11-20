"""
Verification script for SENTINEL visualization system.

Tests:
1. Server creation
2. Data serialization
3. Streaming manager
4. Integration helper
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    create_server,
    serialize_frame_data,
    create_streaming_manager,
    StreamingIntegration,
    PerformanceMonitor
)
from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState
)


def test_server_creation():
    """Test server creation."""
    print("Testing server creation...")
    
    config = {
        'system': {'name': 'SENTINEL'},
        'recording': {'storage_path': 'scenarios/'}
    }
    
    server = create_server(config)
    assert server is not None
    assert server.app is not None
    
    print("  ✓ Server created successfully")


def test_data_serialization():
    """Test data serialization."""
    print("Testing data serialization...")
    
    # Create mock data
    bev = BEVOutput(
        timestamp=1.0,
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )
    
    segmentation = SegmentationOutput(
        timestamp=1.0,
        class_map=np.random.randint(0, 9, (640, 640), dtype=np.int8),
        confidence=np.random.rand(640, 640).astype(np.float32)
    )
    
    detection = Detection3D(
        bbox_3d=(10.0, 2.0, 0.0, 4.5, 1.8, 2.0, 0.1),
        class_name='vehicle',
        confidence=0.95,
        velocity=(5.0, 0.0, 0.0),
        track_id=1
    )
    
    # Serialize
    data = serialize_frame_data(
        timestamp=1.0,
        bev=bev,
        segmentation=segmentation,
        detections=[detection]
    )
    
    assert data['type'] == 'frame_data'
    assert data['timestamp'] == 1.0
    assert 'bev' in data
    assert 'segmentation' in data
    assert 'detections' in data
    assert len(data['detections']) == 1
    
    print("  ✓ Data serialization working")


def test_streaming_manager():
    """Test streaming manager."""
    print("Testing streaming manager...")
    
    config = {
        'system': {'name': 'SENTINEL'},
        'recording': {'storage_path': 'scenarios/'}
    }
    
    server = create_server(config)
    streaming = create_streaming_manager(server, target_fps=30)
    
    assert streaming is not None
    assert streaming.target_fps == 30
    assert streaming.performance_monitor is not None
    
    # Test data updates
    bev = BEVOutput(
        timestamp=1.0,
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )
    
    streaming.update_bev(bev)
    assert streaming.latest_bev is not None
    
    print("  ✓ Streaming manager working")


def test_performance_monitor():
    """Test performance monitor."""
    print("Testing performance monitor...")
    
    monitor = PerformanceMonitor(window_size=30)
    
    # Record some frames
    for i in range(10):
        monitor.record_frame()
        monitor.record_latency('bev', 15.0)
        monitor.record_latency('detection', 20.0)
    
    fps = monitor.get_fps()
    latencies = monitor.get_latencies()
    metrics = monitor.get_metrics()
    
    assert fps >= 0
    assert 'bev' in latencies
    assert 'detection' in latencies
    assert 'fps' in metrics
    assert 'latency' in metrics
    
    print("  ✓ Performance monitor working")


def test_integration_helper():
    """Test integration helper."""
    print("Testing integration helper...")
    
    config = {
        'system': {'name': 'SENTINEL'},
        'recording': {'storage_path': 'scenarios/'}
    }
    
    server = create_server(config)
    streaming = create_streaming_manager(server, target_fps=30)
    integration = StreamingIntegration(streaming)
    
    assert integration is not None
    assert integration.streaming is not None
    
    # Test push data
    bev = BEVOutput(
        timestamp=1.0,
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )
    
    integration.push_frame_data(
        timestamp=1.0,
        bev=bev,
        latencies={'bev': 15.0}
    )
    
    assert integration.streaming.latest_bev is not None
    
    print("  ✓ Integration helper working")


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("SENTINEL Visualization Verification")
    print("=" * 70)
    print()
    
    try:
        test_server_creation()
        test_data_serialization()
        test_streaming_manager()
        test_performance_monitor()
        test_integration_helper()
        
        print()
        print("=" * 70)
        print("✓ All verification tests passed!")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Verification failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
