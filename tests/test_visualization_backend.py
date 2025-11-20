"""
Tests for visualization backend server.
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import shutil

from src.visualization.backend import create_server, serialize_frame_data
from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert, Hazard, Risk
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        'system': {
            'name': 'SENTINEL',
            'version': '1.0'
        },
        'recording': {
            'storage_path': tempfile.mkdtemp()
        }
    }


@pytest.fixture
def server(test_config):
    """Create test server."""
    srv = create_server(test_config)
    yield srv
    # Cleanup
    storage_path = Path(test_config['recording']['storage_path'])
    if storage_path.exists():
        shutil.rmtree(storage_path)


@pytest.fixture
def mock_bev():
    """Create mock BEV output."""
    return BEVOutput(
        timestamp=1.0,
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )


@pytest.fixture
def mock_segmentation():
    """Create mock segmentation output."""
    return SegmentationOutput(
        timestamp=1.0,
        class_map=np.random.randint(0, 9, (640, 640), dtype=np.int8),
        confidence=np.random.rand(640, 640).astype(np.float32)
    )


@pytest.fixture
def mock_detection():
    """Create mock 3D detection."""
    return Detection3D(
        bbox_3d=(10.0, 2.0, 0.0, 4.5, 1.8, 2.0, 0.1),
        class_name='vehicle',
        confidence=0.95,
        velocity=(5.0, 0.0, 0.0),
        track_id=1
    )


@pytest.fixture
def mock_driver_state():
    """Create mock driver state."""
    return DriverState(
        face_detected=True,
        landmarks=np.random.rand(68, 2).astype(np.float32),
        head_pose={'roll': 0.1, 'pitch': -0.05, 'yaw': 0.02},
        gaze={'pitch': -0.1, 'yaw': 0.05, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.15},
        drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.95, 'duration': 0.0},
        readiness_score=85.0
    )


def test_server_creation(server):
    """Test server creation."""
    assert server is not None
    assert server.app is not None
    assert server.connection_manager is not None


def test_serialize_bev(mock_bev):
    """Test BEV serialization."""
    data = serialize_frame_data(timestamp=1.0, bev=mock_bev)
    
    assert data['type'] == 'frame_data'
    assert data['timestamp'] == 1.0
    assert 'bev' in data
    assert 'image' in data['bev']
    assert isinstance(data['bev']['image'], str)  # Base64 encoded


def test_serialize_segmentation(mock_segmentation):
    """Test segmentation serialization."""
    data = serialize_frame_data(timestamp=1.0, segmentation=mock_segmentation)
    
    assert 'segmentation' in data
    assert 'class_map' in data['segmentation']
    assert 'overlay' in data['segmentation']
    assert isinstance(data['segmentation']['overlay'], str)  # Base64 encoded


def test_serialize_detection(mock_detection):
    """Test detection serialization."""
    data = serialize_frame_data(timestamp=1.0, detections=[mock_detection])
    
    assert 'detections' in data
    assert len(data['detections']) == 1
    
    det = data['detections'][0]
    assert 'bbox_3d' in det
    assert det['class_name'] == 'vehicle'
    assert det['confidence'] == 0.95
    assert det['track_id'] == 1


def test_serialize_driver_state(mock_driver_state):
    """Test driver state serialization."""
    data = serialize_frame_data(timestamp=1.0, driver_state=mock_driver_state)
    
    assert 'driver_state' in data
    ds = data['driver_state']
    assert ds['face_detected'] is True
    assert ds['readiness_score'] == 85.0
    assert 'head_pose' in ds
    assert 'gaze' in ds


def test_serialize_complete_frame(mock_bev, mock_segmentation, mock_detection, mock_driver_state):
    """Test complete frame serialization."""
    performance = {
        'fps': 30.0,
        'latency': {'total': 85.0}
    }
    
    data = serialize_frame_data(
        timestamp=1.0,
        bev=mock_bev,
        segmentation=mock_segmentation,
        detections=[mock_detection],
        driver_state=mock_driver_state,
        performance=performance
    )
    
    assert data['type'] == 'frame_data'
    assert 'bev' in data
    assert 'segmentation' in data
    assert 'detections' in data
    assert 'driver_state' in data
    assert 'performance' in data
    assert data['performance']['fps'] == 30.0


@pytest.mark.asyncio
async def test_stream_data(server, mock_bev):
    """Test data streaming."""
    data = serialize_frame_data(timestamp=1.0, bev=mock_bev)
    
    # Stream data (no clients connected, should not raise error)
    await server.stream_data(data)
    
    # Check latest data is stored
    assert server.latest_data is not None
    assert server.latest_data['timestamp'] == 1.0


def test_config_merge(server):
    """Test configuration merging."""
    base = {'a': 1, 'b': {'c': 2, 'd': 3}}
    update = {'b': {'c': 5}, 'e': 6}
    
    server._merge_config(base, update)
    
    assert base['a'] == 1
    assert base['b']['c'] == 5
    assert base['b']['d'] == 3
    assert base['e'] == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
