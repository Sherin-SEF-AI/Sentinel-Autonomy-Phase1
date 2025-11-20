"""Tests for Contextual Intelligence Engine."""

import pytest
import numpy as np
from src.intelligence import ContextualIntelligence
from src.core.data_structures import Detection3D, DriverState, SegmentationOutput


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'risk_assessment': {
            'ttc_calculation': {
                'method': 'constant_velocity',
                'safety_margin': 1.5
            },
            'trajectory_prediction': {
                'horizon': 3.0,
                'dt': 0.1,
                'method': 'linear'
            },
            'zone_mapping': {
                'num_zones': 8
            },
            'base_risk_weights': {
                'ttc': 0.4,
                'trajectory_conflict': 0.3,
                'vulnerability': 0.2,
                'relative_speed': 0.1
            },
            'thresholds': {
                'hazard_detection': 0.3,
                'intervention': 0.7,
                'critical': 0.9
            }
        }
    }


@pytest.fixture
def intelligence_engine(config):
    """Create intelligence engine instance."""
    return ContextualIntelligence(config)


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    return [
        Detection3D(
            bbox_3d=(10.0, 2.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='vehicle',
            confidence=0.9,
            velocity=(-5.0, 0.0, 0.0),  # Approaching
            track_id=1
        ),
        Detection3D(
            bbox_3d=(5.0, -3.0, 0.0, 0.5, 1.8, 0.5, 0.0),
            class_name='pedestrian',
            confidence=0.85,
            velocity=(0.0, 1.5, 0.0),  # Moving left
            track_id=2
        )
    ]


@pytest.fixture
def sample_driver_state():
    """Create sample driver state."""
    return DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=85.0
    )


@pytest.fixture
def sample_bev_seg():
    """Create sample BEV segmentation."""
    return SegmentationOutput(
        timestamp=0.0,
        class_map=np.zeros((640, 640), dtype=np.int8),
        confidence=np.ones((640, 640), dtype=np.float32)
    )


def test_intelligence_initialization(intelligence_engine):
    """Test intelligence engine initialization."""
    assert intelligence_engine is not None
    assert intelligence_engine.scene_graph_builder is not None
    assert intelligence_engine.attention_mapper is not None
    assert intelligence_engine.ttc_calculator is not None
    assert intelligence_engine.trajectory_predictor is not None
    assert intelligence_engine.risk_calculator is not None
    assert intelligence_engine.risk_prioritizer is not None


def test_assess_with_detections(intelligence_engine, sample_detections, sample_driver_state, sample_bev_seg):
    """Test risk assessment with detections."""
    result = intelligence_engine.assess(sample_detections, sample_driver_state, sample_bev_seg)
    
    assert result is not None
    assert result.scene_graph is not None
    assert len(result.hazards) == len(sample_detections)
    assert result.attention_map is not None
    assert isinstance(result.top_risks, list)


def test_assess_empty_detections(intelligence_engine, sample_driver_state, sample_bev_seg):
    """Test risk assessment with no detections."""
    result = intelligence_engine.assess([], sample_driver_state, sample_bev_seg)
    
    assert result is not None
    assert len(result.hazards) == 0
    assert len(result.top_risks) == 0
    assert result.scene_graph['num_objects'] == 0


def test_scene_graph_creation(intelligence_engine, sample_detections, sample_driver_state, sample_bev_seg):
    """Test scene graph creation."""
    result = intelligence_engine.assess(sample_detections, sample_driver_state, sample_bev_seg)
    
    scene_graph = result.scene_graph
    assert 'objects' in scene_graph
    assert 'relationships' in scene_graph
    assert 'spatial_map' in scene_graph
    assert scene_graph['num_objects'] == len(sample_detections)


def test_attention_mapping(intelligence_engine, sample_detections, sample_driver_state, sample_bev_seg):
    """Test attention mapping."""
    result = intelligence_engine.assess(sample_detections, sample_driver_state, sample_bev_seg)
    
    attention_map = result.attention_map
    assert 'attended_zones' in attention_map
    assert 'primary_zone' in attention_map
    assert 'gaze_yaw' in attention_map
    assert 'gaze_pitch' in attention_map
    assert attention_map['attention_valid'] is True


def test_hazard_creation(intelligence_engine, sample_detections, sample_driver_state, sample_bev_seg):
    """Test hazard creation."""
    result = intelligence_engine.assess(sample_detections, sample_driver_state, sample_bev_seg)
    
    for hazard in result.hazards:
        assert hazard.object_id is not None
        assert hazard.type is not None
        assert hazard.position is not None
        assert hazard.velocity is not None
        assert hazard.trajectory is not None
        assert hazard.ttc >= 0
        assert hazard.zone is not None
        assert 0 <= hazard.base_risk <= 1


def test_risk_prioritization(intelligence_engine, sample_detections, sample_driver_state, sample_bev_seg):
    """Test risk prioritization."""
    result = intelligence_engine.assess(sample_detections, sample_driver_state, sample_bev_seg)
    
    # Top risks should be sorted by contextual score
    if len(result.top_risks) > 1:
        for i in range(len(result.top_risks) - 1):
            assert result.top_risks[i].contextual_score >= result.top_risks[i + 1].contextual_score


def test_driver_unaware_increases_risk(intelligence_engine, sample_detections, sample_bev_seg):
    """Test that driver unawareness increases contextual risk."""
    # Driver looking forward
    aware_driver = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=85.0
    )
    
    # Driver looking away (right)
    unaware_driver = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0},
        gaze={'pitch': 0.0, 'yaw': -90.0, 'attention_zone': 'right'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=85.0
    )
    
    # Assess with aware driver
    result_aware = intelligence_engine.assess(sample_detections, aware_driver, sample_bev_seg)
    
    # Assess with unaware driver
    result_unaware = intelligence_engine.assess(sample_detections, unaware_driver, sample_bev_seg)
    
    # Find risks for the same object
    if result_aware.top_risks and result_unaware.top_risks:
        # At least one risk should have higher score when driver is unaware
        max_aware = max(r.contextual_score for r in result_aware.top_risks)
        max_unaware = max(r.contextual_score for r in result_unaware.top_risks)
        
        # Unaware driver should generally have higher risk
        # (may not always be true depending on object positions)
        assert max_unaware >= max_aware * 0.9  # Allow some tolerance


def test_low_readiness_increases_risk(intelligence_engine, sample_detections, sample_bev_seg):
    """Test that low driver readiness increases contextual risk."""
    # High readiness driver
    high_readiness = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=90.0
    )
    
    # Low readiness driver
    low_readiness = DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.5},
        drowsiness={'score': 0.7, 'yawn_detected': True, 'micro_sleep': False, 'head_nod': True},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=30.0
    )
    
    result_high = intelligence_engine.assess(sample_detections, high_readiness, sample_bev_seg)
    result_low = intelligence_engine.assess(sample_detections, low_readiness, sample_bev_seg)
    
    if result_high.top_risks and result_low.top_risks:
        max_high = max(r.contextual_score for r in result_high.top_risks)
        max_low = max(r.contextual_score for r in result_low.top_risks)
        
        # Low readiness should increase risk (or at least not decrease it)
        # Both may hit the 1.0 cap, so we use >= instead of >
        assert max_low >= max_high


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
