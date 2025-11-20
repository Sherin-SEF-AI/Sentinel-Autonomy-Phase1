"""Tests for Data Recording Module."""

import pytest
import time
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.recording import (
    RecordingTrigger, FrameRecorder, ScenarioExporter, 
    ScenarioPlayback, ScenarioRecorder
)
from src.core.data_structures import (
    CameraBundle, BEVOutput, Detection3D, DriverState,
    RiskAssessment, Hazard, Risk, Alert
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def recording_config(temp_storage):
    """Create recording configuration."""
    return {
        'enabled': True,
        'triggers': {
            'risk_threshold': 0.7,
            'ttc_threshold': 1.5
        },
        'storage_path': temp_storage,
        'max_duration': 30.0
    }


@pytest.fixture
def mock_camera_bundle():
    """Create mock camera bundle."""
    return CameraBundle(
        timestamp=time.time(),
        interior=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        front_left=np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8),
        front_right=np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    )


@pytest.fixture
def mock_bev_output():
    """Create mock BEV output."""
    return BEVOutput(
        timestamp=time.time(),
        image=np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )


@pytest.fixture
def mock_detections():
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


@pytest.fixture
def mock_driver_state():
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


@pytest.fixture
def mock_risk_assessment():
    """Create mock risk assessment."""
    hazard = Hazard(
        object_id=1,
        type='vehicle',
        position=(10.0, 2.0, 0.0),
        velocity=(5.0, 0.0, 0.0),
        trajectory=[(10.0, 2.0, 0.0)],
        ttc=2.0,
        zone='front',
        base_risk=0.5
    )
    
    risk = Risk(
        hazard=hazard,
        contextual_score=0.6,
        driver_aware=True,
        urgency='medium',
        intervention_needed=False
    )
    
    return RiskAssessment(
        scene_graph={'objects': [1]},
        hazards=[hazard],
        attention_map={'current_zone': 'front'},
        top_risks=[risk]
    )


class TestRecordingTrigger:
    """Test RecordingTrigger class."""
    
    def test_initialization(self, recording_config):
        """Test trigger initialization."""
        trigger = RecordingTrigger(recording_config)
        assert trigger.risk_threshold == 0.7
        assert trigger.ttc_threshold == 1.5
    
    def test_high_risk_trigger(self, recording_config, mock_driver_state):
        """Test high risk score trigger."""
        trigger = RecordingTrigger(recording_config)
        
        # Create high risk assessment
        hazard = Hazard(
            object_id=1, type='vehicle', position=(10.0, 2.0, 0.0),
            velocity=(5.0, 0.0, 0.0), trajectory=[], ttc=2.0,
            zone='front', base_risk=0.8
        )
        risk = Risk(
            hazard=hazard, contextual_score=0.85, driver_aware=True,
            urgency='critical', intervention_needed=True
        )
        risk_assessment = RiskAssessment(
            scene_graph={}, hazards=[hazard], attention_map={}, top_risks=[risk]
        )
        
        triggers = trigger.check_triggers(
            time.time(), risk_assessment, mock_driver_state, []
        )
        
        assert len(triggers) > 0
        assert any(t.trigger_type == 'high_risk' for t in triggers)
    
    def test_low_ttc_trigger(self, recording_config, mock_driver_state):
        """Test low TTC trigger."""
        trigger = RecordingTrigger(recording_config)
        
        # Create low TTC hazard
        hazard = Hazard(
            object_id=1, type='vehicle', position=(10.0, 2.0, 0.0),
            velocity=(5.0, 0.0, 0.0), trajectory=[], ttc=1.2,
            zone='front', base_risk=0.5
        )
        risk_assessment = RiskAssessment(
            scene_graph={}, hazards=[hazard], attention_map={}, top_risks=[]
        )
        
        triggers = trigger.check_triggers(
            time.time(), risk_assessment, mock_driver_state, []
        )
        
        assert len(triggers) > 0
        assert any(t.trigger_type == 'low_ttc' for t in triggers)
    
    def test_intervention_trigger(self, recording_config, mock_driver_state, mock_risk_assessment):
        """Test system intervention trigger."""
        trigger = RecordingTrigger(recording_config)
        
        alert = Alert(
            timestamp=time.time(), urgency='critical',
            modalities=['visual', 'audio'], message='Test alert',
            hazard_id=1, dismissed=False
        )
        
        triggers = trigger.check_triggers(
            time.time(), mock_risk_assessment, mock_driver_state, [alert]
        )
        
        assert len(triggers) > 0
        assert any(t.trigger_type == 'intervention' for t in triggers)


class TestFrameRecorder:
    """Test FrameRecorder class."""
    
    def test_initialization(self, recording_config):
        """Test recorder initialization."""
        recorder = FrameRecorder(recording_config)
        assert recorder.buffer_size == 90
        assert not recorder.is_recording
    
    def test_buffer_recording(
        self, recording_config, mock_camera_bundle, mock_bev_output,
        mock_detections, mock_driver_state, mock_risk_assessment
    ):
        """Test frame buffering."""
        recorder = FrameRecorder(recording_config, buffer_size=10)
        
        # Add frames to buffer
        for i in range(15):
            recorder.save_frame(
                time.time() + i * 0.033,
                mock_camera_bundle, mock_bev_output, mock_detections,
                mock_driver_state, mock_risk_assessment, []
            )
        
        # Buffer should contain only last 10 frames
        assert len(recorder.frame_buffer) == 10
    
    def test_start_stop_recording(
        self, recording_config, mock_camera_bundle, mock_bev_output,
        mock_detections, mock_driver_state, mock_risk_assessment
    ):
        """Test recording start/stop."""
        recorder = FrameRecorder(recording_config)
        
        # Add some buffer frames
        for i in range(5):
            recorder.save_frame(
                time.time() + i * 0.033,
                mock_camera_bundle, mock_bev_output, mock_detections,
                mock_driver_state, mock_risk_assessment, []
            )
        
        # Start recording
        recorder.start_recording(time.time())
        assert recorder.is_recording
        
        # Add recording frames
        for i in range(10):
            recorder.save_frame(
                time.time() + (5 + i) * 0.033,
                mock_camera_bundle, mock_bev_output, mock_detections,
                mock_driver_state, mock_risk_assessment, []
            )
        
        # Stop recording
        recorder.stop_recording()
        assert not recorder.is_recording
        
        # Should have buffer + recording frames
        frames = recorder.get_recorded_frames()
        assert len(frames) == 15  # 5 buffer + 10 recording


class TestScenarioRecorder:
    """Test ScenarioRecorder integration."""
    
    def test_initialization(self, recording_config):
        """Test recorder initialization."""
        recorder = ScenarioRecorder(recording_config)
        assert recorder.enabled
        assert not recorder.is_recording()
    
    def test_automatic_trigger(
        self, recording_config, mock_camera_bundle, mock_bev_output,
        mock_detections, mock_driver_state
    ):
        """Test automatic recording trigger."""
        recorder = ScenarioRecorder(recording_config)
        
        # Process normal frames
        for i in range(10):
            recorder.process_frame(
                time.time() + i * 0.033,
                mock_camera_bundle, mock_bev_output, mock_detections,
                mock_driver_state, mock_risk_assessment, []
            )
        
        assert not recorder.is_recording()
        
        # Process high-risk frame
        hazard = Hazard(
            object_id=1, type='vehicle', position=(10.0, 2.0, 0.0),
            velocity=(5.0, 0.0, 0.0), trajectory=[], ttc=1.2,
            zone='front', base_risk=0.8
        )
        risk = Risk(
            hazard=hazard, contextual_score=0.85, driver_aware=True,
            urgency='critical', intervention_needed=True
        )
        high_risk_assessment = RiskAssessment(
            scene_graph={}, hazards=[hazard], attention_map={}, top_risks=[risk]
        )
        
        recorder.process_frame(
            time.time() + 10 * 0.033,
            mock_camera_bundle, mock_bev_output, mock_detections,
            mock_driver_state, high_risk_assessment, []
        )
        
        assert recorder.is_recording()
    
    def test_manual_recording(self, recording_config):
        """Test manual recording control."""
        recorder = ScenarioRecorder(recording_config)
        
        timestamp = time.time()
        recorder.start_recording(timestamp)
        assert recorder.is_recording()
        
        recorder.stop_recording()
        assert not recorder.is_recording()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
