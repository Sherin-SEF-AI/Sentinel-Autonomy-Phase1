"""Test suite for main SENTINEL system orchestration module."""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import SentinelSystem, main
from src.core.data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput,
    Detection3D, DriverState, RiskAssessment, Alert, Hazard, Risk
)


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration for testing."""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'visualization.enabled': True,
        'visualization.port': 8080,
        'system.log_level': 'INFO',
        'system.version': '1.0'
    }.get(key, default))
    return config


@pytest.fixture
def mock_camera_bundle():
    """Fixture providing mock camera bundle."""
    return CameraBundle(
        timestamp=time.time(),
        interior=np.zeros((480, 640, 3), dtype=np.uint8),
        front_left=np.zeros((720, 1280, 3), dtype=np.uint8),
        front_right=np.zeros((720, 1280, 3), dtype=np.uint8)
    )


@pytest.fixture
def mock_bev_output():
    """Fixture providing mock BEV output."""
    return BEVOutput(
        timestamp=time.time(),
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        mask=np.ones((640, 640), dtype=bool)
    )


@pytest.fixture
def mock_segmentation_output():
    """Fixture providing mock segmentation output."""
    return SegmentationOutput(
        timestamp=time.time(),
        class_map=np.zeros((640, 640), dtype=np.int8),
        confidence=np.ones((640, 640), dtype=np.float32)
    )


@pytest.fixture
def mock_driver_state():
    """Fixture providing mock driver state."""
    return DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'forward'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=85.0
    )


@pytest.fixture
def mock_risk_assessment():
    """Fixture providing mock risk assessment."""
    hazard = Hazard(
        object_id=1,
        type='vehicle',
        position=(5.0, 0.0, 0.0),
        velocity=(10.0, 0.0, 0.0),
        trajectory=[(5.0, 0.0, 0.0)],
        ttc=2.5,
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
        scene_graph={},
        hazards=[hazard],
        attention_map={},
        top_risks=[risk]
    )


@pytest.fixture
def mock_alerts():
    """Fixture providing mock alerts."""
    return [
        Alert(
            timestamp=time.time(),
            urgency='warning',
            modalities=['visual', 'audio'],
            message='Vehicle ahead',
            hazard_id=1,
            dismissed=False
        )
    ]


@pytest.fixture
@patch('src.main.CameraManager')
@patch('src.main.BEVGenerator')
@patch('src.main.SemanticSegmentor')
@patch('src.main.ObjectDetector')
@patch('src.main.DriverMonitor')
@patch('src.main.ContextualIntelligence')
@patch('src.main.AlertSystem')
@patch('src.main.ScenarioRecorder')
@patch('src.main.VisualizationServer')
@patch('src.main.StreamingManager')
def sentinel_system(mock_streaming, mock_viz, mock_recorder, mock_alerts_sys,
                    mock_intelligence, mock_dms, mock_detector, mock_segmentor,
                    mock_bev, mock_camera, mock_config):
    """Fixture creating a SentinelSystem instance with mocked modules."""
    # Mock module constructors
    mock_camera.return_value = Mock()
    mock_bev.return_value = Mock()
    mock_segmentor.return_value = Mock()
    mock_detector.return_value = Mock()
    mock_dms.return_value = Mock()
    mock_intelligence.return_value = Mock()
    mock_alerts_sys.return_value = Mock()
    mock_recorder.return_value = Mock()
    mock_viz.return_value = Mock()
    mock_streaming.return_value = Mock()
    
    # Create system
    with patch.object(SentinelSystem, '_restore_system_state'):
        system = SentinelSystem(mock_config)
    
    return system


class TestSentinelSystemInitialization:
    """Test suite for SentinelSystem initialization."""
    
    @patch('src.main.CameraManager')
    @patch('src.main.BEVGenerator')
    @patch('src.main.SemanticSegmentor')
    @patch('src.main.ObjectDetector')
    @patch('src.main.DriverMonitor')
    @patch('src.main.ContextualIntelligence')
    @patch('src.main.AlertSystem')
    @patch('src.main.ScenarioRecorder')
    @patch('src.main.VisualizationServer')
    @patch('src.main.StreamingManager')
    def test_initialization_success(self, mock_streaming, mock_viz, mock_recorder,
                                    mock_alerts_sys, mock_intelligence, mock_dms,
                                    mock_detector, mock_segmentor, mock_bev,
                                    mock_camera, mock_config):
        """Test that SentinelSystem initializes correctly with valid configuration."""
        with patch.object(SentinelSystem, '_restore_system_state'):
            system = SentinelSystem(mock_config)
        
        assert system is not None
        assert system.config == mock_config
        assert system.running is False
        assert system.frame_count == 0
        assert system.camera_manager is not None
        assert system.bev_generator is not None
        assert system.segmentor is not None
        assert system.detector is not None
        assert system.dms is not None
        assert system.intelligence is not None
        assert system.alert_system is not None
        assert system.recorder is not None
    
    @patch('src.main.CameraManager')
    def test_initialization_module_failure(self, mock_camera, mock_config):
        """Test initialization handles module creation failures gracefully."""
        mock_camera.side_effect = Exception("Camera initialization failed")
        
        with patch.object(SentinelSystem, '_restore_system_state'):
            with pytest.raises(Exception) as exc_info:
                SentinelSystem(mock_config)
        
        assert "Camera initialization failed" in str(exc_info.value)
    
    @patch('src.main.CameraManager')
    @patch('src.main.BEVGenerator')
    @patch('src.main.SemanticSegmentor')
    @patch('src.main.ObjectDetector')
    @patch('src.main.DriverMonitor')
    @patch('src.main.ContextualIntelligence')
    @patch('src.main.AlertSystem')
    @patch('src.main.ScenarioRecorder')
    def test_initialization_without_visualization(self, mock_recorder, mock_alerts_sys,
                                                  mock_intelligence, mock_dms, mock_detector,
                                                  mock_segmentor, mock_bev, mock_camera):
        """Test initialization with visualization disabled."""
        config = Mock()
        config.get = Mock(side_effect=lambda key, default=None: {
            'visualization.enabled': False
        }.get(key, default))
        
        with patch.object(SentinelSystem, '_restore_system_state'):
            system = SentinelSystem(config)
        
        assert system.viz_server is None
        assert system.streaming_manager is None


class TestSentinelSystemStartStop:
    """Test suite for system start and stop operations."""
    
    def test_start_system(self, sentinel_system):
        """Test starting the SENTINEL system."""
        sentinel_system.camera_manager.start = Mock()
        sentinel_system.viz_server.start = Mock()
        
        sentinel_system.start()
        
        assert sentinel_system.running is True
        assert sentinel_system.start_time is not None
        sentinel_system.camera_manager.start.assert_called_once()
        sentinel_system.viz_server.start.assert_called_once()
    
    def test_start_already_running(self, sentinel_system):
        """Test starting system when already running."""
        sentinel_system.running = True
        sentinel_system.camera_manager.start = Mock()
        
        sentinel_system.start()
        
        # Should not call start again
        sentinel_system.camera_manager.start.assert_not_called()
    
    def test_stop_system(self, sentinel_system):
        """Test stopping the SENTINEL system gracefully."""
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        sentinel_system.frame_count = 100
        
        sentinel_system.camera_manager.stop = Mock()
        sentinel_system.viz_server.stop = Mock()
        sentinel_system.recorder.is_recording = False
        
        with patch.object(sentinel_system, '_save_system_state'):
            with patch.object(sentinel_system, '_close_resources'):
                with patch.object(sentinel_system, '_log_final_statistics'):
                    sentinel_system.stop()
        
        assert sentinel_system.running is False
        assert sentinel_system.shutdown_event.is_set()
        sentinel_system.camera_manager.stop.assert_called_once()
        sentinel_system.viz_server.stop.assert_called_once()
    
    def test_stop_with_active_recording(self, sentinel_system):
        """Test stopping system with active recording."""
        sentinel_system.running = True
        sentinel_system.recorder.is_recording = True
        sentinel_system.recorder.stop_recording = Mock()
        sentinel_system.recorder.export_scenario = Mock(return_value="/path/to/scenario")
        sentinel_system.camera_manager.stop = Mock()
        sentinel_system.viz_server.stop = Mock()
        
        with patch.object(sentinel_system, '_save_system_state'):
            with patch.object(sentinel_system, '_close_resources'):
                with patch.object(sentinel_system, '_log_final_statistics'):
                    sentinel_system.stop()
        
        sentinel_system.recorder.stop_recording.assert_called_once()
        sentinel_system.recorder.export_scenario.assert_called_once()


class TestSentinelSystemProcessing:
    """Test suite for main processing loop."""
    
    def test_processing_loop_single_iteration(self, sentinel_system, mock_camera_bundle,
                                              mock_bev_output, mock_segmentation_output,
                                              mock_driver_state, mock_risk_assessment,
                                              mock_alerts):
        """Test single iteration of processing loop."""
        # Setup mocks
        sentinel_system.camera_manager.get_frame_bundle = Mock(return_value=mock_camera_bundle)
        sentinel_system.bev_generator.generate = Mock(return_value=mock_bev_output)
        sentinel_system.segmentor.segment = Mock(return_value=mock_segmentation_output)
        sentinel_system.detector.detect = Mock(return_value=({}, []))
        sentinel_system.dms.analyze = Mock(return_value=mock_driver_state)
        sentinel_system.intelligence.assess = Mock(return_value=mock_risk_assessment)
        sentinel_system.alert_system.process = Mock(return_value=mock_alerts)
        sentinel_system.recorder.should_record = Mock(return_value=False)
        sentinel_system.recorder.is_recording = False
        
        # Run one iteration
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        
        # Start processing thread
        processing_thread = threading.Thread(target=sentinel_system._processing_loop, daemon=True)
        processing_thread.start()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop
        sentinel_system.running = False
        sentinel_system.shutdown_event.set()
        processing_thread.join(timeout=1.0)
        
        # Verify calls were made
        assert sentinel_system.camera_manager.get_frame_bundle.called
        assert sentinel_system.frame_count > 0
    
    def test_processing_loop_no_camera_bundle(self, sentinel_system):
        """Test processing loop handles missing camera bundle."""
        sentinel_system.camera_manager.get_frame_bundle = Mock(return_value=None)
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        
        # Run one iteration
        processing_thread = threading.Thread(target=sentinel_system._processing_loop, daemon=True)
        processing_thread.start()
        
        time.sleep(0.05)
        
        sentinel_system.running = False
        sentinel_system.shutdown_event.set()
        processing_thread.join(timeout=1.0)
        
        # Should not process frames
        assert sentinel_system.frame_count == 0
    
    def test_processing_loop_with_recording_trigger(self, sentinel_system, mock_camera_bundle,
                                                    mock_bev_output, mock_segmentation_output,
                                                    mock_driver_state, mock_risk_assessment,
                                                    mock_alerts):
        """Test processing loop triggers recording on high risk."""
        sentinel_system.camera_manager.get_frame_bundle = Mock(return_value=mock_camera_bundle)
        sentinel_system.bev_generator.generate = Mock(return_value=mock_bev_output)
        sentinel_system.segmentor.segment = Mock(return_value=mock_segmentation_output)
        sentinel_system.detector.detect = Mock(return_value=({}, []))
        sentinel_system.dms.analyze = Mock(return_value=mock_driver_state)
        sentinel_system.intelligence.assess = Mock(return_value=mock_risk_assessment)
        sentinel_system.alert_system.process = Mock(return_value=mock_alerts)
        sentinel_system.recorder.should_record = Mock(return_value=True)
        sentinel_system.recorder.is_recording = False
        sentinel_system.recorder.start_recording = Mock()
        sentinel_system.recorder.save_frame = Mock()
        
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        
        processing_thread = threading.Thread(target=sentinel_system._processing_loop, daemon=True)
        processing_thread.start()
        
        time.sleep(0.1)
        
        sentinel_system.running = False
        sentinel_system.shutdown_event.set()
        processing_thread.join(timeout=1.0)
        
        # Verify recording was triggered
        assert sentinel_system.recorder.start_recording.called


class TestSentinelSystemPerformanceMonitoring:
    """Test suite for performance monitoring."""
    
    def test_performance_monitoring_loop(self, sentinel_system):
        """Test performance monitoring collects metrics."""
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=sentinel_system._performance_monitoring_loop,
            daemon=True
        )
        monitor_thread.start()
        
        # Let it collect some samples
        time.sleep(1.5)
        
        # Stop
        sentinel_system.running = False
        sentinel_system.shutdown_event.set()
        monitor_thread.join(timeout=2.0)
        
        # Verify metrics were collected
        assert len(sentinel_system.cpu_usage_history) > 0
        assert len(sentinel_system.memory_usage_history) > 0
    
    def test_log_performance_metrics(self, sentinel_system):
        """Test performance metrics logging."""
        sentinel_system.start_time = time.time()
        sentinel_system.frame_count = 100
        sentinel_system.cpu_usage_history = [50.0, 55.0, 52.0]
        sentinel_system.memory_usage_history = [1024.0, 1050.0, 1030.0]
        sentinel_system.module_latencies['camera'] = [0.005, 0.006, 0.005]
        sentinel_system.module_latencies['bev'] = [0.015, 0.014, 0.016]
        
        # Should not raise exception
        sentinel_system._log_performance_metrics()
    
    def test_log_final_statistics(self, sentinel_system):
        """Test final statistics logging."""
        sentinel_system.start_time = time.time() - 10.0  # 10 seconds ago
        sentinel_system.frame_count = 300
        sentinel_system.cpu_usage_history = [50.0] * 10
        sentinel_system.memory_usage_history = [1024.0] * 10
        sentinel_system.module_latencies['camera'] = [0.005] * 100
        sentinel_system.module_latencies['bev'] = [0.015] * 100
        
        # Should not raise exception
        sentinel_system._log_final_statistics()


class TestSentinelSystemStateManagement:
    """Test suite for state save/restore."""
    
    @patch('builtins.open', create=True)
    @patch('pickle.dump')
    @patch('pathlib.Path.mkdir')
    def test_save_system_state(self, mock_mkdir, mock_pickle_dump, mock_open, sentinel_system):
        """Test saving system state."""
        sentinel_system.frame_count = 100
        sentinel_system.start_time = time.time()
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        sentinel_system._save_system_state()
        
        mock_pickle_dump.assert_called_once()
        saved_state = mock_pickle_dump.call_args[0][0]
        assert 'frame_count' in saved_state
        assert 'total_runtime' in saved_state
        assert 'timestamp' in saved_state
    
    @patch('builtins.open', create=True)
    @patch('pickle.load')
    @patch('pathlib.Path.exists')
    def test_restore_system_state_success(self, mock_exists, mock_pickle_load, mock_open, mock_config):
        """Test restoring system state from file."""
        mock_exists.return_value = True
        mock_pickle_load.return_value = {
            'frame_count': 100,
            'total_runtime': 10.0,
            'timestamp': time.time() - 60  # 1 minute ago
        }
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        with patch('src.main.CameraManager'), \
             patch('src.main.BEVGenerator'), \
             patch('src.main.SemanticSegmentor'), \
             patch('src.main.ObjectDetector'), \
             patch('src.main.DriverMonitor'), \
             patch('src.main.ContextualIntelligence'), \
             patch('src.main.AlertSystem'), \
             patch('src.main.ScenarioRecorder'), \
             patch('src.main.VisualizationServer'), \
             patch('src.main.StreamingManager'):
            system = SentinelSystem(mock_config)
        
        # Should not raise exception
        assert system is not None
    
    @patch('pathlib.Path.exists')
    def test_restore_system_state_no_file(self, mock_exists, mock_config):
        """Test restore handles missing state file."""
        mock_exists.return_value = False
        
        with patch('src.main.CameraManager'), \
             patch('src.main.BEVGenerator'), \
             patch('src.main.SemanticSegmentor'), \
             patch('src.main.ObjectDetector'), \
             patch('src.main.DriverMonitor'), \
             patch('src.main.ContextualIntelligence'), \
             patch('src.main.AlertSystem'), \
             patch('src.main.ScenarioRecorder'), \
             patch('src.main.VisualizationServer'), \
             patch('src.main.StreamingManager'):
            system = SentinelSystem(mock_config)
        
        assert system is not None
    
    def test_periodic_state_save(self, sentinel_system):
        """Test periodic state save is triggered at correct intervals."""
        with patch.object(sentinel_system, '_save_system_state') as mock_save:
            # Should save at frame 100
            sentinel_system.frame_count = 100
            sentinel_system._periodic_state_save()
            mock_save.assert_called_once()
            
            # Should not save at frame 101
            mock_save.reset_mock()
            sentinel_system.frame_count = 101
            sentinel_system._periodic_state_save()
            mock_save.assert_not_called()


class TestSentinelSystemResourceManagement:
    """Test suite for resource management."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_close_resources_with_gpu(self, mock_empty_cache, mock_cuda_available, sentinel_system):
        """Test closing resources clears GPU cache."""
        mock_cuda_available.return_value = True
        
        sentinel_system._close_resources()
        
        mock_empty_cache.assert_called_once()
    
    def test_close_resources_without_torch(self, sentinel_system):
        """Test closing resources handles missing PyTorch."""
        with patch.dict('sys.modules', {'torch': None}):
            # Should not raise exception
            sentinel_system._close_resources()


class TestMainFunction:
    """Test suite for main entry point function."""
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml'])
    def test_main_function_success(self, mock_system_class, mock_logger_setup, mock_config_class):
        """Test main function with valid arguments."""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = '1.0'
        mock_config_class.return_value = mock_config
        
        mock_logger = Mock()
        mock_logger_setup.setup.return_value = mock_logger
        
        mock_system = Mock()
        mock_system.running = False
        mock_system_class.return_value = mock_system
        
        # Run main (will exit quickly since system.running = False)
        result = main()
        
        assert result == 0
        mock_config_class.assert_called_once()
        mock_system_class.assert_called_once()
    
    @patch('src.main.ConfigManager')
    @patch('sys.argv', ['main.py', '--config', 'nonexistent.yaml'])
    def test_main_function_config_not_found(self, mock_config_class):
        """Test main function handles missing config file."""
        mock_config_class.side_effect = FileNotFoundError("Config not found")
        
        result = main()
        
        assert result == 1
    
    @patch('src.main.ConfigManager')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml'])
    def test_main_function_invalid_config(self, mock_config_class):
        """Test main function handles invalid configuration."""
        mock_config = Mock()
        mock_config.validate.return_value = False
        mock_config_class.return_value = mock_config
        
        result = main()
        
        assert result == 1
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml', '--log-level', 'DEBUG'])
    def test_main_function_with_log_level(self, mock_system_class, mock_logger_setup, mock_config_class):
        """Test main function with custom log level."""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = '1.0'
        mock_config_class.return_value = mock_config
        
        mock_logger = Mock()
        mock_logger_setup.setup.return_value = mock_logger
        
        mock_system = Mock()
        mock_system.running = False
        mock_system_class.return_value = mock_system
        
        result = main()
        
        assert result == 0
        mock_logger_setup.setup.assert_called_once()
        # Verify DEBUG level was used
        call_args = mock_logger_setup.setup.call_args
        assert call_args[1]['log_level'] == 'DEBUG'


@pytest.mark.performance
class TestSentinelSystemPerformance:
    """Performance tests for SENTINEL system."""
    
    def test_initialization_performance(self, mock_config):
        """Test that system initialization completes within reasonable time."""
        with patch('src.main.CameraManager'), \
             patch('src.main.BEVGenerator'), \
             patch('src.main.SemanticSegmentor'), \
             patch('src.main.ObjectDetector'), \
             patch('src.main.DriverMonitor'), \
             patch('src.main.ContextualIntelligence'), \
             patch('src.main.AlertSystem'), \
             patch('src.main.ScenarioRecorder'), \
             patch('src.main.VisualizationServer'), \
             patch('src.main.StreamingManager'), \
             patch.object(SentinelSystem, '_restore_system_state'):
            
            start_time = time.perf_counter()
            system = SentinelSystem(mock_config)
            end_time = time.perf_counter()
            
            initialization_time_ms = (end_time - start_time) * 1000
            assert initialization_time_ms < 1000, \
                f"Initialization took {initialization_time_ms:.2f}ms, expected < 1000ms"
    
    def test_state_save_performance(self, sentinel_system):
        """Test that state save completes quickly."""
        sentinel_system.frame_count = 1000
        sentinel_system.start_time = time.time()
        
        with patch('builtins.open', create=True), \
             patch('pickle.dump'), \
             patch('pathlib.Path.mkdir'):
            
            start_time = time.perf_counter()
            sentinel_system._save_system_state()
            end_time = time.perf_counter()
            
            save_time_ms = (end_time - start_time) * 1000
            assert save_time_ms < 50, \
                f"State save took {save_time_ms:.2f}ms, expected < 50ms"
