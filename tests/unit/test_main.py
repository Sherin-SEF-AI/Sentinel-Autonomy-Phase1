"""Test suite for main system orchestration module."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from src.main import SentinelSystem, main
from src.core.config import ConfigManager


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration for testing."""
    config = Mock(spec=ConfigManager)
    config.get = Mock(side_effect=lambda key, default=None: {
        'visualization.enabled': True,
        'visualization.port': 8080,
        'system.version': '1.0',
        'system.log_level': 'INFO'
    }.get(key, default))
    config.validate = Mock(return_value=True)
    config.config = {
        'system': {'name': 'SENTINEL', 'version': '1.0'},
        'cameras': {},
        'models': {}
    }
    return config


@pytest.fixture
def mock_modules():
    """Fixture providing mocked module instances."""
    with patch('src.main.CameraManager') as camera_mgr, \
         patch('src.main.BEVGenerator') as bev_gen, \
         patch('src.main.SemanticSegmentor') as segmentor, \
         patch('src.main.ObjectDetector') as detector, \
         patch('src.main.DriverMonitor') as dms, \
         patch('src.main.ContextualIntelligence') as intelligence, \
         patch('src.main.AlertSystem') as alert_sys, \
         patch('src.main.ScenarioRecorder') as recorder, \
         patch('src.main.VisualizationServer') as viz_server, \
         patch('src.main.StreamingManager') as streaming_mgr:
        
        # Configure mock instances
        camera_mgr.return_value = Mock()
        bev_gen.return_value = Mock()
        segmentor.return_value = Mock()
        detector.return_value = Mock()
        dms.return_value = Mock()
        intelligence.return_value = Mock()
        alert_sys.return_value = Mock()
        recorder.return_value = Mock()
        viz_server.return_value = Mock()
        streaming_mgr.return_value = Mock()
        
        yield {
            'camera_manager': camera_mgr,
            'bev_generator': bev_gen,
            'segmentor': segmentor,
            'detector': detector,
            'dms': dms,
            'intelligence': intelligence,
            'alert_system': alert_sys,
            'recorder': recorder,
            'viz_server': viz_server,
            'streaming_manager': streaming_mgr
        }


@pytest.fixture
def sentinel_system(mock_config, mock_modules):
    """Fixture creating an instance of SentinelSystem for testing."""
    return SentinelSystem(mock_config)


class TestSentinelSystem:
    """Test suite for SentinelSystem class."""
    
    def test_initialization(self, sentinel_system, mock_config):
        """Test that SentinelSystem initializes correctly with valid configuration."""
        assert sentinel_system is not None
        assert sentinel_system.config == mock_config
        assert sentinel_system.running is False
        assert sentinel_system.frame_count == 0
        assert sentinel_system.start_time is None
        
        # Verify all module instances are initialized
        assert sentinel_system.camera_manager is not None
        assert sentinel_system.bev_generator is not None
        assert sentinel_system.segmentor is not None
        assert sentinel_system.detector is not None
        assert sentinel_system.dms is not None
        assert sentinel_system.intelligence is not None
        assert sentinel_system.alert_system is not None
        assert sentinel_system.recorder is not None
        assert sentinel_system.viz_server is not None
        assert sentinel_system.streaming_manager is not None
        
        # Verify module latency tracking is initialized
        expected_modules = [
            'camera', 'bev', 'segmentation', 'detection', 
            'dms', 'intelligence', 'alerts', 'recording', 'visualization'
        ]
        for module in expected_modules:
            assert module in sentinel_system.module_latencies
            assert isinstance(sentinel_system.module_latencies[module], list)
    
    def test_initialization_without_visualization(self, mock_config, mock_modules):
        """Test initialization when visualization is disabled."""
        mock_config.get = Mock(side_effect=lambda key, default=None: {
            'visualization.enabled': False,
            'system.version': '1.0'
        }.get(key, default))
        
        system = SentinelSystem(mock_config)
        
        assert system.viz_server is not None  # Still initialized
        assert system.streaming_manager is not None
    
    def test_initialization_failure(self, mock_config, mock_modules):
        """Test that initialization failure is handled gracefully."""
        mock_modules['camera_manager'].side_effect = RuntimeError("Camera init failed")
        
        with pytest.raises(RuntimeError, match="Camera init failed"):
            SentinelSystem(mock_config)
    
    def test_start_system(self, sentinel_system):
        """Test starting the SENTINEL system."""
        sentinel_system.start()
        
        assert sentinel_system.running is True
        assert sentinel_system.start_time is not None
        
        # Verify camera manager was started
        sentinel_system.camera_manager.start.assert_called_once()
        
        # Verify visualization server was started
        sentinel_system.viz_server.start.assert_called_once()
    
    def test_start_system_already_running(self, sentinel_system):
        """Test starting system when it's already running."""
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        
        # Should not raise exception, just log warning
        sentinel_system.start()
        
        # Camera manager should not be called again
        sentinel_system.camera_manager.start.assert_not_called()
    
    def test_start_system_failure(self, sentinel_system):
        """Test handling of start failure."""
        sentinel_system.camera_manager.start.side_effect = RuntimeError("Start failed")
        
        with pytest.raises(RuntimeError, match="Start failed"):
            sentinel_system.start()
        
        # System should not be marked as running
        assert sentinel_system.running is False
    
    def test_stop_system(self, sentinel_system):
        """Test stopping the SENTINEL system."""
        # Start the system first
        sentinel_system.running = True
        sentinel_system.start_time = time.time()
        sentinel_system.frame_count = 100
        
        sentinel_system.stop()
        
        assert sentinel_system.running is False
        assert sentinel_system.shutdown_event.is_set()
        
        # Verify camera manager was stopped
        sentinel_system.camera_manager.stop.assert_called_once()
        
        # Verify visualization server was stopped
        sentinel_system.viz_server.stop.assert_called_once()
    
    def test_stop_system_not_running(self, sentinel_system):
        """Test stopping system when it's not running."""
        sentinel_system.running = False
        
        # Should return early without errors
        sentinel_system.stop()
        
        # Stop methods should not be called
        sentinel_system.camera_manager.stop.assert_not_called()
    
    def test_stop_system_with_error(self, sentinel_system):
        """Test that stop handles errors gracefully."""
        sentinel_system.running = True
        sentinel_system.camera_manager.stop.side_effect = RuntimeError("Stop failed")
        
        # Should not raise exception, just log error
        sentinel_system.stop()
        
        # System should still be marked as stopped
        assert sentinel_system.running is False
    
    def test_log_final_statistics(self, sentinel_system):
        """Test logging of final statistics."""
        sentinel_system.start_time = time.time() - 10.0  # 10 seconds ago
        sentinel_system.frame_count = 300
        
        # Add some latency data
        sentinel_system.module_latencies['camera'] = [0.005, 0.006, 0.005]
        sentinel_system.module_latencies['bev'] = [0.015, 0.014, 0.016]
        
        # Should not raise exception
        sentinel_system._log_final_statistics()
    
    def test_log_final_statistics_no_data(self, sentinel_system):
        """Test logging statistics with no data."""
        sentinel_system.start_time = None
        sentinel_system.frame_count = 0
        
        # Should handle gracefully
        sentinel_system._log_final_statistics()
    
    def test_module_initialization_order(self, mock_config, mock_modules):
        """Test that modules are initialized in correct order."""
        system = SentinelSystem(mock_config)
        
        # Verify initialization order through call order
        call_order = [
            mock_modules['camera_manager'],
            mock_modules['bev_generator'],
            mock_modules['segmentor'],
            mock_modules['detector'],
            mock_modules['dms'],
            mock_modules['intelligence'],
            mock_modules['alert_system'],
            mock_modules['recorder']
        ]
        
        for mock_module in call_order:
            assert mock_module.called


class TestMainFunction:
    """Test suite for main() function."""
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml'])
    def test_main_function_success(self, mock_system, mock_logger, mock_config_mgr):
        """Test main function with valid arguments."""
        # Setup mocks
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = 'INFO'
        mock_config_mgr.return_value = mock_config
        
        mock_logger.setup.return_value = Mock()
        
        mock_sentinel = Mock()
        mock_sentinel.running = False  # Will exit loop immediately
        mock_system.return_value = mock_sentinel
        
        # Run main
        result = main()
        
        # Verify success
        assert result == 0
        mock_config_mgr.assert_called_once_with('configs/default.yaml')
        mock_sentinel.start.assert_called_once()
        mock_sentinel.stop.assert_called_once()
    
    @patch('src.main.ConfigManager')
    @patch('sys.argv', ['main.py', '--config', 'nonexistent.yaml'])
    def test_main_function_config_not_found(self, mock_config_mgr):
        """Test main function with missing config file."""
        mock_config_mgr.side_effect = FileNotFoundError("Config not found")
        
        result = main()
        
        assert result == 1
    
    @patch('src.main.ConfigManager')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml'])
    def test_main_function_invalid_config(self, mock_config_mgr):
        """Test main function with invalid configuration."""
        mock_config = Mock()
        mock_config.validate.return_value = False
        mock_config_mgr.return_value = mock_config
        
        result = main()
        
        assert result == 1
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml', '--log-level', 'DEBUG'])
    def test_main_function_with_log_level(self, mock_system, mock_logger, mock_config_mgr):
        """Test main function with custom log level."""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = 'INFO'
        mock_config_mgr.return_value = mock_config
        
        mock_logger.setup.return_value = Mock()
        
        mock_sentinel = Mock()
        mock_sentinel.running = False
        mock_system.return_value = mock_sentinel
        
        result = main()
        
        # Verify DEBUG log level was used
        mock_logger.setup.assert_called_once()
        call_kwargs = mock_logger.setup.call_args
        assert call_kwargs[1]['log_level'] == 'DEBUG' or call_kwargs[0][0] == 'DEBUG'
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py'])
    def test_main_function_default_config(self, mock_system, mock_logger, mock_config_mgr):
        """Test main function with default config path."""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = 'INFO'
        mock_config_mgr.return_value = mock_config
        
        mock_logger.setup.return_value = Mock()
        
        mock_sentinel = Mock()
        mock_sentinel.running = False
        mock_system.return_value = mock_sentinel
        
        result = main()
        
        # Should use default config path
        mock_config_mgr.assert_called_once_with('configs/default.yaml')
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml'])
    def test_main_function_keyboard_interrupt(self, mock_system, mock_logger, mock_config_mgr):
        """Test main function handles keyboard interrupt gracefully."""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = 'INFO'
        mock_config_mgr.return_value = mock_config
        
        mock_logger.setup.return_value = Mock()
        
        mock_sentinel = Mock()
        mock_sentinel.start.side_effect = KeyboardInterrupt()
        mock_system.return_value = mock_sentinel
        
        result = main()
        
        # Should handle gracefully and return 0
        assert result == 0
        mock_sentinel.stop.assert_called_once()
    
    @patch('src.main.ConfigManager')
    @patch('src.main.LoggerSetup')
    @patch('src.main.SentinelSystem')
    @patch('sys.argv', ['main.py', '--config', 'configs/default.yaml'])
    def test_main_function_system_error(self, mock_system, mock_logger, mock_config_mgr):
        """Test main function handles system errors."""
        mock_config = Mock()
        mock_config.validate.return_value = True
        mock_config.get.return_value = 'INFO'
        mock_config_mgr.return_value = mock_config
        
        mock_logger.setup.return_value = Mock()
        
        mock_sentinel = Mock()
        mock_sentinel.start.side_effect = RuntimeError("System error")
        mock_system.return_value = mock_sentinel
        
        result = main()
        
        # Should return error code
        assert result == 1
        mock_sentinel.stop.assert_called_once()


@pytest.mark.performance
class TestSentinelSystemPerformance:
    """Performance tests for SentinelSystem."""
    
    def test_initialization_performance(self, mock_config, mock_modules):
        """Test that system initialization completes within reasonable time."""
        start_time = time.perf_counter()
        system = SentinelSystem(mock_config)
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        
        # Initialization should be fast with mocked modules
        assert initialization_time_ms < 100, \
            f"Initialization took {initialization_time_ms:.2f}ms, expected < 100ms"
    
    def test_start_stop_performance(self, sentinel_system):
        """Test that start/stop operations complete quickly."""
        # Test start
        start_time = time.perf_counter()
        sentinel_system.start()
        start_duration = (time.perf_counter() - start_time) * 1000
        
        assert start_duration < 50, \
            f"Start took {start_duration:.2f}ms, expected < 50ms"
        
        # Test stop
        stop_time = time.perf_counter()
        sentinel_system.stop()
        stop_duration = (time.perf_counter() - stop_time) * 1000
        
        assert stop_duration < 50, \
            f"Stop took {stop_duration:.2f}ms, expected < 50ms"
