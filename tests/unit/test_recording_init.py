"""Test suite for recording module initialization and exports."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestRecordingModuleImports:
    """Test suite for recording module imports and exports."""
    
    def test_recording_trigger_import(self):
        """Test that RecordingTrigger can be imported from recording module."""
        from src.recording import RecordingTrigger
        assert RecordingTrigger is not None
    
    def test_frame_recorder_import(self):
        """Test that FrameRecorder can be imported from recording module."""
        from src.recording import FrameRecorder
        assert FrameRecorder is not None
    
    def test_scenario_exporter_import(self):
        """Test that ScenarioExporter can be imported from recording module."""
        from src.recording import ScenarioExporter
        assert ScenarioExporter is not None
    
    def test_scenario_playback_import(self):
        """Test that ScenarioPlayback can be imported from recording module."""
        from src.recording import ScenarioPlayback
        assert ScenarioPlayback is not None
    
    def test_scenario_recorder_import(self):
        """Test that ScenarioRecorder can be imported from recording module."""
        from src.recording import ScenarioRecorder
        assert ScenarioRecorder is not None
    
    def test_all_exports(self):
        """Test that __all__ contains all expected exports."""
        from src.recording import __all__
        
        expected_exports = [
            'RecordingTrigger',
            'FrameRecorder',
            'ScenarioExporter',
            'ScenarioPlayback',
            'ScenarioRecorder'
        ]
        
        assert set(__all__) == set(expected_exports)
        assert len(__all__) == len(expected_exports)
    
    def test_all_exports_are_importable(self):
        """Test that all items in __all__ can be imported."""
        import src.recording as recording_module
        
        for export_name in recording_module.__all__:
            assert hasattr(recording_module, export_name), f"{export_name} not found in module"
            exported_item = getattr(recording_module, export_name)
            assert exported_item is not None, f"{export_name} is None"
    
    def test_module_docstring(self):
        """Test that module has appropriate docstring."""
        import src.recording as recording_module
        
        assert recording_module.__doc__ is not None
        assert "recording" in recording_module.__doc__.lower()
    
    def test_no_unexpected_exports(self):
        """Test that module doesn't export unexpected items."""
        import src.recording as recording_module
        
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(recording_module) if not attr.startswith('_')]
        
        # Filter out __all__ itself and expected exports
        expected = set(recording_module.__all__) | {'__all__'}
        actual = set(public_attrs)
        
        # All public attributes should be in __all__ or be __all__ itself
        unexpected = actual - expected
        
        # Allow some standard module attributes
        allowed_extras = set()
        unexpected = unexpected - allowed_extras
        
        assert len(unexpected) == 0, f"Unexpected public exports: {unexpected}"


class TestRecordingModuleIntegration:
    """Integration tests for recording module components."""
    
    @patch('src.recording.trigger.logging.getLogger')
    def test_recording_trigger_instantiation(self, mock_logger):
        """Test that RecordingTrigger can be instantiated."""
        from src.recording import RecordingTrigger
        
        config = {
            'risk_threshold': 0.7,
            'ttc_threshold': 1.5
        }
        
        trigger = RecordingTrigger(config)
        assert trigger is not None
    
    @patch('src.recording.recorder.logging.getLogger')
    def test_frame_recorder_instantiation(self, mock_logger):
        """Test that FrameRecorder can be instantiated."""
        from src.recording import FrameRecorder
        
        config = {
            'storage_path': 'scenarios/',
            'max_duration': 30.0
        }
        
        recorder = FrameRecorder(config)
        assert recorder is not None
    
    @patch('src.recording.exporter.logging.getLogger')
    def test_scenario_exporter_instantiation(self, mock_logger):
        """Test that ScenarioExporter can be instantiated."""
        from src.recording import ScenarioExporter
        
        config = {
            'storage_path': 'scenarios/',
            'compression': 'h264'
        }
        
        exporter = ScenarioExporter(config)
        assert exporter is not None
    
    @patch('src.recording.playback.logging.getLogger')
    def test_scenario_playback_instantiation(self, mock_logger):
        """Test that ScenarioPlayback can be instantiated."""
        from src.recording import ScenarioPlayback
        
        playback = ScenarioPlayback('scenarios/test_scenario')
        assert playback is not None
    
    @patch('src.recording.scenario_recorder.logging.getLogger')
    def test_scenario_recorder_instantiation(self, mock_logger):
        """Test that ScenarioRecorder can be instantiated."""
        from src.recording import ScenarioRecorder
        
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
        assert recorder is not None


class TestRecordingModuleAPI:
    """Test the public API of the recording module."""
    
    def test_recording_trigger_has_required_methods(self):
        """Test that RecordingTrigger has required public methods."""
        from src.recording import RecordingTrigger
        
        required_methods = ['should_trigger', 'reset']
        
        for method_name in required_methods:
            assert hasattr(RecordingTrigger, method_name), f"RecordingTrigger missing {method_name}"
    
    def test_frame_recorder_has_required_methods(self):
        """Test that FrameRecorder has required public methods."""
        from src.recording import FrameRecorder
        
        required_methods = ['start_recording', 'add_frame', 'stop_recording']
        
        for method_name in required_methods:
            assert hasattr(FrameRecorder, method_name), f"FrameRecorder missing {method_name}"
    
    def test_scenario_exporter_has_required_methods(self):
        """Test that ScenarioExporter has required public methods."""
        from src.recording import ScenarioExporter
        
        required_methods = ['export']
        
        for method_name in required_methods:
            assert hasattr(ScenarioExporter, method_name), f"ScenarioExporter missing {method_name}"
    
    def test_scenario_playback_has_required_methods(self):
        """Test that ScenarioPlayback has required public methods."""
        from src.recording import ScenarioPlayback
        
        required_methods = ['load', 'get_frame', 'get_metadata']
        
        for method_name in required_methods:
            assert hasattr(ScenarioPlayback, method_name), f"ScenarioPlayback missing {method_name}"
    
    def test_scenario_recorder_has_required_methods(self):
        """Test that ScenarioRecorder has required public methods."""
        from src.recording import ScenarioRecorder
        
        required_methods = ['process', 'is_recording']
        
        for method_name in required_methods:
            assert hasattr(ScenarioRecorder, method_name), f"ScenarioRecorder missing {method_name}"


class TestRecordingModuleCompatibility:
    """Test compatibility and version requirements."""
    
    def test_module_can_be_imported_without_errors(self):
        """Test that the module can be imported without raising exceptions."""
        try:
            import src.recording
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import recording module: {e}")
    
    def test_submodules_are_accessible(self):
        """Test that submodules are accessible through the main module."""
        import src.recording
        
        # These should be accessible as attributes
        assert hasattr(src.recording, 'RecordingTrigger')
        assert hasattr(src.recording, 'FrameRecorder')
        assert hasattr(src.recording, 'ScenarioExporter')
        assert hasattr(src.recording, 'ScenarioPlayback')
        assert hasattr(src.recording, 'ScenarioRecorder')
    
    def test_direct_import_vs_module_import(self):
        """Test that direct imports match module attribute access."""
        from src.recording import RecordingTrigger as DirectImport
        import src.recording
        
        ModuleImport = src.recording.RecordingTrigger
        
        assert DirectImport is ModuleImport, "Direct import doesn't match module attribute"
    
    @pytest.mark.performance
    def test_import_performance(self):
        """Test that module imports complete quickly."""
        import time
        import sys
        
        # Remove module from cache if present
        if 'src.recording' in sys.modules:
            del sys.modules['src.recording']
        
        start_time = time.perf_counter()
        import src.recording
        end_time = time.perf_counter()
        
        import_time_ms = (end_time - start_time) * 1000
        assert import_time_ms < 100, f"Import took {import_time_ms:.2f}ms, expected < 100ms"
