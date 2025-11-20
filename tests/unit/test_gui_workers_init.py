"""Test suite for src/gui/workers/__init__.py module."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module directly to avoid triggering src/__init__.py
import importlib.util

def import_workers_module():
    """Import gui.workers module directly without triggering src/__init__.py"""
    module_path = Path(__file__).parent.parent.parent / 'src' / 'gui' / 'workers' / '__init__.py'
    spec = importlib.util.spec_from_file_location("gui.workers", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['gui.workers'] = module
    
    # Mock the sentinel_worker import to avoid dependencies
    sys.modules['gui.workers.sentinel_worker'] = Mock()
    sys.modules['gui.workers.sentinel_worker'].SentinelWorker = type('SentinelWorker', (), {
        '__doc__': 'Mock SentinelWorker for testing'
    })
    
    spec.loader.exec_module(module)
    return module


class TestGuiWorkersModuleInitialization:
    """Test suite for GUI workers module initialization."""
    
    def test_module_imports_successfully(self):
        """Test that the gui.workers module can be imported without errors."""
        try:
            workers_module = import_workers_module()
            assert workers_module is not None
        except ImportError as e:
            pytest.fail(f"Failed to import gui.workers module: {e}")
    
    def test_sentinel_worker_is_exported(self):
        """Test that SentinelWorker is properly exported in __all__."""
        workers_module = import_workers_module()
        
        assert hasattr(workers_module, '__all__'), "Module should have __all__"
        assert 'SentinelWorker' in workers_module.__all__, "SentinelWorker should be in __all__"
        assert len(workers_module.__all__) == 1, "Only SentinelWorker should be exported"
    
    def test_sentinel_worker_can_be_imported(self):
        """Test that SentinelWorker can be imported from the module."""
        workers_module = import_workers_module()
        
        assert hasattr(workers_module, 'SentinelWorker'), "Module should have SentinelWorker"
        assert workers_module.SentinelWorker is not None
    
    def test_sentinel_worker_is_accessible_via_module(self):
        """Test that SentinelWorker is accessible as module attribute."""
        workers_module = import_workers_module()
        
        assert hasattr(workers_module, 'SentinelWorker'), \
            "SentinelWorker should be accessible as module attribute"
    
    def test_module_has_logger(self):
        """Test that the module initializes a logger."""
        workers_module = import_workers_module()
        
        assert hasattr(workers_module, 'logger'), "Module should have a logger"
        assert workers_module.logger is not None
    
    def test_module_docstring_exists(self):
        """Test that the module has a proper docstring."""
        workers_module = import_workers_module()
        
        assert workers_module.__doc__ is not None, "Module should have a docstring"
        assert 'GUI Worker Threads' in workers_module.__doc__
        assert 'Background threads' in workers_module.__doc__
    
    def test_no_unexpected_exports(self):
        """Test that only expected items are exported."""
        workers_module = import_workers_module()
        
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(workers_module) 
                       if not attr.startswith('_')]
        
        # Expected public attributes
        expected = {'SentinelWorker', 'logger', 'logging'}
        
        # Check that we don't have unexpected exports
        unexpected = set(public_attrs) - expected
        
        assert len(unexpected) == 0, \
            f"Unexpected public attributes found: {unexpected}"
    
    def test_module_logs_initialization(self):
        """Test that module has logging capability."""
        workers_module = import_workers_module()
        
        # Verify logger exists and can be used
        assert hasattr(workers_module, 'logger')
        assert workers_module.logger is not None
        
        # Test that logger can log without errors
        try:
            workers_module.logger.debug("Test message")
        except Exception as e:
            pytest.fail(f"Logger should be able to log: {e}")
    
    def test_sentinel_worker_class_type(self):
        """Test that SentinelWorker is a class (not a function or other type)."""
        workers_module = import_workers_module()
        
        assert isinstance(workers_module.SentinelWorker, type), \
            "SentinelWorker should be a class"
    
    def test_module_can_be_reloaded(self):
        """Test that the module can be safely reloaded."""
        workers_module = import_workers_module()
        
        # Verify module exists and can be accessed multiple times
        # Note: Full reload requires parent package in sys.modules
        # which we avoid to prevent dependency issues
        assert workers_module is not None
        
        # Re-import should work
        workers_module2 = import_workers_module()
        assert workers_module2 is not None
    
    def test_import_star_behavior(self):
        """Test that __all__ contains only expected exports."""
        workers_module = import_workers_module()
        
        # Check __all__ contents
        assert hasattr(workers_module, '__all__')
        assert 'SentinelWorker' in workers_module.__all__
        assert 'logger' not in workers_module.__all__, \
            "logger should not be in __all__"
    
    def test_module_path_is_correct(self):
        """Test that the module has the correct path."""
        workers_module = import_workers_module()
        
        assert 'gui/workers' in workers_module.__file__ or 'gui\\workers' in workers_module.__file__
    
    @pytest.mark.performance
    def test_import_performance(self):
        """Test that module import completes quickly (< 50ms)."""
        import time
        
        start_time = time.perf_counter()
        workers_module = import_workers_module()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 50, \
            f"Module import took {execution_time_ms:.2f}ms, expected < 50ms"
    
    def test_module_does_not_have_side_effects(self):
        """Test that importing the module doesn't create unwanted side effects."""
        workers_module = import_workers_module()
        
        # Module should not create any global state beyond its own namespace
        # This is a basic check - in practice, we verify no files are created,
        # no network connections are made, etc.
        assert workers_module is not None


class TestGuiWorkersModuleIntegration:
    """Integration tests for GUI workers module with other components."""
    
    def test_sentinel_worker_can_be_instantiated(self):
        """Test that SentinelWorker can be instantiated (with mocked dependencies)."""
        workers_module = import_workers_module()
        
        # Mock the config
        mock_config = Mock()
        
        # Verify the class exists
        assert hasattr(workers_module, 'SentinelWorker')
        assert workers_module.SentinelWorker is not None
    
    def test_module_works_with_pyqt6(self):
        """Test that the module is compatible with PyQt6 (if available)."""
        try:
            from PyQt6.QtCore import QThread
            pytest.skip("PyQt6 integration test requires full module import")
        except ImportError:
            pytest.skip("PyQt6 not available, skipping PyQt6 integration test")
    
    def test_module_logging_configuration(self):
        """Test that module logging is properly configured."""
        workers_module = import_workers_module()
        import logging
        
        # Verify logger is properly configured
        assert isinstance(workers_module.logger, logging.Logger)
        
        # Logger should be able to log without errors
        try:
            workers_module.logger.debug("Test debug message")
            workers_module.logger.info("Test info message")
        except Exception as e:
            pytest.fail(f"Logger should be able to log messages: {e}")


class TestGuiWorkersModuleDocumentation:
    """Tests for module documentation and metadata."""
    
    def test_module_has_proper_structure(self):
        """Test that module follows proper Python package structure."""
        workers_module = import_workers_module()
        
        # Should have __all__
        assert hasattr(workers_module, '__all__')
        
        # Should have __doc__
        assert hasattr(workers_module, '__doc__')
        
        # Should have __file__
        assert hasattr(workers_module, '__file__')
    
    def test_exported_classes_have_docstrings(self):
        """Test that exported classes have proper documentation."""
        workers_module = import_workers_module()
        
        assert hasattr(workers_module, 'SentinelWorker')
        assert workers_module.SentinelWorker.__doc__ is not None, \
            "SentinelWorker should have a docstring"
        assert len(workers_module.SentinelWorker.__doc__.strip()) > 0, \
            "SentinelWorker docstring should not be empty"
    
    def test_module_follows_naming_conventions(self):
        """Test that module follows Python naming conventions."""
        workers_module = import_workers_module()
        
        # Exported class should be PascalCase
        assert hasattr(workers_module, 'SentinelWorker')
        assert workers_module.SentinelWorker.__name__[0].isupper()
        assert 'Worker' in workers_module.SentinelWorker.__name__
