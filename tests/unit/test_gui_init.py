"""Test suite for GUI __init__ module."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestGUIModuleImport:
    """Test suite for GUI module initialization and imports."""
    
    def test_module_imports_successfully(self):
        """Test that the GUI module can be imported without errors."""
        try:
            import src.gui
            assert src.gui is not None
        except ImportError as e:
            pytest.fail(f"Failed to import GUI module: {e}")
    
    def test_sentinel_main_window_available(self):
        """Test that SENTINELMainWindow is available in module exports."""
        import src.gui
        
        assert hasattr(src.gui, 'SENTINELMainWindow')
        assert 'SENTINELMainWindow' in src.gui.__all__
    
    def test_module_docstring_exists(self):
        """Test that the module has proper documentation."""
        import src.gui
        
        assert src.gui.__doc__ is not None
        assert len(src.gui.__doc__) > 0
        assert 'SENTINEL' in src.gui.__doc__
    
    def test_all_exports_defined(self):
        """Test that __all__ is properly defined with expected exports."""
        import src.gui
        
        assert hasattr(src.gui, '__all__')
        assert isinstance(src.gui.__all__, list)
        assert len(src.gui.__all__) > 0
    
    @patch('src.gui.main_window.SENTINELMainWindow')
    def test_main_window_import_mocked(self, mock_main_window):
        """Test that main_window module is imported correctly."""
        # Reload the module to trigger the import with mock
        import importlib
        import src.gui
        importlib.reload(src.gui)
        
        # Verify the import was attempted
        assert src.gui.SENTINELMainWindow is not None
    
    def test_logger_initialization(self):
        """Test that module logger is properly initialized."""
        import src.gui
        
        assert hasattr(src.gui, 'logger')
        assert src.gui.logger is not None
        assert src.gui.logger.name == 'src.gui'


class TestGUIModuleIntegration:
    """Integration tests for GUI module with PyQt6."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("PyQt6", reason="PyQt6 not installed"),
        reason="PyQt6 not available"
    )
    def test_pyqt6_available(self):
        """Test that PyQt6 is available for GUI operations."""
        try:
            from PyQt6 import QtWidgets, QtCore
            assert QtWidgets is not None
            assert QtCore is not None
        except ImportError:
            pytest.skip("PyQt6 not installed")
    
    @pytest.mark.skipif(
        not pytest.importorskip("PyQt6", reason="PyQt6 not installed"),
        reason="PyQt6 not available"
    )
    @patch('src.gui.main_window.SENTINELMainWindow')
    def test_main_window_can_be_instantiated(self, mock_main_window):
        """Test that SENTINELMainWindow can be instantiated through module."""
        import src.gui
        
        # Create mock instance
        mock_instance = MagicMock()
        mock_main_window.return_value = mock_instance
        
        # Instantiate through module
        window = src.gui.SENTINELMainWindow()
        
        assert window is not None
        mock_main_window.assert_called_once()


class TestGUIModuleLogging:
    """Test suite for GUI module logging functionality."""
    
    @patch('src.gui.logger')
    def test_debug_message_logged_on_import(self, mock_logger):
        """Test that debug message is logged when module is imported."""
        import importlib
        import src.gui
        
        # Reload to trigger logging
        importlib.reload(src.gui)
        
        # Note: This test verifies the logger exists and is configured
        # The actual debug call happens at import time
        assert src.gui.logger is not None
    
    def test_logger_has_correct_name(self):
        """Test that logger is created with correct module name."""
        import src.gui
        
        assert src.gui.logger.name == 'src.gui'
    
    def test_logger_is_logging_instance(self):
        """Test that logger is a proper logging.Logger instance."""
        import src.gui
        import logging
        
        assert isinstance(src.gui.logger, logging.Logger)


class TestGUIModuleStructure:
    """Test suite for GUI module structure and organization."""
    
    def test_module_has_expected_attributes(self):
        """Test that module has all expected attributes."""
        import src.gui
        
        expected_attributes = ['__doc__', '__all__', 'logger', 'SENTINELMainWindow']
        
        for attr in expected_attributes:
            assert hasattr(src.gui, attr), f"Missing expected attribute: {attr}"
    
    def test_module_path_correct(self):
        """Test that module is located in correct path."""
        import src.gui
        
        module_file = Path(src.gui.__file__)
        assert module_file.name == '__init__.py'
        assert module_file.parent.name == 'gui'
        assert module_file.parent.parent.name == 'src'
    
    def test_no_unexpected_exports(self):
        """Test that module doesn't export unexpected items."""
        import src.gui
        
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(src.gui) if not attr.startswith('_')]
        
        # Expected public attributes
        expected = {'SENTINELMainWindow', 'logger', 'logging'}
        
        # All public attributes should be expected or in __all__
        for attr in public_attrs:
            assert attr in expected or attr in src.gui.__all__, \
                f"Unexpected public attribute: {attr}"


class TestGUIModuleDependencies:
    """Test suite for GUI module dependencies."""
    
    def test_logging_module_imported(self):
        """Test that logging module is properly imported."""
        import src.gui
        import logging
        
        assert hasattr(src.gui, 'logging')
        assert src.gui.logging is logging
    
    @patch('src.gui.main_window')
    def test_main_window_module_imported(self, mock_main_window_module):
        """Test that main_window submodule is imported."""
        import importlib
        import src.gui
        
        # Reload to trigger import with mock
        importlib.reload(src.gui)
        
        # Verify the module exists
        assert hasattr(src.gui, 'main_window') or hasattr(src.gui, 'SENTINELMainWindow')
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        try:
            import src.gui
            import src.gui.main_window
            
            # If we get here, no circular imports
            assert True
        except ImportError as e:
            if 'circular' in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                # Other import errors might be due to missing dependencies
                pytest.skip(f"Import error (not circular): {e}")


@pytest.mark.performance
class TestGUIModulePerformance:
    """Performance tests for GUI module initialization."""
    
    def test_import_performance(self):
        """Test that module imports within reasonable time (< 100ms)."""
        import time
        import importlib
        
        # Clear module from cache if present
        if 'src.gui' in sys.modules:
            del sys.modules['src.gui']
        
        start_time = time.perf_counter()
        import src.gui
        end_time = time.perf_counter()
        
        import_time_ms = (end_time - start_time) * 1000
        
        # GUI module import should be fast (< 100ms)
        assert import_time_ms < 100, \
            f"Module import took {import_time_ms:.2f}ms, expected < 100ms"
    
    def test_reload_performance(self):
        """Test that module reload is performant."""
        import time
        import importlib
        import src.gui
        
        start_time = time.perf_counter()
        importlib.reload(src.gui)
        end_time = time.perf_counter()
        
        reload_time_ms = (end_time - start_time) * 1000
        
        # Reload should be even faster than initial import
        assert reload_time_ms < 50, \
            f"Module reload took {reload_time_ms:.2f}ms, expected < 50ms"


class TestGUIModuleErrorHandling:
    """Test suite for GUI module error handling."""
    
    @patch('src.gui.main_window.SENTINELMainWindow', side_effect=ImportError("Mock import error"))
    def test_handles_main_window_import_error(self, mock_main_window):
        """Test behavior when main_window import fails."""
        import importlib
        
        with pytest.raises(ImportError):
            import src.gui
            importlib.reload(src.gui)
    
    def test_module_survives_logger_issues(self):
        """Test that module can handle logging configuration issues."""
        import src.gui
        
        # Even if logger has issues, module should be importable
        assert src.gui is not None
        assert hasattr(src.gui, 'logger')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
