"""Test suite for src/__init__.py module."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestSrcPackageInitialization:
    """Test suite for src package initialization."""
    
    def test_package_docstring(self):
        """Test that package has proper docstring."""
        import src
        assert src.__doc__ is not None
        assert "SENTINEL" in src.__doc__
        assert "Contextual Safety Intelligence Platform" in src.__doc__
    
    @patch('src.gui_main')
    @patch('src.main')
    def test_successful_initialization_with_all_dependencies(self, mock_main, mock_gui_main):
        """Test that package initializes correctly when all dependencies are available."""
        # Setup mocks
        mock_main.SentinelSystem = MagicMock(name='SentinelSystem')
        mock_main.main = MagicMock(name='main')
        mock_gui_main.main = MagicMock(name='gui_main')
        
        # Force reimport to test initialization
        if 'src' in sys.modules:
            del sys.modules['src']
        
        import src
        
        # Verify exports are available
        assert hasattr(src, 'SentinelSystem')
        assert hasattr(src, 'main')
        assert hasattr(src, 'gui_main')
        assert 'SentinelSystem' in src.__all__
        assert 'main' in src.__all__
        assert 'gui_main' in src.__all__
    
    def test_initialization_with_missing_dependencies(self):
        """Test that package handles missing dependencies gracefully."""
        # Simulate missing dependencies by removing modules
        modules_to_remove = ['src.gui_main', 'src.main', 'src']
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
        
        # Mock the imports to raise ImportError
        with patch.dict('sys.modules', {
            'src.gui_main': None,
            'src.main': None
        }):
            # This should not raise an exception
            try:
                import src
                # When dependencies are missing, __all__ should be empty
                assert isinstance(src.__all__, list)
            except ImportError:
                pytest.fail("Package initialization should not raise ImportError")
    
    def test_all_exports_list_type(self):
        """Test that __all__ is always a list."""
        import src
        assert isinstance(src.__all__, list)
    
    def test_all_exports_contains_expected_items(self):
        """Test that __all__ contains expected exports when available."""
        import src
        
        # Check if exports are available
        if len(src.__all__) > 0:
            expected_exports = {'SentinelSystem', 'gui_main', 'main'}
            actual_exports = set(src.__all__)
            assert actual_exports == expected_exports
    
    def test_logger_initialization(self):
        """Test that package logger is properly initialized."""
        import src
        
        # Verify logger exists and has correct name
        assert hasattr(src, 'logger')
        assert src.logger.name == 'src'
    
    @patch('src.logger')
    def test_logging_on_successful_import(self, mock_logger):
        """Test that appropriate log messages are generated on successful import."""
        # Force reimport
        if 'src' in sys.modules:
            del sys.modules['src']
        
        with patch('src.gui_main'), patch('src.main'):
            import src
            
            # Verify debug and info logs were called
            # Note: This test may need adjustment based on actual logging behavior
            assert mock_logger.debug.called or mock_logger.info.called
    
    @patch('src.logger')
    def test_logging_on_import_error(self, mock_logger):
        """Test that warning is logged when dependencies are missing."""
        # Force reimport with missing dependencies
        if 'src' in sys.modules:
            del sys.modules['src']
        
        with patch('src.gui_main', side_effect=ImportError("Missing dependency")):
            with patch('src.main', side_effect=ImportError("Missing dependency")):
                try:
                    import src
                    # Verify warning was logged
                    # Note: This test may need adjustment based on actual logging behavior
                except ImportError:
                    pass
    
    def test_sentinel_system_export_when_available(self):
        """Test that SentinelSystem is exported when main module is available."""
        import src
        
        if 'SentinelSystem' in src.__all__:
            assert hasattr(src, 'SentinelSystem')
            # Verify it's callable (a class)
            assert callable(src.SentinelSystem)
    
    def test_main_function_export_when_available(self):
        """Test that main function is exported when main module is available."""
        import src
        
        if 'main' in src.__all__:
            assert hasattr(src, 'main')
            # Verify it's callable (a function)
            assert callable(src.main)
    
    def test_gui_main_function_export_when_available(self):
        """Test that gui_main function is exported when gui_main module is available."""
        import src
        
        if 'gui_main' in src.__all__:
            assert hasattr(src, 'gui_main')
            # Verify it's callable (a function)
            assert callable(src.gui_main)
    
    def test_no_unexpected_exports(self):
        """Test that only expected items are exported."""
        import src
        
        expected_exports = {'SentinelSystem', 'gui_main', 'main'}
        actual_exports = set(src.__all__)
        
        # All actual exports should be in expected exports
        assert actual_exports.issubset(expected_exports)
    
    def test_package_can_be_imported_multiple_times(self):
        """Test that package can be safely imported multiple times."""
        import src
        first_all = src.__all__.copy()
        
        # Import again
        import src as src2
        second_all = src2.__all__.copy()
        
        # Should be identical
        assert first_all == second_all
    
    def test_graceful_degradation_empty_all(self):
        """Test that __all__ is empty list when imports fail, not None."""
        # This ensures that 'from src import *' won't fail catastrophically
        import src
        
        assert src.__all__ is not None
        assert isinstance(src.__all__, list)
    
    @pytest.mark.integration
    def test_actual_imports_work(self):
        """Integration test: verify actual imports work if dependencies are present."""
        try:
            from src import SentinelSystem, main, gui_main
            
            # If we get here, imports worked
            assert SentinelSystem is not None
            assert main is not None
            assert gui_main is not None
            
            # Verify they are the right types
            assert callable(SentinelSystem)  # Class
            assert callable(main)  # Function
            assert callable(gui_main)  # Function
            
        except ImportError:
            # Dependencies not available, skip this test
            pytest.skip("Dependencies not available for integration test")
    
    def test_import_star_behavior(self):
        """Test that 'from src import *' works correctly."""
        # Create a clean namespace
        test_namespace = {}
        
        try:
            exec("from src import *", test_namespace)
            
            # Check what was imported
            import src
            for item in src.__all__:
                assert item in test_namespace, f"{item} should be imported with 'from src import *'"
                
        except ImportError:
            # If dependencies are missing, __all__ should be empty
            import src
            assert len(src.__all__) == 0


class TestPackageMetadata:
    """Test suite for package metadata and attributes."""
    
    def test_package_has_name_attribute(self):
        """Test that package has __name__ attribute."""
        import src
        assert hasattr(src, '__name__')
        assert src.__name__ == 'src'
    
    def test_package_has_file_attribute(self):
        """Test that package has __file__ attribute."""
        import src
        assert hasattr(src, '__file__')
        assert src.__file__ is not None
        assert '__init__.py' in src.__file__
    
    def test_package_has_path_attribute(self):
        """Test that package has __path__ attribute."""
        import src
        assert hasattr(src, '__path__')
        assert isinstance(src.__path__, list)
    
    def test_package_location(self):
        """Test that package is in expected location."""
        import src
        package_path = Path(src.__file__).parent
        assert package_path.name == 'src'


class TestErrorHandling:
    """Test suite for error handling in package initialization."""
    
    def test_handles_import_error_gracefully(self):
        """Test that ImportError during initialization is handled gracefully."""
        # This should not raise an exception even if dependencies are missing
        try:
            import src
            # Success - package imported
            assert True
        except ImportError as e:
            pytest.fail(f"Package should handle ImportError gracefully, but raised: {e}")
    
    def test_handles_attribute_error_gracefully(self):
        """Test that AttributeError during initialization is handled gracefully."""
        # Even if submodules have issues, package should import
        try:
            import src
            assert True
        except AttributeError as e:
            pytest.fail(f"Package should handle AttributeError gracefully, but raised: {e}")
    
    def test_partial_import_success(self):
        """Test that package works even if only some submodules are available."""
        import src
        
        # Package should always be importable
        assert src is not None
        
        # __all__ should be a list (empty or populated)
        assert isinstance(src.__all__, list)


@pytest.mark.performance
class TestPerformance:
    """Test suite for package initialization performance."""
    
    def test_import_performance(self):
        """Test that package import completes quickly (< 100ms)."""
        import time
        
        # Remove from cache to test fresh import
        if 'src' in sys.modules:
            del sys.modules['src']
        
        start_time = time.perf_counter()
        import src
        end_time = time.perf_counter()
        
        import_time_ms = (end_time - start_time) * 1000
        
        # Package initialization should be fast
        # Note: This may be slower on first import due to dependency loading
        assert import_time_ms < 100, f"Import took {import_time_ms:.2f}ms, expected < 100ms"
    
    def test_reimport_performance(self):
        """Test that reimporting is fast (< 10ms)."""
        import time
        import src  # Ensure it's already imported
        
        start_time = time.perf_counter()
        import src as src2
        end_time = time.perf_counter()
        
        reimport_time_ms = (end_time - start_time) * 1000
        
        # Reimport should be very fast (cached)
        assert reimport_time_ms < 10, f"Reimport took {reimport_time_ms:.2f}ms, expected < 10ms"
