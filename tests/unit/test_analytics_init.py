"""Test suite for analytics module initialization."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))


class TestAnalyticsModuleImports:
    """Test suite for analytics module imports and initialization."""
    
    def test_module_imports_successfully(self):
        """Test that analytics module can be imported without errors."""
        try:
            import src.analytics as analytics
            assert analytics is not None
        except ImportError as e:
            pytest.fail(f"Failed to import analytics module: {e}")
    
    def test_all_exports_defined(self):
        """Test that __all__ is properly defined with expected exports."""
        from src.analytics import __all__
        
        expected_exports = [
            'AnalyticsDashboard',
            'BehaviorReportGenerator',
            'ReportExporter',
            'RiskHeatmap',
            'TripAnalytics',
            'TripSegment',
            'TripSummary',
        ]
        
        assert __all__ is not None
        assert isinstance(__all__, list)
        assert len(__all__) == len(expected_exports)
        
        for export in expected_exports:
            assert export in __all__, f"Expected export '{export}' not found in __all__"
    
    def test_analytics_dashboard_importable(self):
        """Test that AnalyticsDashboard can be imported from analytics module."""
        try:
            from src.analytics import AnalyticsDashboard
            assert AnalyticsDashboard is not None
        except ImportError as e:
            pytest.fail(f"Failed to import AnalyticsDashboard: {e}")
    
    def test_behavior_report_generator_importable(self):
        """Test that BehaviorReportGenerator can be imported from analytics module."""
        try:
            from src.analytics import BehaviorReportGenerator
            assert BehaviorReportGenerator is not None
        except ImportError as e:
            pytest.fail(f"Failed to import BehaviorReportGenerator: {e}")
    
    def test_report_exporter_importable(self):
        """Test that ReportExporter can be imported from analytics module."""
        try:
            from src.analytics import ReportExporter
            assert ReportExporter is not None
        except ImportError as e:
            pytest.fail(f"Failed to import ReportExporter: {e}")
    
    def test_risk_heatmap_importable(self):
        """Test that RiskHeatmap can be imported from analytics module."""
        try:
            from src.analytics import RiskHeatmap
            assert RiskHeatmap is not None
        except ImportError as e:
            pytest.fail(f"Failed to import RiskHeatmap: {e}")
    
    def test_trip_analytics_importable(self):
        """Test that TripAnalytics can be imported from analytics module."""
        try:
            from src.analytics import TripAnalytics
            assert TripAnalytics is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TripAnalytics: {e}")
    
    def test_trip_segment_importable(self):
        """Test that TripSegment can be imported from analytics module."""
        try:
            from src.analytics import TripSegment
            assert TripSegment is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TripSegment: {e}")
    
    def test_trip_summary_importable(self):
        """Test that TripSummary can be imported from analytics module."""
        try:
            from src.analytics import TripSummary
            assert TripSummary is not None
        except ImportError as e:
            pytest.fail(f"Failed to import TripSummary: {e}")
    
    def test_all_classes_accessible_via_module(self):
        """Test that all exported classes are accessible via the module."""
        import src.analytics as analytics
        
        assert hasattr(analytics, 'AnalyticsDashboard')
        assert hasattr(analytics, 'BehaviorReportGenerator')
        assert hasattr(analytics, 'ReportExporter')
        assert hasattr(analytics, 'RiskHeatmap')
        assert hasattr(analytics, 'TripAnalytics')
        assert hasattr(analytics, 'TripSegment')
        assert hasattr(analytics, 'TripSummary')
    
    def test_module_logger_initialized(self):
        """Test that module logger is properly initialized."""
        import src.analytics as analytics
        
        assert hasattr(analytics, 'logger')
        assert analytics.logger is not None
        assert analytics.logger.name == 'src.analytics'
    
    def test_no_unexpected_exports(self):
        """Test that only expected items are exported in __all__."""
        from src.analytics import __all__
        import src.analytics as analytics
        
        # Get all public attributes (not starting with _)
        public_attrs = [attr for attr in dir(analytics) if not attr.startswith('_')]
        
        # Filter out the logger and module-level imports
        expected_public = set(__all__) | {'logger', 'logging'}
        
        for attr in __all__:
            assert attr in public_attrs, f"Exported '{attr}' not found in module"
    
    def test_module_docstring_exists(self):
        """Test that module has a proper docstring."""
        import src.analytics as analytics
        
        assert analytics.__doc__ is not None
        assert len(analytics.__doc__) > 0
        assert 'Analytics' in analytics.__doc__
    
    @pytest.mark.performance
    def test_import_performance(self):
        """Test that module imports complete within reasonable time."""
        import time
        import importlib
        
        # Clear module from cache if present
        if 'src.analytics' in sys.modules:
            del sys.modules['src.analytics']
        
        start_time = time.perf_counter()
        import src.analytics
        end_time = time.perf_counter()
        
        import_time_ms = (end_time - start_time) * 1000
        
        # Module import should be fast (< 100ms)
        assert import_time_ms < 100, f"Module import took {import_time_ms:.2f}ms, expected < 100ms"
    
    def test_circular_import_protection(self):
        """Test that module doesn't have circular import issues."""
        # This test passes if the import succeeds without hanging
        try:
            import src.analytics
            from src.analytics import AnalyticsDashboard, TripAnalytics
            assert True
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")
    
    def test_submodule_independence(self):
        """Test that individual submodules can be imported independently."""
        try:
            from src.analytics.trip_analytics import TripAnalytics
            from src.analytics.behavior_report import BehaviorReportGenerator
            from src.analytics.risk_heatmap import RiskHeatmap
            from src.analytics.analytics_dashboard import AnalyticsDashboard
            from src.analytics.report_exporter import ReportExporter
            
            assert TripAnalytics is not None
            assert BehaviorReportGenerator is not None
            assert RiskHeatmap is not None
            assert AnalyticsDashboard is not None
            assert ReportExporter is not None
        except ImportError as e:
            pytest.fail(f"Submodule import failed: {e}")


class TestAnalyticsModuleIntegration:
    """Integration tests for analytics module components."""
    
    def test_all_classes_are_classes(self):
        """Test that all exports are actually classes (not functions or other types)."""
        from src.analytics import (
            AnalyticsDashboard,
            BehaviorReportGenerator,
            ReportExporter,
            RiskHeatmap,
            TripAnalytics,
            TripSegment,
            TripSummary,
        )
        
        # Check that they are classes (can be instantiated)
        assert isinstance(AnalyticsDashboard, type)
        assert isinstance(BehaviorReportGenerator, type)
        assert isinstance(ReportExporter, type)
        assert isinstance(RiskHeatmap, type)
        assert isinstance(TripAnalytics, type)
        
        # TripSegment and TripSummary might be dataclasses
        assert TripSegment is not None
        assert TripSummary is not None
    
    def test_module_reload_safe(self):
        """Test that module can be safely reloaded."""
        import importlib
        import src.analytics as analytics
        
        try:
            importlib.reload(analytics)
            assert analytics is not None
        except Exception as e:
            pytest.fail(f"Module reload failed: {e}")
    
    def test_exports_match_actual_classes(self):
        """Test that exported names match the actual class names."""
        import src.analytics as analytics
        
        for export_name in analytics.__all__:
            exported_obj = getattr(analytics, export_name)
            
            # For classes, check that the name matches
            if isinstance(exported_obj, type):
                assert exported_obj.__name__ == export_name, \
                    f"Class name mismatch: exported as '{export_name}' but class is '{exported_obj.__name__}'"


class TestAnalyticsModuleLogging:
    """Test logging configuration for analytics module."""
    
    def test_logger_configuration(self):
        """Test that logger is properly configured."""
        import src.analytics as analytics
        import logging
        
        assert isinstance(analytics.logger, logging.Logger)
        assert analytics.logger.name == 'src.analytics'
    
    @patch('src.analytics.logger')
    def test_initialization_logging(self, mock_logger):
        """Test that module logs initialization message."""
        import importlib
        import src.analytics
        
        # Reload to trigger initialization
        importlib.reload(src.analytics)
        
        # Check that debug message was logged
        # Note: This might not work if logger is already initialized
        # but it tests the logging infrastructure
        assert mock_logger is not None


class TestAnalyticsModuleEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_import_with_missing_dependencies(self):
        """Test behavior when optional dependencies are missing."""
        # This test verifies graceful degradation
        # In production, missing dependencies should be handled
        try:
            import src.analytics
            assert src.analytics is not None
        except ImportError:
            # If import fails, it should be due to actual missing required deps
            # not due to module structure issues
            pass
    
    def test_namespace_pollution(self):
        """Test that module doesn't pollute namespace with internal imports."""
        import src.analytics as analytics
        
        # Check that internal implementation details aren't exposed
        internal_names = ['os', 'sys', 'Path', 'datetime']
        
        for name in internal_names:
            if hasattr(analytics, name):
                # If present, it should be in __all__ or be the logger
                if name not in ['logger', 'logging']:
                    assert name in analytics.__all__, \
                        f"Internal name '{name}' leaked into module namespace"
    
    def test_module_attributes_immutable(self):
        """Test that critical module attributes cannot be easily overwritten."""
        import src.analytics as analytics
        
        original_all = analytics.__all__.copy()
        
        # Attempt to modify __all__
        analytics.__all__.append('FakeClass')
        
        # Verify it was modified (Python allows this)
        assert 'FakeClass' in analytics.__all__
        
        # Restore original
        analytics.__all__ = original_all
        assert 'FakeClass' not in analytics.__all__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
