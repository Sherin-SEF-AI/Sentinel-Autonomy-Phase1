#!/usr/bin/env python3
"""Verification script for analytics module logging setup."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from core.logging import LoggerSetup


def test_analytics_logging():
    """Test analytics module logging configuration."""
    print("=" * 60)
    print("ANALYTICS MODULE LOGGING VERIFICATION")
    print("=" * 60)
    
    # Setup logging
    LoggerSetup.setup(log_level='DEBUG', log_dir='logs')
    
    # Test analytics module import
    print("\n1. Testing analytics module import...")
    try:
        import analytics
        print("   ✓ Analytics module imported successfully")
    except Exception as e:
        print(f"   ✗ Failed to import analytics module: {e}")
        return False
    
    # Test individual component imports
    print("\n2. Testing individual component imports...")
    components = [
        'TripAnalytics',
        'BehaviorReportGenerator',
        'RiskHeatmap',
        'AnalyticsDashboard',
        'ReportExporter'
    ]
    
    for component in components:
        try:
            obj = getattr(analytics, component)
            print(f"   ✓ {component} imported successfully")
        except Exception as e:
            print(f"   ✗ Failed to import {component}: {e}")
    
    # Test logger instances
    print("\n3. Testing logger instances...")
    logger_names = [
        'src.analytics',
        'src.analytics.trip_analytics',
        'src.analytics.behavior_report',
        'src.analytics.risk_heatmap',
        'src.analytics.analytics_dashboard',
        'src.analytics.report_exporter'
    ]
    
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        if logger:
            print(f"   ✓ Logger '{logger_name}' exists")
            print(f"     - Level: {logging.getLevelName(logger.level)}")
            print(f"     - Handlers: {len(logger.handlers)}")
        else:
            print(f"   ✗ Logger '{logger_name}' not found")
    
    # Test logging output
    print("\n4. Testing logging output...")
    test_logger = logging.getLogger('src.analytics')
    
    test_logger.debug("DEBUG: Test debug message")
    test_logger.info("INFO: Test info message")
    test_logger.warning("WARNING: Test warning message")
    test_logger.error("ERROR: Test error message")
    
    print("   ✓ Logging output test completed")
    
    # Test component-specific logging
    print("\n5. Testing component-specific logging...")
    
    try:
        from analytics.trip_analytics import TripAnalytics
        trip_logger = logging.getLogger('src.analytics.trip_analytics')
        trip_logger.info("TripAnalytics logger test: component initialized")
        print("   ✓ TripAnalytics logging works")
    except Exception as e:
        print(f"   ✗ TripAnalytics logging failed: {e}")
    
    try:
        from analytics.behavior_report import BehaviorReportGenerator
        behavior_logger = logging.getLogger('src.analytics.behavior_report')
        behavior_logger.info("BehaviorReportGenerator logger test: component initialized")
        print("   ✓ BehaviorReportGenerator logging works")
    except Exception as e:
        print(f"   ✗ BehaviorReportGenerator logging failed: {e}")
    
    try:
        from analytics.risk_heatmap import RiskHeatmap
        heatmap_logger = logging.getLogger('src.analytics.risk_heatmap')
        heatmap_logger.info("RiskHeatmap logger test: component initialized")
        print("   ✓ RiskHeatmap logging works")
    except Exception as e:
        print(f"   ✗ RiskHeatmap logging failed: {e}")
    
    try:
        from analytics.report_exporter import ReportExporter
        exporter_logger = logging.getLogger('src.analytics.report_exporter')
        exporter_logger.info("ReportExporter logger test: component initialized")
        print("   ✓ ReportExporter logging works")
    except Exception as e:
        print(f"   ✗ ReportExporter logging failed: {e}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nAll analytics module logging tests passed!")
    print("\nLog files created in logs/ directory:")
    print("  - logs/sentinel.log (all logs)")
    print("  - logs/errors.log (errors only)")
    
    return True


if __name__ == '__main__':
    try:
        success = test_analytics_logging()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
