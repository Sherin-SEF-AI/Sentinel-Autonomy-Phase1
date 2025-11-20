#!/usr/bin/env python3
"""
Simple verification script for Vehicle Telemetry Dock logging.

Tests logging configuration without full imports.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_logging_config():
    """Verify logging configuration."""
    print("="*60)
    print("Vehicle Telemetry Dock Logging Verification")
    print("="*60)
    
    # Load logging config
    import yaml
    config_path = Path(__file__).parent.parent / 'configs' / 'logging.yaml'
    
    print(f"\nLoading logging config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for vehicle telemetry dock logger
    loggers = config.get('loggers', {})
    
    print("\n" + "="*60)
    print("VERIFICATION: Logger Configuration")
    print("="*60)
    
    checks = {
        'src.gui.widgets.vehicle_telemetry_dock': False
    }
    
    for logger_name in checks.keys():
        if logger_name in loggers:
            checks[logger_name] = True
            logger_config = loggers[logger_name]
            print(f"✓ Found logger: {logger_name}")
            print(f"  Level: {logger_config.get('level', 'N/A')}")
            print(f"  Handlers: {logger_config.get('handlers', [])}")
            print(f"  Propagate: {logger_config.get('propagate', True)}")
        else:
            print(f"✗ Missing logger: {logger_name}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"Logger Configuration: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


def verify_source_code():
    """Verify logging statements in source code."""
    print("\n" + "="*60)
    print("VERIFICATION: Source Code Logging Statements")
    print("="*60)
    
    source_file = Path(__file__).parent.parent / 'src' / 'gui' / 'widgets' / 'vehicle_telemetry_dock.py'
    
    print(f"\nAnalyzing: {source_file}")
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    checks = {
        'import logging': False,
        'logger = logging.getLogger(__name__)': False,
        'Vehicle Telemetry Dock initialized': False,
        'Setting up Vehicle Telemetry Dock UI': False,
        'Telemetry update received': False,
        'Telemetry display updated successfully': False,
        'Gear changed': False,
        'Turn signal changed': False,
        'Steering angle updated': False,
        'Telemetry display update failed': False
    }
    
    for statement in checks.keys():
        if statement in content:
            checks[statement] = True
            print(f"✓ Found: {statement}")
        else:
            print(f"✗ Missing: {statement}")
    
    all_passed = all(checks.values())
    
    print(f"\nSource Code Logging: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


def main():
    """Run verification."""
    try:
        config_pass = verify_logging_config()
        source_pass = verify_source_code()
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Logger Configuration: {'PASS' if config_pass else 'FAIL'}")
        print(f"Source Code Logging: {'PASS' if source_pass else 'FAIL'}")
        
        if config_pass and source_pass:
            print("\n✓ ALL VERIFICATIONS PASSED")
            print("\nThe Vehicle Telemetry Dock has comprehensive logging:")
            print("  - Initialization logging (DEBUG)")
            print("  - Telemetry update logging (DEBUG)")
            print("  - State change logging (DEBUG)")
            print("  - Error handling (ERROR)")
            print("\nLogging is configured in configs/logging.yaml")
            print("Log output goes to logs/sentinel.log")
            return 0
        else:
            print("\n✗ SOME VERIFICATIONS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
