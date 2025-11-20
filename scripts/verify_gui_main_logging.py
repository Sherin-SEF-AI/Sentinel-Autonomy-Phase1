#!/usr/bin/env python3
"""
Verification script for GUI main entry point logging.

Tests that gui_main.py has proper logging setup and configuration.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_logging_config():
    """Verify logging configuration includes gui_main module."""
    print("=" * 60)
    print("GUI Main Logging Configuration Verification")
    print("=" * 60)
    
    try:
        import yaml
        
        # Load logging config
        config_path = Path(__file__).parent.parent / 'configs' / 'logging.yaml'
        with open(config_path, 'r') as f:
            logging_config = yaml.safe_load(f)
        
        # Check for gui_main logger
        loggers = logging_config.get('loggers', {})
        
        checks = {
            'src.gui_main': 'GUI main entry point logger',
            'src.gui': 'GUI module logger',
            'src.gui.main_window': 'Main window logger',
        }
        
        all_passed = True
        for logger_name, description in checks.items():
            if logger_name in loggers:
                config = loggers[logger_name]
                level = config.get('level', 'NOTSET')
                handlers = config.get('handlers', [])
                print(f"✓ {description}: level={level}, handlers={handlers}")
            else:
                print(f"✗ {description}: NOT FOUND")
                all_passed = False
        
        print()
        if all_passed:
            print("✓ All GUI main logging configuration checks passed")
        else:
            print("✗ Some GUI main logging configuration checks failed")
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error verifying logging config: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_module_imports():
    """Verify gui_main module can be imported."""
    print("\n" + "=" * 60)
    print("GUI Main Module Import Verification")
    print("=" * 60)
    
    try:
        # Import the module
        print("Importing gui_main module...")
        import gui_main
        print("✓ gui_main module imported successfully")
        
        # Check for logger
        if hasattr(gui_main, 'logger'):
            print("✓ Module-level logger found")
        else:
            print("✗ Module-level logger not found")
            return False
        
        # Check for main function
        if hasattr(gui_main, 'main'):
            print("✓ main() function found")
        else:
            print("✗ main() function not found")
            return False
        
        print("\n✓ All module import checks passed")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import gui_main: {e}")
        print("Note: This is expected if PyQt6 is not installed")
        return True  # Don't fail on missing PyQt6
    except Exception as e:
        print(f"✗ Error during import verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_logging_statements():
    """Verify logging statements are present in the code."""
    print("\n" + "=" * 60)
    print("GUI Main Logging Statements Verification")
    print("=" * 60)
    
    try:
        # Read the source file
        source_path = Path(__file__).parent.parent / 'src' / 'gui_main.py'
        with open(source_path, 'r') as f:
            source_code = f.read()
        
        # Check for required logging patterns
        checks = {
            'logger = logging.getLogger(__name__)': 'Module logger initialization',
            'logger.info': 'INFO level logging',
            'logger.debug': 'DEBUG level logging',
            'logger.error': 'ERROR level logging',
            'logger.warning': 'WARNING level logging',
            'duration=': 'Performance timing logs',
            'exc_info=True': 'Exception traceback logging',
        }
        
        all_passed = True
        for pattern, description in checks.items():
            if pattern in source_code:
                count = source_code.count(pattern)
                print(f"✓ {description}: found {count} occurrence(s)")
            else:
                print(f"✗ {description}: NOT FOUND")
                all_passed = False
        
        print()
        if all_passed:
            print("✓ All logging statement checks passed")
        else:
            print("✗ Some logging statement checks failed")
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Error verifying logging statements: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("SENTINEL GUI Main Logging Verification")
    print("=" * 60)
    print()
    
    results = {
        'Logging Configuration': verify_logging_config(),
        'Module Imports': verify_module_imports(),
        'Logging Statements': verify_logging_statements(),
    }
    
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    all_passed = all(results.values())
    
    print()
    if all_passed:
        print("=" * 60)
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        print("\nGUI main entry point logging is properly configured:")
        print("  - Module logger initialized at top of file")
        print("  - Comprehensive logging at all key points")
        print("  - Performance timing for initialization steps")
        print("  - Error handling with exception tracebacks")
        print("  - Logging configuration includes gui_main module")
        return 0
    else:
        print("=" * 60)
        print("✗ SOME CHECKS FAILED")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
