#!/usr/bin/env python3
"""
Simple verification script for AlertsPanel logging implementation.

This script checks that logging statements are present in the source code.
"""

import sys
from pathlib import Path


def verify_logging_statements():
    """Verify that logging statements are present in alerts_panel.py"""
    print("=" * 80)
    print("ALERTS PANEL LOGGING VERIFICATION (Source Code Analysis)")
    print("=" * 80)
    
    # Read the source file
    source_file = Path(__file__).parent.parent / 'src' / 'gui' / 'widgets' / 'alerts_panel.py'
    
    if not source_file.exists():
        print(f"\n✗ FAIL: Source file not found: {source_file}")
        return 1
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # Define required logging patterns
    required_patterns = {
        'Logger initialization': 'self.logger = logging.getLogger',
        'Initialization started': 'AlertsPanel initialization started',
        'Audio settings initialized': 'Audio settings initialized',
        'Flash timer initialized': 'Flash timer initialized',
        'Audio players initialized': 'Audio players initialized',
        'UI initialization completed': 'AlertsPanel UI initialization completed',
        'Initialization success': 'AlertsPanel initialized successfully',
        'Adding alert': 'Adding alert: urgency=',
        'Statistics updated': 'Statistics updated:',
        'Alert entry created': 'Alert entry created:',
        'Audio alert triggered': 'Audio alert triggered',
        'Critical effects triggered': 'Critical alert effects triggered',
        'Alert added successfully': 'Alert added successfully:',
        'Adding alert to display': 'Adding alert to display:',
        'Alert displayed successfully': 'Alert displayed successfully:',
        'Playing alert sound': 'Playing {urgency} alert sound',
        'Flash timer started': 'Flash timer started:',
        'Window brought to front': 'Window brought to front',
        'Audio alerts muted': 'Audio alerts muted',
        'Audio alerts unmuted': 'Audio alerts unmuted',
        'Volume changed': 'Volume changed to',
        'Alert filter changed': 'Alert filter changed:',
        'Refreshing display': 'Refreshing display with filter:',
        'Display refreshed': 'Display refreshed:',
        'Exporting alert log': 'Exporting alert log to file:',
        'Alert log exported': 'Alert log exported successfully:',
        'Export failed': 'Failed to export alert log',
        'Marking false positive': 'Marking alert as false positive:',
        'False positive marked': 'Alert marked as false positive:',
        'Clearing history': 'Clearing alert history:',
        'History cleared': 'Alert history cleared:',
        'Audio enabled/disabled': 'Audio alerts',
        'Audio volume changed': 'Audio volume changed:',
    }
    
    print("\nChecking for required logging statements...\n")
    
    passed = 0
    failed = 0
    
    for description, pattern in required_patterns.items():
        if pattern in content:
            print(f"  ✓ PASS: {description}")
            passed += 1
        else:
            print(f"  ✗ FAIL: {description} (pattern: '{pattern}')")
            failed += 1
    
    # Check for logger usage patterns
    print("\nChecking logger usage patterns...\n")
    
    logger_patterns = {
        'DEBUG level': 'self.logger.debug(',
        'INFO level': 'self.logger.info(',
        'WARNING level': 'self.logger.warning(',
        'ERROR level': 'self.logger.error(',
    }
    
    for description, pattern in logger_patterns.items():
        count = content.count(pattern)
        if count > 0:
            print(f"  ✓ PASS: {description} ({count} occurrences)")
            passed += 1
        else:
            print(f"  ✗ FAIL: {description} (not found)")
            failed += 1
    
    # Check for exception handling with logging
    print("\nChecking exception handling...\n")
    
    exception_patterns = {
        'Try-except blocks': 'except Exception as e:',
        'Error logging with exception': 'self.logger.error(',
    }
    
    for description, pattern in exception_patterns.items():
        count = content.count(pattern)
        if count > 0:
            print(f"  ✓ PASS: {description} ({count} occurrences)")
            passed += 1
        else:
            print(f"  ✗ FAIL: {description} (not found)")
            failed += 1
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    total = passed + failed
    print(f"\nTotal checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if failed == 0:
        print("\n✓ ALL LOGGING CHECKS PASSED")
        print("\nThe AlertsPanel has comprehensive logging coverage:")
        print("  - Initialization and setup")
        print("  - Alert addition and display")
        print("  - Audio playback and effects")
        print("  - User interactions")
        print("  - Display management")
        print("  - Export and false positive marking")
        print("  - Configuration changes")
        print("  - Error handling")
        return 0
    else:
        print(f"\n✗ {failed} LOGGING CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(verify_logging_statements())
