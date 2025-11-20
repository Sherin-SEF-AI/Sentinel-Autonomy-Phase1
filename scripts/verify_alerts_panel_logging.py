#!/usr/bin/env python3
"""
Verification script for AlertsPanel logging implementation.

This script tests that all logging statements are properly configured
and functioning in the AlertsPanel widget.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from io import StringIO
from typing import List
from dataclasses import dataclass

# Set environment to avoid display issues
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication

# Define Alert dataclass to match the actual structure
@dataclass
class Alert:
    """Alert dataclass matching core.data_structures.Alert"""
    timestamp: float
    urgency: str  # 'info', 'warning', 'critical'
    modalities: List[str]  # ['visual', 'audio', 'haptic']
    message: str
    hazard_id: int
    dismissed: bool


def setup_test_logging():
    """Set up logging to capture log messages"""
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(handler)
    
    # Configure alerts panel logger
    alerts_logger = logging.getLogger('sentinel.gui.alerts_panel')
    alerts_logger.setLevel(logging.DEBUG)
    alerts_logger.addHandler(handler)
    
    return log_stream


def create_test_alert(urgency: str, message: str, hazard_id: int):
    """Create a test alert"""
    return Alert(
        timestamp=datetime.now().timestamp(),
        urgency=urgency,
        modalities=['visual', 'audio'],
        message=message,
        hazard_id=hazard_id,
        dismissed=False
    )


def verify_logging():
    """Verify AlertsPanel logging implementation"""
    print("=" * 80)
    print("ALERTS PANEL LOGGING VERIFICATION")
    print("=" * 80)
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Set up test logging
    log_stream = setup_test_logging()
    
    # Import AlertsPanel directly (after logging is configured)
    # We need to manually load the module to avoid import issues
    import importlib.util
    alerts_panel_path = project_root / 'src' / 'gui' / 'widgets' / 'alerts_panel.py'
    
    spec = importlib.util.spec_from_file_location("alerts_panel", alerts_panel_path)
    alerts_panel_module = importlib.util.module_from_spec(spec)
    
    # Mock the core.data_structures import
    sys.modules['core.data_structures'] = type(sys)('core.data_structures')
    sys.modules['core.data_structures'].Alert = Alert
    
    spec.loader.exec_module(alerts_panel_module)
    AlertsPanel = alerts_panel_module.AlertsPanel
    
    print("\n1. Testing AlertsPanel initialization...")
    panel = AlertsPanel()
    
    # Check initialization logs
    log_output = log_stream.getvalue()
    
    checks = {
        'initialization_started': 'AlertsPanel initialization started' in log_output,
        'audio_settings': 'Audio settings initialized' in log_output,
        'flash_timer': 'Flash timer initialized' in log_output,
        'audio_players': 'Audio players initialized' in log_output,
        'ui_completed': 'AlertsPanel UI initialization completed' in log_output,
        'initialization_success': 'AlertsPanel initialized successfully' in log_output,
    }
    
    for check, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Clear log stream
    log_stream.truncate(0)
    log_stream.seek(0)
    
    print("\n2. Testing alert addition logging...")
    
    # Add critical alert
    critical_alert = create_test_alert('critical', 'Collision imminent!', 1)
    panel.add_alert(critical_alert)
    
    log_output = log_stream.getvalue()
    
    alert_checks = {
        'alert_adding': 'Adding alert: urgency=critical' in log_output,
        'statistics_updated': 'Statistics updated' in log_output,
        'alert_entry_created': 'Alert entry created' in log_output,
        'audio_triggered': 'Audio alert triggered' in log_output,
        'critical_effects': 'Critical alert effects triggered' in log_output,
        'alert_added_success': 'Alert added successfully' in log_output,
        'display_added': 'Adding alert to display' in log_output,
        'display_success': 'Alert displayed successfully' in log_output,
    }
    
    for check, passed in alert_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Clear log stream
    log_stream.truncate(0)
    log_stream.seek(0)
    
    print("\n3. Testing warning alert logging...")
    
    # Add warning alert
    warning_alert = create_test_alert('warning', 'Lane departure detected', 2)
    panel.add_alert(warning_alert)
    
    log_output = log_stream.getvalue()
    
    warning_checks = {
        'warning_urgency': 'urgency=warning' in log_output,
        'warning_added': 'Alert added successfully' in log_output,
    }
    
    for check, passed in warning_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Clear log stream
    log_stream.truncate(0)
    log_stream.seek(0)
    
    print("\n4. Testing audio control logging...")
    
    # Test mute
    panel.set_audio_enabled(False)
    log_output = log_stream.getvalue()
    
    audio_checks = {
        'audio_disabled': 'Audio alerts disabled' in log_output,
    }
    
    # Clear and test unmute
    log_stream.truncate(0)
    log_stream.seek(0)
    
    panel.set_audio_enabled(True)
    log_output = log_stream.getvalue()
    audio_checks['audio_enabled'] = 'Audio alerts enabled' in log_output
    
    # Clear and test volume
    log_stream.truncate(0)
    log_stream.seek(0)
    
    panel.set_volume(0.5)
    log_output = log_stream.getvalue()
    audio_checks['volume_changed'] = 'Audio volume changed' in log_output
    
    for check, passed in audio_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Clear log stream
    log_stream.truncate(0)
    log_stream.seek(0)
    
    print("\n5. Testing filter and display refresh logging...")
    
    # Change filter
    panel.filter_combo.setCurrentText("Critical")
    
    log_output = log_stream.getvalue()
    
    filter_checks = {
        'filter_changed': 'Alert filter changed' in log_output,
        'display_refreshed': 'Refreshing display with filter' in log_output,
    }
    
    for check, passed in filter_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Clear log stream
    log_stream.truncate(0)
    log_stream.seek(0)
    
    print("\n6. Testing false positive marking logging...")
    
    # Mark alert as false positive
    panel.mark_false_positive(0)
    
    log_output = log_stream.getvalue()
    
    fp_checks = {
        'fp_marking': 'Marking alert as false positive' in log_output,
        'fp_marked': 'Alert marked as false positive' in log_output,
    }
    
    for check, passed in fp_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    # Clear log stream
    log_stream.truncate(0)
    log_stream.seek(0)
    
    print("\n7. Testing clear history logging...")
    
    # Clear history
    panel.clear_history()
    
    log_output = log_stream.getvalue()
    
    clear_checks = {
        'clearing_history': 'Clearing alert history' in log_output,
        'history_cleared': 'Alert history cleared' in log_output,
    }
    
    for check, passed in clear_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {check}")
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Count total checks
    all_checks = {**checks, **alert_checks, **warning_checks, **audio_checks, 
                  **filter_checks, **fp_checks, **clear_checks}
    
    passed_count = sum(1 for v in all_checks.values() if v)
    total_count = len(all_checks)
    
    print(f"\nTotal checks: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Success rate: {passed_count/total_count*100:.1f}%")
    
    if passed_count == total_count:
        print("\n✓ ALL LOGGING CHECKS PASSED")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} LOGGING CHECKS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(verify_logging())
