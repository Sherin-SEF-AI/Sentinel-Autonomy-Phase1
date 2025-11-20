#!/usr/bin/env python3
"""
Verification script for Vehicle Telemetry Dock logging.

Tests that all logging statements are properly configured and working.
"""

import sys
import logging
from pathlib import Path
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from src.core.data_structures import VehicleTelemetry
from src.gui.widgets.vehicle_telemetry_dock import VehicleTelemetryDock


def setup_test_logging():
    """Set up logging to capture output."""
    # Create string buffer to capture logs
    log_buffer = StringIO()
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.StreamHandler(log_buffer)
        ]
    )
    
    return log_buffer


def verify_initialization_logging(log_output: str) -> bool:
    """Verify initialization logging."""
    print("\n" + "="*60)
    print("VERIFICATION: Initialization Logging")
    print("="*60)
    
    checks = {
        "Vehicle Telemetry Dock initialized": False,
        "Setting up Vehicle Telemetry Dock UI": False,
        "SteeringIndicator initialized": False,
        "BarIndicator initialized": False,
        "GearIndicator initialized": False,
        "TurnSignalIndicator initialized": False,
        "Vehicle Telemetry Dock UI setup completed": False
    }
    
    for message in checks.keys():
        if message in log_output:
            checks[message] = True
            print(f"✓ Found: {message}")
        else:
            print(f"✗ Missing: {message}")
    
    all_passed = all(checks.values())
    print(f"\nInitialization Logging: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def verify_telemetry_update_logging(log_output: str) -> bool:
    """Verify telemetry update logging."""
    print("\n" + "="*60)
    print("VERIFICATION: Telemetry Update Logging")
    print("="*60)
    
    checks = {
        "Telemetry update received": False,
        "speed=": False,
        "steering=": False,
        "brake=": False,
        "throttle=": False,
        "gear=": False,
        "signal=": False,
        "Telemetry display updated successfully": False
    }
    
    for message in checks.keys():
        if message in log_output:
            checks[message] = True
            print(f"✓ Found: {message}")
        else:
            print(f"✗ Missing: {message}")
    
    all_passed = all(checks.values())
    print(f"\nTelemetry Update Logging: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def verify_state_change_logging(log_output: str) -> bool:
    """Verify state change logging."""
    print("\n" + "="*60)
    print("VERIFICATION: State Change Logging")
    print("="*60)
    
    checks = {
        "Gear changed": False,
        "Turn signal changed": False,
        "Steering angle updated": False
    }
    
    for message in checks.keys():
        if message in log_output:
            checks[message] = True
            print(f"✓ Found: {message}")
        else:
            print(f"✗ Missing: {message}")
    
    all_passed = all(checks.values())
    print(f"\nState Change Logging: {'PASS' if all_passed else 'FAIL'}")
    return all_passed


def main():
    """Run verification tests."""
    print("="*60)
    print("Vehicle Telemetry Dock Logging Verification")
    print("="*60)
    
    # Set up logging
    log_buffer = setup_test_logging()
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    try:
        # Test 1: Initialization
        print("\nTest 1: Creating Vehicle Telemetry Dock...")
        dock = VehicleTelemetryDock()
        dock.show()
        
        # Process events
        app.processEvents()
        
        # Test 2: Update telemetry
        print("\nTest 2: Updating telemetry...")
        telemetry = VehicleTelemetry(
            timestamp=0.0,
            speed=15.5,
            steering_angle=0.2,
            brake_pressure=2.5,
            throttle_position=0.6,
            gear=3,
            turn_signal='left'
        )
        dock.update_telemetry(telemetry)
        app.processEvents()
        
        # Test 3: Change gear
        print("\nTest 3: Changing gear...")
        telemetry.gear = 4
        dock.update_telemetry(telemetry)
        app.processEvents()
        
        # Test 4: Change turn signal
        print("\nTest 4: Changing turn signal...")
        telemetry.turn_signal = 'right'
        dock.update_telemetry(telemetry)
        app.processEvents()
        
        # Test 5: Update steering
        print("\nTest 5: Updating steering angle...")
        telemetry.steering_angle = -0.3
        dock.update_telemetry(telemetry)
        app.processEvents()
        
        # Get log output
        log_output = log_buffer.getvalue()
        
        # Verify logging
        init_pass = verify_initialization_logging(log_output)
        update_pass = verify_telemetry_update_logging(log_output)
        state_pass = verify_state_change_logging(log_output)
        
        # Summary
        print("\n" + "="*60)
        print("VERIFICATION SUMMARY")
        print("="*60)
        print(f"Initialization Logging: {'PASS' if init_pass else 'FAIL'}")
        print(f"Telemetry Update Logging: {'PASS' if update_pass else 'FAIL'}")
        print(f"State Change Logging: {'PASS' if state_pass else 'FAIL'}")
        
        all_passed = init_pass and update_pass and state_pass
        
        if all_passed:
            print("\n✓ ALL VERIFICATIONS PASSED")
            return 0
        else:
            print("\n✗ SOME VERIFICATIONS FAILED")
            print("\nFull log output:")
            print("-"*60)
            print(log_output)
            return 1
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        app.quit()


if __name__ == '__main__':
    sys.exit(main())
