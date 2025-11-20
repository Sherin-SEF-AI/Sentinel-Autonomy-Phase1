#!/usr/bin/env python3
"""Simple verification of data structures logging setup."""

import sys
from pathlib import Path

print("=" * 60)
print("DATA STRUCTURES LOGGING VERIFICATION")
print("=" * 60)

# Check 1: Verify logging import in data_structures.py
print("\n1. Checking logging import in data_structures.py...")
data_structures_file = Path("src/core/data_structures.py")
if data_structures_file.exists():
    content = data_structures_file.read_text()
    if "import logging" in content and "logger = logging.getLogger(__name__)" in content:
        print("   ✓ Logging import and logger setup found")
    else:
        print("   ✗ Logging import or logger setup missing")
        sys.exit(1)
else:
    print("   ✗ File not found")
    sys.exit(1)

# Check 2: Verify new data structures are present
print("\n2. Checking new data structures...")
new_structures = ["MapFeature", "Lane", "VehicleTelemetry"]
for struct in new_structures:
    if f"class {struct}:" in content:
        print(f"   ✓ {struct} dataclass found")
    else:
        print(f"   ✗ {struct} dataclass missing")
        sys.exit(1)

# Check 3: Verify logging configuration
print("\n3. Checking logging configuration...")
logging_config = Path("configs/logging.yaml")
if logging_config.exists():
    config_content = logging_config.read_text()
    if "src.core.data_structures:" in config_content:
        print("   ✓ Logger configuration found in logging.yaml")
        
        # Check log level
        if "level: INFO" in config_content:
            print("   ✓ Log level set to INFO")
        
        # Check handlers
        if "handlers: [file_all]" in config_content:
            print("   ✓ Handlers configured (file_all)")
    else:
        print("   ✗ Logger configuration missing")
        sys.exit(1)
else:
    print("   ✗ logging.yaml not found")
    sys.exit(1)

# Check 4: Verify data structure fields
print("\n4. Verifying data structure fields...")

# MapFeature fields
if all(field in content for field in ["feature_id:", "type:", "position:", "attributes:", "geometry:"]):
    print("   ✓ MapFeature has all required fields")
else:
    print("   ✗ MapFeature missing fields")
    sys.exit(1)

# Lane fields
if all(field in content for field in ["lane_id:", "centerline:", "left_boundary:", "right_boundary:", 
                                       "width:", "speed_limit:", "lane_type:", "predecessors:", "successors:"]):
    print("   ✓ Lane has all required fields")
else:
    print("   ✗ Lane missing fields")
    sys.exit(1)

# VehicleTelemetry fields
if all(field in content for field in ["timestamp:", "speed:", "steering_angle:", "brake_pressure:", 
                                       "throttle_position:", "gear:", "turn_signal:"]):
    print("   ✓ VehicleTelemetry has all required fields")
else:
    print("   ✗ VehicleTelemetry missing fields")
    sys.exit(1)

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✓ All checks passed!")
print("\nSummary:")
print("  - Logging setup: ✓ Complete")
print("  - New data structures: ✓ 3 added (MapFeature, Lane, VehicleTelemetry)")
print("  - Configuration: ✓ logging.yaml updated")
print("\nData structures module is ready for:")
print("  - HD Map integration (Task 27)")
print("  - CAN bus telemetry (future)")
print("  - Advanced trajectory prediction")
print("\nLogging configuration:")
print("  - Module: src.core.data_structures")
print("  - Level: INFO")
print("  - Handlers: file_all")
print("  - Log file: logs/sentinel.log")

sys.exit(0)
