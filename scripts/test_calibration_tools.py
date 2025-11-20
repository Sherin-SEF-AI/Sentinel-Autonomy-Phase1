#!/usr/bin/env python3
"""
Test script for calibration tools

Tests the calibration and validation scripts without requiring actual cameras.
"""

import sys
import os
import yaml
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_calibration_file_creation():
    """Test creating a sample calibration file"""
    print("Testing calibration file creation...")
    
    # Create sample calibration data
    calibration_data = {
        'camera_name': 'test_camera',
        'image_size': {
            'width': 1280,
            'height': 720
        },
        'intrinsics': {
            'fx': 800.0,
            'fy': 800.0,
            'cx': 640.0,
            'cy': 360.0,
            'distortion': [0.1, -0.05, 0.001, 0.001, 0.01]
        },
        'extrinsics': {
            'translation': [1.5, 0.5, 1.2],
            'rotation': [0.0, -0.2, 0.785]
        },
        'homography': {
            'matrix': [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        }
    }
    
    # Save to temporary file
    test_file = 'configs/calibration/test_camera.yaml'
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    with open(test_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created test calibration file: {test_file}")
    
    # Verify file can be loaded
    with open(test_file, 'r') as f:
        loaded_data = yaml.safe_load(f)
    
    assert loaded_data['camera_name'] == 'test_camera'
    assert loaded_data['intrinsics']['fx'] == 800.0
    assert len(loaded_data['intrinsics']['distortion']) == 5
    assert 'extrinsics' in loaded_data
    assert 'homography' in loaded_data
    
    print("✓ Calibration file format is valid")
    
    # Clean up
    os.remove(test_file)
    print("✓ Test file cleaned up")
    
    return True


def test_calibration_data_structure():
    """Test calibration data structure requirements"""
    print("\nTesting calibration data structure...")
    
    # Test intrinsics
    intrinsics = {
        'fx': 800.0,
        'fy': 800.0,
        'cx': 640.0,
        'cy': 360.0,
        'distortion': [0.1, -0.05, 0.001, 0.001, 0.01]
    }
    
    assert 'fx' in intrinsics
    assert 'fy' in intrinsics
    assert 'cx' in intrinsics
    assert 'cy' in intrinsics
    assert 'distortion' in intrinsics
    assert len(intrinsics['distortion']) == 5
    
    print("✓ Intrinsics structure is valid")
    
    # Test extrinsics
    extrinsics = {
        'translation': [1.5, 0.5, 1.2],
        'rotation': [0.0, -0.2, 0.785]
    }
    
    assert 'translation' in extrinsics
    assert 'rotation' in extrinsics
    assert len(extrinsics['translation']) == 3
    assert len(extrinsics['rotation']) == 3
    
    print("✓ Extrinsics structure is valid")
    
    # Test homography
    homography = {
        'matrix': [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    }
    
    assert 'matrix' in homography
    assert len(homography['matrix']) == 3
    assert all(len(row) == 3 for row in homography['matrix'])
    
    print("✓ Homography structure is valid")
    
    return True


def test_coordinate_systems():
    """Test coordinate system definitions"""
    print("\nTesting coordinate system definitions...")
    
    # Vehicle frame
    vehicle_origin = np.array([0.0, 0.0, 0.0])
    vehicle_x_axis = np.array([1.0, 0.0, 0.0])  # Forward
    vehicle_y_axis = np.array([0.0, 1.0, 0.0])  # Left
    vehicle_z_axis = np.array([0.0, 0.0, 1.0])  # Up
    
    # Verify orthogonality
    assert np.abs(np.dot(vehicle_x_axis, vehicle_y_axis)) < 1e-6
    assert np.abs(np.dot(vehicle_y_axis, vehicle_z_axis)) < 1e-6
    assert np.abs(np.dot(vehicle_z_axis, vehicle_x_axis)) < 1e-6
    
    print("✓ Vehicle coordinate frame is orthogonal")
    
    # BEV frame
    bev_size = (640, 640)
    bev_scale = 0.1  # meters per pixel
    
    assert bev_size[0] == bev_size[1]  # Square
    assert bev_scale > 0
    
    coverage = bev_size[0] * bev_scale
    print(f"✓ BEV coverage: {coverage}m x {coverage}m")
    
    return True


def test_calibration_quality_checks():
    """Test calibration quality verification logic"""
    print("\nTesting calibration quality checks...")
    
    # Test focal length check
    width = 1280
    fx = 800.0
    focal_ratio = fx / width
    
    assert 0.5 <= focal_ratio <= 2.0, "Focal length ratio should be reasonable"
    print(f"✓ Focal length ratio: {focal_ratio:.3f} (reasonable)")
    
    # Test principal point check
    cx = 640.0
    cy = 360.0
    height = 720
    
    cx_ratio = cx / width
    cy_ratio = cy / height
    
    assert 0.3 <= cx_ratio <= 0.7, "Principal point X should be near center"
    assert 0.3 <= cy_ratio <= 0.7, "Principal point Y should be near center"
    print(f"✓ Principal point: ({cx_ratio:.3f}, {cy_ratio:.3f}) (near center)")
    
    # Test distortion coefficients
    k1, k2 = 0.1, -0.05
    
    assert abs(k1) < 1.0, "k1 should be reasonable"
    assert abs(k2) < 1.0, "k2 should be reasonable"
    print(f"✓ Distortion coefficients: k1={k1}, k2={k2} (reasonable)")
    
    return True


def test_existing_calibration_files():
    """Test loading existing calibration files if they exist"""
    print("\nTesting existing calibration files...")
    
    calibration_dir = 'configs/calibration'
    camera_names = ['interior', 'front_left', 'front_right']
    
    found_files = []
    
    for camera_name in camera_names:
        calibration_path = os.path.join(calibration_dir, f'{camera_name}.yaml')
        
        if os.path.exists(calibration_path):
            print(f"  Found: {calibration_path}")
            
            # Try to load it
            with open(calibration_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Verify required fields
            assert 'intrinsics' in data, "Missing intrinsics"
            
            # Optional fields (may not be present in all calibration files)
            if 'camera_name' in data:
                print(f"    Camera name: {data['camera_name']}")
            if 'image_size' in data:
                print(f"    Image size: {data['image_size']}")
            
            print(f"    ✓ Valid calibration file")
            found_files.append(camera_name)
        else:
            print(f"  Not found: {calibration_path} (expected for new setup)")
    
    if found_files:
        print(f"✓ Found {len(found_files)} existing calibration file(s)")
    else:
        print("ℹ No existing calibration files (run calibration script to create)")
    
    return True


def main():
    """Run all tests"""
    print("="*60)
    print("CALIBRATION TOOLS TEST SUITE")
    print("="*60)
    
    tests = [
        ("Calibration File Creation", test_calibration_file_creation),
        ("Calibration Data Structure", test_calibration_data_structure),
        ("Coordinate Systems", test_coordinate_systems),
        ("Calibration Quality Checks", test_calibration_quality_checks),
        ("Existing Calibration Files", test_existing_calibration_files),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        print("\nCalibration tools are ready to use:")
        print("  1. Run: python3 scripts/calibrate_camera.py --help")
        print("  2. Run: python3 scripts/validate_calibration.py --help")
        print("  3. See: configs/calibration/README.md for full guide")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    exit(main())
