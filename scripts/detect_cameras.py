#!/usr/bin/env python3
"""
Camera Detection Utility

Detects available cameras and provides configuration recommendations.
"""

import cv2
import sys
from pathlib import Path

def detect_cameras(max_cameras=10):
    """Detect available camera devices."""
    available = []

    print("Scanning for available cameras...")
    print("=" * 60)

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it works
            ret, frame = cap.read()
            if ret and frame is not None:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                available.append({
                    'id': i,
                    'resolution': (width, height),
                    'fps': fps
                })

                print(f"✓ Camera {i} FOUND:")
                print(f"  - Resolution: {width}x{height}")
                print(f"  - FPS: {fps}")
                print()
            cap.release()

    return available

def generate_config(cameras):
    """Generate configuration for detected cameras."""
    if not cameras:
        print("ERROR: No cameras detected!")
        print("\nTroubleshooting:")
        print("1. Check if cameras are connected")
        print("2. Check permissions: sudo chmod 666 /dev/video*")
        print("3. Close other applications using cameras")
        return None

    print("=" * 60)
    print(f"Found {len(cameras)} camera(s)")
    print("=" * 60)
    print()
    print("Recommended Configuration:")
    print()
    print("cameras:")

    # Map cameras to roles
    camera_roles = ['interior', 'front_left', 'front_right']

    for idx, cam in enumerate(cameras):
        if idx < len(camera_roles):
            role = camera_roles[idx]
            print(f"  {role}:")
            print(f"    device: {cam['id']}")
            print(f"    resolution: [{cam['resolution'][0]}, {cam['resolution'][1]}]")
            print(f"    fps: {cam['fps']}")
            print(f"    calibration: \"configs/calibration/{role}.yaml\"")
            print()

    # If fewer than 3 cameras, suggest configuration
    if len(cameras) < 3:
        print(f"\nNote: System designed for 3 cameras, but {len(cameras)} detected.")
        print("The system will run with available cameras.")
        print("Missing cameras will use placeholder frames.")

    return cameras

def main():
    """Main detection routine."""
    print()
    print("SENTINEL Camera Detection Utility")
    print("=" * 60)
    print()

    cameras = detect_cameras()
    config = generate_config(cameras)

    if config:
        print()
        print("=" * 60)
        print("Copy the configuration above to configs/default.yaml")
        print("=" * 60)
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
