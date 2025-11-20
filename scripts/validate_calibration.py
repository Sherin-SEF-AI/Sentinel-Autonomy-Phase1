#!/usr/bin/env python3
"""
Calibration Validation Script for SENTINEL System

This script validates camera calibration by:
1. Visualizing undistorted images
2. Visualizing BEV transformation
3. Verifying calibration quality metrics

Usage:
    python scripts/validate_calibration.py --camera interior
    python scripts/validate_calibration.py --camera front_left --show-bev
    python scripts/validate_calibration.py --all
"""

import argparse
import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from typing import Dict, Optional, Tuple


class CalibrationValidator:
    """Validates camera calibration"""
    
    def __init__(self, calibration_path: str, device_id: int):
        self.calibration_path = calibration_path
        self.device_id = device_id
        self.calibration_data = None
        
        # Load calibration
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """Load calibration data from YAML file"""
        if not os.path.exists(self.calibration_path):
            print(f"Error: Calibration file not found: {self.calibration_path}")
            return False
        
        with open(self.calibration_path, 'r') as f:
            self.calibration_data = yaml.safe_load(f)
        
        print(f"\n=== Loaded calibration: {self.calibration_data['camera_name']} ===")
        print(f"Image size: {self.calibration_data['image_size']['width']}x{self.calibration_data['image_size']['height']}")
        print(f"Focal length: fx={self.calibration_data['intrinsics']['fx']:.2f}, fy={self.calibration_data['intrinsics']['fy']:.2f}")
        print(f"Principal point: cx={self.calibration_data['intrinsics']['cx']:.2f}, cy={self.calibration_data['intrinsics']['cy']:.2f}")
        
        if 'extrinsics' in self.calibration_data:
            print(f"Position: {self.calibration_data['extrinsics']['translation']}")
            print(f"Orientation: {self.calibration_data['extrinsics']['rotation']}")
        
        if 'homography' in self.calibration_data:
            print("BEV homography: Available")
        
        return True
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix"""
        intrinsics = self.calibration_data['intrinsics']
        return np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def get_distortion_coeffs(self) -> np.ndarray:
        """Get distortion coefficients"""
        return np.array(self.calibration_data['intrinsics']['distortion'], dtype=np.float32)
    
    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get homography matrix for BEV transformation"""
        if 'homography' not in self.calibration_data:
            return None
        
        H = self.calibration_data['homography']['matrix']
        return np.array(H, dtype=np.float32)
    
    def validate_undistortion(self):
        """Visualize undistorted images"""
        print(f"\n=== Validating Undistortion ===")
        print("Opening camera...")
        
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.device_id}")
            return False
        
        # Set resolution
        width = self.calibration_data['image_size']['width']
        height = self.calibration_data['image_size']['height']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        camera_matrix = self.get_camera_matrix()
        dist_coeffs = self.get_distortion_coeffs()
        
        # Compute optimal camera matrix
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (width, height),
            1,
            (width, height)
        )
        
        print("\nInstructions:")
        print("- Compare original (left) vs undistorted (right) images")
        print("- Look for straight lines at image edges")
        print("- Press 'q' to quit")
        print("- Press 's' to save comparison image\n")
        
        save_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Undistort image
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Crop to ROI
            x, y, w, h = roi
            if w > 0 and h > 0:
                undistorted = undistorted[y:y+h, x:x+w]
                undistorted = cv2.resize(undistorted, (frame.shape[1], frame.shape[0]))
            
            # Create side-by-side comparison
            comparison = np.hstack([frame, undistorted])
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Undistorted", (width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw grid for reference
            for i in range(0, width, width // 8):
                cv2.line(comparison, (i, 0), (i, height), (255, 255, 0), 1)
                cv2.line(comparison, (width + i, 0), (width + i, height), (255, 255, 0), 1)
            for i in range(0, height, height // 6):
                cv2.line(comparison, (0, i), (width * 2, i), (255, 255, 0), 1)
            
            cv2.imshow('Undistortion Validation', comparison)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"undistortion_validation_{save_count}.jpg"
                cv2.imwrite(filename, comparison)
                print(f"Saved: {filename}")
                save_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nUndistortion validation complete")
        return True
    
    def validate_bev_transformation(self):
        """Visualize BEV transformation"""
        print(f"\n=== Validating BEV Transformation ===")
        
        homography = self.get_homography_matrix()
        if homography is None:
            print("No homography matrix available for this camera")
            return False
        
        print("Opening camera...")
        
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.device_id}")
            return False
        
        # Set resolution
        width = self.calibration_data['image_size']['width']
        height = self.calibration_data['image_size']['height']
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        camera_matrix = self.get_camera_matrix()
        dist_coeffs = self.get_distortion_coeffs()
        
        # BEV output size
        bev_size = (640, 640)
        
        print("\nInstructions:")
        print("- View BEV transformation in real-time")
        print("- Check if ground plane is properly transformed")
        print("- Verify scale and orientation")
        print("- Press 'q' to quit")
        print("- Press 's' to save BEV image\n")
        
        save_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Undistort first
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            # Apply homography to get BEV
            bev = cv2.warpPerspective(undistorted, homography, bev_size)
            
            # Create visualization
            # Resize original for display
            display_original = cv2.resize(undistorted, (640, 480))
            display_bev = bev.copy()
            
            # Draw grid on BEV
            grid_spacing = 64  # 10 pixels = 1 meter at 0.1m/pixel scale
            for i in range(0, bev_size[0], grid_spacing):
                cv2.line(display_bev, (i, 0), (i, bev_size[1]), (0, 255, 0), 1)
            for i in range(0, bev_size[1], grid_spacing):
                cv2.line(display_bev, (0, i), (bev_size[0], i), (0, 255, 0), 1)
            
            # Draw vehicle position (center bottom)
            vehicle_x, vehicle_y = bev_size[0] // 2, int(bev_size[1] * 0.75)
            cv2.circle(display_bev, (vehicle_x, vehicle_y), 10, (0, 0, 255), -1)
            cv2.putText(display_bev, "Vehicle", (vehicle_x - 30, vehicle_y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Add labels
            cv2.putText(display_original, "Camera View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_bev, "Bird's Eye View", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_bev, "Grid: 6.4m spacing", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Stack vertically
            # Pad original to match BEV width
            pad_height = bev_size[1] - display_original.shape[0]
            if pad_height > 0:
                display_original = cv2.copyMakeBorder(
                    display_original, 0, pad_height, 0, 0,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            
            comparison = np.vstack([display_original, display_bev])
            
            cv2.imshow('BEV Transformation Validation', comparison)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"bev_validation_{save_count}.jpg"
                cv2.imwrite(filename, comparison)
                print(f"Saved: {filename}")
                save_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\nBEV transformation validation complete")
        return True
    
    def verify_calibration_quality(self) -> Dict:
        """Verify calibration quality metrics"""
        print(f"\n=== Verifying Calibration Quality ===")
        
        metrics = {
            'camera_name': self.calibration_data['camera_name'],
            'has_intrinsics': True,
            'has_extrinsics': 'extrinsics' in self.calibration_data,
            'has_homography': 'homography' in self.calibration_data,
            'quality': 'UNKNOWN'
        }
        
        # Check focal length reasonableness
        fx = self.calibration_data['intrinsics']['fx']
        fy = self.calibration_data['intrinsics']['fy']
        width = self.calibration_data['image_size']['width']
        height = self.calibration_data['image_size']['height']
        
        # Typical focal length should be 0.5-2.0 times image width
        focal_ratio = fx / width
        
        print(f"Focal length ratio (fx/width): {focal_ratio:.3f}")
        
        if 0.5 <= focal_ratio <= 2.0:
            print("✓ Focal length is reasonable")
            metrics['focal_length_ok'] = True
        else:
            print("✗ Focal length may be incorrect")
            metrics['focal_length_ok'] = False
        
        # Check principal point
        cx = self.calibration_data['intrinsics']['cx']
        cy = self.calibration_data['intrinsics']['cy']
        
        cx_ratio = cx / width
        cy_ratio = cy / height
        
        print(f"Principal point: ({cx_ratio:.3f}, {cy_ratio:.3f}) relative to image center")
        
        if 0.3 <= cx_ratio <= 0.7 and 0.3 <= cy_ratio <= 0.7:
            print("✓ Principal point is near image center")
            metrics['principal_point_ok'] = True
        else:
            print("⚠ Principal point is far from image center")
            metrics['principal_point_ok'] = False
        
        # Check distortion coefficients
        dist = self.calibration_data['intrinsics']['distortion']
        k1, k2 = dist[0], dist[1]
        
        print(f"Radial distortion: k1={k1:.6f}, k2={k2:.6f}")
        
        if abs(k1) < 1.0 and abs(k2) < 1.0:
            print("✓ Distortion coefficients are reasonable")
            metrics['distortion_ok'] = True
        else:
            print("⚠ Large distortion coefficients detected")
            metrics['distortion_ok'] = False
        
        # Overall quality assessment
        checks = [
            metrics.get('focal_length_ok', False),
            metrics.get('principal_point_ok', False),
            metrics.get('distortion_ok', False)
        ]
        
        if all(checks):
            metrics['quality'] = 'GOOD'
            print("\n✓ Overall calibration quality: GOOD")
        elif sum(checks) >= 2:
            metrics['quality'] = 'ACCEPTABLE'
            print("\n⚠ Overall calibration quality: ACCEPTABLE")
        else:
            metrics['quality'] = 'POOR'
            print("\n✗ Overall calibration quality: POOR - Consider recalibrating")
        
        return metrics


def validate_camera(calibration_path: str, device_id: int, show_bev: bool = False):
    """Validate a single camera calibration"""
    validator = CalibrationValidator(calibration_path, device_id)
    
    # Verify quality metrics
    metrics = validator.verify_calibration_quality()
    
    # Interactive validation
    print("\n" + "-"*60)
    print("INTERACTIVE VALIDATION")
    print("-"*60)
    
    # Undistortion validation
    response = input("\nValidate undistortion? (y/n) [y]: ").lower()
    if response != 'n':
        validator.validate_undistortion()
    
    # BEV validation (if applicable)
    if show_bev and metrics['has_homography']:
        response = input("\nValidate BEV transformation? (y/n) [y]: ").lower()
        if response != 'n':
            validator.validate_bev_transformation()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Validate camera calibration for SENTINEL system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate interior camera
  python scripts/validate_calibration.py --camera interior --device 0

  # Validate external camera with BEV
  python scripts/validate_calibration.py --camera front_left --device 1 --show-bev

  # Validate all cameras
  python scripts/validate_calibration.py --all
        """
    )
    
    parser.add_argument(
        '--camera',
        choices=['interior', 'front_left', 'front_right'],
        help='Camera to validate'
    )
    parser.add_argument(
        '--device',
        type=int,
        help='Camera device ID'
    )
    parser.add_argument(
        '--show-bev',
        action='store_true',
        help='Show BEV transformation validation'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Validate all cameras sequentially'
    )
    parser.add_argument(
        '--calibration-dir',
        default='configs/calibration',
        help='Directory containing calibration files'
    )
    
    args = parser.parse_args()
    
    if not args.all and (args.camera is None or args.device is None):
        parser.error("Either --all or both --camera and --device must be specified")
    
    results = []
    
    if args.all:
        print("\n" + "="*60)
        print("VALIDATING ALL CAMERAS")
        print("="*60)
        
        cameras = [
            ('interior', 0, False),
            ('front_left', 1, True),
            ('front_right', 2, True)
        ]
        
        for camera_name, device_id, show_bev in cameras:
            calibration_path = os.path.join(args.calibration_dir, f'{camera_name}.yaml')
            
            if not os.path.exists(calibration_path):
                print(f"\nSkipping {camera_name}: calibration file not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"VALIDATING {camera_name.upper().replace('_', ' ')}")
            print("="*60)
            
            metrics = validate_camera(calibration_path, device_id, show_bev)
            results.append(metrics)
    
    else:
        calibration_path = os.path.join(args.calibration_dir, f'{args.camera}.yaml')
        
        if not os.path.exists(calibration_path):
            print(f"Error: Calibration file not found: {calibration_path}")
            return 1
        
        metrics = validate_camera(calibration_path, args.device, args.show_bev)
        results.append(metrics)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for metrics in results:
        print(f"\n{metrics['camera_name']}:")
        print(f"  Quality: {metrics['quality']}")
        print(f"  Intrinsics: {'✓' if metrics['has_intrinsics'] else '✗'}")
        print(f"  Extrinsics: {'✓' if metrics['has_extrinsics'] else '✗'}")
        print(f"  Homography: {'✓' if metrics['has_homography'] else '✗'}")
    
    # Check if all cameras have good quality
    all_good = all(m['quality'] == 'GOOD' for m in results)
    
    if all_good:
        print("\n✓ All calibrations are GOOD")
        print("\nNext steps:")
        print("1. Calibration is ready for use")
        print("2. Update configs/default.yaml if needed")
        print("3. Run system: python src/main.py")
        return 0
    else:
        print("\n⚠ Some calibrations need attention")
        print("\nRecommendations:")
        print("1. Review cameras with ACCEPTABLE or POOR quality")
        print("2. Consider recalibrating if quality is POOR")
        print("3. Ensure proper lighting and checkerboard visibility")
        return 1


if __name__ == '__main__':
    exit(main())
