#!/usr/bin/env python3
"""
Camera Calibration Script for SENTINEL System

This script performs camera calibration for the SENTINEL multi-camera system:
1. Captures checkerboard images from cameras
2. Computes camera intrinsics (focal length, principal point, distortion)
3. Computes camera extrinsics (position and orientation relative to vehicle frame)
4. Computes homography matrices for BEV transformation
5. Saves calibration data to YAML files

Usage:
    python scripts/calibrate_camera.py --camera interior --device 0
    python scripts/calibrate_camera.py --camera front_left --device 1 --bev
    python scripts/calibrate_camera.py --camera front_right --device 2 --bev
"""

import argparse
import cv2
import numpy as np
import yaml
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time


class CameraCalibrator:
    """Handles camera calibration process"""
    
    def __init__(
        self,
        camera_name: str,
        device_id: int,
        checkerboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025,  # 25mm squares
        num_images: int = 20
    ):
        self.camera_name = camera_name
        self.device_id = device_id
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.num_images = num_images
        
        # Storage for calibration data
        self.object_points = []  # 3D points in real world space
        self.image_points = []   # 2D points in image plane
        self.images = []
        self.image_size = None
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def prepare_object_points(self) -> np.ndarray:
        """Prepare 3D object points for checkerboard"""
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:self.checkerboard_size[0],
            0:self.checkerboard_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def capture_calibration_images(self) -> bool:
        """Capture checkerboard images from camera"""
        print(f"\n=== Capturing calibration images for {self.camera_name} ===")
        print(f"Device ID: {self.device_id}")
        print(f"Target: {self.num_images} images")
        print(f"Checkerboard: {self.checkerboard_size[0]}x{self.checkerboard_size[1]}")
        print("\nInstructions:")
        print("- Position checkerboard in camera view")
        print("- Press SPACE to capture image")
        print("- Press 'q' to quit")
        print("- Move checkerboard to different positions/angles\n")
        
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.device_id}")
            return False
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        objp = self.prepare_object_points()
        captured_count = 0
        
        while captured_count < self.num_images:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            if self.image_size is None:
                self.image_size = (frame.shape[1], frame.shape[0])
            
            # Convert to grayscale for corner detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Draw corners on display frame
            display_frame = frame.copy()
            if ret:
                cv2.drawChessboardCorners(display_frame, self.checkerboard_size, corners, ret)
                cv2.putText(
                    display_frame,
                    "Checkerboard detected! Press SPACE to capture",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display_frame,
                    "No checkerboard detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            cv2.putText(
                display_frame,
                f"Captured: {captured_count}/{self.num_images}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            cv2.imshow(f'Calibration - {self.camera_name}', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                self.object_points.append(objp)
                self.image_points.append(corners_refined)
                self.images.append(frame.copy())
                captured_count += 1
                print(f"Captured image {captured_count}/{self.num_images}")
                
                # Visual feedback
                cv2.imshow(f'Calibration - {self.camera_name}', display_frame)
                cv2.waitKey(500)
                
            elif key == ord('q'):
                print("\nCalibration cancelled by user")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_count < self.num_images:
            print(f"\nWarning: Only captured {captured_count}/{self.num_images} images")
            return captured_count >= 10  # Minimum 10 images
        
        print(f"\nSuccessfully captured {captured_count} images")
        return True
    
    def compute_intrinsics(self) -> bool:
        """Compute camera intrinsic parameters"""
        print(f"\n=== Computing intrinsics for {self.camera_name} ===")
        
        if len(self.object_points) < 10:
            print("Error: Not enough calibration images")
            return False
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.image_points,
            self.image_size,
            None,
            None
        )
        
        if not ret:
            print("Error: Camera calibration failed")
            return False
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.object_points)):
            imgpoints2, _ = cv2.projectPoints(
                self.object_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.object_points)
        
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs.ravel()}")
        print(f"Mean reprojection error: {mean_error:.4f} pixels")
        
        return True
    
    def compute_extrinsics(
        self,
        vehicle_origin: Tuple[float, float, float],
        vehicle_orientation: Tuple[float, float, float]
    ) -> Dict:
        """
        Compute camera extrinsics relative to vehicle frame
        
        Args:
            vehicle_origin: (x, y, z) position in meters from vehicle origin
            vehicle_orientation: (roll, pitch, yaw) in radians
        """
        print(f"\n=== Computing extrinsics for {self.camera_name} ===")
        
        extrinsics = {
            'translation': list(vehicle_origin),
            'rotation': list(vehicle_orientation)
        }
        
        print(f"Translation (x, y, z): {vehicle_origin}")
        print(f"Rotation (roll, pitch, yaw): {vehicle_orientation}")
        
        return extrinsics
    
    def compute_homography(
        self,
        ground_points_image: np.ndarray,
        ground_points_bev: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix for BEV transformation
        
        Args:
            ground_points_image: 4+ points in image coordinates (Nx2)
            ground_points_bev: Corresponding points in BEV coordinates (Nx2)
        """
        print(f"\n=== Computing homography for {self.camera_name} ===")
        
        if len(ground_points_image) < 4 or len(ground_points_bev) < 4:
            print("Error: Need at least 4 point correspondences")
            return None
        
        H, mask = cv2.findHomography(ground_points_image, ground_points_bev, cv2.RANSAC, 5.0)
        
        if H is None:
            print("Error: Homography computation failed")
            return None
        
        print(f"Homography matrix:\n{H}")
        print(f"Inliers: {np.sum(mask)}/{len(mask)}")
        
        return H
    
    def save_calibration(
        self,
        output_path: str,
        extrinsics: Optional[Dict] = None,
        homography: Optional[np.ndarray] = None
    ):
        """Save calibration data to YAML file"""
        print(f"\n=== Saving calibration to {output_path} ===")
        
        calibration_data = {
            'camera_name': self.camera_name,
            'image_size': {
                'width': int(self.image_size[0]),
                'height': int(self.image_size[1])
            },
            'intrinsics': {
                'fx': float(self.camera_matrix[0, 0]),
                'fy': float(self.camera_matrix[1, 1]),
                'cx': float(self.camera_matrix[0, 2]),
                'cy': float(self.camera_matrix[1, 2]),
                'distortion': [float(x) for x in self.dist_coeffs.ravel()]
            }
        }
        
        if extrinsics is not None:
            calibration_data['extrinsics'] = extrinsics
        
        if homography is not None:
            calibration_data['homography'] = {
                'matrix': [[float(homography[i, j]) for j in range(3)] for i in range(3)]
            }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Calibration saved successfully")


def calibrate_interior_camera(device_id: int, output_dir: str):
    """Calibrate interior (DMS) camera"""
    print("\n" + "="*60)
    print("INTERIOR CAMERA CALIBRATION")
    print("="*60)
    
    calibrator = CameraCalibrator(
        camera_name='interior',
        device_id=device_id,
        num_images=20
    )
    
    # Capture images
    if not calibrator.capture_calibration_images():
        print("Failed to capture calibration images")
        return False
    
    # Compute intrinsics
    if not calibrator.compute_intrinsics():
        print("Failed to compute intrinsics")
        return False
    
    # For interior camera, we don't need extrinsics or homography
    # (it's not used for BEV generation)
    output_path = os.path.join(output_dir, 'interior.yaml')
    calibrator.save_calibration(output_path)
    
    return True


def calibrate_external_camera(
    camera_name: str,
    device_id: int,
    output_dir: str,
    compute_bev: bool = True
):
    """Calibrate external camera (front_left or front_right)"""
    print("\n" + "="*60)
    print(f"{camera_name.upper().replace('_', ' ')} CAMERA CALIBRATION")
    print("="*60)
    
    calibrator = CameraCalibrator(
        camera_name=camera_name,
        device_id=device_id,
        num_images=20
    )
    
    # Capture images
    if not calibrator.capture_calibration_images():
        print("Failed to capture calibration images")
        return False
    
    # Compute intrinsics
    if not calibrator.compute_intrinsics():
        print("Failed to compute intrinsics")
        return False
    
    # Get extrinsics from user
    print("\n" + "-"*60)
    print("EXTRINSICS CONFIGURATION")
    print("-"*60)
    print("Enter camera position relative to vehicle origin (rear axle center):")
    
    try:
        x = float(input(f"  X (forward, meters) [{1.5 if 'front' in camera_name else 0.0}]: ") or (1.5 if 'front' in camera_name else 0.0))
        y = float(input(f"  Y (left, meters) [{0.5 if 'left' in camera_name else -0.5}]: ") or (0.5 if 'left' in camera_name else -0.5))
        z = float(input(f"  Z (up, meters) [1.2]: ") or 1.2)
        
        print("\nEnter camera orientation (radians):")
        roll = float(input(f"  Roll [0.0]: ") or 0.0)
        pitch = float(input(f"  Pitch [-0.2]: ") or -0.2)
        yaw = float(input(f"  Yaw [{0.785 if 'left' in camera_name else -0.785}]: ") or (0.785 if 'left' in camera_name else -0.785))
        
    except ValueError:
        print("Invalid input, using default values")
        x, y, z = (1.5, 0.5 if 'left' in camera_name else -0.5, 1.2)
        roll, pitch, yaw = (0.0, -0.2, 0.785 if 'left' in camera_name else -0.785)
    
    extrinsics = calibrator.compute_extrinsics(
        vehicle_origin=(x, y, z),
        vehicle_orientation=(roll, pitch, yaw)
    )
    
    homography = None
    if compute_bev:
        print("\n" + "-"*60)
        print("BEV HOMOGRAPHY COMPUTATION")
        print("-"*60)
        print("For BEV transformation, we need ground plane point correspondences.")
        print("Using default homography based on camera position...")
        
        # Create a default homography based on camera extrinsics
        # This is a simplified approach - in production, you'd want to manually
        # select ground points or use a more sophisticated method
        homography = create_default_homography(
            calibrator.camera_matrix,
            (x, y, z),
            (roll, pitch, yaw)
        )
    
    output_path = os.path.join(output_dir, f'{camera_name}.yaml')
    calibrator.save_calibration(output_path, extrinsics, homography)
    
    return True


def create_default_homography(
    camera_matrix: np.ndarray,
    translation: Tuple[float, float, float],
    rotation: Tuple[float, float, float]
) -> np.ndarray:
    """
    Create a default homography matrix based on camera parameters
    This is a simplified approach for demonstration
    """
    # Create rotation matrix from euler angles
    roll, pitch, yaw = rotation
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    
    # Translation vector
    t = np.array(translation).reshape(3, 1)
    
    # Simplified homography (assumes ground plane at z=0)
    # H = K * [r1 r2 t] where r1, r2 are first two columns of R
    H = camera_matrix @ np.hstack([R[:, 0:2], t])
    
    # Normalize
    H = H / H[2, 2]
    
    return H


def main():
    parser = argparse.ArgumentParser(
        description='Camera calibration for SENTINEL system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate interior camera
  python scripts/calibrate_camera.py --camera interior --device 0

  # Calibrate external camera with BEV
  python scripts/calibrate_camera.py --camera front_left --device 1 --bev

  # Calibrate all cameras
  python scripts/calibrate_camera.py --all
        """
    )
    
    parser.add_argument(
        '--camera',
        choices=['interior', 'front_left', 'front_right'],
        help='Camera to calibrate'
    )
    parser.add_argument(
        '--device',
        type=int,
        help='Camera device ID'
    )
    parser.add_argument(
        '--bev',
        action='store_true',
        help='Compute BEV homography (for external cameras)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Calibrate all cameras sequentially'
    )
    parser.add_argument(
        '--output-dir',
        default='configs/calibration',
        help='Output directory for calibration files'
    )
    
    args = parser.parse_args()
    
    if not args.all and (args.camera is None or args.device is None):
        parser.error("Either --all or both --camera and --device must be specified")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all:
        print("\n" + "="*60)
        print("CALIBRATING ALL CAMERAS")
        print("="*60)
        
        cameras = [
            ('interior', 0, False),
            ('front_left', 1, True),
            ('front_right', 2, True)
        ]
        
        for camera_name, device_id, compute_bev in cameras:
            if camera_name == 'interior':
                success = calibrate_interior_camera(device_id, args.output_dir)
            else:
                success = calibrate_external_camera(
                    camera_name,
                    device_id,
                    args.output_dir,
                    compute_bev
                )
            
            if not success:
                print(f"\nFailed to calibrate {camera_name}")
                return 1
            
            print(f"\n{camera_name} calibration complete!")
            time.sleep(2)
    
    else:
        if args.camera == 'interior':
            success = calibrate_interior_camera(args.device, args.output_dir)
        else:
            success = calibrate_external_camera(
                args.camera,
                args.device,
                args.output_dir,
                args.bev
            )
        
        if not success:
            print(f"\nFailed to calibrate {args.camera}")
            return 1
    
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Calibration files saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Review calibration files")
    print("2. Run validation: python scripts/validate_calibration.py")
    print("3. Update configs/default.yaml with calibration paths")
    
    return 0


if __name__ == '__main__':
    exit(main())
