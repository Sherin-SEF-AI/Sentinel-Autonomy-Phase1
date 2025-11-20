"""Camera calibration data loading."""

import yaml
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    distortion: np.ndarray  # Distortion coefficients [k1, k2, p1, p2, k3]
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert to camera matrix.
        
        Returns:
            3x3 camera matrix
        """
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)


@dataclass
class CameraExtrinsics:
    """Camera extrinsic parameters."""
    translation: np.ndarray  # Translation vector [x, y, z] in meters
    rotation: np.ndarray  # Rotation angles [roll, pitch, yaw] in radians
    
    def to_rotation_matrix(self) -> np.ndarray:
        """
        Convert rotation angles to rotation matrix.
        
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = self.rotation
        
        # Rotation matrix around X axis (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Rotation matrix around Y axis (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation matrix around Z axis (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def to_transform_matrix(self) -> np.ndarray:
        """
        Convert to 4x4 transformation matrix.
        
        Returns:
            4x4 transformation matrix
        """
        R = self.to_rotation_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.translation
        return T


@dataclass
class CameraCalibration:
    """Complete camera calibration data."""
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    homography: np.ndarray  # 3x3 homography matrix for BEV transformation


class CalibrationLoader:
    """Loads camera calibration data from YAML files."""
    
    def __init__(self):
        """Initialize calibration loader."""
        self.logger = logging.getLogger("CalibrationLoader")
        self.calibrations: Dict[int, CameraCalibration] = {}
    
    def load(self, camera_id: int, calibration_path: str) -> Optional[CameraCalibration]:
        """
        Load calibration data from YAML file.
        
        Args:
            camera_id: Camera identifier
            calibration_path: Path to calibration YAML file
        
        Returns:
            CameraCalibration object or None if loading fails
        """
        try:
            with open(calibration_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse intrinsics
            intrinsics_data = data.get('intrinsics', {})
            intrinsics = CameraIntrinsics(
                fx=float(intrinsics_data.get('fx', 800.0)),
                fy=float(intrinsics_data.get('fy', 800.0)),
                cx=float(intrinsics_data.get('cx', 640.0)),
                cy=float(intrinsics_data.get('cy', 360.0)),
                distortion=np.array(intrinsics_data.get('distortion', [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            )
            
            # Parse extrinsics
            extrinsics_data = data.get('extrinsics', {})
            extrinsics = CameraExtrinsics(
                translation=np.array(extrinsics_data.get('translation', [0.0, 0.0, 0.0]), dtype=np.float32),
                rotation=np.array(extrinsics_data.get('rotation', [0.0, 0.0, 0.0]), dtype=np.float32)
            )
            
            # Parse homography
            homography_data = data.get('homography', {})
            homography_matrix = homography_data.get('matrix', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            homography = np.array(homography_matrix, dtype=np.float32)
            
            # Create calibration object
            calibration = CameraCalibration(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                homography=homography
            )
            
            # Cache calibration
            self.calibrations[camera_id] = calibration
            
            self.logger.info(f"Loaded calibration for camera {camera_id} from {calibration_path}")
            return calibration
            
        except FileNotFoundError:
            self.logger.error(f"Calibration file not found: {calibration_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading calibration from {calibration_path}: {e}")
            return None
    
    def get_calibration(self, camera_id: int) -> Optional[CameraCalibration]:
        """
        Get cached calibration for camera.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            CameraCalibration object or None if not loaded
        """
        return self.calibrations.get(camera_id)
    
    def load_all(self, camera_configs: Dict[str, Dict]) -> bool:
        """
        Load calibrations for all cameras from configuration.
        
        Args:
            camera_configs: Dictionary of camera configurations with calibration paths
        
        Returns:
            True if all calibrations loaded successfully, False otherwise
        """
        camera_name_to_id = {
            'interior': 0,
            'front_left': 1,
            'front_right': 2
        }
        
        success = True
        for camera_name, config in camera_configs.items():
            camera_id = camera_name_to_id.get(camera_name)
            if camera_id is None:
                self.logger.warning(f"Unknown camera name: {camera_name}")
                continue
            
            calibration_path = config.get('calibration')
            if not calibration_path:
                self.logger.warning(f"No calibration path for camera {camera_name}")
                success = False
                continue
            
            if self.load(camera_id, calibration_path) is None:
                success = False
        
        return success
