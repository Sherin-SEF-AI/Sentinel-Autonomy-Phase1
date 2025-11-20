"""Head pose estimation using PnP algorithm."""

import logging
from typing import Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class HeadPoseEstimator:
    """Head pose estimation using PnP (Perspective-n-Point) algorithm."""
    
    def __init__(self):
        """Initialize head pose estimator with 3D face model."""
        # 3D model points for key facial landmarks (in cm)
        # Based on average human face dimensions
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip (30)
            (0.0, -3.3, -2.5),         # Chin (8)
            (-2.25, 1.65, -1.5),       # Left eye left corner (36)
            (2.25, 1.65, -1.5),        # Right eye right corner (45)
            (-1.5, -1.5, -1.5),        # Left mouth corner (48)
            (1.5, -1.5, -1.5)          # Right mouth corner (54)
        ], dtype=np.float64)
        
        # Indices of landmarks in 68-point model that correspond to model points
        self.landmark_indices = [30, 8, 36, 45, 48, 54]
        
        logger.info("HeadPoseEstimator initialized")
    
    def estimate_head_pose(self, landmarks: np.ndarray, frame_shape: tuple) -> Dict[str, float]:
        """
        Estimate head pose (roll, pitch, yaw) from facial landmarks.
        
        Args:
            landmarks: Facial landmarks array (68, 2)
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            Dictionary with roll, pitch, yaw angles in degrees
        """
        if landmarks is None or len(landmarks) < 68:
            return self._default_pose()
        
        try:
            # Extract 2D image points for key landmarks
            image_points = np.array([
                landmarks[idx] for idx in self.landmark_indices
            ], dtype=np.float64)
            
            # Camera internals (assuming standard webcam)
            h, w = frame_shape[:2]
            focal_length = w
            center = (w / 2, h / 2)
            
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Assuming no lens distortion
            dist_coeffs = np.zeros((4, 1))
            
            # Solve PnP
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return self._default_pose()
            
            # Convert rotation vector to rotation matrix
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            # Calculate Euler angles from rotation matrix
            roll, pitch, yaw = self._rotation_matrix_to_euler_angles(rotation_mat)
            
            return {
                'roll': float(roll),
                'pitch': float(pitch),
                'yaw': float(yaw)
            }
        
        except Exception as e:
            logger.error(f"Head pose estimation failed: {e}")
            return self._default_pose()
    
    def _rotation_matrix_to_euler_angles(self, R: np.ndarray) -> tuple:
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw).
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        # Calculate yaw
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        # Convert to degrees
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        
        return roll, pitch, yaw
    
    def _default_pose(self) -> Dict[str, float]:
        """Return default pose when estimation fails."""
        return {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0
        }
