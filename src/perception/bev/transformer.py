"""Perspective transformation for BEV generation."""

import numpy as np
import cv2
from typing import Dict, Any
import logging


class PerspectiveTransformer:
    """Transforms camera perspective to bird's eye view using homography."""
    
    def __init__(self, calibration: Dict[str, Any], output_size: tuple):
        """
        Initialize perspective transformer.
        
        Args:
            calibration: Camera calibration data with intrinsics, extrinsics, and homography
            output_size: Target BEV output size (width, height)
        """
        self.logger = logging.getLogger(__name__)
        self.output_size = output_size
        
        # Extract calibration parameters
        intrinsics = calibration.get('intrinsics', {})
        self.fx = intrinsics.get('fx', 800.0)
        self.fy = intrinsics.get('fy', 800.0)
        self.cx = intrinsics.get('cx', 640.0)
        self.cy = intrinsics.get('cy', 360.0)
        self.distortion = np.array(intrinsics.get('distortion', [0.0, 0.0, 0.0, 0.0, 0.0]))
        
        # Camera matrix
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Homography matrix
        homography_data = calibration.get('homography', {})
        homography_matrix = homography_data.get('matrix', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.homography = np.array(homography_matrix, dtype=np.float32)
        
        self.logger.info(f"PerspectiveTransformer initialized with output size {output_size}")
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply camera undistortion using intrinsic parameters.
        
        Args:
            frame: Input camera frame
            
        Returns:
            Undistorted frame
        """
        if np.allclose(self.distortion, 0.0):
            # No distortion, return original
            return frame
        
        return cv2.undistort(frame, self.camera_matrix, self.distortion)
    
    def warp_to_bev(self, frame: np.ndarray) -> np.ndarray:
        """
        Warp frame to BEV coordinate system using homography.
        
        Args:
            frame: Undistorted camera frame
            
        Returns:
            Warped BEV frame
        """
        bev_frame = cv2.warpPerspective(
            frame,
            self.homography,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return bev_frame
    
    def transform(self, frame: np.ndarray) -> np.ndarray:
        """
        Complete transformation pipeline: undistort and warp to BEV.
        
        Args:
            frame: Input camera frame
            
        Returns:
            BEV transformed frame
        """
        # Apply undistortion
        undistorted = self.undistort(frame)
        
        # Warp to BEV
        bev = self.warp_to_bev(undistorted)
        
        return bev
