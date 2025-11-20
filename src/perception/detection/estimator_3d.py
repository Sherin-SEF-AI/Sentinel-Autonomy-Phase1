"""3D bounding box estimation from 2D detections."""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.core.data_structures import Detection2D, Detection3D


class Estimator3D:
    """Estimates 3D bounding boxes from 2D detections using camera extrinsics."""
    
    def __init__(self, calibration_data: Dict):
        """
        Initialize 3D estimator.

        Args:
            calibration_data: Dictionary mapping camera_id to calibration parameters:
                - intrinsics: Camera intrinsic matrix (3x3)
                - extrinsics: Camera extrinsic parameters
                    - translation: [x, y, z] in vehicle frame
                    - rotation: [roll, pitch, yaw] in radians
        """
        self.logger = logging.getLogger(__name__)
        self.calibration_data = calibration_data

        # Track cameras we've already warned about (to avoid log spam)
        self._warned_cameras = set()

        # Default object dimensions (w, h, l) in meters for each class
        self.default_dimensions = {
            'vehicle': (1.8, 1.5, 4.5),
            'pedestrian': (0.6, 1.7, 0.6),
            'cyclist': (0.6, 1.7, 1.8),
            'traffic_light': (0.3, 0.8, 0.3),
            'traffic_sign': (0.1, 0.6, 0.6)
        }

        self.logger.info("Estimator3D initialized")
    
    def estimate(self, detection_2d: Detection2D) -> Optional[Detection3D]:
        """
        Estimate 3D bounding box from 2D detection.
        
        Args:
            detection_2d: 2D detection in camera view
        
        Returns:
            Detection3D object or None if estimation fails
        """
        try:
            camera_id = detection_2d.camera_id

            if camera_id not in self.calibration_data:
                # Only warn once per camera to avoid log spam
                if camera_id not in self._warned_cameras:
                    self.logger.warning(f"No calibration data for camera {camera_id} - "
                                      f"3D estimation disabled for this camera")
                    self._warned_cameras.add(camera_id)
                return None
            
            calib = self.calibration_data[camera_id]
            
            # Get 2D bbox center and dimensions
            x1, y1, x2, y2 = detection_2d.bbox
            bbox_center_2d = ((x1 + x2) / 2, (y1 + y2) / 2)
            bbox_height_2d = y2 - y1
            
            # Estimate distance from bbox height (simple pinhole model)
            # Assumes object is on ground plane
            default_dims = self.default_dimensions.get(
                detection_2d.class_name, 
                (1.0, 1.5, 1.0)
            )
            w, h, l = default_dims
            
            # Get camera intrinsics
            intrinsics = calib.get('intrinsics', {})
            fy = intrinsics.get('fy', 800.0)
            
            # Estimate distance: distance = (real_height * focal_length) / pixel_height
            distance = (h * fy) / max(bbox_height_2d, 1.0)
            
            # Clip distance to reasonable range
            distance = np.clip(distance, 1.0, 100.0)
            
            # Project 2D point to 3D in camera frame
            fx = intrinsics.get('fx', 800.0)
            cx = intrinsics.get('cx', 640.0)
            cy = intrinsics.get('cy', 360.0)
            
            # 3D position in camera frame
            x_cam = (bbox_center_2d[0] - cx) * distance / fx
            y_cam = (bbox_center_2d[1] - cy) * distance / fy
            z_cam = distance
            
            # Transform to vehicle frame using extrinsics
            extrinsics = calib.get('extrinsics', {})
            translation = np.array(extrinsics.get('translation', [0, 0, 0]))
            rotation = extrinsics.get('rotation', [0, 0, 0])
            
            # Create rotation matrix from roll, pitch, yaw
            R = self._rotation_matrix_from_euler(rotation)
            
            # Transform point
            point_cam = np.array([x_cam, y_cam, z_cam])
            point_vehicle = R @ point_cam + translation
            
            x_veh, y_veh, z_veh = point_vehicle
            
            # Estimate orientation (simplified - assumes object faces forward)
            theta = 0.0
            
            # Create Detection3D
            detection_3d = Detection3D(
                bbox_3d=(float(x_veh), float(y_veh), float(z_veh), 
                        float(w), float(h), float(l), float(theta)),
                class_name=detection_2d.class_name,
                confidence=detection_2d.confidence,
                velocity=(0.0, 0.0, 0.0),  # Will be estimated by tracker
                track_id=-1  # Will be assigned by tracker
            )
            
            return detection_3d
            
        except Exception as e:
            self.logger.error(f"3D estimation failed: {e}")
            return None
    
    def _rotation_matrix_from_euler(self, euler: List[float]) -> np.ndarray:
        """
        Create rotation matrix from Euler angles (roll, pitch, yaw).
        
        Args:
            euler: [roll, pitch, yaw] in radians
        
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = euler
        
        # Rotation around X (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Rotation around Y (pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Rotation around Z (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R
    
    def estimate_batch(self, detections_2d: List[Detection2D]) -> List[Detection3D]:
        """
        Estimate 3D bounding boxes for multiple 2D detections.
        
        Args:
            detections_2d: List of 2D detections
        
        Returns:
            List of Detection3D objects
        """
        detections_3d = []
        
        for det_2d in detections_2d:
            det_3d = self.estimate(det_2d)
            if det_3d is not None:
                detections_3d.append(det_3d)
        
        return detections_3d
