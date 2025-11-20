"""Main object detector integrating all detection components."""

import logging
from typing import Dict, List, Tuple
import numpy as np
import time

from src.core.interfaces import IObjectDetector
from src.core.data_structures import Detection2D, Detection3D
from .detector_2d import Detector2D
from .estimator_3d import Estimator3D
from .fusion import MultiViewFusion
from .tracker import ObjectTracker


class ObjectDetector(IObjectDetector):
    """
    Multi-view object detector with 3D estimation, fusion, and tracking.
    
    Integrates:
    - 2D detection per camera (YOLOv8)
    - 3D bounding box estimation
    - Multi-view fusion
    - Object tracking (DeepSORT)
    - Velocity estimation (via Kalman filter)
    """
    
    def __init__(self, config: Dict, calibration_data: Dict):
        """
        Initialize object detector.
        
        Args:
            config: Detection configuration containing:
                - detection: 2D detector config
                - fusion: Fusion config
                - tracking: Tracking config
            calibration_data: Camera calibration data
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        detection_config = config.get('detection', {})
        self.detector_2d = Detector2D(detection_config)
        
        self.estimator_3d = Estimator3D(calibration_data)
        
        fusion_config = config.get('fusion', {})
        self.fusion = MultiViewFusion(fusion_config)
        
        tracking_config = config.get('tracking', {})
        self.tracker = ObjectTracker(tracking_config)
        
        # Error recovery
        self.error_count = 0
        self.max_errors = 3
        self.last_valid_output = ({},[])
        
        self.logger.info("ObjectDetector initialized")
    
    def detect(self, frames: Dict[int, np.ndarray]) -> Tuple[Dict[int, List[Detection2D]], 
                                                              List[Detection3D]]:
        """
        Detect and track objects from multiple camera views.
        
        Args:
            frames: Dictionary mapping camera_id to frame (H, W, 3)
        
        Returns:
            Tuple of:
                - Dictionary mapping camera_id to list of Detection2D objects
                - List of tracked Detection3D objects
        """
        try:
            start_time = time.time()
            
            # Step 1: 2D detection per camera
            detections_2d = self.detector_2d.detect_batch(frames)
            
            # Step 2: 3D estimation for all 2D detections
            all_detections_3d = []
            for camera_id, dets_2d in detections_2d.items():
                for det_2d in dets_2d:
                    det_3d = self.estimator_3d.estimate(det_2d)
                    if det_3d is not None:
                        all_detections_3d.append(det_3d)
            
            # Step 3: Multi-view fusion
            fused_detections = self.fusion.fuse(all_detections_3d)
            
            # Step 4: Tracking and velocity estimation
            tracked_detections = self.tracker.update(fused_detections)
            
            # Log performance
            elapsed = (time.time() - start_time) * 1000
            self.logger.debug(f"Detection pipeline: {elapsed:.1f}ms, "
                            f"{len(tracked_detections)} objects tracked")
            
            # Reset error count on success
            self.error_count = 0
            self.last_valid_output = (detections_2d, tracked_detections)
            
            return detections_2d, tracked_detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            self.error_count += 1
            
            # Error recovery
            if self.error_count >= self.max_errors:
                self.logger.warning("Max errors reached, attempting to reload detector")
                self._recover()
                self.error_count = 0
            
            # Return last valid output
            return self.last_valid_output
    
    def _recover(self):
        """Attempt to recover from errors by reloading components."""
        try:
            self.logger.info("Attempting error recovery...")
            
            # Reload 2D detector
            detection_config = self.config.get('detection', {})
            self.detector_2d = Detector2D(detection_config)
            
            self.logger.info("Error recovery successful")
            
        except Exception as e:
            self.logger.error(f"Error recovery failed: {e}")
    
    def get_statistics(self) -> Dict:
        """
        Get detection statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'active_tracks': len(self.tracker.tracks),
            'confirmed_tracks': sum(1 for t in self.tracker.tracks if t.state == 'confirmed'),
            'frame_count': self.tracker.frame_count,
            'error_count': self.error_count
        }
