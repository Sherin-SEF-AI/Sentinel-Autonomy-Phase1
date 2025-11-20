"""Camera manager integrating capture, synchronization, and calibration."""

import numpy as np
from typing import Dict, Optional
import logging

from ..core.interfaces import ICameraManager
from ..core.data_structures import CameraBundle
from .capture import CameraCapture
from .sync import TimestampSync
from .calibration import CalibrationLoader, CameraCalibration


class CameraManager(ICameraManager):
    """
    Manages multiple cameras with synchronization and calibration.
    
    Implements graceful degradation when cameras fail and automatic reconnection.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize camera manager.
        
        Args:
            config: Configuration dictionary with camera settings
        """
        self.config = config
        self.logger = logging.getLogger("CameraManager")
        
        # Camera name to ID mapping
        self.camera_name_to_id = {
            'interior': 0,
            'front_left': 1,
            'front_right': 2
        }
        
        # Initialize components
        self.captures: Dict[int, CameraCapture] = {}
        self.sync = TimestampSync(tolerance_ms=5.0)
        self.calibration_loader = CalibrationLoader()
        
        # State
        self.is_running = False
        self.required_cameras = set([0, 1, 2])  # All cameras required for full operation
        self.degraded_mode = False
        
        # Statistics
        self.frame_count = 0
        self.failed_sync_count = 0
    
    def start(self) -> None:
        """Start camera capture threads."""
        if self.is_running:
            self.logger.warning("Camera manager already running")
            return
        
        # Load calibrations
        camera_configs = self.config.get('cameras', {})
        if not self.calibration_loader.load_all(camera_configs):
            self.logger.warning("Some calibrations failed to load")
        
        # Initialize and start cameras
        for camera_name, camera_config in camera_configs.items():
            camera_id = self.camera_name_to_id.get(camera_name)
            if camera_id is None:
                self.logger.warning(f"Unknown camera name: {camera_name}")
                continue
            
            # Create camera capture
            capture = CameraCapture(
                camera_id=camera_id,
                device=camera_config.get('device', camera_id),
                resolution=tuple(camera_config.get('resolution', [640, 480])),
                fps=camera_config.get('fps', 30),
                buffer_size=5
            )
            
            # Start capture
            if capture.start():
                self.captures[camera_id] = capture
                self.logger.info(f"Started camera {camera_name} (id={camera_id})")
            else:
                self.logger.error(f"Failed to start camera {camera_name} (id={camera_id})")
        
        # Check if we have minimum required cameras
        if len(self.captures) == 0:
            raise RuntimeError("No cameras could be initialized")
        
        if len(self.captures) < len(self.required_cameras):
            self.logger.warning(
                f"Running in degraded mode: {len(self.captures)}/{len(self.required_cameras)} cameras active"
            )
            self.degraded_mode = True
        
        self.is_running = True
        self.logger.info("Camera manager started")
    
    def stop(self) -> None:
        """Stop camera capture."""
        if not self.is_running:
            return
        
        # Stop all cameras
        for camera_id, capture in self.captures.items():
            capture.stop()
            self.logger.info(f"Stopped camera {camera_id}")
        
        self.captures.clear()
        self.is_running = False
        self.logger.info("Camera manager stopped")
    
    def get_frame_bundle(self) -> Optional[CameraBundle]:
        """
        Get synchronized frame bundle.
        
        Returns:
            CameraBundle with synchronized frames or None if synchronization fails
        """
        if not self.is_running:
            return None
        
        # Collect latest frames from all cameras
        frames = {}
        for camera_id, capture in self.captures.items():
            frame_data = capture.get_latest_frame()
            if frame_data is not None:
                frames[camera_id] = frame_data
        
        # Check if we have frames from all active cameras
        if len(frames) != len(self.captures):
            return None
        
        # Synchronize frames
        sync_result = self.sync.synchronize(frames)
        if sync_result is None:
            self.failed_sync_count += 1
            if self.failed_sync_count % 10 == 0:
                self.logger.warning(f"Frame synchronization failed ({self.failed_sync_count} times)")
            return None
        
        synchronized_frames, timestamp = sync_result
        
        # Check for camera health and attempt reconnection if needed
        self._check_and_reconnect_cameras()
        
        # Create CameraBundle
        try:
            bundle = self._create_camera_bundle(synchronized_frames, timestamp)
            self.frame_count += 1
            return bundle
        except Exception as e:
            self.logger.error(f"Error creating camera bundle: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """
        Check if all cameras are operational.
        
        Returns:
            True if all required cameras are healthy, False otherwise
        """
        if not self.is_running:
            return False
        
        # Check health of all active cameras
        healthy_cameras = sum(1 for capture in self.captures.values() if capture.is_healthy())
        
        # In normal mode, all cameras must be healthy
        if not self.degraded_mode:
            return healthy_cameras == len(self.required_cameras)
        
        # In degraded mode, at least one camera must be healthy
        return healthy_cameras > 0
    
    def get_calibration(self, camera_id: int) -> Optional[CameraCalibration]:
        """
        Get calibration for specific camera.
        
        Args:
            camera_id: Camera identifier
        
        Returns:
            CameraCalibration object or None if not available
        """
        return self.calibration_loader.get_calibration(camera_id)
    
    def _create_camera_bundle(self, frames: Dict[int, np.ndarray], timestamp: float) -> CameraBundle:
        """
        Create CameraBundle from synchronized frames.
        
        Args:
            frames: Dictionary of synchronized frames
            timestamp: Reference timestamp
        
        Returns:
            CameraBundle object
        """
        # Get frames for each camera (use black frame if missing)
        interior = frames.get(0, self._create_black_frame((480, 640, 3)))
        front_left = frames.get(1, self._create_black_frame((720, 1280, 3)))
        front_right = frames.get(2, self._create_black_frame((720, 1280, 3)))
        
        return CameraBundle(
            timestamp=timestamp,
            interior=interior,
            front_left=front_left,
            front_right=front_right
        )
    
    def _create_black_frame(self, shape: tuple) -> np.ndarray:
        """Create a black frame of specified shape."""
        return np.zeros(shape, dtype=np.uint8)
    
    def _check_and_reconnect_cameras(self) -> None:
        """Check camera health and attempt reconnection for failed cameras."""
        # Check each camera
        for camera_id, capture in list(self.captures.items()):
            if not capture.is_healthy():
                self.logger.warning(f"Camera {camera_id} unhealthy, monitoring for reconnection")
                
                # Camera will attempt automatic reconnection in its capture thread
                # We just log the status here
        
        # Check if any required cameras are missing and not in captures
        missing_cameras = self.required_cameras - set(self.captures.keys())
        if missing_cameras:
            self.logger.debug(f"Missing cameras: {missing_cameras}")
    
    def get_statistics(self) -> Dict:
        """
        Get camera manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'is_running': self.is_running,
            'degraded_mode': self.degraded_mode,
            'frame_count': self.frame_count,
            'failed_sync_count': self.failed_sync_count,
            'active_cameras': len(self.captures),
            'healthy_cameras': sum(1 for c in self.captures.values() if c.is_healthy()),
            'sync_stats': self.sync.get_statistics()
        }
        
        # Per-camera stats
        for camera_id, capture in self.captures.items():
            stats[f'camera_{camera_id}_healthy'] = capture.is_healthy()
            stats[f'camera_{camera_id}_frame_count'] = capture.frame_count
        
        return stats
