"""Multi-threaded camera capture with health monitoring."""

import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Tuple
import logging


class CameraCapture:
    """Multi-threaded camera capture from USB devices."""
    
    def __init__(self, camera_id: int, device: int, resolution: Tuple[int, int], 
                 fps: int, buffer_size: int = 5):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Unique identifier for this camera (0=interior, 1=front_left, 2=front_right)
            device: USB device index
            resolution: (width, height) tuple
            fps: Target frames per second
            buffer_size: Size of circular buffer for frames
        """
        self.camera_id = camera_id
        self.device = device
        self.resolution = resolution
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.logger = logging.getLogger(f"CameraCapture-{camera_id}")
        
        # Thread-safe circular buffer
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_connected = False
        self.last_frame_time = 0.0
        self.frame_count = 0
        
        # Capture thread
        self.capture_thread: Optional[threading.Thread] = None
        
        # Health monitoring
        self.health_check_interval = 1.0  # seconds
        self.last_health_check = 0.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Performance tracking
        self.stats_interval = 10.0  # Log stats every 10 seconds
        self.last_stats_time = 0.0
        self.frames_since_stats = 0
        
        self.logger.debug(
            f"CameraCapture initialized: camera_id={camera_id}, device={device}, "
            f"resolution={resolution}, fps={fps}, buffer_size={buffer_size}"
        )
    
    def start(self) -> bool:
        """
        Start camera capture thread.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        self.logger.debug(f"Starting camera capture: camera_id={self.camera_id}, device={self.device}")
        
        if self.is_running:
            self.logger.warning(f"Camera already running: camera_id={self.camera_id}")
            return True
        
        # Initialize camera
        start_time = time.time()
        if not self._initialize_camera():
            self.logger.error(f"Camera initialization failed: camera_id={self.camera_id}, device={self.device}")
            return False
        
        init_duration = time.time() - start_time
        
        # Start capture thread
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.last_stats_time = time.time()
        
        self.logger.info(
            f"Camera capture started: camera_id={self.camera_id}, device={self.device}, "
            f"resolution={self.resolution}, fps={self.fps}, init_duration={init_duration:.3f}s"
        )
        return True
    
    def stop(self) -> None:
        """Stop camera capture thread."""
        if not self.is_running:
            self.logger.debug(f"Camera already stopped: camera_id={self.camera_id}")
            return
        
        self.logger.debug(f"Stopping camera capture: camera_id={self.camera_id}")
        self.is_running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                self.logger.warning(f"Capture thread did not terminate: camera_id={self.camera_id}")
        
        # Release camera
        self._release_camera()
        
        self.logger.info(
            f"Camera capture stopped: camera_id={self.camera_id}, total_frames={self.frame_count}"
        )
    
    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Get the latest frame from buffer.
        
        Returns:
            Tuple of (frame, timestamp) or None if no frame available
        """
        with self.buffer_lock:
            if len(self.frame_buffer) == 0:
                self.logger.debug(f"Frame buffer empty: camera_id={self.camera_id}")
                return None
            return self.frame_buffer[-1]
    
    def is_healthy(self) -> bool:
        """
        Check if camera is healthy.
        
        Returns:
            True if camera is connected and capturing frames
        """
        current_time = time.time()
        
        # Check if we've received frames recently
        if self.last_frame_time > 0:
            time_since_last_frame = current_time - self.last_frame_time
            if time_since_last_frame > 2.0:  # No frame for 2 seconds
                self.logger.warning(
                    f"Camera unhealthy - no frames: camera_id={self.camera_id}, "
                    f"time_since_last_frame={time_since_last_frame:.2f}s"
                )
                return False
        
        is_healthy = self.is_connected and self.is_running
        if not is_healthy:
            self.logger.debug(
                f"Camera unhealthy: camera_id={self.camera_id}, "
                f"is_connected={self.is_connected}, is_running={self.is_running}"
            )
        
        return is_healthy
    
    def _initialize_camera(self) -> bool:
        """Initialize camera device."""
        self.logger.debug(f"Initializing camera device: camera_id={self.camera_id}, device={self.device}")
        
        try:
            self.cap = cv2.VideoCapture(self.device)
            
            if not self.cap.isOpened():
                self.logger.error(
                    f"Failed to open camera device: camera_id={self.camera_id}, device={self.device}"
                )
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # Warn if settings don't match requested
            if (actual_width, actual_height) != self.resolution:
                self.logger.warning(
                    f"Camera resolution mismatch: camera_id={self.camera_id}, "
                    f"requested={self.resolution}, actual=({actual_width}, {actual_height})"
                )
            
            if actual_fps != self.fps:
                self.logger.warning(
                    f"Camera FPS mismatch: camera_id={self.camera_id}, "
                    f"requested={self.fps}, actual={actual_fps}"
                )
            
            self.logger.info(
                f"Camera initialized: camera_id={self.camera_id}, "
                f"resolution={actual_width}x{actual_height}, fps={actual_fps}"
            )
            
            self.is_connected = True
            self.consecutive_failures = 0
            return True
            
        except Exception as e:
            self.logger.error(
                f"Error initializing camera: camera_id={self.camera_id}, device={self.device}, error={e}",
                exc_info=True
            )
            return False
    
    def _release_camera(self) -> None:
        """Release camera resources."""
        if self.cap:
            self.logger.debug(f"Releasing camera: camera_id={self.camera_id}")
            self.cap.release()
            self.cap = None
        self.is_connected = False
        self.logger.debug(f"Camera released: camera_id={self.camera_id}")
    
    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        self.logger.debug(f"Capture loop started: camera_id={self.camera_id}")
        target_frame_time = 1.0 / self.fps
        
        while self.is_running:
            loop_start = time.time()
            
            # Capture frame
            success = self._capture_frame()
            
            if not success:
                self.consecutive_failures += 1
                
                if self.consecutive_failures == 1:
                    self.logger.warning(
                        f"Frame capture failed: camera_id={self.camera_id}, "
                        f"consecutive_failures={self.consecutive_failures}"
                    )
                
                # Check if we need to reconnect
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self.logger.warning(
                        f"Too many consecutive failures, attempting reconnection: "
                        f"camera_id={self.camera_id}, failures={self.consecutive_failures}"
                    )
                    self._attempt_reconnection()
                
                # Sleep before retry
                time.sleep(0.1)
                continue
            
            # Reset failure counter on success
            if self.consecutive_failures > 0:
                self.logger.info(
                    f"Frame capture recovered: camera_id={self.camera_id}, "
                    f"previous_failures={self.consecutive_failures}"
                )
            self.consecutive_failures = 0
            self.frames_since_stats += 1
            
            # Health check
            current_time = time.time()
            if current_time - self.last_health_check > self.health_check_interval:
                self._perform_health_check()
                self.last_health_check = current_time
            
            # Periodic statistics logging
            if current_time - self.last_stats_time > self.stats_interval:
                self._log_statistics(current_time)
                self.last_stats_time = current_time
                self.frames_since_stats = 0
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.logger.debug(f"Capture loop exited: camera_id={self.camera_id}")
    
    def _capture_frame(self) -> bool:
        """
        Capture a single frame.
        
        Returns:
            True if frame captured successfully, False otherwise
        """
        if not self.cap or not self.cap.isOpened():
            self.logger.debug(f"Camera not ready: camera_id={self.camera_id}")
            return False
        
        try:
            capture_start = time.time()
            ret, frame = self.cap.read()
            capture_duration = time.time() - capture_start
            
            if not ret or frame is None:
                self.logger.debug(
                    f"Failed to read frame: camera_id={self.camera_id}, ret={ret}, "
                    f"frame_is_none={frame is None}"
                )
                return False
            
            # Get timestamp
            timestamp = time.time()
            
            # Add to buffer
            with self.buffer_lock:
                self.frame_buffer.append((frame.copy(), timestamp))
            
            self.last_frame_time = timestamp
            self.frame_count += 1
            
            # Log slow captures (> 10ms is concerning for 30fps)
            if capture_duration > 0.010:
                self.logger.debug(
                    f"Slow frame capture: camera_id={self.camera_id}, "
                    f"duration={capture_duration*1000:.2f}ms"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(
                f"Error capturing frame: camera_id={self.camera_id}, error={e}",
                exc_info=True
            )
            return False
    
    def _perform_health_check(self) -> None:
        """Perform periodic health check."""
        if not self.is_connected:
            self.logger.debug(f"Health check - camera disconnected: camera_id={self.camera_id}")
            return
        
        # Check if camera is still accessible
        if self.cap and not self.cap.isOpened():
            self.logger.error(f"Camera device lost during health check: camera_id={self.camera_id}")
            self.is_connected = False
        else:
            self.logger.debug(
                f"Health check passed: camera_id={self.camera_id}, "
                f"frame_count={self.frame_count}, buffer_size={len(self.frame_buffer)}"
            )
    
    def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to camera."""
        self.logger.info(f"Attempting camera reconnection: camera_id={self.camera_id}")
        reconnect_start = time.time()
        
        # Release current camera
        self._release_camera()
        
        # Wait a bit
        time.sleep(1.0)
        
        # Try to reinitialize
        if self._initialize_camera():
            reconnect_duration = time.time() - reconnect_start
            self.logger.info(
                f"Camera reconnected successfully: camera_id={self.camera_id}, "
                f"duration={reconnect_duration:.3f}s"
            )
        else:
            self.logger.error(f"Camera reconnection failed: camera_id={self.camera_id}")
    
    def _log_statistics(self, current_time: float) -> None:
        """Log periodic statistics."""
        elapsed = current_time - self.last_stats_time
        actual_fps = self.frames_since_stats / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Camera statistics: camera_id={self.camera_id}, "
            f"total_frames={self.frame_count}, actual_fps={actual_fps:.1f}, "
            f"target_fps={self.fps}, buffer_size={len(self.frame_buffer)}/{self.buffer_size}, "
            f"consecutive_failures={self.consecutive_failures}"
        )
