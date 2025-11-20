"""Timestamp synchronization for multi-camera frames."""

import numpy as np
from typing import Dict, Optional, Tuple
import logging


class TimestampSync:
    """Software-based timestamp synchronization for camera frames."""
    
    def __init__(self, tolerance_ms: float = 5.0):
        """
        Initialize timestamp synchronization.
        
        Args:
            tolerance_ms: Maximum allowed timestamp difference in milliseconds
        """
        self.tolerance_ms = tolerance_ms
        self.tolerance_sec = tolerance_ms / 1000.0
        
        self.logger = logging.getLogger("TimestampSync")
        
        # Statistics
        self.sync_attempts = 0
        self.sync_successes = 0
        self.frames_dropped = 0
    
    def synchronize(self, frames: Dict[int, Tuple[np.ndarray, float]]) -> Optional[Tuple[Dict[int, np.ndarray], float]]:
        """
        Synchronize frames from multiple cameras.
        
        Args:
            frames: Dictionary mapping camera_id to (frame, timestamp) tuple
        
        Returns:
            Tuple of (synchronized_frames_dict, reference_timestamp) or None if sync fails
        """
        self.sync_attempts += 1
        
        if not frames:
            return None
        
        # Extract timestamps
        timestamps = {cam_id: ts for cam_id, (_, ts) in frames.items()}
        
        # Find reference timestamp (median)
        ts_values = list(timestamps.values())
        reference_ts = np.median(ts_values)
        
        # Check if all timestamps are within tolerance
        synchronized_frames = {}
        max_deviation = 0.0
        
        for cam_id, (frame, ts) in frames.items():
            deviation = abs(ts - reference_ts)
            max_deviation = max(max_deviation, deviation)
            
            if deviation <= self.tolerance_sec:
                synchronized_frames[cam_id] = frame
            else:
                self.logger.debug(
                    f"Camera {cam_id} frame dropped: deviation={deviation*1000:.2f}ms "
                    f"(tolerance={self.tolerance_ms}ms)"
                )
                self.frames_dropped += 1
        
        # Check if we have all cameras synchronized
        if len(synchronized_frames) != len(frames):
            self.logger.warning(
                f"Synchronization failed: only {len(synchronized_frames)}/{len(frames)} "
                f"cameras within tolerance (max_deviation={max_deviation*1000:.2f}ms)"
            )
            return None
        
        self.sync_successes += 1
        
        if self.sync_attempts % 100 == 0:
            success_rate = (self.sync_successes / self.sync_attempts) * 100
            self.logger.debug(
                f"Sync stats: {self.sync_successes}/{self.sync_attempts} successful "
                f"({success_rate:.1f}%), {self.frames_dropped} frames dropped"
            )
        
        return synchronized_frames, reference_ts
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get synchronization statistics.
        
        Returns:
            Dictionary with sync statistics
        """
        return {
            'sync_attempts': self.sync_attempts,
            'sync_successes': self.sync_successes,
            'frames_dropped': self.frames_dropped,
            'success_rate': (self.sync_successes / self.sync_attempts * 100) if self.sync_attempts > 0 else 0.0
        }
    
    def reset_statistics(self) -> None:
        """Reset synchronization statistics."""
        self.sync_attempts = 0
        self.sync_successes = 0
        self.frames_dropped = 0
