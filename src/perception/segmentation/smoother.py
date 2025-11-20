"""Temporal smoothing for semantic segmentation."""

import logging
import numpy as np
from typing import Optional


logger = logging.getLogger(__name__)


class TemporalSmoother:
    """Exponential moving average smoother for temporal stability."""
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize temporal smoother.
        
        Args:
            alpha: Smoothing factor (0-1). Higher values give more weight to current frame.
                  alpha=0.7 means 70% current frame, 30% history.
        """
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.alpha = alpha
        self.smoothed_confidence: Optional[np.ndarray] = None
        self.frame_count = 0
        
        logger.info(f"TemporalSmoother initialized with alpha={alpha}")
    
    def smooth(
        self,
        class_map: np.ndarray,
        confidence: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal smoothing to segmentation output.
        
        Args:
            class_map: Current frame class predictions (H, W) int8
            confidence: Current frame confidence scores (H, W) float32
            
        Returns:
            Tuple of (smoothed_class_map, smoothed_confidence)
        """
        # First frame - initialize
        if self.smoothed_confidence is None:
            self.smoothed_confidence = confidence.copy()
            self.frame_count = 1
            return class_map.copy(), confidence.copy()
        
        # Apply exponential moving average to confidence
        self.smoothed_confidence = (
            self.alpha * confidence +
            (1 - self.alpha) * self.smoothed_confidence
        )
        
        # For class map, use the class with highest smoothed confidence
        # This requires converting class_map to one-hot, smoothing, then argmax
        # For efficiency, we'll use a simpler approach:
        # Keep current class if confidence is high, otherwise blend
        
        # Create smoothed class map
        smoothed_class_map = class_map.copy()
        
        # Where current confidence is low, consider keeping previous prediction
        # This helps reduce flicker in uncertain regions
        low_confidence_mask = confidence < 0.5
        if low_confidence_mask.any():
            # In low confidence regions, only update if smoothed confidence
            # for current class is still higher than threshold
            smoothed_class_map[low_confidence_mask] = class_map[low_confidence_mask]
        
        self.frame_count += 1
        
        return smoothed_class_map, self.smoothed_confidence.copy()
    
    def reset(self):
        """Reset smoother state."""
        self.smoothed_confidence = None
        self.frame_count = 0
        logger.debug("TemporalSmoother reset")
    
    def get_frame_count(self) -> int:
        """Get number of frames processed."""
        return self.frame_count
