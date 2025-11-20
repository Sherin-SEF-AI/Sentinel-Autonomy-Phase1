"""Semantic segmentation implementation for BEV images."""

import logging
import time
import numpy as np
from typing import Optional

from ...core.interfaces import ISemanticSegmentor
from ...core.data_structures import SegmentationOutput
from .model import BEVSegmentationModel
from .smoother import TemporalSmoother


logger = logging.getLogger(__name__)


class SemanticSegmentor(ISemanticSegmentor):
    """
    Semantic segmentation for BEV images.
    
    Implements the ISemanticSegmentor interface with:
    - Deep learning model inference
    - FP16 precision for performance
    - Temporal smoothing for stability
    - Error recovery for robustness
    """
    
    def __init__(self, config: dict):
        """
        Initialize semantic segmentor.
        
        Args:
            config: Configuration dictionary with keys:
                - architecture: Model architecture name
                - weights: Path to model weights
                - device: Device to run on ('cuda' or 'cpu')
                - precision: Precision mode ('fp16' or 'fp32')
                - temporal_smoothing: Whether to enable temporal smoothing
                - smoothing_alpha: Alpha value for temporal smoothing
        """
        self.config = config
        
        # Extract configuration
        model_path = config.get('weights', 'models/bev_segmentation.pth')
        device = config.get('device', 'cuda')
        use_fp16 = config.get('precision', 'fp16') == 'fp16'
        self.use_temporal_smoothing = config.get('temporal_smoothing', True)
        smoothing_alpha = config.get('smoothing_alpha', 0.7)
        
        # Initialize model
        self.model = BEVSegmentationModel(
            model_path=model_path,
            device=device,
            use_fp16=use_fp16
        )
        
        # Initialize temporal smoother
        self.smoother = None
        if self.use_temporal_smoothing:
            self.smoother = TemporalSmoother(alpha=smoothing_alpha)
            logger.info("Temporal smoothing enabled")
        
        # Error recovery state
        self.error_count = 0
        self.max_errors = 3
        self.last_valid_output: Optional[SegmentationOutput] = None

        # Performance tracking
        self.inference_times = []
        self.target_inference_time = 0.015  # 15ms target
        self._slow_inference_count = 0  # Track slow inferences for rate-limited warnings

        logger.info("SemanticSegmentor initialized")
    
    def segment(self, bev_image: np.ndarray) -> SegmentationOutput:
        """
        Segment BEV image into semantic classes.
        
        Args:
            bev_image: BEV image (640, 640, 3) in BGR format, uint8
            
        Returns:
            SegmentationOutput with class_map and confidence
        """
        timestamp = time.time()
        start_time = time.perf_counter()
        
        try:
            # Validate input
            if bev_image is None or bev_image.size == 0:
                raise ValueError("Invalid BEV image")
            
            if bev_image.shape[:2] != (640, 640):
                logger.warning(
                    f"Expected BEV size (640, 640), got {bev_image.shape[:2]}"
                )
            
            # Run inference
            class_map, confidence = self.model.infer(bev_image)
            
            # Apply temporal smoothing if enabled
            if self.use_temporal_smoothing and self.smoother is not None:
                class_map, confidence = self.smoother.smooth(class_map, confidence)
            
            # Create output
            output = SegmentationOutput(
                timestamp=timestamp,
                class_map=class_map,
                confidence=confidence
            )
            
            # Store as last valid output for error recovery
            self.last_valid_output = output
            
            # Reset error count on success
            if self.error_count > 0:
                logger.info("Recovered from inference errors")
                self.error_count = 0
            
            # Track performance
            inference_time = time.perf_counter() - start_time
            self.inference_times.append(inference_time)

            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)

            # Log performance warning if too slow (rate-limited to avoid log spam)
            if inference_time > self.target_inference_time:
                self._slow_inference_count += 1
                # Only log every 100 slow inferences
                if self._slow_inference_count % 100 == 1:
                    avg_time = np.mean(self.inference_times) if self.inference_times else inference_time
                    logger.warning(
                        f"Segmentation inference time {inference_time*1000:.1f}ms "
                        f"exceeds target {self.target_inference_time*1000:.1f}ms "
                        f"(avg: {avg_time*1000:.1f}ms, count: {self._slow_inference_count})"
                    )
            
            return output
            
        except Exception as e:
            logger.error(f"Segmentation inference failed: {e}")
            return self._handle_error(timestamp)
    
    def _handle_error(self, timestamp: float) -> SegmentationOutput:
        """
        Handle inference error with recovery strategy.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            SegmentationOutput (either last valid or fallback)
        """
        self.error_count += 1
        
        # If we have a recent valid output, return it
        if self.last_valid_output is not None:
            logger.warning(
                f"Using last valid segmentation output (error count: {self.error_count})"
            )
            # Update timestamp but keep the segmentation
            return SegmentationOutput(
                timestamp=timestamp,
                class_map=self.last_valid_output.class_map.copy(),
                confidence=self.last_valid_output.confidence.copy()
            )
        
        # If too many errors, try to reload model
        if self.error_count >= self.max_errors:
            logger.error(
                f"Segmentation failed {self.error_count} times, attempting model reload"
            )
            try:
                self._reload_model()
                self.error_count = 0
            except Exception as e:
                logger.error(f"Model reload failed: {e}")
        
        # Return fallback output (all road class)
        logger.warning("Returning fallback segmentation output")
        return self._create_fallback_output(timestamp)
    
    def _reload_model(self):
        """Reload the segmentation model."""
        logger.info("Reloading segmentation model...")
        
        model_path = self.config.get('weights', 'models/bev_segmentation.pth')
        device = self.config.get('device', 'cuda')
        use_fp16 = self.config.get('precision', 'fp16') == 'fp16'
        
        # Clear GPU cache
        self.model.clear_cache()
        
        # Reinitialize model
        self.model = BEVSegmentationModel(
            model_path=model_path,
            device=device,
            use_fp16=use_fp16
        )
        
        logger.info("Model reloaded successfully")
    
    def _create_fallback_output(self, timestamp: float) -> SegmentationOutput:
        """
        Create fallback segmentation output.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            SegmentationOutput with default values
        """
        # Create output with all pixels as 'road' class (index 0)
        class_map = np.zeros((640, 640), dtype=np.int8)
        confidence = np.ones((640, 640), dtype=np.float32) * 0.5
        
        return SegmentationOutput(
            timestamp=timestamp,
            class_map=class_map,
            confidence=confidence
        )
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                'mean_inference_time': 0.0,
                'p95_inference_time': 0.0,
                'fps': 0.0
            }
        
        times = np.array(self.inference_times)
        mean_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        fps = 1.0 / mean_time if mean_time > 0 else 0.0
        
        return {
            'mean_inference_time': mean_time,
            'p95_inference_time': p95_time,
            'fps': fps,
            'target_met': p95_time <= self.target_inference_time
        }
    
    def reset(self):
        """Reset segmentor state."""
        if self.smoother is not None:
            self.smoother.reset()
        self.error_count = 0
        self.last_valid_output = None
        self.inference_times = []
        logger.info("SemanticSegmentor reset")
