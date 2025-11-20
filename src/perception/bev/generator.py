"""BEV Generator implementation integrating all BEV components."""

import numpy as np
import time
from typing import List, Dict, Any
import logging

from src.core.interfaces import IBEVGenerator
from src.core.data_structures import BEVOutput
from .transformer import PerspectiveTransformer
from .stitcher import ViewStitcher
from .mask_generator import MaskGenerator


class BEVGenerator(IBEVGenerator):
    """
    Generates bird's eye view from multiple camera perspectives.
    
    Integrates perspective transformation, view stitching, and mask generation
    to create a unified top-down view of the vehicle's surroundings.
    """
    
    def __init__(self, config: Dict[str, Any], calibrations: Dict[str, Dict[str, Any]]):
        """
        Initialize BEV generator.
        
        Args:
            config: BEV configuration from system config
            calibrations: Dictionary mapping camera names to calibration data
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Extract BEV parameters
        self.output_size = tuple(config.get('output_size', [640, 640]))
        self.scale = config.get('scale', 0.1)
        self.vehicle_position = tuple(config.get('vehicle_position', [320, 480]))
        self.blend_width = config.get('blend_width', 50)
        
        # Initialize transformers for each camera
        self.transformers = {}
        for camera_name, calibration in calibrations.items():
            self.transformers[camera_name] = PerspectiveTransformer(
                calibration, self.output_size
            )
        
        # Initialize stitcher
        self.stitcher = ViewStitcher(self.output_size, self.blend_width)
        
        # Initialize mask generator
        self.mask_generator = MaskGenerator(
            self.output_size,
            self.vehicle_position,
            scale=self.scale
        )
        
        # Performance tracking
        self.processing_times = []
        self.target_time_ms = 15.0
        
        self.logger.info(f"BEVGenerator initialized with output size {self.output_size}")
    
    def generate(self, frames: List[np.ndarray]) -> BEVOutput:
        """
        Transform camera views to BEV.
        
        Args:
            frames: List of camera frames (typically [front_left, front_right])
            
        Returns:
            BEVOutput with stitched BEV image and valid region mask
        """
        start_time = time.time()
        
        try:
            # Transform each frame to BEV
            bev_views = []
            camera_names = list(self.transformers.keys())
            
            for i, frame in enumerate(frames):
                if i < len(camera_names):
                    camera_name = camera_names[i]
                    transformer = self.transformers[camera_name]
                    bev_view = transformer.transform(frame)
                    bev_views.append(bev_view)
            
            # Stitch views together
            stitched_bev = self.stitcher.stitch(bev_views)
            
            # Generate valid region mask
            mask = self.mask_generator.generate(stitched_bev)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.processing_times.append(processing_time)
            
            # Keep only last 100 measurements
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # Log performance warning if exceeding target
            if processing_time > self.target_time_ms:
                self.logger.warning(
                    f"BEV generation took {processing_time:.2f}ms, "
                    f"exceeding target of {self.target_time_ms}ms"
                )
            
            return BEVOutput(
                timestamp=time.time(),
                image=stitched_bev,
                mask=mask
            )
            
        except Exception as e:
            self.logger.error(f"BEV generation failed: {e}", exc_info=True)
            # Return empty BEV on error
            return BEVOutput(
                timestamp=time.time(),
                image=np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8),
                mask=np.zeros((self.output_size[1], self.output_size[0]), dtype=bool)
            )
    
    def get_average_processing_time(self) -> float:
        """
        Get average processing time in milliseconds.
        
        Returns:
            Average processing time
        """
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get detailed performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {
                'avg_ms': 0.0,
                'min_ms': 0.0,
                'max_ms': 0.0,
                'p95_ms': 0.0
            }
        
        times = sorted(self.processing_times)
        p95_idx = int(len(times) * 0.95)
        
        return {
            'avg_ms': sum(times) / len(times),
            'min_ms': times[0],
            'max_ms': times[-1],
            'p95_ms': times[p95_idx] if p95_idx < len(times) else times[-1]
        }
