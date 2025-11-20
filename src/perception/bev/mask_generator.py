"""Mask generation for BEV valid regions."""

import numpy as np
import cv2
from typing import Tuple
import logging


class MaskGenerator:
    """Generates valid region masks for BEV output."""
    
    def __init__(self, output_size: Tuple[int, int], vehicle_position: Tuple[int, int], 
                 vehicle_size: Tuple[float, float] = (2.0, 4.5), scale: float = 0.1):
        """
        Initialize mask generator.
        
        Args:
            output_size: BEV output size (width, height)
            vehicle_position: Vehicle position in BEV image (x, y)
            vehicle_size: Vehicle dimensions in meters (width, length)
            scale: Meters per pixel in BEV
        """
        self.logger = logging.getLogger(__name__)
        self.output_size = output_size
        self.vehicle_position = vehicle_position
        self.vehicle_size = vehicle_size
        self.scale = scale
        
        # Convert vehicle size to pixels
        self.vehicle_width_px = int(vehicle_size[0] / scale)
        self.vehicle_length_px = int(vehicle_size[1] / scale)
        
        # Pre-compute static vehicle mask
        self.vehicle_mask = self._create_vehicle_mask()
        
        self.logger.info(f"MaskGenerator initialized with output size {output_size}")
    
    def _create_vehicle_mask(self) -> np.ndarray:
        """
        Create mask for vehicle body region to exclude.
        
        Returns:
            Binary mask where 0 = vehicle body, 1 = valid region
        """
        mask = np.ones((self.output_size[1], self.output_size[0]), dtype=np.uint8)
        
        # Calculate vehicle rectangle
        vx, vy = self.vehicle_position
        half_width = self.vehicle_width_px // 2
        
        # Vehicle occupies region centered at vehicle_position
        x1 = max(0, vx - half_width)
        x2 = min(self.output_size[0], vx + half_width)
        y1 = max(0, vy - self.vehicle_length_px)
        y2 = min(self.output_size[1], vy)
        
        # Set vehicle region to 0 (invalid)
        mask[y1:y2, x1:x2] = 0
        
        return mask
    
    def _detect_sky_region(self, bev_image: np.ndarray) -> np.ndarray:
        """
        Detect sky regions in BEV (typically black/empty areas at edges).
        
        Args:
            bev_image: BEV image
            
        Returns:
            Binary mask where 0 = sky, 1 = valid region
        """
        # Convert to grayscale if needed
        if len(bev_image.shape) == 3:
            gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = bev_image
        
        # Threshold to find non-zero regions (valid content)
        _, valid_mask = cv2.threshold(gray, 1, 1, cv2.THRESH_BINARY)
        
        return valid_mask
    
    def generate(self, bev_image: np.ndarray) -> np.ndarray:
        """
        Generate complete valid region mask for BEV output.
        
        Args:
            bev_image: BEV image to generate mask for
            
        Returns:
            Binary mask (bool) where True = valid region, False = invalid
        """
        # Start with vehicle mask
        mask = self.vehicle_mask.copy()
        
        # Detect and exclude sky regions
        sky_mask = self._detect_sky_region(bev_image)
        
        # Combine masks (both must be valid)
        combined_mask = cv2.bitwise_and(mask, sky_mask)
        
        # Apply morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        return combined_mask.astype(bool)
    
    def apply_mask(self, bev_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply mask to BEV image.
        
        Args:
            bev_image: BEV image
            mask: Binary mask
            
        Returns:
            Masked BEV image
        """
        if len(bev_image.shape) == 3:
            # Expand mask to match channels
            mask_3ch = np.stack([mask] * 3, axis=2)
            return bev_image * mask_3ch
        else:
            return bev_image * mask
