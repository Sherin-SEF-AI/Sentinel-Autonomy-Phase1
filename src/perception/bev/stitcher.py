"""Multi-view stitching for BEV generation."""

import numpy as np
import cv2
from typing import List, Tuple
import logging


class ViewStitcher:
    """Stitches multiple BEV views using multi-band blending."""
    
    def __init__(self, output_size: Tuple[int, int], blend_width: int = 50):
        """
        Initialize view stitcher.
        
        Args:
            output_size: Final BEV output size (width, height)
            blend_width: Width of blending region in pixels
        """
        self.logger = logging.getLogger(__name__)
        self.output_size = output_size
        self.blend_width = blend_width
        self.num_levels = 4  # Pyramid levels for multi-band blending
        
        self.logger.info(f"ViewStitcher initialized with output size {output_size}, blend width {blend_width}")
    
    def _create_laplacian_pyramid(self, image: np.ndarray, levels: int) -> List[np.ndarray]:
        """
        Create Laplacian pyramid for multi-band blending.
        
        Args:
            image: Input image
            levels: Number of pyramid levels
            
        Returns:
            List of Laplacian pyramid levels
        """
        gaussian_pyramid = [image]
        for i in range(levels - 1):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image)
        
        laplacian_pyramid = []
        for i in range(levels - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)
        
        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid
    
    def _reconstruct_from_pyramid(self, pyramid: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct image from Laplacian pyramid.
        
        Args:
            pyramid: Laplacian pyramid levels
            
        Returns:
            Reconstructed image
        """
        image = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[0])
            image = cv2.pyrUp(image, dstsize=size)
            image = cv2.add(image, pyramid[i])
        return image
    
    def _identify_overlap_regions(self, views: List[np.ndarray]) -> List[np.ndarray]:
        """
        Identify overlapping regions between views.
        
        Args:
            views: List of BEV views
            
        Returns:
            List of binary masks indicating valid regions for each view
        """
        masks = []
        for view in views:
            # Create mask where view has non-zero content
            if len(view.shape) == 3:
                mask = np.any(view > 0, axis=2).astype(np.uint8) * 255
            else:
                mask = (view > 0).astype(np.uint8) * 255
            masks.append(mask)
        
        return masks
    
    def _create_blend_mask(self, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create blending masks for two overlapping views.
        
        Args:
            mask1: Binary mask for first view
            mask2: Binary mask for second view
            
        Returns:
            Tuple of (weight_mask1, weight_mask2) for blending
        """
        # Find overlap region
        overlap = cv2.bitwise_and(mask1, mask2)
        
        if np.sum(overlap) == 0:
            # No overlap, use original masks
            return mask1.astype(np.float32) / 255.0, mask2.astype(np.float32) / 255.0
        
        # Create distance transforms
        dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
        dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
        
        # Normalize distances in overlap region
        overlap_float = overlap.astype(np.float32) / 255.0
        total_dist = dist1 + dist2 + 1e-6
        
        weight1 = np.where(overlap_float > 0, dist1 / total_dist, mask1.astype(np.float32) / 255.0)
        weight2 = np.where(overlap_float > 0, dist2 / total_dist, mask2.astype(np.float32) / 255.0)
        
        return weight1, weight2
    
    def _multiband_blend(self, view1: np.ndarray, view2: np.ndarray, 
                         mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Blend two views using multi-band blending.
        
        Args:
            view1: First BEV view
            view2: Second BEV view
            mask1: Mask for first view
            mask2: Mask for second view
            
        Returns:
            Blended result
        """
        # Create blend weights
        weight1, weight2 = self._create_blend_mask(mask1, mask2)
        
        # Expand weights to match image channels
        if len(view1.shape) == 3:
            weight1 = np.expand_dims(weight1, axis=2)
            weight2 = np.expand_dims(weight2, axis=2)
        
        # Create Laplacian pyramids
        pyramid1 = self._create_laplacian_pyramid(view1.astype(np.float32), self.num_levels)
        pyramid2 = self._create_laplacian_pyramid(view2.astype(np.float32), self.num_levels)
        
        # Create weight pyramids
        weight_pyramid1 = self._create_laplacian_pyramid(weight1, self.num_levels)
        weight_pyramid2 = self._create_laplacian_pyramid(weight2, self.num_levels)
        
        # Blend pyramids
        blended_pyramid = []
        for i in range(self.num_levels):
            # Resize weights to match pyramid level
            w1 = cv2.resize(weight1, (pyramid1[i].shape[1], pyramid1[i].shape[0]))
            w2 = cv2.resize(weight2, (pyramid2[i].shape[1], pyramid2[i].shape[0]))
            
            if len(pyramid1[i].shape) == 3 and len(w1.shape) == 2:
                w1 = np.expand_dims(w1, axis=2)
                w2 = np.expand_dims(w2, axis=2)
            
            blended = pyramid1[i] * w1 + pyramid2[i] * w2
            blended_pyramid.append(blended)
        
        # Reconstruct
        result = self._reconstruct_from_pyramid(blended_pyramid)
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def stitch(self, views: List[np.ndarray]) -> np.ndarray:
        """
        Stitch multiple BEV views into a single composite image.
        
        Args:
            views: List of BEV views to stitch
            
        Returns:
            Stitched BEV image at output_size resolution
        """
        if len(views) == 0:
            return np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        
        if len(views) == 1:
            # Single view, just resize if needed
            if views[0].shape[:2] != (self.output_size[1], self.output_size[0]):
                return cv2.resize(views[0], self.output_size)
            return views[0]
        
        # Identify overlap regions
        masks = self._identify_overlap_regions(views)
        
        # Start with first view
        result = views[0].copy()
        result_mask = masks[0].copy()
        
        # Progressively blend additional views
        for i in range(1, len(views)):
            result = self._multiband_blend(result, views[i], result_mask, masks[i])
            result_mask = cv2.bitwise_or(result_mask, masks[i])
        
        # Ensure output is correct size
        if result.shape[:2] != (self.output_size[1], self.output_size[0]):
            result = cv2.resize(result, self.output_size)
        
        return result
