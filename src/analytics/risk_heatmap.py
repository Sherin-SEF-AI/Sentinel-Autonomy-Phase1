"""
Risk Heatmap Generator

Generates spatial risk heatmaps by aggregating risk scores by location.
Visualizes risk distribution and exports heatmap images.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskHeatmap:
    """
    Generates spatial risk heatmaps.
    
    Capabilities:
    - Aggregate risk by location
    - Generate heatmap visualization
    - Export heatmap images
    - Support multiple resolution levels
    """
    
    def __init__(self, config: dict):
        """
        Initialize risk heatmap generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Heatmap parameters
        heatmap_config = config.get('analytics', {}).get('heatmap', {})
        self.grid_size = heatmap_config.get('grid_size', 2.0)  # meters per cell
        self.max_range = heatmap_config.get('max_range', 100.0)  # meters
        self.decay_factor = heatmap_config.get('decay_factor', 0.95)  # temporal decay
        
        # Calculate grid dimensions
        self.grid_width = int(2 * self.max_range / self.grid_size)
        self.grid_height = int(2 * self.max_range / self.grid_size)
        
        # Risk accumulation grid
        self.risk_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.count_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        
        # Location-based risk history
        self.location_risks: Dict[Tuple[int, int], List[float]] = defaultdict(list)
        
        logger.info(f"RiskHeatmap initialized: grid={self.grid_width}x{self.grid_height}, "
                   f"cell_size={self.grid_size}m")
    
    def add_risk_point(self,
                       position: Tuple[float, float],
                       risk_score: float,
                       radius: float = 5.0):
        """
        Add a risk data point to the heatmap.
        
        Args:
            position: Position (x, y) in meters
            risk_score: Risk score (0-1)
            radius: Influence radius in meters
        """
        # Convert position to grid coordinates
        grid_x, grid_y = self._position_to_grid(position)
        
        if not self._is_valid_grid(grid_x, grid_y):
            return
        
        # Calculate influence radius in grid cells
        radius_cells = int(radius / self.grid_size)
        
        # Add risk to surrounding cells with Gaussian falloff
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                
                if not self._is_valid_grid(gx, gy):
                    continue
                
                # Calculate distance-based weight
                distance = np.sqrt(dx**2 + dy**2) * self.grid_size
                if distance > radius:
                    continue
                
                weight = np.exp(-(distance**2) / (2 * (radius/2)**2))
                weighted_risk = risk_score * weight
                
                # Accumulate risk
                self.risk_grid[gy, gx] += weighted_risk
                self.count_grid[gy, gx] += 1
                
                # Store in history
                self.location_risks[(gx, gy)].append(weighted_risk)
    
    def add_trajectory_risk(self,
                           trajectory: List[Tuple[float, float]],
                           risk_scores: List[float]):
        """
        Add risk along a trajectory.
        
        Args:
            trajectory: List of (x, y) positions
            risk_scores: Risk score for each position
        """
        if len(trajectory) != len(risk_scores):
            logger.warning("Trajectory and risk_scores length mismatch")
            return
        
        for pos, risk in zip(trajectory, risk_scores):
            self.add_risk_point(pos, risk, radius=3.0)
    
    def get_heatmap(self, normalize: bool = True) -> np.ndarray:
        """
        Get the current risk heatmap.
        
        Args:
            normalize: Whether to normalize to 0-1 range
        
        Returns:
            Heatmap array (grid_height, grid_width)
        """
        # Calculate average risk per cell
        heatmap = np.zeros_like(self.risk_grid)
        mask = self.count_grid > 0
        heatmap[mask] = self.risk_grid[mask] / self.count_grid[mask]
        
        if normalize and np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def get_heatmap_colored(self, colormap: str = 'hot') -> np.ndarray:
        """
        Get colored heatmap visualization.
        
        Args:
            colormap: Colormap name ('hot', 'jet', 'viridis')
        
        Returns:
            RGB image array (grid_height, grid_width, 3)
        """
        heatmap = self.get_heatmap(normalize=True)
        
        # Apply colormap
        if colormap == 'hot':
            colored = self._apply_hot_colormap(heatmap)
        elif colormap == 'jet':
            colored = self._apply_jet_colormap(heatmap)
        elif colormap == 'viridis':
            colored = self._apply_viridis_colormap(heatmap)
        else:
            # Default to grayscale
            colored = np.stack([heatmap] * 3, axis=-1)
        
        return (colored * 255).astype(np.uint8)
    
    def _apply_hot_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply hot colormap (black -> red -> yellow -> white)."""
        colored = np.zeros((*heatmap.shape, 3), dtype=np.float32)
        
        # Red channel: increases from 0 to 1
        colored[:, :, 0] = np.clip(heatmap * 3, 0, 1)
        
        # Green channel: increases after red saturates
        colored[:, :, 1] = np.clip((heatmap - 0.33) * 3, 0, 1)
        
        # Blue channel: increases after green saturates
        colored[:, :, 2] = np.clip((heatmap - 0.66) * 3, 0, 1)
        
        return colored
    
    def _apply_jet_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply jet colormap (blue -> cyan -> green -> yellow -> red)."""
        colored = np.zeros((*heatmap.shape, 3), dtype=np.float32)
        
        # Blue channel
        colored[:, :, 2] = np.where(heatmap < 0.5,
                                     1.0 - heatmap * 2,
                                     0.0)
        
        # Green channel
        colored[:, :, 1] = np.where(heatmap < 0.5,
                                     heatmap * 2,
                                     1.0 - (heatmap - 0.5) * 2)
        
        # Red channel
        colored[:, :, 0] = np.where(heatmap > 0.5,
                                     (heatmap - 0.5) * 2,
                                     0.0)
        
        return colored
    
    def _apply_viridis_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply viridis-like colormap (purple -> blue -> green -> yellow)."""
        colored = np.zeros((*heatmap.shape, 3), dtype=np.float32)
        
        # Simplified viridis approximation
        t = heatmap
        
        colored[:, :, 0] = 0.267 + 0.005 * t + 0.322 * t**2  # R
        colored[:, :, 1] = 0.005 + 0.503 * t + 0.206 * t**2  # G
        colored[:, :, 2] = 0.329 + 0.718 * t - 0.670 * t**2  # B
        
        return np.clip(colored, 0, 1)
    
    def get_high_risk_locations(self, threshold: float = 0.7) -> List[Tuple[float, float, float]]:
        """
        Get locations with high risk scores.
        
        Args:
            threshold: Risk threshold (0-1)
        
        Returns:
            List of (x, y, risk) tuples
        """
        heatmap = self.get_heatmap(normalize=False)
        high_risk_cells = np.where(heatmap > threshold)
        
        locations = []
        for gy, gx in zip(high_risk_cells[0], high_risk_cells[1]):
            position = self._grid_to_position(gx, gy)
            risk = heatmap[gy, gx]
            locations.append((position[0], position[1], risk))
        
        return locations
    
    def export_heatmap_image(self, filepath: str, colormap: str = 'hot'):
        """
        Export heatmap as image file.
        
        Args:
            filepath: Output file path (PNG)
            colormap: Colormap to use
        """
        try:
            import cv2
            
            # Get colored heatmap
            heatmap_img = self.get_heatmap_colored(colormap)
            
            # Flip vertically for correct orientation
            heatmap_img = np.flipud(heatmap_img)
            
            # Convert RGB to BGR for OpenCV
            heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(filepath, heatmap_img)
            logger.info(f"Heatmap exported to {filepath}")
            
        except ImportError:
            logger.error("OpenCV not available for image export")
        except Exception as e:
            logger.error(f"Failed to export heatmap: {e}")
    
    def overlay_on_map(self,
                       map_image: np.ndarray,
                       alpha: float = 0.5,
                       colormap: str = 'hot') -> np.ndarray:
        """
        Overlay heatmap on a map image.
        
        Args:
            map_image: Background map image (H, W, 3)
            alpha: Transparency of heatmap (0-1)
            colormap: Colormap to use
        
        Returns:
            Overlaid image
        """
        try:
            import cv2
            
            # Get colored heatmap
            heatmap_img = self.get_heatmap_colored(colormap)
            
            # Resize heatmap to match map image
            if map_image.shape[:2] != heatmap_img.shape[:2]:
                heatmap_img = cv2.resize(heatmap_img, 
                                        (map_image.shape[1], map_image.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
            
            # Blend images
            overlaid = cv2.addWeighted(map_image, 1 - alpha, heatmap_img, alpha, 0)
            
            return overlaid
            
        except ImportError:
            logger.error("OpenCV not available for overlay")
            return map_image
        except Exception as e:
            logger.error(f"Failed to overlay heatmap: {e}")
            return map_image
    
    def apply_temporal_decay(self):
        """Apply temporal decay to risk values."""
        self.risk_grid *= self.decay_factor
        
        # Remove very small values
        self.risk_grid[self.risk_grid < 0.01] = 0
        
        # Update counts
        self.count_grid[self.risk_grid == 0] = 0
    
    def clear(self):
        """Clear all risk data."""
        self.risk_grid.fill(0)
        self.count_grid.fill(0)
        self.location_risks.clear()
        logger.info("Risk heatmap cleared")
    
    def get_statistics(self) -> Dict:
        """
        Get heatmap statistics.
        
        Returns:
            Dictionary with statistics
        """
        heatmap = self.get_heatmap(normalize=False)
        
        non_zero = heatmap[heatmap > 0]
        
        if len(non_zero) == 0:
            return {
                'total_cells': 0,
                'max_risk': 0.0,
                'mean_risk': 0.0,
                'high_risk_cells': 0
            }
        
        return {
            'total_cells': len(non_zero),
            'max_risk': float(np.max(non_zero)),
            'mean_risk': float(np.mean(non_zero)),
            'std_risk': float(np.std(non_zero)),
            'high_risk_cells': int(np.sum(non_zero > 0.7))
        }
    
    def _position_to_grid(self, position: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world position to grid coordinates."""
        x, y = position
        
        # Center grid at origin
        grid_x = int((x + self.max_range) / self.grid_size)
        grid_y = int((y + self.max_range) / self.grid_size)
        
        return grid_x, grid_y
    
    def _grid_to_position(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world position."""
        x = grid_x * self.grid_size - self.max_range + self.grid_size / 2
        y = grid_y * self.grid_size - self.max_range + self.grid_size / 2
        
        return x, y
    
    def _is_valid_grid(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid coordinates are valid."""
        return (0 <= grid_x < self.grid_width and 
                0 <= grid_y < self.grid_height)
