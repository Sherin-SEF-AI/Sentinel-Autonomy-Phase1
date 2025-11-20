"""Scene graph builder for spatial representation of detected objects."""

import logging
import time
from typing import List, Dict, Any
import numpy as np

from src.core.data_structures import Detection3D

logger = logging.getLogger(__name__)


class SceneGraphBuilder:
    """Builds spatial representation of all detected objects and their relationships."""
    
    def __init__(self):
        """Initialize scene graph builder."""
        self.logger = logging.getLogger("SceneGraphBuilder")
        self.logger.info("SceneGraphBuilder initialized")
    
    def build(self, detections: List[Detection3D]) -> Dict[str, Any]:
        """
        Build scene graph from detected objects.
        
        Args:
            detections: List of 3D detections
            
        Returns:
            Scene graph containing objects and their spatial relationships
        """
        start_time = time.time()
        self.logger.debug(f"Scene graph build started: num_detections={len(detections)}")
        
        if not detections:
            self.logger.debug("Scene graph build completed: no detections, empty graph returned")
            return {
                'objects': [],
                'relationships': [],
                'spatial_map': {},
                'num_objects': 0
            }
        
        # Extract object information
        objects = []
        for det in detections:
            x, y, z, w, h, l, theta = det.bbox_3d
            obj = {
                'id': det.track_id,
                'type': det.class_name,
                'position': (x, y, z),
                'dimensions': (w, h, l),
                'orientation': theta,
                'velocity': det.velocity,
                'confidence': det.confidence
            }
            objects.append(obj)
        
        self.logger.debug(f"Objects extracted: count={len(objects)}")
        
        # Calculate spatial relationships
        relationships = self._calculate_relationships(objects)
        self.logger.debug(f"Spatial relationships calculated: count={len(relationships)}")
        
        # Create spatial map (grid-based representation)
        spatial_map = self._create_spatial_map(objects)
        occupied_cells = len(spatial_map.get('grid', {}))
        self.logger.debug(f"Spatial map created: occupied_cells={occupied_cells}")
        
        scene_graph = {
            'objects': objects,
            'relationships': relationships,
            'spatial_map': spatial_map,
            'num_objects': len(objects)
        }
        
        duration_ms = (time.time() - start_time) * 1000
        self.logger.debug(
            f"Scene graph build completed: num_objects={len(objects)}, "
            f"num_relationships={len(relationships)}, duration={duration_ms:.2f}ms"
        )
        
        # Warn if exceeding performance target (should be < 2ms for scene graph)
        if duration_ms > 2.0:
            self.logger.warning(
                f"Scene graph build exceeded target: duration={duration_ms:.2f}ms, target=2.0ms"
            )
        
        return scene_graph
    
    def _calculate_relationships(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate spatial relationships between objects.
        
        Args:
            objects: List of object dictionaries
            
        Returns:
            List of relationship dictionaries
        """
        self.logger.debug(f"Calculating relationships: num_objects={len(objects)}")
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                
                # Calculate distance between objects
                pos1 = np.array(obj1['position'])
                pos2 = np.array(obj2['position'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # Calculate relative position
                relative_pos = pos2 - pos1
                
                # Determine spatial relationship
                relationship = {
                    'object1_id': obj1['id'],
                    'object2_id': obj2['id'],
                    'distance': float(distance),
                    'relative_position': tuple(relative_pos.tolist()),
                    'proximity': self._categorize_proximity(distance)
                }
                
                relationships.append(relationship)
                
                # Log very close objects (potential collision risk)
                if distance < 2.0:
                    self.logger.debug(
                        f"Very close objects detected: obj1_id={obj1['id']}, "
                        f"obj2_id={obj2['id']}, distance={distance:.2f}m"
                    )
        
        return relationships
    
    def _categorize_proximity(self, distance: float) -> str:
        """
        Categorize proximity based on distance.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Proximity category
        """
        if distance < 2.0:
            return 'very_close'
        elif distance < 5.0:
            return 'close'
        elif distance < 10.0:
            return 'near'
        else:
            return 'far'
    
    def _create_spatial_map(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create grid-based spatial map of objects.
        
        Args:
            objects: List of object dictionaries
            
        Returns:
            Spatial map dictionary
        """
        self.logger.debug(f"Creating spatial map: num_objects={len(objects)}")
        
        # Define grid parameters (matching BEV: 0.1m per pixel, 64m x 64m area)
        grid_size = 64  # meters
        grid_resolution = 0.5  # meters per cell
        grid_cells = int(grid_size / grid_resolution)
        
        # Initialize grid
        grid = {}
        out_of_bounds_count = 0
        
        for obj in objects:
            x, y, z = obj['position']
            
            # Convert to grid coordinates (origin at vehicle center)
            grid_x = int((x + grid_size / 2) / grid_resolution)
            grid_y = int((y + grid_size / 2) / grid_resolution)
            
            # Check bounds
            if 0 <= grid_x < grid_cells and 0 <= grid_y < grid_cells:
                cell_key = f"{grid_x},{grid_y}"
                if cell_key not in grid:
                    grid[cell_key] = []
                grid[cell_key].append(obj['id'])
            else:
                out_of_bounds_count += 1
                self.logger.debug(
                    f"Object out of grid bounds: id={obj['id']}, "
                    f"position=({x:.2f}, {y:.2f}, {z:.2f}), "
                    f"grid_coords=({grid_x}, {grid_y})"
                )
        
        if out_of_bounds_count > 0:
            self.logger.debug(
                f"Objects outside grid bounds: count={out_of_bounds_count}/{len(objects)}"
            )
        
        spatial_map = {
            'grid': grid,
            'grid_size': grid_size,
            'grid_resolution': grid_resolution,
            'grid_cells': grid_cells
        }
        
        return spatial_map
