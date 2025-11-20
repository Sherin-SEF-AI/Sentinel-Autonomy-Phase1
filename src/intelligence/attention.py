"""Attention mapping for driver gaze to spatial zones."""

import logging
from typing import Dict, Any, List
import numpy as np

from src.core.data_structures import DriverState


class AttentionMapper:
    """Maps driver gaze to spatial zones around the vehicle."""
    
    # Zone boundaries in degrees (yaw angle from vehicle forward direction)
    ZONE_BOUNDARIES = {
        'front': (-30, 30),
        'front_left': (30, 75),
        'left': (75, 105),
        'rear_left': (105, 150),
        'rear': (150, 210),  # 150 to -150 (wraps around)
        'rear_right': (-150, -105),
        'right': (-105, -75),
        'front_right': (-75, -30)
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize attention mapper.
        
        Args:
            config: Configuration dictionary with zone_mapping settings
        """
        self.logger = logging.getLogger(__name__)
        self.num_zones = config.get('num_zones', 8)
        
        # Validate configuration
        if self.num_zones != 8:
            self.logger.warning(f"Only 8 zones supported, got {self.num_zones}")
            self.num_zones = 8
    
    def map_attention(self, driver_state: DriverState) -> Dict[str, Any]:
        """
        Map driver gaze to spatial zones.
        
        Args:
            driver_state: Current driver state with gaze information
            
        Returns:
            Attention map with attended zones and gaze information
        """
        if not driver_state.face_detected:
            return {
                'attended_zones': [],
                'primary_zone': None,
                'gaze_yaw': None,
                'gaze_pitch': None,
                'attention_valid': False
            }
        
        # Extract gaze angles
        gaze_yaw = driver_state.gaze.get('yaw', 0.0)
        gaze_pitch = driver_state.gaze.get('pitch', 0.0)
        
        # Determine which zone(s) the driver is looking at
        attended_zones = self._determine_zones(gaze_yaw)
        
        # Primary zone is the first one (most aligned)
        primary_zone = attended_zones[0] if attended_zones else None
        
        attention_map = {
            'attended_zones': attended_zones,
            'primary_zone': primary_zone,
            'gaze_yaw': float(gaze_yaw),
            'gaze_pitch': float(gaze_pitch),
            'attention_valid': True,
            'zone_boundaries': self.ZONE_BOUNDARIES
        }
        
        return attention_map
    
    def _determine_zones(self, gaze_yaw: float) -> List[str]:
        """
        Determine which zones the driver is looking at based on gaze yaw.
        
        Args:
            gaze_yaw: Gaze yaw angle in degrees
            
        Returns:
            List of zone names (primary zone first)
        """
        # Normalize yaw to [-180, 180]
        yaw = self._normalize_angle(gaze_yaw)
        
        attended_zones = []
        
        # Check each zone
        for zone_name, (min_angle, max_angle) in self.ZONE_BOUNDARIES.items():
            if self._is_in_zone(yaw, min_angle, max_angle):
                attended_zones.append(zone_name)
        
        return attended_zones
    
    def _is_in_zone(self, yaw: float, min_angle: float, max_angle: float) -> bool:
        """
        Check if yaw angle is within zone boundaries.
        
        Args:
            yaw: Yaw angle in degrees [-180, 180]
            min_angle: Minimum zone boundary
            max_angle: Maximum zone boundary
            
        Returns:
            True if yaw is in zone
        """
        # Handle wrap-around for rear zone
        if min_angle > max_angle:
            # Zone wraps around 180/-180 boundary
            return yaw >= min_angle or yaw <= max_angle
        else:
            return min_angle <= yaw <= max_angle
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-180, 180] range.
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Normalized angle
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def is_looking_at_zone(self, attention_map: Dict[str, Any], zone: str) -> bool:
        """
        Check if driver is looking at a specific zone.
        
        Args:
            attention_map: Attention map from map_attention()
            zone: Zone name to check
            
        Returns:
            True if driver is looking at the zone
        """
        if not attention_map.get('attention_valid', False):
            return False
        
        return zone in attention_map.get('attended_zones', [])
    
    def get_zone_for_position(self, position: tuple) -> str:
        """
        Get the zone for a given position relative to vehicle.
        
        Args:
            position: (x, y, z) position in vehicle frame
            
        Returns:
            Zone name
        """
        x, y, z = position
        
        # Calculate angle from vehicle forward direction
        # x is forward, y is left
        angle = np.degrees(np.arctan2(y, x))
        
        # Determine zone
        zones = self._determine_zones(angle)
        return zones[0] if zones else 'unknown'
