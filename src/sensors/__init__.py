"""Sensors module for SENTINEL."""

from .gps_tracker import GPSTracker, GPSData, SpeedLimitInfo

__all__ = [
    'GPSTracker',
    'GPSData',
    'SpeedLimitInfo'
]
