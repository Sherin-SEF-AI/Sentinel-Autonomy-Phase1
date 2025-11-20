"""
Fleet manager for cloud-based fleet statistics and analytics.

Provides fleet-wide statistics, vehicle rankings, and trend visualization.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .api_client import CloudAPIClient


class FleetManager:
    """
    Manages fleet-wide statistics and analytics.
    
    Features:
    - Fetch fleet-wide statistics
    - Display aggregate metrics
    - Show vehicle rankings
    - Visualize trends
    """
    
    def __init__(self, api_client: CloudAPIClient):
        """
        Initialize fleet manager.
        
        Args:
            api_client: Cloud API client
        """
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)
        
        # Cached statistics
        self.cached_stats: Optional[Dict[str, Any]] = None
        self.cache_time: float = 0.0
        self.cache_ttl: float = 300.0  # 5 minutes
        
        self.logger.info("FleetManager initialized")
    
    def get_fleet_statistics(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get fleet-wide statistics.
        
        Args:
            force_refresh: Force refresh from server (ignore cache)
        
        Returns:
            Dictionary with fleet statistics or None on error
        """
        try:
            # Check cache
            import time
            now = time.time()
            
            if not force_refresh and self.cached_stats and (now - self.cache_time) < self.cache_ttl:
                return self.cached_stats
            
            # Fetch from cloud
            response = self.api_client.get(
                f'/fleet/{self.api_client.fleet_id}/statistics'
            )
            
            if not response.success:
                self.logger.error(f"Failed to fetch fleet statistics: {response.error}")
                return self.cached_stats  # Return cached data on error
            
            stats = response.data
            
            # Update cache
            self.cached_stats = stats
            self.cache_time = now
            
            self.logger.debug("Fetched fleet statistics")
            return stats
        
        except Exception as e:
            self.logger.error(f"Error fetching fleet statistics: {e}", exc_info=True)
            return self.cached_stats
    
    def get_vehicle_rankings(
        self,
        metric: str = 'safety_score',
        limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get vehicle rankings by metric.
        
        Args:
            metric: Metric to rank by ('safety_score', 'total_distance', 'alert_rate', etc.)
            limit: Maximum number of vehicles to return
        
        Returns:
            List of vehicle rankings or None on error
        """
        try:
            response = self.api_client.get(
                f'/fleet/{self.api_client.fleet_id}/rankings',
                params={
                    'metric': metric,
                    'limit': limit
                }
            )
            
            if not response.success:
                self.logger.error(f"Failed to fetch rankings: {response.error}")
                return None
            
            rankings = response.data.get('rankings', [])
            
            self.logger.debug(f"Fetched top {len(rankings)} vehicles by {metric}")
            return rankings
        
        except Exception as e:
            self.logger.error(f"Error fetching rankings: {e}", exc_info=True)
            return None
    
    def get_fleet_trends(
        self,
        metric: str = 'safety_score',
        days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Get fleet trend data over time.
        
        Args:
            metric: Metric to get trends for
            days: Number of days of historical data
        
        Returns:
            Dictionary with trend data or None on error
        """
        try:
            response = self.api_client.get(
                f'/fleet/{self.api_client.fleet_id}/trends',
                params={
                    'metric': metric,
                    'days': days
                }
            )
            
            if not response.success:
                self.logger.error(f"Failed to fetch trends: {response.error}")
                return None
            
            trends = response.data
            
            self.logger.debug(f"Fetched {metric} trends for {days} days")
            return trends
        
        except Exception as e:
            self.logger.error(f"Error fetching trends: {e}", exc_info=True)
            return None
    
    def get_vehicle_comparison(
        self,
        vehicle_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compare vehicles in the fleet.
        
        Args:
            vehicle_ids: List of vehicle IDs to compare (if None, compares all)
        
        Returns:
            Dictionary with comparison data or None on error
        """
        try:
            params = {}
            if vehicle_ids:
                params['vehicle_ids'] = ','.join(vehicle_ids)
            
            response = self.api_client.get(
                f'/fleet/{self.api_client.fleet_id}/comparison',
                params=params
            )
            
            if not response.success:
                self.logger.error(f"Failed to fetch comparison: {response.error}")
                return None
            
            comparison = response.data
            
            self.logger.debug(f"Fetched vehicle comparison")
            return comparison
        
        except Exception as e:
            self.logger.error(f"Error fetching comparison: {e}", exc_info=True)
            return None
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """
        Get aggregate fleet metrics.
        
        Returns:
            Dictionary with aggregate metrics
        """
        stats = self.get_fleet_statistics()
        
        if not stats:
            return {
                'total_vehicles': 0,
                'total_distance': 0.0,
                'total_trips': 0,
                'average_safety_score': 0.0,
                'total_alerts': 0
            }
        
        return {
            'total_vehicles': stats.get('total_vehicles', 0),
            'active_vehicles': stats.get('active_vehicles', 0),
            'total_distance': stats.get('total_distance', 0.0),
            'total_trips': stats.get('total_trips', 0),
            'average_safety_score': stats.get('average_safety_score', 0.0),
            'total_alerts': stats.get('total_alerts', 0),
            'critical_alerts': stats.get('critical_alerts', 0),
            'average_trip_duration': stats.get('average_trip_duration', 0.0)
        }
    
    def get_vehicle_status(self, vehicle_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get status for a specific vehicle.
        
        Args:
            vehicle_id: Vehicle ID (if None, uses current vehicle)
        
        Returns:
            Dictionary with vehicle status or None on error
        """
        if not vehicle_id:
            vehicle_id = self.api_client.vehicle_id
        
        try:
            response = self.api_client.get(f'/vehicles/{vehicle_id}/status')
            
            if not response.success:
                self.logger.error(f"Failed to fetch vehicle status: {response.error}")
                return None
            
            status = response.data
            
            self.logger.debug(f"Fetched status for vehicle {vehicle_id}")
            return status
        
        except Exception as e:
            self.logger.error(f"Error fetching vehicle status: {e}", exc_info=True)
            return None
    
    def get_fleet_alerts(
        self,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get recent alerts across the fleet.
        
        Args:
            severity: Filter by severity ('critical', 'warning', 'info')
            limit: Maximum number of alerts to return
        
        Returns:
            List of alerts or None on error
        """
        try:
            params = {'limit': limit}
            if severity:
                params['severity'] = severity
            
            response = self.api_client.get(
                f'/fleet/{self.api_client.fleet_id}/alerts',
                params=params
            )
            
            if not response.success:
                self.logger.error(f"Failed to fetch fleet alerts: {response.error}")
                return None
            
            alerts = response.data.get('alerts', [])
            
            self.logger.debug(f"Fetched {len(alerts)} fleet alerts")
            return alerts
        
        except Exception as e:
            self.logger.error(f"Error fetching fleet alerts: {e}", exc_info=True)
            return None
    
    def get_driver_leaderboard(
        self,
        metric: str = 'safety_score',
        limit: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get driver leaderboard.
        
        Args:
            metric: Metric to rank by
            limit: Maximum number of drivers to return
        
        Returns:
            List of driver rankings or None on error
        """
        try:
            response = self.api_client.get(
                f'/fleet/{self.api_client.fleet_id}/drivers/leaderboard',
                params={
                    'metric': metric,
                    'limit': limit
                }
            )
            
            if not response.success:
                self.logger.error(f"Failed to fetch leaderboard: {response.error}")
                return None
            
            leaderboard = response.data.get('leaderboard', [])
            
            self.logger.debug(f"Fetched driver leaderboard")
            return leaderboard
        
        except Exception as e:
            self.logger.error(f"Error fetching leaderboard: {e}", exc_info=True)
            return None
    
    def format_statistics_summary(self) -> str:
        """
        Format fleet statistics as human-readable summary.
        
        Returns:
            Formatted summary string
        """
        metrics = self.get_aggregate_metrics()
        
        summary = f"""
Fleet Statistics Summary
========================
Total Vehicles: {metrics['total_vehicles']}
Active Vehicles: {metrics.get('active_vehicles', 'N/A')}
Total Distance: {metrics['total_distance'] / 1000:.1f} km
Total Trips: {metrics['total_trips']}
Average Safety Score: {metrics['average_safety_score']:.1f}
Total Alerts: {metrics['total_alerts']}
Critical Alerts: {metrics.get('critical_alerts', 'N/A')}
Average Trip Duration: {metrics.get('average_trip_duration', 0) / 60:.1f} minutes
"""
        return summary.strip()
    
    def clear_cache(self) -> None:
        """Clear cached statistics"""
        self.cached_stats = None
        self.cache_time = 0.0
        self.logger.debug("Cleared statistics cache")
