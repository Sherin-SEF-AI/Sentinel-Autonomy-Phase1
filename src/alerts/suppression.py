"""Alert suppression logic for SENTINEL system."""

import time
from typing import List, Dict, Any
from collections import defaultdict

from ..core.data_structures import Alert


class AlertSuppressor:
    """Manages alert suppression to prevent alert fatigue."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert suppressor.
        
        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        
        # Suppression settings
        self.duplicate_window = config.get('suppression', {}).get('duplicate_window', 5.0)
        self.max_simultaneous = config.get('suppression', {}).get('max_simultaneous', 2)
        
        # Track alert history per hazard
        # Key: hazard_id, Value: timestamp of last alert
        self.alert_history: Dict[int, float] = {}
        
        # Track currently active alerts
        self.active_alerts: List[Alert] = []
    
    def suppress_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        Apply suppression logic to filter alerts.
        
        Args:
            alerts: List of candidate alerts
            
        Returns:
            Filtered list of alerts after suppression
        """
        current_time = time.time()
        filtered_alerts = []
        
        # Clean up expired alerts from history
        self._cleanup_history(current_time)
        
        # Sort alerts by urgency (critical > warning > info)
        urgency_priority = {'critical': 3, 'warning': 2, 'info': 1}
        sorted_alerts = sorted(
            alerts,
            key=lambda a: urgency_priority.get(a.urgency, 0),
            reverse=True
        )
        
        for alert in sorted_alerts:
            # Check if this hazard was alerted recently
            if self._is_duplicate(alert, current_time):
                continue
            
            # Check if we've reached max simultaneous alerts
            if len(filtered_alerts) >= self.max_simultaneous:
                break
            
            # Add to filtered list and update history
            filtered_alerts.append(alert)
            self.alert_history[alert.hazard_id] = current_time
        
        # Update active alerts
        self.active_alerts = filtered_alerts
        
        return filtered_alerts
    
    def _is_duplicate(self, alert: Alert, current_time: float) -> bool:
        """
        Check if alert is a duplicate within suppression window.
        
        Args:
            alert: Alert to check
            current_time: Current timestamp
            
        Returns:
            True if alert should be suppressed as duplicate
        """
        if alert.hazard_id not in self.alert_history:
            return False
        
        last_alert_time = self.alert_history[alert.hazard_id]
        time_since_last = current_time - last_alert_time
        
        return time_since_last < self.duplicate_window
    
    def _cleanup_history(self, current_time: float) -> None:
        """
        Remove expired entries from alert history.
        
        Args:
            current_time: Current timestamp
        """
        expired_hazards = [
            hazard_id
            for hazard_id, timestamp in self.alert_history.items()
            if current_time - timestamp > self.duplicate_window
        ]
        
        for hazard_id in expired_hazards:
            del self.alert_history[hazard_id]
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get currently active alerts.
        
        Returns:
            List of active alerts
        """
        return self.active_alerts.copy()
    
    def clear_history(self) -> None:
        """Clear alert history (useful for testing or reset)."""
        self.alert_history.clear()
        self.active_alerts.clear()
