"""Alert logging for SENTINEL system."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from ..core.data_structures import Alert


class AlertLogger:
    """Logs alerts with timestamp and context for analysis."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize alert logger.
        
        Args:
            log_dir: Directory for alert logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up dedicated alert logger
        self.logger = logging.getLogger('sentinel.alerts')
        
        # Create alert-specific log file
        alert_log_file = self.log_dir / "alerts.log"
        
        # Check if handler already exists
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(alert_log_file) 
                   for h in self.logger.handlers):
            handler = logging.FileHandler(alert_log_file, mode='a')
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Alert history for analysis
        self.alert_history: List[Dict[str, Any]] = []
    
    def log_alert(self, alert: Alert, context: Dict[str, Any] = None) -> None:
        """
        Log an alert with timestamp and context.
        
        Args:
            alert: Alert to log
            context: Optional context information (driver state, risk scores, etc.)
        """
        # Create log entry
        log_entry = {
            'timestamp': alert.timestamp,
            'datetime': datetime.fromtimestamp(alert.timestamp).isoformat(),
            'urgency': alert.urgency,
            'message': alert.message,
            'hazard_id': alert.hazard_id,
            'modalities': alert.modalities,
            'dismissed': alert.dismissed
        }
        
        # Add context if provided
        if context:
            log_entry['context'] = context
        
        # Log to file
        self.logger.info(json.dumps(log_entry))
        
        # Store in history
        self.alert_history.append(log_entry)
    
    def log_alerts(self, alerts: List[Alert], context: Dict[str, Any] = None) -> None:
        """
        Log multiple alerts.
        
        Args:
            alerts: List of alerts to log
            context: Optional context information
        """
        for alert in alerts:
            self.log_alert(alert, context)
    
    def get_alert_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            limit: Optional limit on number of alerts to return (most recent)
            
        Returns:
            List of alert log entries
        """
        if limit is None:
            return self.alert_history.copy()
        return self.alert_history[-limit:]
    
    def get_alerts_by_urgency(self, urgency: str) -> List[Dict[str, Any]]:
        """
        Get alerts filtered by urgency level.
        
        Args:
            urgency: Urgency level ('info', 'warning', 'critical')
            
        Returns:
            List of alerts with specified urgency
        """
        return [
            alert for alert in self.alert_history
            if alert['urgency'] == urgency
        ]
    
    def get_alerts_by_hazard(self, hazard_id: int) -> List[Dict[str, Any]]:
        """
        Get all alerts for a specific hazard.
        
        Args:
            hazard_id: Hazard ID to filter by
            
        Returns:
            List of alerts for the specified hazard
        """
        return [
            alert for alert in self.alert_history
            if alert['hazard_id'] == hazard_id
        ]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        total = len(self.alert_history)
        
        if total == 0:
            return {
                'total': 0,
                'by_urgency': {},
                'unique_hazards': 0
            }
        
        # Count by urgency
        by_urgency = {}
        for alert in self.alert_history:
            urgency = alert['urgency']
            by_urgency[urgency] = by_urgency.get(urgency, 0) + 1
        
        # Count unique hazards
        unique_hazards = len(set(alert['hazard_id'] for alert in self.alert_history))
        
        return {
            'total': total,
            'by_urgency': by_urgency,
            'unique_hazards': unique_hazards
        }
    
    def clear_history(self) -> None:
        """Clear alert history (useful for testing)."""
        self.alert_history.clear()
