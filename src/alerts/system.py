"""Alert system implementation for SENTINEL."""

import logging
from typing import List, Dict, Any

from ..core.interfaces import IAlertSystem
from ..core.data_structures import Alert, RiskAssessment, DriverState

from .generator import AlertGenerator
from .suppression import AlertSuppressor
from .logger import AlertLogger
from .dispatch import AlertDispatcher


class AlertSystem(IAlertSystem):
    """
    Complete alert system integrating generation, suppression, logging, and dispatch.
    
    Implements the IAlertSystem interface to provide context-aware safety alerts.
    """
    
    def __init__(self, config: Dict[str, Any], log_dir: str = "logs"):
        """
        Initialize alert system.
        
        Args:
            config: Alert configuration dictionary
            log_dir: Directory for alert logs
        """
        self.config = config
        self.logger = logging.getLogger('sentinel.alerts')
        
        # Initialize components
        self.generator = AlertGenerator(config)
        self.suppressor = AlertSuppressor(config)
        self.alert_logger = AlertLogger(log_dir)
        self.dispatcher = AlertDispatcher(config)
        
        self.logger.info("AlertSystem initialized")
    
    def process(self, risks: RiskAssessment, driver: DriverState) -> List[Alert]:
        """
        Generate and dispatch alerts based on risk assessment and driver state.
        
        This is the main interface method that:
        1. Generates alerts from risk assessment
        2. Applies suppression logic
        3. Logs alerts
        4. Dispatches through appropriate modalities
        
        Args:
            risks: Risk assessment containing top risks
            driver: Current driver state
            
        Returns:
            List of alerts that were generated and dispatched
        """
        # Step 1: Generate candidate alerts
        candidate_alerts = self.generator.generate_alerts(risks, driver)
        
        if not candidate_alerts:
            return []
        
        self.logger.debug(f"Generated {len(candidate_alerts)} candidate alerts")
        
        # Step 2: Apply suppression logic
        filtered_alerts = self.suppressor.suppress_alerts(candidate_alerts)
        
        if len(filtered_alerts) < len(candidate_alerts):
            suppressed_count = len(candidate_alerts) - len(filtered_alerts)
            self.logger.debug(f"Suppressed {suppressed_count} alerts")
        
        if not filtered_alerts:
            return []
        
        # Step 3: Log alerts with context
        context = self._build_context(risks, driver)
        self.alert_logger.log_alerts(filtered_alerts, context)
        
        # Step 4: Dispatch alerts
        self.dispatcher.dispatch_alerts(filtered_alerts)
        
        self.logger.info(f"Processed {len(filtered_alerts)} alerts")
        
        return filtered_alerts
    
    def _build_context(
        self,
        risks: RiskAssessment,
        driver: DriverState
    ) -> Dict[str, Any]:
        """
        Build context information for alert logging.
        
        Args:
            risks: Risk assessment
            driver: Driver state
            
        Returns:
            Context dictionary
        """
        return {
            'driver_readiness': driver.readiness_score,
            'driver_attention_zone': driver.gaze.get('attention_zone', 'unknown'),
            'num_hazards': len(risks.hazards),
            'num_top_risks': len(risks.top_risks),
            'face_detected': driver.face_detected,
            'drowsiness_score': driver.drowsiness.get('score', 0.0),
            'distraction_type': driver.distraction.get('type', 'none')
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get currently active alerts.
        
        Returns:
            List of active alerts
        """
        return self.suppressor.get_active_alerts()
    
    def get_active_visual_alerts(self) -> List[Dict[str, Any]]:
        """
        Get active visual alerts for rendering.
        
        Returns:
            List of visual alert data
        """
        return self.dispatcher.get_active_visual_alerts()
    
    def get_alert_history(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get alert history.
        
        Args:
            limit: Optional limit on number of alerts
            
        Returns:
            List of historical alerts
        """
        return self.alert_logger.get_alert_history(limit)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about alerts.
        
        Returns:
            Dictionary with alert statistics
        """
        return self.alert_logger.get_alert_statistics()
    
    def clear_history(self) -> None:
        """Clear alert history (useful for testing)."""
        self.suppressor.clear_history()
        self.alert_logger.clear_history()
        self.dispatcher.clear_visual_alerts()
        self.logger.info("Alert history cleared")
