"""Alert generation logic for SENTINEL system."""

import time
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from ..core.data_structures import Alert, Risk, RiskAssessment, DriverState

logger = logging.getLogger(__name__)


class AlertGenerator:
    """Generates alerts based on risk assessment and driver state."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize alert generator.
        
        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        
        # Thresholds from config
        self.critical_threshold = config.get('escalation', {}).get('critical_threshold', 0.9)
        self.high_threshold = config.get('escalation', {}).get('high_threshold', 0.7)
        self.medium_threshold = config.get('escalation', {}).get('medium_threshold', 0.5)
        
        # Cognitive load threshold for INFO alerts
        self.cognitive_load_threshold = 0.7
        
        # Statistics
        self.alerts_generated = 0
        self.alerts_by_urgency = {'critical': 0, 'warning': 0, 'info': 0}
        
        logger.info(
            f"AlertGenerator initialized: critical_threshold={self.critical_threshold}, "
            f"high_threshold={self.high_threshold}, medium_threshold={self.medium_threshold}, "
            f"cognitive_load_threshold={self.cognitive_load_threshold}"
        )
    
    def generate_alerts(
        self,
        risks: RiskAssessment,
        driver: DriverState
    ) -> List[Alert]:
        """
        Generate alerts based on risk assessment and driver state.
        
        Args:
            risks: Risk assessment containing top risks
            driver: Current driver state
            
        Returns:
            List of generated alerts
        """
        start_time = time.time()
        alerts = []
        current_time = time.time()
        
        # Calculate cognitive load from driver readiness
        cognitive_load = 1.0 - (driver.readiness_score / 100.0)
        
        logger.debug(
            f"Alert generation started: num_risks={len(risks.top_risks)}, "
            f"driver_readiness={driver.readiness_score:.1f}, cognitive_load={cognitive_load:.2f}"
        )
        
        for risk in risks.top_risks:
            alert = self._generate_alert_for_risk(
                risk,
                driver,
                cognitive_load,
                current_time
            )
            
            if alert is not None:
                alerts.append(alert)
                self.alerts_generated += 1
                self.alerts_by_urgency[alert.urgency] += 1
                
                logger.info(
                    f"Alert generated: urgency={alert.urgency}, hazard_id={alert.hazard_id}, "
                    f"modalities={alert.modalities}, message='{alert.message}'"
                )
        
        duration_ms = (time.time() - start_time) * 1000
        logger.debug(
            f"Alert generation completed: alerts_generated={len(alerts)}, duration={duration_ms:.2f}ms"
        )
        
        # Log performance warning if exceeding target
        if duration_ms > 5.0:
            logger.warning(
                f"Alert generation exceeded target: duration={duration_ms:.2f}ms, target=5.0ms"
            )
        
        return alerts
    
    def _generate_alert_for_risk(
        self,
        risk: Risk,
        driver: DriverState,
        cognitive_load: float,
        timestamp: float
    ) -> Alert:
        """
        Generate an alert for a specific risk.
        
        Args:
            risk: Risk to generate alert for
            driver: Current driver state
            cognitive_load: Driver cognitive load (0-1)
            timestamp: Current timestamp
            
        Returns:
            Alert object or None if no alert should be generated
        """
        contextual_score = risk.contextual_score
        
        logger.debug(
            f"Evaluating risk: hazard_id={risk.hazard.object_id}, type={risk.hazard.type}, "
            f"contextual_score={contextual_score:.2f}, driver_aware={risk.driver_aware}, "
            f"ttc={risk.hazard.ttc:.2f}s"
        )
        
        # CRITICAL alerts: risk > 0.9
        if contextual_score > self.critical_threshold:
            logger.debug(
                f"Critical alert triggered: contextual_score={contextual_score:.2f} > "
                f"threshold={self.critical_threshold}"
            )
            return Alert(
                timestamp=timestamp,
                urgency='critical',
                modalities=['visual', 'audio', 'haptic'],
                message=self._format_message(risk, 'critical'),
                hazard_id=risk.hazard.object_id,
                dismissed=False
            )
        
        # WARNING alerts: risk > 0.7 when driver unaware
        elif contextual_score > self.high_threshold:
            if not risk.driver_aware:
                logger.debug(
                    f"Warning alert triggered: contextual_score={contextual_score:.2f} > "
                    f"threshold={self.high_threshold}, driver_aware=False"
                )
                return Alert(
                    timestamp=timestamp,
                    urgency='warning',
                    modalities=['visual', 'audio'],
                    message=self._format_message(risk, 'warning'),
                    hazard_id=risk.hazard.object_id,
                    dismissed=False
                )
            else:
                logger.debug(
                    f"Warning alert suppressed: driver_aware=True"
                )
        
        # INFO alerts: risk > 0.5 when cognitive load < 0.7
        elif contextual_score > self.medium_threshold:
            if cognitive_load < self.cognitive_load_threshold:
                logger.debug(
                    f"Info alert triggered: contextual_score={contextual_score:.2f} > "
                    f"threshold={self.medium_threshold}, cognitive_load={cognitive_load:.2f} < "
                    f"threshold={self.cognitive_load_threshold}"
                )
                return Alert(
                    timestamp=timestamp,
                    urgency='info',
                    modalities=['visual'],
                    message=self._format_message(risk, 'info'),
                    hazard_id=risk.hazard.object_id,
                    dismissed=False
                )
            else:
                logger.debug(
                    f"Info alert suppressed: cognitive_load={cognitive_load:.2f} >= "
                    f"threshold={self.cognitive_load_threshold}"
                )
        
        return None
    
    def _format_message(self, risk: Risk, urgency: str) -> str:
        """
        Format alert message based on risk and urgency.
        
        Args:
            risk: Risk object
            urgency: Alert urgency level
            
        Returns:
            Formatted alert message
        """
        hazard = risk.hazard
        
        if urgency == 'critical':
            message = f"CRITICAL: {hazard.type} ahead in {hazard.zone} zone! TTC: {hazard.ttc:.1f}s"
        elif urgency == 'warning':
            awareness = "not looking" if not risk.driver_aware else "attention needed"
            message = f"WARNING: {hazard.type} in {hazard.zone} zone ({awareness}). TTC: {hazard.ttc:.1f}s"
        else:  # info
            message = f"INFO: {hazard.type} detected in {hazard.zone} zone. TTC: {hazard.ttc:.1f}s"
        
        logger.debug(f"Alert message formatted: urgency={urgency}, message='{message}'")
        return message
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert generation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_alerts_generated': self.alerts_generated,
            'alerts_by_urgency': self.alerts_by_urgency.copy()
        }
