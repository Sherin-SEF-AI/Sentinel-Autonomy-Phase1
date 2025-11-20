"""Recording trigger logic for SENTINEL system."""

import logging
from typing import Dict, Any, List
from dataclasses import dataclass

from ..core.data_structures import RiskAssessment, DriverState, Alert


@dataclass
class TriggerEvent:
    """Recording trigger event."""
    timestamp: float
    trigger_type: str  # 'high_risk', 'distraction_hazard', 'intervention', 'low_ttc'
    reason: str
    metadata: Dict[str, Any]


class RecordingTrigger:
    """
    Determines when to trigger scenario recording based on system state.
    
    Triggers on:
    - Risk score > 0.7
    - Driver distraction during hazard
    - System intervention (alert generated)
    - TTC < 1.5 seconds
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize recording trigger.
        
        Args:
            config: Recording configuration with trigger thresholds
        """
        self.logger = logging.getLogger(__name__)
        
        # Load trigger thresholds from config
        self.risk_threshold = config.get('triggers', {}).get('risk_threshold', 0.7)
        self.ttc_threshold = config.get('triggers', {}).get('ttc_threshold', 1.5)
        
        self.logger.info(
            f"RecordingTrigger initialized - "
            f"risk_threshold={self.risk_threshold}, "
            f"ttc_threshold={self.ttc_threshold}"
        )
    
    def check_triggers(
        self,
        timestamp: float,
        risk_assessment: RiskAssessment,
        driver_state: DriverState,
        alerts: List[Alert]
    ) -> List[TriggerEvent]:
        """
        Check if any recording triggers are activated.
        
        Args:
            timestamp: Current timestamp
            risk_assessment: Current risk assessment
            driver_state: Current driver state
            alerts: Generated alerts
            
        Returns:
            List of triggered events (empty if no triggers)
        """
        triggers = []
        
        # Trigger 1: High risk score (> 0.7)
        for risk in risk_assessment.top_risks:
            if risk.contextual_score > self.risk_threshold:
                triggers.append(TriggerEvent(
                    timestamp=timestamp,
                    trigger_type='high_risk',
                    reason=f"High contextual risk score: {risk.contextual_score:.2f}",
                    metadata={
                        'risk_score': risk.contextual_score,
                        'hazard_type': risk.hazard.type,
                        'hazard_id': risk.hazard.object_id,
                        'urgency': risk.urgency
                    }
                ))
                self.logger.debug(
                    f"High risk trigger: score={risk.contextual_score:.2f}, "
                    f"hazard={risk.hazard.type}"
                )
        
        # Trigger 2: Driver distraction during hazard
        if len(risk_assessment.hazards) > 0:
            distraction = driver_state.distraction
            if distraction.get('type') != 'none' and distraction.get('duration', 0) > 0:
                triggers.append(TriggerEvent(
                    timestamp=timestamp,
                    trigger_type='distraction_hazard',
                    reason=f"Driver distracted during hazard: {distraction.get('type')}",
                    metadata={
                        'distraction_type': distraction.get('type'),
                        'distraction_duration': distraction.get('duration'),
                        'num_hazards': len(risk_assessment.hazards),
                        'readiness_score': driver_state.readiness_score
                    }
                ))
                self.logger.debug(
                    f"Distraction hazard trigger: type={distraction.get('type')}, "
                    f"hazards={len(risk_assessment.hazards)}"
                )
        
        # Trigger 3: System intervention (alert generated)
        for alert in alerts:
            if not alert.dismissed:
                triggers.append(TriggerEvent(
                    timestamp=timestamp,
                    trigger_type='intervention',
                    reason=f"System intervention: {alert.urgency} alert",
                    metadata={
                        'alert_urgency': alert.urgency,
                        'alert_message': alert.message,
                        'hazard_id': alert.hazard_id,
                        'modalities': alert.modalities
                    }
                ))
                self.logger.debug(
                    f"Intervention trigger: urgency={alert.urgency}, "
                    f"message={alert.message}"
                )
        
        # Trigger 4: Low TTC (< 1.5 seconds)
        for hazard in risk_assessment.hazards:
            if hazard.ttc < self.ttc_threshold and hazard.ttc > 0:
                triggers.append(TriggerEvent(
                    timestamp=timestamp,
                    trigger_type='low_ttc',
                    reason=f"Low time-to-collision: {hazard.ttc:.2f}s",
                    metadata={
                        'ttc': hazard.ttc,
                        'hazard_type': hazard.type,
                        'hazard_id': hazard.object_id,
                        'position': hazard.position,
                        'velocity': hazard.velocity
                    }
                ))
                self.logger.debug(
                    f"Low TTC trigger: ttc={hazard.ttc:.2f}s, "
                    f"hazard={hazard.type}"
                )
        
        if triggers:
            self.logger.info(f"Recording triggered: {len(triggers)} events at t={timestamp:.3f}")
        
        return triggers
    
    def should_record(
        self,
        timestamp: float,
        risk_assessment: RiskAssessment,
        driver_state: DriverState,
        alerts: List[Alert]
    ) -> bool:
        """
        Check if recording should be triggered.
        
        Args:
            timestamp: Current timestamp
            risk_assessment: Current risk assessment
            driver_state: Current driver state
            alerts: Generated alerts
            
        Returns:
            True if recording should be triggered
        """
        triggers = self.check_triggers(timestamp, risk_assessment, driver_state, alerts)
        return len(triggers) > 0
