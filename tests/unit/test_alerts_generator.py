"""Test suite for alerts generator module."""

import pytest
import time
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass
from typing import List, Tuple

from src.alerts.generator import AlertGenerator
from src.core.data_structures import Alert, Risk, RiskAssessment, DriverState, Hazard


@pytest.fixture
def mock_config():
    """Fixture providing mock configuration for testing."""
    return {
        'escalation': {
            'critical_threshold': 0.9,
            'high_threshold': 0.7,
            'medium_threshold': 0.5
        }
    }


@pytest.fixture
def alert_generator(mock_config):
    """Fixture creating an instance of AlertGenerator for testing."""
    return AlertGenerator(mock_config)


@pytest.fixture
def mock_driver_state():
    """Fixture providing mock driver state."""
    return DriverState(
        face_detected=True,
        landmarks=Mock(),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.2},
        drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=80.0
    )


@pytest.fixture
def mock_hazard():
    """Fixture providing mock hazard."""
    return Hazard(
        object_id=1,
        type='vehicle',
        position=(10.0, 0.0, 0.0),
        velocity=(5.0, 0.0, 0.0),
        trajectory=[(10.0, 0.0, 0.0), (8.0, 0.0, 0.0)],
        ttc=2.0,
        zone='front',
        base_risk=0.8
    )


@pytest.fixture
def mock_risk_assessment(mock_hazard):
    """Fixture providing mock risk assessment."""
    risk = Risk(
        hazard=mock_hazard,
        contextual_score=0.75,
        driver_aware=False,
        urgency='high',
        intervention_needed=True
    )
    
    return RiskAssessment(
        scene_graph={},
        hazards=[mock_hazard],
        attention_map={},
        top_risks=[risk]
    )


class TestAlertGenerator:
    """Test suite for AlertGenerator class."""
    
    def test_initialization(self, alert_generator, mock_config):
        """Test that AlertGenerator initializes correctly with valid configuration."""
        assert alert_generator is not None
        assert alert_generator.config == mock_config
        assert alert_generator.critical_threshold == 0.9
        assert alert_generator.high_threshold == 0.7
        assert alert_generator.medium_threshold == 0.5
        assert alert_generator.cognitive_load_threshold == 0.7
    
    def test_initialization_with_default_thresholds(self):
        """Test initialization with missing config values uses defaults."""
        config = {}
        generator = AlertGenerator(config)
        
        assert generator.critical_threshold == 0.9
        assert generator.high_threshold == 0.7
        assert generator.medium_threshold == 0.5
    
    def test_generate_alerts_critical_risk(self, alert_generator, mock_driver_state, mock_hazard):
        """Test generation of CRITICAL alert for very high risk (> 0.9)."""
        # Create critical risk
        critical_risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.95,
            driver_aware=True,
            urgency='critical',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[critical_risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 1
        assert alerts[0].urgency == 'critical'
        assert alerts[0].modalities == ['visual', 'audio', 'haptic']
        assert 'CRITICAL' in alerts[0].message
        assert alerts[0].hazard_id == mock_hazard.object_id
        assert not alerts[0].dismissed
    
    def test_generate_alerts_warning_driver_unaware(self, alert_generator, mock_driver_state, mock_hazard):
        """Test generation of WARNING alert when driver is unaware of high risk."""
        # Create high risk with driver unaware
        warning_risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.75,
            driver_aware=False,
            urgency='high',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[warning_risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 1
        assert alerts[0].urgency == 'warning'
        assert alerts[0].modalities == ['visual', 'audio']
        assert 'WARNING' in alerts[0].message
        assert 'not looking' in alerts[0].message
    
    def test_generate_alerts_no_warning_when_driver_aware(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that WARNING alert is not generated when driver is aware of high risk."""
        # Create high risk with driver aware
        warning_risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.75,
            driver_aware=True,
            urgency='high',
            intervention_needed=False
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[warning_risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 0
    
    def test_generate_alerts_info_low_cognitive_load(self, alert_generator, mock_driver_state, mock_hazard):
        """Test generation of INFO alert when cognitive load is low."""
        # Set high readiness (low cognitive load)
        mock_driver_state.readiness_score = 85.0
        
        # Create medium risk
        info_risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.6,
            driver_aware=True,
            urgency='medium',
            intervention_needed=False
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[info_risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 1
        assert alerts[0].urgency == 'info'
        assert alerts[0].modalities == ['visual']
        assert 'INFO' in alerts[0].message
    
    def test_generate_alerts_no_info_high_cognitive_load(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that INFO alert is suppressed when cognitive load is high."""
        # Set low readiness (high cognitive load)
        mock_driver_state.readiness_score = 20.0
        
        # Create medium risk
        info_risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.6,
            driver_aware=True,
            urgency='medium',
            intervention_needed=False
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[info_risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 0
    
    def test_generate_alerts_multiple_risks(self, alert_generator, mock_driver_state):
        """Test generation of multiple alerts for multiple risks."""
        # Create multiple hazards and risks
        hazard1 = Hazard(
            object_id=1,
            type='vehicle',
            position=(10.0, 0.0, 0.0),
            velocity=(5.0, 0.0, 0.0),
            trajectory=[],
            ttc=2.0,
            zone='front',
            base_risk=0.9
        )
        
        hazard2 = Hazard(
            object_id=2,
            type='pedestrian',
            position=(5.0, 2.0, 0.0),
            velocity=(1.0, 0.0, 0.0),
            trajectory=[],
            ttc=3.0,
            zone='front_left',
            base_risk=0.7
        )
        
        risk1 = Risk(
            hazard=hazard1,
            contextual_score=0.95,
            driver_aware=False,
            urgency='critical',
            intervention_needed=True
        )
        
        risk2 = Risk(
            hazard=hazard2,
            contextual_score=0.75,
            driver_aware=False,
            urgency='high',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[hazard1, hazard2],
            attention_map={},
            top_risks=[risk1, risk2]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 2
        assert alerts[0].urgency == 'critical'
        assert alerts[1].urgency == 'warning'
    
    def test_generate_alerts_empty_risks(self, alert_generator, mock_driver_state):
        """Test that no alerts are generated when there are no risks."""
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[],
            attention_map={},
            top_risks=[]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 0
    
    def test_generate_alerts_below_threshold(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that no alerts are generated for risks below medium threshold."""
        # Create low risk
        low_risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.3,
            driver_aware=True,
            urgency='low',
            intervention_needed=False
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[low_risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 0
    
    def test_format_message_critical(self, alert_generator, mock_hazard):
        """Test message formatting for critical alerts."""
        risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.95,
            driver_aware=False,
            urgency='critical',
            intervention_needed=True
        )
        
        message = alert_generator._format_message(risk, 'critical')
        
        assert 'CRITICAL' in message
        assert mock_hazard.type in message
        assert mock_hazard.zone in message
        assert 'TTC' in message
        assert str(mock_hazard.ttc) in message
    
    def test_format_message_warning_unaware(self, alert_generator, mock_hazard):
        """Test message formatting for warning alerts when driver is unaware."""
        risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.75,
            driver_aware=False,
            urgency='high',
            intervention_needed=True
        )
        
        message = alert_generator._format_message(risk, 'warning')
        
        assert 'WARNING' in message
        assert 'not looking' in message
        assert mock_hazard.type in message
        assert mock_hazard.zone in message
    
    def test_format_message_warning_aware(self, alert_generator, mock_hazard):
        """Test message formatting for warning alerts when driver is aware."""
        risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.75,
            driver_aware=True,
            urgency='high',
            intervention_needed=True
        )
        
        message = alert_generator._format_message(risk, 'warning')
        
        assert 'WARNING' in message
        assert 'attention needed' in message
        assert mock_hazard.type in message
    
    def test_format_message_info(self, alert_generator, mock_hazard):
        """Test message formatting for info alerts."""
        risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.6,
            driver_aware=True,
            urgency='medium',
            intervention_needed=False
        )
        
        message = alert_generator._format_message(risk, 'info')
        
        assert 'INFO' in message
        assert mock_hazard.type in message
        assert mock_hazard.zone in message
        assert 'detected' in message
    
    def test_alert_timestamp_accuracy(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that generated alerts have accurate timestamps."""
        risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.95,
            driver_aware=False,
            urgency='critical',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        before_time = time.time()
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        after_time = time.time()
        
        assert len(alerts) == 1
        assert before_time <= alerts[0].timestamp <= after_time
    
    def test_cognitive_load_calculation(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that cognitive load is correctly calculated from readiness score."""
        # Test with different readiness scores
        test_cases = [
            (100.0, 0.0),  # Perfect readiness = 0 cognitive load
            (50.0, 0.5),   # Medium readiness = 0.5 cognitive load
            (0.0, 1.0),    # Zero readiness = 1.0 cognitive load
        ]
        
        for readiness, expected_load in test_cases:
            mock_driver_state.readiness_score = readiness
            
            # Create medium risk to test cognitive load threshold
            risk = Risk(
                hazard=mock_hazard,
                contextual_score=0.6,
                driver_aware=True,
                urgency='medium',
                intervention_needed=False
            )
            
            risk_assessment = RiskAssessment(
                scene_graph={},
                hazards=[mock_hazard],
                attention_map={},
                top_risks=[risk]
            )
            
            alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
            
            # INFO alerts should only be generated when cognitive load < 0.7
            if expected_load < 0.7:
                assert len(alerts) == 1
            else:
                assert len(alerts) == 0
    
    @pytest.mark.performance
    def test_performance(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that alert generation completes within performance requirements (< 5ms)."""
        # Create multiple risks to test performance
        risks = []
        for i in range(10):
            hazard = Hazard(
                object_id=i,
                type='vehicle',
                position=(10.0 + i, 0.0, 0.0),
                velocity=(5.0, 0.0, 0.0),
                trajectory=[],
                ttc=2.0 + i * 0.5,
                zone='front',
                base_risk=0.5 + i * 0.05
            )
            
            risk = Risk(
                hazard=hazard,
                contextual_score=0.5 + i * 0.05,
                driver_aware=i % 2 == 0,
                urgency='medium',
                intervention_needed=True
            )
            risks.append(risk)
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[r.hazard for r in risks],
            attention_map={},
            top_risks=risks
        )
        
        start_time = time.perf_counter()
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 5, f"Execution took {execution_time_ms:.2f}ms, expected < 5ms"
        assert len(alerts) > 0
    
    def test_alert_dismissed_flag(self, alert_generator, mock_driver_state, mock_hazard):
        """Test that generated alerts have dismissed flag set to False."""
        risk = Risk(
            hazard=mock_hazard,
            contextual_score=0.95,
            driver_aware=False,
            urgency='critical',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[mock_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 1
        assert not alerts[0].dismissed
    
    def test_hazard_id_mapping(self, alert_generator, mock_driver_state):
        """Test that alerts correctly map to hazard IDs."""
        hazard_ids = [101, 202, 303]
        risks = []
        
        for hazard_id in hazard_ids:
            hazard = Hazard(
                object_id=hazard_id,
                type='vehicle',
                position=(10.0, 0.0, 0.0),
                velocity=(5.0, 0.0, 0.0),
                trajectory=[],
                ttc=2.0,
                zone='front',
                base_risk=0.8
            )
            
            risk = Risk(
                hazard=hazard,
                contextual_score=0.95,
                driver_aware=False,
                urgency='critical',
                intervention_needed=True
            )
            risks.append(risk)
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[r.hazard for r in risks],
            attention_map={},
            top_risks=risks
        )
        
        alerts = alert_generator.generate_alerts(risk_assessment, mock_driver_state)
        
        assert len(alerts) == 3
        alert_hazard_ids = [alert.hazard_id for alert in alerts]
        assert alert_hazard_ids == hazard_ids
