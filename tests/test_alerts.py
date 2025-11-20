"""Tests for Alert & Action System."""

import pytest
import time
import numpy as np
from pathlib import Path

from src.alerts import AlertSystem, AlertGenerator, AlertSuppressor, AlertLogger, AlertDispatcher
from src.core.data_structures import (
    RiskAssessment, Risk, Hazard, DriverState, Alert
)


@pytest.fixture
def alert_config():
    """Alert configuration for testing."""
    return {
        'suppression': {
            'duplicate_window': 5.0,
            'max_simultaneous': 2
        },
        'escalation': {
            'critical_threshold': 0.9,
            'high_threshold': 0.7,
            'medium_threshold': 0.5
        },
        'modalities': {
            'visual': {
                'display_duration': 3.0,
                'flash_rate': 2
            },
            'audio': {
                'volume': 0.8,
                'critical_sound': 'sounds/alarm.wav',
                'warning_sound': 'sounds/beep.wav'
            },
            'haptic': {
                'enabled': False
            }
        }
    }


@pytest.fixture
def test_hazard():
    """Create a test hazard."""
    return Hazard(
        object_id=1,
        type='vehicle',
        position=(5.0, 0.0, 0.0),
        velocity=(10.0, 0.0, 0.0),
        trajectory=[],
        ttc=1.5,
        zone='front',
        base_risk=0.8
    )


@pytest.fixture
def test_driver_state():
    """Create a test driver state."""
    return DriverState(
        face_detected=True,
        landmarks=np.zeros((68, 2)),
        head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
        eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
        drowsiness={'score': 0.2, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
        distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
        readiness_score=75.0
    )


class TestAlertGenerator:
    """Tests for AlertGenerator."""
    
    def test_critical_alert_generation(self, alert_config, test_hazard, test_driver_state):
        """Test generation of critical alerts."""
        generator = AlertGenerator(alert_config)
        
        risk = Risk(
            hazard=test_hazard,
            contextual_score=0.95,
            driver_aware=True,
            urgency='critical',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[test_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = generator.generate_alerts(risk_assessment, test_driver_state)
        
        assert len(alerts) == 1
        assert alerts[0].urgency == 'critical'
        assert 'visual' in alerts[0].modalities
        assert 'audio' in alerts[0].modalities
        assert 'haptic' in alerts[0].modalities
    
    def test_warning_alert_when_unaware(self, alert_config, test_hazard, test_driver_state):
        """Test warning alert generation when driver unaware."""
        generator = AlertGenerator(alert_config)
        
        risk = Risk(
            hazard=test_hazard,
            contextual_score=0.75,
            driver_aware=False,
            urgency='high',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[test_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = generator.generate_alerts(risk_assessment, test_driver_state)
        
        assert len(alerts) == 1
        assert alerts[0].urgency == 'warning'
        assert 'visual' in alerts[0].modalities
        assert 'audio' in alerts[0].modalities
    
    def test_no_warning_when_aware(self, alert_config, test_hazard, test_driver_state):
        """Test no warning alert when driver is aware."""
        generator = AlertGenerator(alert_config)
        
        risk = Risk(
            hazard=test_hazard,
            contextual_score=0.75,
            driver_aware=True,  # Driver is aware
            urgency='high',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[test_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = generator.generate_alerts(risk_assessment, test_driver_state)
        
        # Should not generate warning if driver is aware
        assert len(alerts) == 0
    
    def test_info_alert_low_cognitive_load(self, alert_config, test_hazard):
        """Test info alert with low cognitive load."""
        generator = AlertGenerator(alert_config)
        
        # High readiness = low cognitive load
        driver = DriverState(
            face_detected=True,
            landmarks=np.zeros((68, 2)),
            head_pose={'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            gaze={'pitch': 0.0, 'yaw': 0.0, 'attention_zone': 'front'},
            eye_state={'left_ear': 0.3, 'right_ear': 0.3, 'perclos': 0.1},
            drowsiness={'score': 0.1, 'yawn_detected': False, 'micro_sleep': False, 'head_nod': False},
            distraction={'type': 'none', 'confidence': 0.0, 'duration': 0.0},
            readiness_score=85.0  # High readiness
        )
        
        risk = Risk(
            hazard=test_hazard,
            contextual_score=0.55,
            driver_aware=True,
            urgency='medium',
            intervention_needed=False
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[test_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = generator.generate_alerts(risk_assessment, driver)
        
        assert len(alerts) == 1
        assert alerts[0].urgency == 'info'
        assert alerts[0].modalities == ['visual']


class TestAlertSuppressor:
    """Tests for AlertSuppressor."""
    
    def test_duplicate_suppression(self, alert_config):
        """Test suppression of duplicate alerts."""
        suppressor = AlertSuppressor(alert_config)
        
        alert1 = Alert(
            timestamp=time.time(),
            urgency='warning',
            modalities=['visual', 'audio'],
            message='Test alert',
            hazard_id=1,
            dismissed=False
        )
        
        alert2 = Alert(
            timestamp=time.time(),
            urgency='warning',
            modalities=['visual', 'audio'],
            message='Test alert',
            hazard_id=1,  # Same hazard
            dismissed=False
        )
        
        # First alert should pass
        filtered1 = suppressor.suppress_alerts([alert1])
        assert len(filtered1) == 1
        
        # Second alert should be suppressed
        filtered2 = suppressor.suppress_alerts([alert2])
        assert len(filtered2) == 0
    
    def test_max_simultaneous_alerts(self, alert_config):
        """Test maximum simultaneous alerts limit."""
        suppressor = AlertSuppressor(alert_config)
        
        alerts = [
            Alert(
                timestamp=time.time(),
                urgency='critical',
                modalities=['visual', 'audio', 'haptic'],
                message=f'Alert {i}',
                hazard_id=i,
                dismissed=False
            )
            for i in range(5)
        ]
        
        filtered = suppressor.suppress_alerts(alerts)
        
        # Should only allow max_simultaneous (2)
        assert len(filtered) <= 2
    
    def test_priority_ordering(self, alert_config):
        """Test that higher priority alerts are kept."""
        suppressor = AlertSuppressor(alert_config)
        
        alerts = [
            Alert(
                timestamp=time.time(),
                urgency='info',
                modalities=['visual'],
                message='Info alert',
                hazard_id=1,
                dismissed=False
            ),
            Alert(
                timestamp=time.time(),
                urgency='critical',
                modalities=['visual', 'audio', 'haptic'],
                message='Critical alert',
                hazard_id=2,
                dismissed=False
            ),
            Alert(
                timestamp=time.time(),
                urgency='warning',
                modalities=['visual', 'audio'],
                message='Warning alert',
                hazard_id=3,
                dismissed=False
            )
        ]
        
        filtered = suppressor.suppress_alerts(alerts)
        
        # Should keep critical and warning, drop info
        assert len(filtered) == 2
        assert filtered[0].urgency == 'critical'
        assert filtered[1].urgency == 'warning'


class TestAlertLogger:
    """Tests for AlertLogger."""
    
    def test_alert_logging(self, tmp_path):
        """Test alert logging functionality."""
        logger = AlertLogger(log_dir=str(tmp_path))
        
        alert = Alert(
            timestamp=time.time(),
            urgency='warning',
            modalities=['visual', 'audio'],
            message='Test alert',
            hazard_id=1,
            dismissed=False
        )
        
        logger.log_alert(alert)
        
        history = logger.get_alert_history()
        assert len(history) == 1
        assert history[0]['urgency'] == 'warning'
        assert history[0]['message'] == 'Test alert'
    
    def test_alert_statistics(self, tmp_path):
        """Test alert statistics calculation."""
        logger = AlertLogger(log_dir=str(tmp_path))
        
        alerts = [
            Alert(time.time(), 'critical', ['visual'], 'Alert 1', 1, False),
            Alert(time.time(), 'warning', ['visual'], 'Alert 2', 2, False),
            Alert(time.time(), 'warning', ['visual'], 'Alert 3', 3, False),
            Alert(time.time(), 'info', ['visual'], 'Alert 4', 4, False)
        ]
        
        for alert in alerts:
            logger.log_alert(alert)
        
        stats = logger.get_alert_statistics()
        
        assert stats['total'] == 4
        assert stats['by_urgency']['critical'] == 1
        assert stats['by_urgency']['warning'] == 2
        assert stats['by_urgency']['info'] == 1
        assert stats['unique_hazards'] == 4


class TestAlertDispatcher:
    """Tests for AlertDispatcher."""
    
    def test_visual_alert_dispatch(self, alert_config):
        """Test visual alert dispatch."""
        dispatcher = AlertDispatcher(alert_config)
        
        alert = Alert(
            timestamp=time.time(),
            urgency='warning',
            modalities=['visual'],
            message='Test alert',
            hazard_id=1,
            dismissed=False
        )
        
        dispatcher.dispatch_alerts([alert])
        
        visual_alerts = dispatcher.get_active_visual_alerts()
        assert len(visual_alerts) == 1
        assert visual_alerts[0]['alert'].message == 'Test alert'
    
    def test_multi_modal_dispatch(self, alert_config):
        """Test multi-modal alert dispatch."""
        dispatcher = AlertDispatcher(alert_config)
        
        alert = Alert(
            timestamp=time.time(),
            urgency='critical',
            modalities=['visual', 'audio', 'haptic'],
            message='Critical alert',
            hazard_id=1,
            dismissed=False
        )
        
        # Should not raise any errors
        dispatcher.dispatch_alerts([alert])
        
        visual_alerts = dispatcher.get_active_visual_alerts()
        assert len(visual_alerts) == 1


class TestAlertSystem:
    """Tests for complete AlertSystem."""
    
    def test_end_to_end_processing(self, alert_config, test_hazard, test_driver_state, tmp_path):
        """Test complete alert processing pipeline."""
        alert_system = AlertSystem(alert_config, log_dir=str(tmp_path))
        
        risk = Risk(
            hazard=test_hazard,
            contextual_score=0.85,
            driver_aware=False,
            urgency='high',
            intervention_needed=True
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[test_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = alert_system.process(risk_assessment, test_driver_state)
        
        assert len(alerts) > 0
        assert alerts[0].urgency == 'warning'
        
        # Check that alert was logged
        history = alert_system.get_alert_history()
        assert len(history) > 0
    
    def test_no_alerts_for_low_risk(self, alert_config, test_hazard, test_driver_state, tmp_path):
        """Test that low risks don't generate alerts."""
        alert_system = AlertSystem(alert_config, log_dir=str(tmp_path))
        
        risk = Risk(
            hazard=test_hazard,
            contextual_score=0.3,  # Below threshold
            driver_aware=True,
            urgency='low',
            intervention_needed=False
        )
        
        risk_assessment = RiskAssessment(
            scene_graph={},
            hazards=[test_hazard],
            attention_map={},
            top_risks=[risk]
        )
        
        alerts = alert_system.process(risk_assessment, test_driver_state)
        
        assert len(alerts) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
