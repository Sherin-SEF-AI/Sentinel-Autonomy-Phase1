"""
Unit tests for AlertsPanel widget
"""

import pytest
import sys
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.gui.widgets.alerts_panel import AlertsPanel
from src.core.data_structures import Alert


@pytest.fixture(scope='module')
def qapp():
    """Create QApplication instance for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def alerts_panel(qapp):
    """Create AlertsPanel instance for testing"""
    panel = AlertsPanel()
    yield panel
    panel.close()


def test_alerts_panel_initialization(alerts_panel):
    """Test that AlertsPanel initializes correctly"""
    assert alerts_panel is not None
    assert alerts_panel.total_alerts == 0
    assert alerts_panel.critical_alerts == 0
    assert alerts_panel.warning_alerts == 0
    assert alerts_panel.info_alerts == 0
    assert alerts_panel.false_positives == 0
    assert alerts_panel.audio_enabled is True
    assert 0.0 <= alerts_panel.audio_volume <= 1.0


def test_add_critical_alert(alerts_panel):
    """Test adding a critical alert"""
    alert = Alert(
        timestamp=datetime.now().timestamp(),
        urgency='critical',
        modalities=['visual', 'audio'],
        message='Collision imminent! Vehicle ahead braking hard.',
        hazard_id=1,
        dismissed=False
    )
    
    alerts_panel.add_alert(alert)
    
    assert alerts_panel.total_alerts == 1
    assert alerts_panel.critical_alerts == 1
    assert len(alerts_panel.alert_history) == 1


def test_add_warning_alert(alerts_panel):
    """Test adding a warning alert"""
    alert = Alert(
        timestamp=datetime.now().timestamp(),
        urgency='warning',
        modalities=['visual', 'audio'],
        message='Pedestrian detected in blind spot.',
        hazard_id=2,
        dismissed=False
    )
    
    initial_total = alerts_panel.total_alerts
    alerts_panel.add_alert(alert)
    
    assert alerts_panel.total_alerts == initial_total + 1
    assert alerts_panel.warning_alerts >= 1


def test_add_info_alert(alerts_panel):
    """Test adding an info alert"""
    alert = Alert(
        timestamp=datetime.now().timestamp(),
        urgency='info',
        modalities=['visual'],
        message='Lane departure detected.',
        hazard_id=3,
        dismissed=False
    )
    
    initial_total = alerts_panel.total_alerts
    alerts_panel.add_alert(alert)
    
    assert alerts_panel.total_alerts == initial_total + 1
    assert alerts_panel.info_alerts >= 1


def test_statistics_update(alerts_panel):
    """Test that statistics update correctly"""
    initial_total = alerts_panel.total_alerts
    
    # Add multiple alerts
    for i in range(3):
        alert = Alert(
            timestamp=datetime.now().timestamp(),
            urgency='warning',
            modalities=['visual'],
            message=f'Test alert {i}',
            hazard_id=100 + i,
            dismissed=False
        )
        alerts_panel.add_alert(alert)
    
    assert alerts_panel.total_alerts == initial_total + 3


def test_clear_history(alerts_panel):
    """Test clearing alert history"""
    # Add some alerts
    for i in range(5):
        alert = Alert(
            timestamp=datetime.now().timestamp(),
            urgency='info',
            modalities=['visual'],
            message=f'Test alert {i}',
            hazard_id=200 + i,
            dismissed=False
        )
        alerts_panel.add_alert(alert)
    
    # Clear history
    alerts_panel.clear_history()
    
    assert alerts_panel.total_alerts == 0
    assert alerts_panel.critical_alerts == 0
    assert alerts_panel.warning_alerts == 0
    assert alerts_panel.info_alerts == 0
    assert len(alerts_panel.alert_history) == 0


def test_mute_toggle(alerts_panel):
    """Test audio mute toggle"""
    initial_state = alerts_panel.audio_enabled
    
    # Toggle mute
    alerts_panel.mute_button.click()
    assert alerts_panel.audio_enabled != initial_state
    
    # Toggle back
    alerts_panel.mute_button.click()
    assert alerts_panel.audio_enabled == initial_state


def test_volume_control(alerts_panel):
    """Test volume control"""
    # Set volume to 50%
    alerts_panel.set_volume(0.5)
    assert alerts_panel.audio_volume == 0.5
    assert alerts_panel.volume_spin.value() == 50
    
    # Set volume to 100%
    alerts_panel.set_volume(1.0)
    assert alerts_panel.audio_volume == 1.0
    assert alerts_panel.volume_spin.value() == 100


def test_mark_false_positive(alerts_panel):
    """Test marking alert as false positive"""
    # Add an alert
    alert = Alert(
        timestamp=datetime.now().timestamp(),
        urgency='warning',
        modalities=['visual'],
        message='Test false positive',
        hazard_id=999,
        dismissed=False
    )
    
    alerts_panel.add_alert(alert)
    alert_id = alerts_panel.alert_history[-1]['id']
    
    # Mark as false positive
    initial_fp = alerts_panel.false_positives
    alerts_panel.mark_false_positive(alert_id)
    
    assert alerts_panel.false_positives == initial_fp + 1
    assert alerts_panel.alert_history[-1]['false_positive'] is True


def test_get_statistics(alerts_panel):
    """Test getting statistics"""
    stats = alerts_panel.get_statistics()
    
    assert 'total' in stats
    assert 'critical' in stats
    assert 'warning' in stats
    assert 'info' in stats
    assert 'false_positives' in stats
    
    assert isinstance(stats['total'], int)
    assert isinstance(stats['critical'], int)


def test_filter_functionality(alerts_panel):
    """Test alert filtering"""
    # Clear and add mixed alerts
    alerts_panel.clear_history()
    
    # Add different types
    for urgency in ['critical', 'warning', 'info']:
        alert = Alert(
            timestamp=datetime.now().timestamp(),
            urgency=urgency,
            modalities=['visual'],
            message=f'Test {urgency} alert',
            hazard_id=300,
            dismissed=False
        )
        alerts_panel.add_alert(alert)
    
    # Test filter combo
    assert alerts_panel.filter_combo.count() == 4  # All, Critical, Warning, Info
    
    # Change filter
    alerts_panel.filter_combo.setCurrentText('Critical')
    # Display should be refreshed (tested visually)


def test_audio_players_initialized(alerts_panel):
    """Test that audio players are initialized"""
    assert 'critical' in alerts_panel.audio_players
    assert 'warning' in alerts_panel.audio_players
    assert 'info' in alerts_panel.audio_players
    
    assert alerts_panel.audio_players['critical'] is not None
    assert alerts_panel.audio_players['warning'] is not None
    assert alerts_panel.audio_players['info'] is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
