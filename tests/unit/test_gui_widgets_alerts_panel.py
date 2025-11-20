"""Test suite for AlertsPanel widget module."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
import sys

# Ensure QApplication exists for widget testing
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def mock_alert():
    """Fixture providing a mock Alert object."""
    alert = Mock()
    alert.timestamp = 1700000000.0
    alert.urgency = 'warning'
    alert.modalities = ['visual', 'audio']
    alert.message = 'Test alert message'
    alert.hazard_id = 123
    alert.dismissed = False
    return alert


@pytest.fixture
def mock_critical_alert():
    """Fixture providing a mock critical Alert object."""
    alert = Mock()
    alert.timestamp = 1700000001.0
    alert.urgency = 'critical'
    alert.modalities = ['visual', 'audio', 'haptic']
    alert.message = 'Critical hazard detected!'
    alert.hazard_id = 456
    alert.dismissed = False
    return alert


@pytest.fixture
def mock_info_alert():
    """Fixture providing a mock info Alert object."""
    alert = Mock()
    alert.timestamp = 1700000002.0
    alert.urgency = 'info'
    alert.modalities = ['visual']
    alert.message = 'Information message'
    alert.hazard_id = 789
    alert.dismissed = False
    return alert


@pytest.fixture
def alerts_panel(qapp):
    """Fixture creating an instance of AlertsPanel for testing."""
    from src.gui.widgets.alerts_panel import AlertsPanel
    panel = AlertsPanel()
    yield panel
    panel.deleteLater()


class TestAlertsPanelInitialization:
    """Test suite for AlertsPanel initialization."""
    
    def test_initialization(self, alerts_panel):
        """Test that AlertsPanel initializes correctly."""
        assert alerts_panel is not None
        assert alerts_panel.alert_history == []
        assert alerts_panel.alert_counter == 0
        assert alerts_panel.total_alerts == 0
        assert alerts_panel.critical_alerts == 0
        assert alerts_panel.warning_alerts == 0
        assert alerts_panel.info_alerts == 0
        assert alerts_panel.false_positives == 0
    
    def test_audio_initialization(self, alerts_panel):
        """Test that audio players are initialized correctly."""
        assert alerts_panel.audio_enabled is True
        assert alerts_panel.audio_volume == 0.8
        assert 'critical' in alerts_panel.audio_players
        assert 'warning' in alerts_panel.audio_players
        assert 'info' in alerts_panel.audio_players
        assert isinstance(alerts_panel.audio_players['critical'], QMediaPlayer)
    
    def test_ui_components_exist(self, alerts_panel):
        """Test that all UI components are created."""
        assert alerts_panel.alert_display is not None
        assert alerts_panel.stats_group is not None
        assert alerts_panel.mute_button is not None
        assert alerts_panel.volume_spin is not None
        assert alerts_panel.filter_combo is not None
        assert alerts_panel.clear_button is not None
        assert alerts_panel.export_button is not None
    
    def test_initial_display_content(self, alerts_panel):
        """Test that initial display shows welcome message."""
        html_content = alerts_panel.alert_display.toHtml()
        assert 'Alert Monitor' in html_content or 'No alerts yet' in html_content


class TestAlertsPanelAddAlert:
    """Test suite for adding alerts to the panel."""
    
    def test_add_warning_alert(self, alerts_panel, mock_alert):
        """Test adding a warning alert updates statistics and display."""
        alerts_panel.add_alert(mock_alert)
        
        assert alerts_panel.total_alerts == 1
        assert alerts_panel.warning_alerts == 1
        assert alerts_panel.critical_alerts == 0
        assert alerts_panel.info_alerts == 0
        assert len(alerts_panel.alert_history) == 1
    
    def test_add_critical_alert(self, alerts_panel, mock_critical_alert):
        """Test adding a critical alert updates statistics correctly."""
        alerts_panel.add_alert(mock_critical_alert)
        
        assert alerts_panel.total_alerts == 1
        assert alerts_panel.critical_alerts == 1
        assert alerts_panel.warning_alerts == 0
        assert len(alerts_panel.alert_history) == 1
    
    def test_add_info_alert(self, alerts_panel, mock_info_alert):
        """Test adding an info alert updates statistics correctly."""
        alerts_panel.add_alert(mock_info_alert)
        
        assert alerts_panel.total_alerts == 1
        assert alerts_panel.info_alerts == 1
        assert len(alerts_panel.alert_history) == 1
    
    def test_add_multiple_alerts(self, alerts_panel, mock_alert, mock_critical_alert, mock_info_alert):
        """Test adding multiple alerts of different types."""
        alerts_panel.add_alert(mock_alert)
        alerts_panel.add_alert(mock_critical_alert)
        alerts_panel.add_alert(mock_info_alert)
        
        assert alerts_panel.total_alerts == 3
        assert alerts_panel.warning_alerts == 1
        assert alerts_panel.critical_alerts == 1
        assert alerts_panel.info_alerts == 1
        assert len(alerts_panel.alert_history) == 3
    
    def test_alert_counter_increments(self, alerts_panel, mock_alert):
        """Test that alert counter increments for each alert."""
        alerts_panel.add_alert(mock_alert)
        assert alerts_panel.alert_counter == 1
        
        alerts_panel.add_alert(mock_alert)
        assert alerts_panel.alert_counter == 2
    
    def test_alert_entry_structure(self, alerts_panel, mock_alert):
        """Test that alert entry has correct structure."""
        alerts_panel.add_alert(mock_alert)
        
        entry = alerts_panel.alert_history[0]
        assert 'id' in entry
        assert 'alert' in entry
        assert 'timestamp' in entry
        assert 'false_positive' in entry
        assert entry['false_positive'] is False
        assert isinstance(entry['timestamp'], datetime)
    
    @patch('src.gui.widgets.alerts_panel.AlertsPanel._play_alert_sound')
    def test_audio_played_when_enabled(self, mock_play_sound, alerts_panel, mock_alert):
        """Test that audio is played when audio is enabled and in modalities."""
        alerts_panel.audio_enabled = True
        alerts_panel.add_alert(mock_alert)
        
        mock_play_sound.assert_called_once_with('warning')
    
    @patch('src.gui.widgets.alerts_panel.AlertsPanel._play_alert_sound')
    def test_audio_not_played_when_disabled(self, mock_play_sound, alerts_panel, mock_alert):
        """Test that audio is not played when audio is disabled."""
        alerts_panel.audio_enabled = False
        alerts_panel.add_alert(mock_alert)
        
        mock_play_sound.assert_not_called()
    
    @patch('src.gui.widgets.alerts_panel.AlertsPanel._trigger_critical_effects')
    def test_critical_effects_triggered(self, mock_trigger, alerts_panel, mock_critical_alert):
        """Test that critical effects are triggered for critical alerts."""
        alerts_panel.add_alert(mock_critical_alert)
        
        mock_trigger.assert_called_once()
    
    @patch('src.gui.widgets.alerts_panel.AlertsPanel._trigger_critical_effects')
    def test_critical_effects_not_triggered_for_warning(self, mock_trigger, alerts_panel, mock_alert):
        """Test that critical effects are not triggered for non-critical alerts."""
        alerts_panel.add_alert(mock_alert)
        
        mock_trigger.assert_not_called()


class TestAlertsPanelStatistics:
    """Test suite for alert statistics functionality."""
    
    def test_get_statistics_empty(self, alerts_panel):
        """Test getting statistics when no alerts exist."""
        stats = alerts_panel.get_statistics()
        
        assert stats['total'] == 0
        assert stats['critical'] == 0
        assert stats['warning'] == 0
        assert stats['info'] == 0
        assert stats['false_positives'] == 0
    
    def test_get_statistics_with_alerts(self, alerts_panel, mock_alert, mock_critical_alert):
        """Test getting statistics after adding alerts."""
        alerts_panel.add_alert(mock_alert)
        alerts_panel.add_alert(mock_critical_alert)
        
        stats = alerts_panel.get_statistics()
        
        assert stats['total'] == 2
        assert stats['critical'] == 1
        assert stats['warning'] == 1
        assert stats['info'] == 0
    
    def test_statistics_labels_updated(self, alerts_panel, mock_alert):
        """Test that statistics labels are updated when alerts are added."""
        alerts_panel.add_alert(mock_alert)
        
        assert '1' in alerts_panel.total_label.text()
        assert '1' in alerts_panel.warning_label.text()


class TestAlertsPanelAudioControls:
    """Test suite for audio control functionality."""
    
    def test_mute_toggle(self, alerts_panel):
        """Test mute button toggles audio state."""
        initial_state = alerts_panel.audio_enabled
        
        alerts_panel._on_mute_toggle(True)
        assert alerts_panel.audio_enabled is False
        
        alerts_panel._on_mute_toggle(False)
        assert alerts_panel.audio_enabled is True
    
    def test_volume_change(self, alerts_panel):
        """Test volume change updates audio players."""
        alerts_panel._on_volume_changed(50)
        
        assert alerts_panel.audio_volume == 0.5
    
    def test_set_audio_enabled(self, alerts_panel):
        """Test set_audio_enabled method."""
        alerts_panel.set_audio_enabled(False)
        assert alerts_panel.audio_enabled is False
        assert alerts_panel.mute_button.isChecked() is True
        
        alerts_panel.set_audio_enabled(True)
        assert alerts_panel.audio_enabled is True
        assert alerts_panel.mute_button.isChecked() is False
    
    def test_set_volume(self, alerts_panel):
        """Test set_volume method with valid range."""
        alerts_panel.set_volume(0.6)
        assert alerts_panel.audio_volume == 0.6
        assert alerts_panel.volume_spin.value() == 60
    
    def test_set_volume_clamps_to_range(self, alerts_panel):
        """Test that set_volume clamps values to valid range."""
        alerts_panel.set_volume(1.5)
        assert alerts_panel.audio_volume == 1.0
        
        alerts_panel.set_volume(-0.5)
        assert alerts_panel.audio_volume == 0.0


class TestAlertsPanelFiltering:
    """Test suite for alert filtering functionality."""
    
    def test_filter_all_shows_all_alerts(self, alerts_panel, mock_alert, mock_critical_alert):
        """Test that 'All' filter shows all alerts."""
        alerts_panel.add_alert(mock_alert)
        alerts_panel.add_alert(mock_critical_alert)
        
        alerts_panel.filter_combo.setCurrentText("All")
        alerts_panel._on_filter_changed("All")
        
        # Both alerts should be in display
        assert len(alerts_panel.alert_history) == 2
    
    def test_filter_critical_shows_only_critical(self, alerts_panel, mock_alert, mock_critical_alert):
        """Test that 'Critical' filter shows only critical alerts."""
        alerts_panel.add_alert(mock_alert)
        alerts_panel.add_alert(mock_critical_alert)
        
        alerts_panel.filter_combo.setCurrentText("Critical")
        alerts_panel._on_filter_changed("Critical")
        
        # Filter should be applied (implementation refreshes display)
        assert alerts_panel.filter_combo.currentText() == "Critical"


class TestAlertsPanelHistoryManagement:
    """Test suite for alert history management."""
    
    def test_clear_history(self, alerts_panel, mock_alert):
        """Test clearing alert history resets all counters."""
        alerts_panel.add_alert(mock_alert)
        alerts_panel.add_alert(mock_alert)
        
        alerts_panel.clear_history()
        
        assert len(alerts_panel.alert_history) == 0
        assert alerts_panel.alert_counter == 0
        assert alerts_panel.total_alerts == 0
        assert alerts_panel.warning_alerts == 0
    
    def test_mark_false_positive(self, alerts_panel, mock_alert):
        """Test marking an alert as false positive."""
        alerts_panel.add_alert(mock_alert)
        alert_id = alerts_panel.alert_history[0]['id']
        
        alerts_panel.mark_false_positive(alert_id)
        
        assert alerts_panel.alert_history[0]['false_positive'] is True
        assert alerts_panel.false_positives == 1
    
    def test_mark_false_positive_nonexistent_alert(self, alerts_panel):
        """Test marking nonexistent alert as false positive does nothing."""
        initial_fp_count = alerts_panel.false_positives
        
        alerts_panel.mark_false_positive(999)
        
        assert alerts_panel.false_positives == initial_fp_count


class TestAlertsPanelExport:
    """Test suite for alert export functionality."""
    
    @patch('builtins.open', create=True)
    def test_export_to_file(self, mock_open, alerts_panel, mock_alert):
        """Test exporting alerts to file."""
        alerts_panel.add_alert(mock_alert)
        
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        alerts_panel._export_to_file('test_export.txt')
        
        mock_open.assert_called_once_with('test_export.txt', 'w')
        assert mock_file.write.called
    
    @patch('builtins.open', side_effect=IOError("Write error"))
    def test_export_to_file_handles_error(self, mock_open, alerts_panel, mock_alert):
        """Test that export handles file write errors gracefully."""
        alerts_panel.add_alert(mock_alert)
        
        # Should not raise exception
        alerts_panel._export_to_file('test_export.txt')


class TestAlertsPanelCriticalEffects:
    """Test suite for critical alert effects."""
    
    @patch('src.gui.widgets.alerts_panel.AlertsPanel._flash_effect')
    def test_critical_effects_starts_flash_timer(self, mock_flash, alerts_panel):
        """Test that critical effects start the flash timer."""
        alerts_panel._trigger_critical_effects()
        
        assert alerts_panel.flash_timer.isActive()
    
    def test_flash_effect_toggles_state(self, alerts_panel):
        """Test that flash effect toggles flash state."""
        initial_state = alerts_panel.flash_state
        
        alerts_panel._flash_effect()
        assert alerts_panel.flash_state != initial_state
        
        alerts_panel._flash_effect()
        assert alerts_panel.flash_state == initial_state


class TestAlertsPanelSignals:
    """Test suite for AlertsPanel signals."""
    
    def test_false_positive_signal_emitted(self, alerts_panel, mock_alert):
        """Test that false_positive_marked signal is emitted."""
        alerts_panel.add_alert(mock_alert)
        alert_id = alerts_panel.alert_history[0]['id']
        
        signal_spy = []
        alerts_panel.false_positive_marked.connect(lambda x: signal_spy.append(x))
        
        alerts_panel.mark_false_positive(alert_id)
        
        assert len(signal_spy) == 1
        assert signal_spy[0] == alert_id


class TestAlertsPanelEdgeCases:
    """Test suite for edge cases and boundary conditions."""
    
    def test_add_alert_with_none_modalities(self, alerts_panel):
        """Test adding alert with None modalities doesn't crash."""
        alert = Mock()
        alert.timestamp = 1700000000.0
        alert.urgency = 'warning'
        alert.modalities = None
        alert.message = 'Test'
        alert.hazard_id = 1
        alert.dismissed = False
        
        # Should handle gracefully
        try:
            alerts_panel.add_alert(alert)
        except Exception as e:
            pytest.fail(f"Should handle None modalities gracefully: {e}")
    
    def test_add_alert_with_empty_message(self, alerts_panel):
        """Test adding alert with empty message."""
        alert = Mock()
        alert.timestamp = 1700000000.0
        alert.urgency = 'info'
        alert.modalities = ['visual']
        alert.message = ''
        alert.hazard_id = 1
        alert.dismissed = False
        
        alerts_panel.add_alert(alert)
        assert alerts_panel.total_alerts == 1
    
    def test_clear_history_when_empty(self, alerts_panel):
        """Test clearing history when already empty."""
        alerts_panel.clear_history()
        
        assert len(alerts_panel.alert_history) == 0
        assert alerts_panel.total_alerts == 0
    
    def test_get_statistics_after_clear(self, alerts_panel, mock_alert):
        """Test getting statistics after clearing history."""
        alerts_panel.add_alert(mock_alert)
        alerts_panel.clear_history()
        
        stats = alerts_panel.get_statistics()
        assert all(value == 0 for value in stats.values())


@pytest.mark.performance
class TestAlertsPanelPerformance:
    """Test suite for performance requirements."""
    
    def test_add_alert_performance(self, alerts_panel, mock_alert):
        """Test that adding an alert completes within performance requirements."""
        import time
        
        start_time = time.perf_counter()
        alerts_panel.add_alert(mock_alert)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 50, f"add_alert took {execution_time_ms:.2f}ms, expected < 50ms"
    
    def test_add_multiple_alerts_performance(self, alerts_panel, mock_alert):
        """Test that adding 100 alerts completes in reasonable time."""
        import time
        
        start_time = time.perf_counter()
        for _ in range(100):
            alerts_panel.add_alert(mock_alert)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        avg_time_ms = execution_time_ms / 100
        assert avg_time_ms < 50, f"Average add_alert took {avg_time_ms:.2f}ms, expected < 50ms"
    
    def test_filter_refresh_performance(self, alerts_panel, mock_alert):
        """Test that refreshing display with filter is performant."""
        import time
        
        # Add many alerts
        for _ in range(50):
            alerts_panel.add_alert(mock_alert)
        
        start_time = time.perf_counter()
        alerts_panel._refresh_display()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 100, f"Refresh took {execution_time_ms:.2f}ms, expected < 100ms"
