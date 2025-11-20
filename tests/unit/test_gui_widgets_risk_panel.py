"""Test suite for risk_panel module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
import sys
import time

# Ensure QApplication exists for widget testing
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def risk_panel(qapp):
    """Fixture creating an instance of RiskAssessmentPanel for testing."""
    from src.gui.widgets.risk_panel import RiskAssessmentPanel
    panel = RiskAssessmentPanel()
    yield panel
    panel.deleteLater()


@pytest.fixture
def ttc_widget(qapp):
    """Fixture creating an instance of TTCDisplayWidget for testing."""
    from src.gui.widgets.risk_panel import TTCDisplayWidget
    widget = TTCDisplayWidget()
    yield widget
    widget.deleteLater()


@pytest.fixture
def zone_radar(qapp):
    """Fixture creating an instance of ZoneRiskRadarChart for testing."""
    from src.gui.widgets.risk_panel import ZoneRiskRadarChart
    chart = ZoneRiskRadarChart()
    yield chart
    chart.deleteLater()


@pytest.fixture
def sample_hazards():
    """Fixture providing sample hazard data."""
    return [
        {
            'type': 'vehicle',
            'zone': 'front',
            'ttc': 2.5,
            'risk_score': 0.75,
            'attended': False
        },
        {
            'type': 'pedestrian',
            'zone': 'front-left',
            'ttc': 3.2,
            'risk_score': 0.65,
            'attended': True
        },
        {
            'type': 'cyclist',
            'zone': 'right',
            'ttc': 4.0,
            'risk_score': 0.45,
            'attended': True
        }
    ]


class TestRiskAssessmentPanel:
    """Test suite for RiskAssessmentPanel class."""
    
    def test_initialization(self, risk_panel):
        """Test that RiskAssessmentPanel initializes correctly."""
        assert risk_panel is not None
        assert risk_panel._current_risk == 0.0
        assert risk_panel._current_hazards == []
        assert len(risk_panel._zone_risks) == 8
        assert all(r == 0.0 for r in risk_panel._zone_risks)
        assert risk_panel._min_ttc == float('inf')
        assert len(risk_panel._risk_history) == 0
        assert len(risk_panel._time_history) == 0
        assert risk_panel._alert_events == []
    
    def test_has_required_widgets(self, risk_panel):
        """Test that panel contains all required child widgets."""
        assert hasattr(risk_panel, '_risk_gauge')
        assert hasattr(risk_panel, '_ttc_widget')
        assert hasattr(risk_panel, '_hazards_list')
        assert hasattr(risk_panel, '_zone_radar')
        assert hasattr(risk_panel, '_risk_timeline')
        assert hasattr(risk_panel, '_risk_curve')
        assert hasattr(risk_panel, '_alert_scatter')
    
    def test_update_risk_score_valid(self, risk_panel):
        """Test updating risk score with valid value."""
        risk_panel.update_risk_score(0.75)
        
        assert risk_panel._current_risk == 0.75
        assert len(risk_panel._risk_history) == 1
        assert len(risk_panel._time_history) == 1
        assert risk_panel._risk_history[0] == 0.75
    
    def test_update_risk_score_boundary_values(self, risk_panel):
        """Test updating risk score with boundary values."""
        # Test minimum
        risk_panel.update_risk_score(0.0)
        assert risk_panel._current_risk == 0.0
        
        # Test maximum
        risk_panel.update_risk_score(1.0)
        assert risk_panel._current_risk == 1.0
    
    def test_update_risk_score_history_limit(self, risk_panel):
        """Test that risk history respects maxlen of 300."""
        # Add 350 values
        for i in range(350):
            risk_panel.update_risk_score(0.5)
        
        # Should only keep last 300
        assert len(risk_panel._risk_history) == 300
        assert len(risk_panel._time_history) == 300
    
    def test_update_hazards_valid(self, risk_panel, sample_hazards):
        """Test updating hazards list with valid data."""
        risk_panel.update_hazards(sample_hazards)
        
        assert len(risk_panel._current_hazards) == 3
        assert risk_panel._hazards_list.count() == 3
    
    def test_update_hazards_top_three_only(self, risk_panel, sample_hazards):
        """Test that only top 3 hazards are displayed."""
        # Create 5 hazards
        many_hazards = sample_hazards + [
            {'type': 'truck', 'zone': 'rear', 'ttc': 5.0, 'risk_score': 0.3, 'attended': True},
            {'type': 'bus', 'zone': 'left', 'ttc': 6.0, 'risk_score': 0.2, 'attended': True}
        ]
        
        risk_panel.update_hazards(many_hazards)
        
        # Should only show top 3
        assert len(risk_panel._current_hazards) == 3
        assert risk_panel._hazards_list.count() == 3
    
    def test_update_hazards_empty_list(self, risk_panel):
        """Test updating with empty hazards list."""
        risk_panel.update_hazards([])
        
        assert len(risk_panel._current_hazards) == 0
        assert risk_panel._hazards_list.count() == 0
    
    def test_update_zone_risks_valid(self, risk_panel):
        """Test updating zone risks with valid 8-element list."""
        zone_risks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        risk_panel.update_zone_risks(zone_risks)
        
        assert risk_panel._zone_risks == zone_risks
    
    def test_update_zone_risks_invalid_length(self, risk_panel):
        """Test updating zone risks with invalid length."""
        # Should log warning and not update
        original_risks = risk_panel._zone_risks.copy()
        risk_panel.update_zone_risks([0.1, 0.2, 0.3])  # Only 3 values
        
        # Should not have changed
        assert risk_panel._zone_risks == original_risks
    
    def test_update_ttc_valid(self, risk_panel):
        """Test updating TTC with valid value."""
        risk_panel.update_ttc(2.5)
        
        assert risk_panel._min_ttc == 2.5
    
    def test_update_ttc_infinity(self, risk_panel):
        """Test updating TTC with infinity."""
        risk_panel.update_ttc(float('inf'))
        
        assert risk_panel._min_ttc == float('inf')
    
    def test_add_alert_event(self, risk_panel):
        """Test adding alert event."""
        risk_panel.add_alert_event('warning')
        
        assert len(risk_panel._alert_events) == 1
        assert risk_panel._alert_events[0][1] == 'warning'
    
    def test_add_alert_event_multiple_urgencies(self, risk_panel):
        """Test adding multiple alert events with different urgencies."""
        risk_panel.add_alert_event('info')
        risk_panel.add_alert_event('warning')
        risk_panel.add_alert_event('critical')
        
        assert len(risk_panel._alert_events) == 3
        urgencies = [event[1] for event in risk_panel._alert_events]
        assert urgencies == ['info', 'warning', 'critical']
    
    def test_alert_event_cleanup(self, risk_panel):
        """Test that old alert events are cleaned up (> 5 minutes)."""
        # Add an old event (mock time)
        old_time = time.time() - 400  # 6.67 minutes ago
        risk_panel._alert_events.append((old_time, 'warning'))
        
        # Add a new event
        risk_panel.add_alert_event('info')
        
        # Old event should be removed
        assert len(risk_panel._alert_events) == 1
        assert risk_panel._alert_events[0][1] == 'info'
    
    @pytest.mark.performance
    def test_update_performance(self, risk_panel, sample_hazards):
        """Test that updates complete within performance requirements."""
        start_time = time.perf_counter()
        
        # Perform multiple updates
        risk_panel.update_risk_score(0.75)
        risk_panel.update_hazards(sample_hazards)
        risk_panel.update_zone_risks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        risk_panel.update_ttc(2.5)
        risk_panel.add_alert_event('warning')
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should complete quickly (< 50ms for UI updates)
        assert execution_time_ms < 50, f"Updates took {execution_time_ms:.2f}ms, expected < 50ms"


class TestTTCDisplayWidget:
    """Test suite for TTCDisplayWidget class."""
    
    def test_initialization(self, ttc_widget):
        """Test that TTCDisplayWidget initializes correctly."""
        assert ttc_widget is not None
        assert ttc_widget._ttc == float('inf')
        assert ttc_widget._animation_phase == 0.0
        assert ttc_widget._animation_timer.isActive()
    
    def test_set_ttc_valid(self, ttc_widget):
        """Test setting TTC with valid value."""
        ttc_widget.set_ttc(2.5)
        assert ttc_widget._ttc == 2.5
    
    def test_set_ttc_infinity(self, ttc_widget):
        """Test setting TTC to infinity."""
        ttc_widget.set_ttc(float('inf'))
        assert ttc_widget._ttc == float('inf')
    
    def test_set_ttc_zero(self, ttc_widget):
        """Test setting TTC to zero (imminent collision)."""
        ttc_widget.set_ttc(0.0)
        assert ttc_widget._ttc == 0.0
    
    def test_get_color_for_ttc_green(self, ttc_widget):
        """Test color selection for safe TTC (> 3s)."""
        color = ttc_widget._get_color_for_ttc(4.0)
        assert color.red() == 76
        assert color.green() == 175
        assert color.blue() == 80
    
    def test_get_color_for_ttc_yellow(self, ttc_widget):
        """Test color selection for warning TTC (1.5-3s)."""
        color = ttc_widget._get_color_for_ttc(2.0)
        assert color.red() == 255
        assert color.green() == 193
        assert color.blue() == 7
    
    def test_get_color_for_ttc_red(self, ttc_widget):
        """Test color selection for critical TTC (< 1.5s)."""
        color = ttc_widget._get_color_for_ttc(1.0)
        assert color.red() == 244
        assert color.green() == 67
        assert color.blue() == 54
    
    def test_animation_updates(self, ttc_widget):
        """Test that animation phase updates."""
        initial_phase = ttc_widget._animation_phase
        
        # Trigger animation
        ttc_widget._animate()
        
        assert ttc_widget._animation_phase != initial_phase
    
    def test_animation_wraps(self, ttc_widget):
        """Test that animation phase wraps at 1.0."""
        ttc_widget._animation_phase = 0.95
        
        # Trigger multiple animations
        for _ in range(3):
            ttc_widget._animate()
        
        # Should have wrapped
        assert ttc_widget._animation_phase < 1.0
    
    def test_paint_event_executes(self, ttc_widget):
        """Test that paintEvent executes without errors."""
        # Create a mock paint event
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        event = QPaintEvent(QRect(0, 0, 200, 200))
        
        # Should not raise exception
        try:
            ttc_widget.paintEvent(event)
        except Exception as e:
            pytest.fail(f"paintEvent raised exception: {e}")


class TestHazardListItem:
    """Test suite for HazardListItem class."""
    
    def test_initialization_vehicle(self, qapp):
        """Test initialization with vehicle hazard."""
        from src.gui.widgets.risk_panel import HazardListItem
        
        hazard = {
            'type': 'vehicle',
            'zone': 'front',
            'ttc': 2.5,
            'risk_score': 0.75,
            'attended': False
        }
        
        item = HazardListItem(hazard)
        assert item is not None
        assert item._hazard == hazard
        item.deleteLater()
    
    def test_initialization_pedestrian(self, qapp):
        """Test initialization with pedestrian hazard."""
        from src.gui.widgets.risk_panel import HazardListItem
        
        hazard = {
            'type': 'pedestrian',
            'zone': 'front-left',
            'ttc': 3.0,
            'risk_score': 0.65,
            'attended': True
        }
        
        item = HazardListItem(hazard)
        assert item is not None
        item.deleteLater()
    
    def test_get_icon_vehicle(self, qapp):
        """Test icon selection for vehicle."""
        from src.gui.widgets.risk_panel import HazardListItem
        
        hazard = {'type': 'vehicle', 'zone': 'front', 'ttc': 2.5, 'risk_score': 0.5, 'attended': True}
        item = HazardListItem(hazard)
        
        icon = item._get_icon()
        assert icon == '🚗'
        item.deleteLater()
    
    def test_get_icon_pedestrian(self, qapp):
        """Test icon selection for pedestrian."""
        from src.gui.widgets.risk_panel import HazardListItem
        
        hazard = {'type': 'pedestrian', 'zone': 'front', 'ttc': 2.5, 'risk_score': 0.5, 'attended': True}
        item = HazardListItem(hazard)
        
        icon = item._get_icon()
        assert icon == '🚶'
        item.deleteLater()
    
    def test_get_icon_unknown(self, qapp):
        """Test icon selection for unknown type."""
        from src.gui.widgets.risk_panel import HazardListItem
        
        hazard = {'type': 'unknown', 'zone': 'front', 'ttc': 2.5, 'risk_score': 0.5, 'attended': True}
        item = HazardListItem(hazard)
        
        icon = item._get_icon()
        assert icon == '⚠️'
        item.deleteLater()
    
    def test_hazard_with_infinity_ttc(self, qapp):
        """Test hazard display with infinite TTC."""
        from src.gui.widgets.risk_panel import HazardListItem
        
        hazard = {
            'type': 'vehicle',
            'zone': 'rear',
            'ttc': float('inf'),
            'risk_score': 0.3,
            'attended': True
        }
        
        item = HazardListItem(hazard)
        assert item is not None
        item.deleteLater()


class TestZoneRiskRadarChart:
    """Test suite for ZoneRiskRadarChart class."""
    
    def test_initialization(self, zone_radar):
        """Test that ZoneRiskRadarChart initializes correctly."""
        assert zone_radar is not None
        assert len(zone_radar._zone_risks) == 8
        assert all(r == 0.0 for r in zone_radar._zone_risks)
        assert len(zone_radar._zone_labels) == 8
    
    def test_zone_labels_correct(self, zone_radar):
        """Test that zone labels are correct."""
        expected_labels = [
            "Front", "Front-Right", "Right", "Rear-Right",
            "Rear", "Rear-Left", "Left", "Front-Left"
        ]
        assert zone_radar._zone_labels == expected_labels
    
    def test_set_zone_risks_valid(self, zone_radar):
        """Test setting zone risks with valid 8-element list."""
        risks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        zone_radar.set_zone_risks(risks)
        
        assert zone_radar._zone_risks == risks
    
    def test_set_zone_risks_invalid_length(self, zone_radar):
        """Test setting zone risks with invalid length."""
        original_risks = zone_radar._zone_risks.copy()
        
        # Should log warning and not update
        zone_radar.set_zone_risks([0.1, 0.2, 0.3])
        
        assert zone_radar._zone_risks == original_risks
    
    def test_set_zone_risks_all_zeros(self, zone_radar):
        """Test setting all zone risks to zero."""
        risks = [0.0] * 8
        zone_radar.set_zone_risks(risks)
        
        assert zone_radar._zone_risks == risks
    
    def test_set_zone_risks_all_max(self, zone_radar):
        """Test setting all zone risks to maximum."""
        risks = [1.0] * 8
        zone_radar.set_zone_risks(risks)
        
        assert zone_radar._zone_risks == risks
    
    def test_paint_event_executes(self, zone_radar):
        """Test that paintEvent executes without errors."""
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        # Set some risk values
        zone_radar.set_zone_risks([0.2, 0.4, 0.6, 0.8, 0.3, 0.5, 0.7, 0.9])
        
        event = QPaintEvent(QRect(0, 0, 250, 250))
        
        # Should not raise exception
        try:
            zone_radar.paintEvent(event)
        except Exception as e:
            pytest.fail(f"paintEvent raised exception: {e}")
    
    def test_paint_event_with_zeros(self, zone_radar):
        """Test painting with all zero risks."""
        from PyQt6.QtGui import QPaintEvent
        from PyQt6.QtCore import QRect
        
        zone_radar.set_zone_risks([0.0] * 8)
        event = QPaintEvent(QRect(0, 0, 250, 250))
        
        try:
            zone_radar.paintEvent(event)
        except Exception as e:
            pytest.fail(f"paintEvent with zeros raised exception: {e}")


class TestIntegration:
    """Integration tests for risk panel components."""
    
    def test_full_update_sequence(self, risk_panel, sample_hazards):
        """Test complete update sequence with all components."""
        # Update all components
        risk_panel.update_risk_score(0.75)
        risk_panel.update_hazards(sample_hazards)
        risk_panel.update_zone_risks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        risk_panel.update_ttc(2.5)
        risk_panel.add_alert_event('warning')
        
        # Verify all updates applied
        assert risk_panel._current_risk == 0.75
        assert len(risk_panel._current_hazards) == 3
        assert risk_panel._zone_risks == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        assert risk_panel._min_ttc == 2.5
        assert len(risk_panel._alert_events) == 1
    
    def test_rapid_updates(self, risk_panel):
        """Test rapid successive updates."""
        # Simulate rapid updates (30 Hz)
        for i in range(30):
            risk_panel.update_risk_score(0.5 + i * 0.01)
        
        # Should handle all updates
        assert len(risk_panel._risk_history) == 30
        assert risk_panel._current_risk == 0.5 + 29 * 0.01
    
    @pytest.mark.performance
    def test_full_panel_performance(self, risk_panel, sample_hazards):
        """Test performance of full panel update cycle."""
        start_time = time.perf_counter()
        
        # Perform 30 complete update cycles (1 second at 30 Hz)
        for _ in range(30):
            risk_panel.update_risk_score(0.75)
            risk_panel.update_hazards(sample_hazards)
            risk_panel.update_zone_risks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            risk_panel.update_ttc(2.5)
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / 30
        
        # Should maintain 30 Hz (< 33ms per cycle)
        assert avg_time_ms < 33, f"Average cycle took {avg_time_ms:.2f}ms, expected < 33ms"
