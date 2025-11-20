"""
Unit tests for Risk Assessment Panel

Tests all components of the risk assessment panel.
"""

import sys
import pytest
from unittest.mock import Mock, patch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Ensure QApplication exists for widget tests
@pytest.fixture(scope='module')
def qapp():
    """Create QApplication for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def risk_panel(qapp):
    """Create RiskAssessmentPanel for testing"""
    # Import directly from widgets to avoid main module dependencies
    sys.path.insert(0, 'src')
    from gui.widgets.risk_panel import RiskAssessmentPanel
    panel = RiskAssessmentPanel()
    yield panel
    panel.deleteLater()


@pytest.fixture
def ttc_widget(qapp):
    """Create TTCDisplayWidget for testing"""
    sys.path.insert(0, 'src')
    from gui.widgets.risk_panel import TTCDisplayWidget
    widget = TTCDisplayWidget()
    yield widget
    widget.deleteLater()


@pytest.fixture
def zone_radar(qapp):
    """Create ZoneRiskRadarChart for testing"""
    sys.path.insert(0, 'src')
    from gui.widgets.risk_panel import ZoneRiskRadarChart
    widget = ZoneRiskRadarChart()
    yield widget
    widget.deleteLater()


@pytest.fixture
def hazard_item(qapp):
    """Create HazardListItem for testing"""
    sys.path.insert(0, 'src')
    from gui.widgets.risk_panel import HazardListItem
    hazard = {
        'type': 'vehicle',
        'zone': 'front',
        'ttc': 2.5,
        'risk_score': 0.75,
        'attended': False
    }
    widget = HazardListItem(hazard)
    yield widget
    widget.deleteLater()


class TestRiskAssessmentPanel:
    """Test RiskAssessmentPanel"""
    
    def test_initialization(self, risk_panel):
        """Test panel initialization"""
        assert risk_panel is not None
        assert risk_panel._current_risk == 0.0
        assert len(risk_panel._current_hazards) == 0
        assert len(risk_panel._zone_risks) == 8
        assert risk_panel._min_ttc == float('inf')
    
    def test_update_risk_score(self, risk_panel):
        """Test updating risk score"""
        risk_panel.update_risk_score(0.75)
        assert risk_panel._current_risk == 0.75
        assert len(risk_panel._risk_history) == 1
        assert len(risk_panel._time_history) == 1
    
    def test_update_risk_score_multiple(self, risk_panel):
        """Test multiple risk score updates"""
        for i in range(10):
            risk_panel.update_risk_score(i * 0.1)
        
        assert len(risk_panel._risk_history) == 10
        assert risk_panel._current_risk == 0.9
    
    def test_update_hazards(self, risk_panel):
        """Test updating hazards list"""
        hazards = [
            {
                'type': 'vehicle',
                'zone': 'front',
                'ttc': 2.5,
                'risk_score': 0.8,
                'attended': False
            },
            {
                'type': 'pedestrian',
                'zone': 'front-left',
                'ttc': 3.0,
                'risk_score': 0.6,
                'attended': True
            }
        ]
        
        risk_panel.update_hazards(hazards)
        assert len(risk_panel._current_hazards) == 2
        assert risk_panel._hazards_list.count() == 2
    
    def test_update_hazards_top_three(self, risk_panel):
        """Test that only top 3 hazards are shown"""
        hazards = [
            {'type': 'vehicle', 'zone': 'front', 'ttc': 2.0, 'risk_score': 0.9, 'attended': False},
            {'type': 'vehicle', 'zone': 'right', 'ttc': 2.5, 'risk_score': 0.8, 'attended': False},
            {'type': 'pedestrian', 'zone': 'left', 'ttc': 3.0, 'risk_score': 0.7, 'attended': True},
            {'type': 'cyclist', 'zone': 'rear', 'ttc': 4.0, 'risk_score': 0.5, 'attended': True},
            {'type': 'vehicle', 'zone': 'front-right', 'ttc': 5.0, 'risk_score': 0.3, 'attended': True}
        ]
        
        risk_panel.update_hazards(hazards)
        assert len(risk_panel._current_hazards) == 3
        assert risk_panel._hazards_list.count() == 3
    
    def test_update_zone_risks(self, risk_panel):
        """Test updating zone risks"""
        zone_risks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        risk_panel.update_zone_risks(zone_risks)
        assert risk_panel._zone_risks == zone_risks
    
    def test_update_zone_risks_invalid_length(self, risk_panel, caplog):
        """Test updating zone risks with invalid length"""
        zone_risks = [0.1, 0.2, 0.3]  # Only 3 values
        risk_panel.update_zone_risks(zone_risks)
        assert "Expected 8 zone risks" in caplog.text
    
    def test_update_ttc(self, risk_panel):
        """Test updating TTC"""
        risk_panel.update_ttc(2.5)
        assert risk_panel._min_ttc == 2.5
    
    def test_add_alert_event(self, risk_panel):
        """Test adding alert events"""
        risk_panel.add_alert_event('critical')
        assert len(risk_panel._alert_events) == 1
        assert risk_panel._alert_events[0][1] == 'critical'
    
    def test_add_multiple_alert_events(self, risk_panel):
        """Test adding multiple alert events"""
        for urgency in ['info', 'warning', 'critical']:
            risk_panel.add_alert_event(urgency)
        
        assert len(risk_panel._alert_events) == 3


class TestTTCDisplayWidget:
    """Test TTCDisplayWidget"""
    
    def test_initialization(self, ttc_widget):
        """Test widget initialization"""
        assert ttc_widget is not None
        assert ttc_widget._ttc == float('inf')
    
    def test_set_ttc(self, ttc_widget):
        """Test setting TTC value"""
        ttc_widget.set_ttc(2.5)
        assert ttc_widget._ttc == 2.5
    
    def test_get_color_green(self, ttc_widget):
        """Test color for safe TTC (>3s)"""
        color = ttc_widget._get_color_for_ttc(4.0)
        assert color.red() == 76
        assert color.green() == 175
        assert color.blue() == 80
    
    def test_get_color_yellow(self, ttc_widget):
        """Test color for warning TTC (1.5-3s)"""
        color = ttc_widget._get_color_for_ttc(2.0)
        assert color.red() == 255
        assert color.green() == 193
        assert color.blue() == 7
    
    def test_get_color_red(self, ttc_widget):
        """Test color for critical TTC (<1.5s)"""
        color = ttc_widget._get_color_for_ttc(1.0)
        assert color.red() == 244
        assert color.green() == 67
        assert color.blue() == 54


class TestHazardListItem:
    """Test HazardListItem"""
    
    def test_initialization(self, hazard_item):
        """Test item initialization"""
        assert hazard_item is not None
        assert hazard_item._hazard['type'] == 'vehicle'
    
    def test_get_icon_vehicle(self, hazard_item):
        """Test icon for vehicle"""
        icon = hazard_item._get_icon()
        assert icon == '🚗'
    
    def test_get_icon_pedestrian(self, qapp):
        """Test icon for pedestrian"""
        sys.path.insert(0, 'src')
        from gui.widgets.risk_panel import HazardListItem
        hazard = {'type': 'pedestrian', 'zone': 'front', 'ttc': 2.0, 'risk_score': 0.5, 'attended': True}
        item = HazardListItem(hazard)
        icon = item._get_icon()
        assert icon == '🚶'
        item.deleteLater()
    
    def test_get_icon_cyclist(self, qapp):
        """Test icon for cyclist"""
        sys.path.insert(0, 'src')
        from gui.widgets.risk_panel import HazardListItem
        hazard = {'type': 'cyclist', 'zone': 'front', 'ttc': 2.0, 'risk_score': 0.5, 'attended': True}
        item = HazardListItem(hazard)
        icon = item._get_icon()
        assert icon == '🚴'
        item.deleteLater()
    
    def test_get_icon_unknown(self, qapp):
        """Test icon for unknown type"""
        sys.path.insert(0, 'src')
        from gui.widgets.risk_panel import HazardListItem
        hazard = {'type': 'unknown', 'zone': 'front', 'ttc': 2.0, 'risk_score': 0.5, 'attended': True}
        item = HazardListItem(hazard)
        icon = item._get_icon()
        assert icon == '⚠️'
        item.deleteLater()


class TestZoneRiskRadarChart:
    """Test ZoneRiskRadarChart"""
    
    def test_initialization(self, zone_radar):
        """Test chart initialization"""
        assert zone_radar is not None
        assert len(zone_radar._zone_risks) == 8
        assert len(zone_radar._zone_labels) == 8
        assert all(risk == 0.0 for risk in zone_radar._zone_risks)
    
    def test_set_zone_risks(self, zone_radar):
        """Test setting zone risks"""
        risks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        zone_radar.set_zone_risks(risks)
        assert zone_radar._zone_risks == risks
    
    def test_set_zone_risks_invalid_length(self, zone_radar, caplog):
        """Test setting zone risks with invalid length"""
        risks = [0.1, 0.2, 0.3]
        zone_radar.set_zone_risks(risks)
        assert "Expected 8 zone risks" in caplog.text
    
    def test_zone_labels(self, zone_radar):
        """Test zone labels are correct"""
        expected_labels = [
            "Front", "Front-Right", "Right", "Rear-Right",
            "Rear", "Rear-Left", "Left", "Front-Left"
        ]
        assert zone_radar._zone_labels == expected_labels


class TestRiskPanelIntegration:
    """Integration tests for risk panel"""
    
    def test_full_update_cycle(self, risk_panel):
        """Test complete update cycle"""
        # Update risk score
        risk_panel.update_risk_score(0.75)
        
        # Update hazards
        hazards = [
            {'type': 'vehicle', 'zone': 'front', 'ttc': 2.0, 'risk_score': 0.8, 'attended': False},
            {'type': 'pedestrian', 'zone': 'left', 'ttc': 3.0, 'risk_score': 0.6, 'attended': True}
        ]
        risk_panel.update_hazards(hazards)
        
        # Update zone risks
        zone_risks = [0.8, 0.2, 0.3, 0.1, 0.1, 0.2, 0.6, 0.4]
        risk_panel.update_zone_risks(zone_risks)
        
        # Update TTC
        risk_panel.update_ttc(2.0)
        
        # Add alert
        risk_panel.add_alert_event('warning')
        
        # Verify all updates
        assert risk_panel._current_risk == 0.75
        assert len(risk_panel._current_hazards) == 2
        assert risk_panel._zone_risks == zone_risks
        assert risk_panel._min_ttc == 2.0
        assert len(risk_panel._alert_events) == 1
    
    def test_risk_history_accumulation(self, risk_panel):
        """Test that risk history accumulates correctly"""
        for i in range(50):
            risk_panel.update_risk_score(i * 0.02)
        
        assert len(risk_panel._risk_history) == 50
        assert len(risk_panel._time_history) == 50
    
    def test_risk_history_max_length(self, risk_panel):
        """Test that risk history respects max length"""
        # Add more than 300 points (5 minutes at 1 Hz)
        for i in range(350):
            risk_panel.update_risk_score(0.5)
        
        # Should be capped at 300
        assert len(risk_panel._risk_history) == 300
        assert len(risk_panel._time_history) == 300
