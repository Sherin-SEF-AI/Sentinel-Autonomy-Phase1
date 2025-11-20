"""
Unit tests for ScenariosDockWidget
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from gui.widgets import ScenariosDockWidget, ScenarioReplayDialog, ScenarioListItem


@pytest.fixture(scope='session')
def qapp():
    """Create QApplication instance for tests"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def temp_scenarios_dir():
    """Create temporary scenarios directory with test data"""
    temp_dir = tempfile.mkdtemp()
    scenarios_path = os.path.join(temp_dir, 'scenarios')
    Path(scenarios_path).mkdir(exist_ok=True)
    
    # Create test scenarios
    test_scenarios = [
        {
            'name': '20241116_103045',
            'trigger_type': 'critical',
            'trigger_reason': 'High risk collision',
            'duration': 15.5,
            'num_frames': 465
        },
        {
            'name': '20241116_094523',
            'trigger_type': 'high_risk',
            'trigger_reason': 'TTC below threshold',
            'duration': 12.3,
            'num_frames': 369
        },
        {
            'name': '20241115_203345',
            'trigger_type': 'distracted',
            'trigger_reason': 'Driver distraction',
            'duration': 8.7,
            'num_frames': 261
        }
    ]
    
    for scenario in test_scenarios:
        scenario_dir = os.path.join(scenarios_path, scenario['name'])
        Path(scenario_dir).mkdir(exist_ok=True)
        
        # Create metadata
        metadata = {
            'timestamp': datetime.strptime(scenario['name'], '%Y%m%d_%H%M%S').isoformat(),
            'duration': scenario['duration'],
            'num_frames': scenario['num_frames'],
            'trigger': {
                'type': scenario['trigger_type'],
                'reason': scenario['trigger_reason']
            },
            'files': {
                'interior': 'interior.mp4',
                'front_left': 'front_left.mp4'
            }
        }
        
        with open(os.path.join(scenario_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # Create annotations
        annotations = {
            'frames': [
                {
                    'timestamp': i * 0.033,
                    'detections_3d': [],
                    'driver_state': {},
                    'risk_assessment': {},
                    'alerts': []
                }
                for i in range(10)
            ]
        }
        
        with open(os.path.join(scenario_dir, 'annotations.json'), 'w') as f:
            json.dump(annotations, f)
    
    yield scenarios_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


def test_scenarios_dock_initialization(qapp, temp_scenarios_dir):
    """Test ScenariosDockWidget initialization"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    assert dock is not None
    assert dock.windowTitle() == "Scenarios"
    assert dock.scenarios_path == temp_scenarios_dir


def test_scenarios_list_loading(qapp, temp_scenarios_dir):
    """Test loading scenarios into list"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    # Should have loaded 3 scenarios
    assert len(dock.scenarios_data) == 3
    assert dock.scenarios_list.count() == 3


def test_search_filtering(qapp, temp_scenarios_dir):
    """Test search functionality"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    # Initial count
    assert dock.scenarios_list.count() == 3
    
    # Search for specific scenario
    dock.search_box.setText("103045")
    assert dock.scenarios_list.count() == 1
    
    # Clear search
    dock.search_box.setText("")
    assert dock.scenarios_list.count() == 3
    
    # Search by trigger reason
    dock.search_box.setText("collision")
    assert dock.scenarios_list.count() == 1


def test_type_filtering(qapp, temp_scenarios_dir):
    """Test filter by trigger type"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    # Filter by critical
    dock.filter_combo.setCurrentText("Critical")
    assert dock.scenarios_list.count() == 1
    
    # Filter by high risk
    dock.filter_combo.setCurrentText("High Risk")
    assert dock.scenarios_list.count() == 1
    
    # Filter by distracted
    dock.filter_combo.setCurrentText("Distracted")
    assert dock.scenarios_list.count() == 1
    
    # Show all
    dock.filter_combo.setCurrentText("All")
    assert dock.scenarios_list.count() == 3


def test_scenario_selection(qapp, temp_scenarios_dir):
    """Test scenario selection"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    # Initially no selection
    assert dock.get_selected_scenario() is None
    assert not dock.export_button.isEnabled()
    assert not dock.delete_button.isEnabled()
    
    # Select first item
    dock.scenarios_list.setCurrentRow(0)
    
    # Should have selection
    assert dock.get_selected_scenario() is not None
    assert dock.export_button.isEnabled()
    assert dock.delete_button.isEnabled()


def test_get_scenario_metadata(qapp, temp_scenarios_dir):
    """Test getting scenario metadata"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    # Get metadata for first scenario
    scenario_name = list(dock.scenarios_data.keys())[0]
    metadata = dock.get_scenario_metadata(scenario_name)
    
    assert metadata is not None
    assert 'timestamp' in metadata
    assert 'duration' in metadata
    assert 'trigger' in metadata
    assert 'files' in metadata


def test_scenario_list_item(qapp):
    """Test ScenarioListItem widget"""
    metadata = {
        'timestamp': '2024-11-16T10:30:45',
        'duration': 15.5,
        'trigger': {
            'type': 'critical',
            'reason': 'High risk collision'
        }
    }
    
    item = ScenarioListItem('20241116_103045', metadata)
    
    assert item is not None
    assert item.scenario_name == '20241116_103045'
    assert item.metadata == metadata


def test_refresh_scenarios(qapp, temp_scenarios_dir):
    """Test refreshing scenarios list"""
    dock = ScenariosDockWidget(temp_scenarios_dir)
    
    initial_count = len(dock.scenarios_data)
    
    # Refresh should reload
    dock.refresh_scenarios()
    
    assert len(dock.scenarios_data) == initial_count


def test_empty_scenarios_directory(qapp):
    """Test with empty scenarios directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        scenarios_path = os.path.join(temp_dir, 'scenarios')
        Path(scenarios_path).mkdir(exist_ok=True)
        
        dock = ScenariosDockWidget(scenarios_path)
        
        assert len(dock.scenarios_data) == 0
        assert dock.scenarios_list.count() == 0
        assert "No scenarios found" in dock.status_label.text()


def test_nonexistent_scenarios_directory(qapp):
    """Test with nonexistent scenarios directory"""
    dock = ScenariosDockWidget('/nonexistent/path')
    
    assert len(dock.scenarios_data) == 0
    assert dock.scenarios_list.count() == 0
    assert "not found" in dock.status_label.text()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
