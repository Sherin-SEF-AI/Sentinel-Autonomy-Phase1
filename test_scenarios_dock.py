"""
Test script for ScenariosDockWidget

This script tests the scenarios dock widget functionality.
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import Qt

from gui.widgets import ScenariosDockWidget, ScenarioReplayDialog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_test_scenarios():
    """Create test scenario data for testing"""
    scenarios_path = 'test_scenarios'
    Path(scenarios_path).mkdir(exist_ok=True)
    
    # Create a few test scenarios
    test_scenarios = [
        {
            'name': '20241116_103045',
            'trigger_type': 'critical',
            'trigger_reason': 'High risk collision detected',
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
            'name': '20241116_085612',
            'trigger_type': 'distracted',
            'trigger_reason': 'Driver distraction during hazard',
            'duration': 8.7,
            'num_frames': 261
        },
        {
            'name': '20241115_203345',
            'trigger_type': 'near_miss',
            'trigger_reason': 'Near miss event',
            'duration': 10.2,
            'num_frames': 306
        },
        {
            'name': '20241115_154523',
            'trigger_type': 'manual',
            'trigger_reason': 'Manual recording',
            'duration': 20.0,
            'num_frames': 600
        }
    ]
    
    for scenario in test_scenarios:
        scenario_dir = os.path.join(scenarios_path, scenario['name'])
        Path(scenario_dir).mkdir(exist_ok=True)
        
        # Create metadata.json
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
                'front_left': 'front_left.mp4',
                'front_right': 'front_right.mp4',
                'bev': 'bev.mp4'
            }
        }
        
        metadata_path = os.path.join(scenario_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create annotations.json (simplified)
        annotations = {
            'frames': []
        }
        
        for i in range(min(10, scenario['num_frames'])):  # Just create a few frames
            frame_data = {
                'timestamp': i * 0.033,
                'detections_3d': [],
                'driver_state': {},
                'risk_assessment': {},
                'alerts': []
            }
            annotations['frames'].append(frame_data)
        
        annotations_path = os.path.join(scenario_dir, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        logger.info(f"Created test scenario: {scenario['name']}")
    
    logger.info(f"Created {len(test_scenarios)} test scenarios in {scenarios_path}")
    return scenarios_path


def test_scenarios_dock():
    """Test the scenarios dock widget"""
    logger.info("Starting ScenariosDockWidget test")
    
    # Create test scenarios
    scenarios_path = create_test_scenarios()
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create main window
    main_window = QMainWindow()
    main_window.setWindowTitle("Scenarios Dock Widget Test")
    main_window.setGeometry(100, 100, 800, 600)
    
    # Create scenarios dock widget
    scenarios_dock = ScenariosDockWidget(scenarios_path)
    
    # Connect signals
    def on_scenario_selected(scenario_name):
        logger.info(f"Scenario selected: {scenario_name}")
        metadata = scenarios_dock.get_scenario_metadata(scenario_name)
        if metadata:
            logger.info(f"  Duration: {metadata.get('duration', 0):.1f}s")
            logger.info(f"  Trigger: {metadata.get('trigger', {}).get('type', 'Unknown')}")
    
    def on_replay_requested(scenario_name):
        logger.info(f"Replay requested for: {scenario_name}")
        
        # Note: Replay dialog requires actual video files, which we don't have in this test
        # In a real scenario, this would open the replay dialog
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            main_window,
            "Replay",
            f"Replay dialog would open for scenario:\n{scenario_name}\n\n"
            "Note: Actual video files are required for replay."
        )
    
    scenarios_dock.scenario_selected.connect(on_scenario_selected)
    scenarios_dock.scenario_replay_requested.connect(on_replay_requested)
    
    # Add dock to main window
    main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, scenarios_dock)
    
    # Show window
    main_window.show()
    
    logger.info("ScenariosDockWidget test window opened")
    logger.info("Test features:")
    logger.info("  - List shows all test scenarios")
    logger.info("  - Search box filters scenarios")
    logger.info("  - Filter combo filters by trigger type")
    logger.info("  - Double-click scenario to request replay")
    logger.info("  - Select scenario and click Export/Delete")
    logger.info("  - Click Refresh to reload scenarios")
    
    # Run application
    sys.exit(app.exec())


if __name__ == '__main__':
    test_scenarios_dock()
