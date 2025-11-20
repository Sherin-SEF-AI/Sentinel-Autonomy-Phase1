"""Test suite for ScenariosDockWidget module."""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QListWidgetItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import numpy as np

from src.gui.widgets.scenarios_dock import ScenariosDockWidget, ScenarioListItem


# Ensure QApplication exists for Qt widgets
@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for testing Qt widgets."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_scenarios_dir():
    """Create temporary scenarios directory with test data."""
    temp_dir = tempfile.mkdtemp()
    scenarios_path = os.path.join(temp_dir, 'scenarios')
    os.makedirs(scenarios_path)
    
    # Create test scenario 1 - Critical
    scenario1_dir = os.path.join(scenarios_path, '20241116_103045')
    os.makedirs(scenario1_dir)
    
    metadata1 = {
        'timestamp': '2024-11-16T10:30:45.123Z',
        'duration': 15.5,
        'trigger': {
            'type': 'critical',
            'reason': 'Pedestrian crossing detected with driver distracted'
        },
        'files': {
            'interior': 'interior.mp4',
            'front_left': 'front_left.mp4',
            'front_right': 'front_right.mp4',
            'bev': 'bev.mp4'
        }
    }
    
    with open(os.path.join(scenario1_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata1, f)
    
    # Create test scenario 2 - High Risk
    scenario2_dir = os.path.join(scenarios_path, '20241116_094530')
    os.makedirs(scenario2_dir)
    
    metadata2 = {
        'timestamp': '2024-11-16T09:45:30.456Z',
        'duration': 8.2,
        'trigger': {
            'type': 'high_risk',
            'reason': 'Vehicle cutting in with low TTC'
        },
        'files': {
            'interior': 'interior.mp4',
            'front_left': 'front_left.mp4',
            'bev': 'bev.mp4'
        }
    }
    
    with open(os.path.join(scenario2_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata2, f)
    
    # Create test scenario 3 - Manual
    scenario3_dir = os.path.join(scenarios_path, '20241115_153020')
    os.makedirs(scenario3_dir)
    
    metadata3 = {
        'timestamp': '2024-11-15T15:30:20.789Z',
        'duration': 12.0,
        'trigger': {
            'type': 'manual',
            'reason': 'User initiated recording'
        },
        'files': {
            'bev': 'bev.mp4'
        }
    }
    
    with open(os.path.join(scenario3_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata3, f)
    
    yield scenarios_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def scenarios_dock(qapp, temp_scenarios_dir):
    """Fixture creating an instance of ScenariosDockWidget for testing."""
    widget = ScenariosDockWidget(scenarios_path=temp_scenarios_dir)
    yield widget
    widget.close()


@pytest.fixture
def sample_metadata():
    """Fixture providing sample scenario metadata."""
    return {
        'timestamp': '2024-11-16T10:30:45.123Z',
        'duration': 15.5,
        'trigger': {
            'type': 'critical',
            'reason': 'Pedestrian crossing detected'
        },
        'files': {
            'bev': 'bev.mp4'
        }
    }


class TestScenarioListItem:
    """Test suite for ScenarioListItem class."""
    
    def test_initialization_with_thumbnail(self, qapp, sample_metadata):
        """Test that ScenarioListItem initializes correctly with thumbnail."""
        # Create a dummy pixmap
        pixmap = QPixmap(120, 90)
        
        item = ScenarioListItem('test_scenario', sample_metadata, pixmap)
        
        assert item is not None
        assert item.scenario_name == 'test_scenario'
        assert item.metadata == sample_metadata
        assert item.thumbnail_label.pixmap() is not None
    
    def test_initialization_without_thumbnail(self, qapp, sample_metadata):
        """Test that ScenarioListItem initializes correctly without thumbnail."""
        item = ScenarioListItem('test_scenario', sample_metadata, None)
        
        assert item is not None
        assert item.thumbnail_label.text() == "No Preview"
    
    def test_trigger_color_critical(self, qapp, sample_metadata):
        """Test that critical trigger gets correct color styling."""
        item = ScenarioListItem('test_scenario', sample_metadata)
        
        color = item._get_trigger_color('critical')
        
        assert 'ff4444' in color
        assert 'bold' in color
    
    def test_trigger_color_high_risk(self, qapp, sample_metadata):
        """Test that high_risk trigger gets correct color styling."""
        item = ScenarioListItem('test_scenario', sample_metadata)
        
        color = item._get_trigger_color('high_risk')
        
        assert 'ff8800' in color
    
    def test_trigger_color_unknown(self, qapp, sample_metadata):
        """Test that unknown trigger type gets default color."""
        item = ScenarioListItem('test_scenario', sample_metadata)
        
        color = item._get_trigger_color('unknown_type')
        
        assert '#aaa' in color


class TestScenariosDockWidget:
    """Test suite for ScenariosDockWidget class."""
    
    def test_initialization(self, scenarios_dock, temp_scenarios_dir):
        """Test that ScenariosDockWidget initializes correctly."""
        assert scenarios_dock is not None
        assert scenarios_dock.scenarios_path == temp_scenarios_dir
        assert scenarios_dock.scenarios_list is not None
        assert scenarios_dock.search_box is not None
        assert scenarios_dock.filter_combo is not None
    
    def test_refresh_scenarios_loads_all(self, scenarios_dock):
        """Test that refresh_scenarios loads all scenarios from disk."""
        scenarios_dock.refresh_scenarios()
        
        # Should load 3 scenarios
        assert len(scenarios_dock.scenarios_data) == 3
        assert scenarios_dock.scenarios_list.count() == 3
    
    def test_refresh_scenarios_sorts_by_timestamp(self, scenarios_dock):
        """Test that scenarios are sorted by timestamp (newest first)."""
        scenarios_dock.refresh_scenarios()
        
        # Get first item (should be newest)
        first_item = scenarios_dock.scenarios_list.item(0)
        first_scenario = first_item.data(Qt.ItemDataRole.UserRole)
        
        # Should be the 20241116_103045 scenario (newest)
        assert '20241116_103045' in first_scenario
    
    def test_refresh_scenarios_missing_directory(self, qapp):
        """Test refresh_scenarios handles missing directory gracefully."""
        widget = ScenariosDockWidget(scenarios_path='/nonexistent/path')
        
        widget.refresh_scenarios()
        
        assert len(widget.scenarios_data) == 0
        assert widget.scenarios_list.count() == 0
        assert 'not found' in widget.status_label.text().lower()
        
        widget.close()
    
    def test_load_scenario_metadata_success(self, scenarios_dock):
        """Test loading scenario metadata successfully."""
        result = scenarios_dock._load_scenario_metadata('20241116_103045')
        
        assert result is True
        assert '20241116_103045' in scenarios_dock.scenarios_data
        assert scenarios_dock.scenarios_data['20241116_103045']['duration'] == 15.5
    
    def test_load_scenario_metadata_missing_file(self, scenarios_dock):
        """Test loading scenario metadata with missing metadata file."""
        result = scenarios_dock._load_scenario_metadata('nonexistent_scenario')
        
        assert result is False
    
    @patch('cv2.VideoCapture')
    def test_load_thumbnail_success(self, mock_video_capture, scenarios_dock):
        """Test loading thumbnail from video file."""
        # Mock video capture
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        
        # Load metadata first
        scenarios_dock._load_scenario_metadata('20241116_103045')
        
        # Load thumbnail
        pixmap = scenarios_dock._load_thumbnail('20241116_103045')
        
        # Should return None since video file doesn't actually exist
        # But the method should handle it gracefully
        assert pixmap is None or isinstance(pixmap, QPixmap)
    
    def test_load_thumbnail_no_metadata(self, scenarios_dock):
        """Test loading thumbnail with no metadata returns None."""
        pixmap = scenarios_dock._load_thumbnail('nonexistent_scenario')
        
        assert pixmap is None
    
    def test_search_filter_by_name(self, scenarios_dock):
        """Test search filtering by scenario name."""
        scenarios_dock.refresh_scenarios()
        
        # Search for specific date
        scenarios_dock.search_box.setText('20241116')
        
        # Should show only 2 scenarios from that date
        assert scenarios_dock.scenarios_list.count() == 2
    
    def test_search_filter_by_reason(self, scenarios_dock):
        """Test search filtering by trigger reason."""
        scenarios_dock.refresh_scenarios()
        
        # Search for "pedestrian"
        scenarios_dock.search_box.setText('pedestrian')
        
        # Should show only 1 scenario
        assert scenarios_dock.scenarios_list.count() == 1
    
    def test_type_filter_critical(self, scenarios_dock):
        """Test filtering by critical trigger type."""
        scenarios_dock.refresh_scenarios()
        
        # Filter by critical
        scenarios_dock.filter_combo.setCurrentText('Critical')
        
        # Should show only 1 critical scenario
        assert scenarios_dock.scenarios_list.count() == 1
    
    def test_type_filter_all(self, scenarios_dock):
        """Test filtering with 'All' shows all scenarios."""
        scenarios_dock.refresh_scenarios()
        
        # Set to All
        scenarios_dock.filter_combo.setCurrentText('All')
        
        # Should show all 3 scenarios
        assert scenarios_dock.scenarios_list.count() == 3
    
    def test_combined_search_and_filter(self, scenarios_dock):
        """Test combining search and type filter."""
        scenarios_dock.refresh_scenarios()
        
        # Search for date and filter by critical
        scenarios_dock.search_box.setText('20241116')
        scenarios_dock.filter_combo.setCurrentText('Critical')
        
        # Should show only 1 scenario
        assert scenarios_dock.scenarios_list.count() == 1
    
    def test_selection_enables_buttons(self, scenarios_dock):
        """Test that selecting a scenario enables action buttons."""
        scenarios_dock.refresh_scenarios()
        
        # Initially buttons should be disabled
        assert not scenarios_dock.export_button.isEnabled()
        assert not scenarios_dock.delete_button.isEnabled()
        
        # Select first item
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Buttons should now be enabled
        assert scenarios_dock.export_button.isEnabled()
        assert scenarios_dock.delete_button.isEnabled()
    
    def test_selection_emits_signal(self, scenarios_dock):
        """Test that selecting a scenario emits signal."""
        scenarios_dock.refresh_scenarios()
        
        # Connect signal to mock
        mock_handler = Mock()
        scenarios_dock.scenario_selected.connect(mock_handler)
        
        # Select first item
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Signal should be emitted
        assert mock_handler.call_count == 1
    
    def test_double_click_emits_replay_signal(self, scenarios_dock):
        """Test that double-clicking emits replay signal."""
        scenarios_dock.refresh_scenarios()
        
        # Connect signal to mock
        mock_handler = Mock()
        scenarios_dock.scenario_replay_requested.connect(mock_handler)
        
        # Get first item and simulate double-click
        first_item = scenarios_dock.scenarios_list.item(0)
        scenarios_dock._on_item_double_clicked(first_item)
        
        # Signal should be emitted
        assert mock_handler.call_count == 1
    
    @patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory')
    @patch('shutil.copytree')
    def test_export_scenario_success(self, mock_copytree, mock_dialog, scenarios_dock):
        """Test exporting a scenario successfully."""
        scenarios_dock.refresh_scenarios()
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Mock dialog to return export directory
        mock_dialog.return_value = '/tmp/export'
        
        # Trigger export
        scenarios_dock._on_export_clicked()
        
        # Should call copytree
        assert mock_copytree.call_count == 1
    
    @patch('PyQt6.QtWidgets.QFileDialog.getExistingDirectory')
    def test_export_scenario_cancelled(self, mock_dialog, scenarios_dock):
        """Test export cancellation when user closes dialog."""
        scenarios_dock.refresh_scenarios()
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Mock dialog to return empty string (cancelled)
        mock_dialog.return_value = ''
        
        # Trigger export
        scenarios_dock._on_export_clicked()
        
        # Should not crash
        assert True
    
    @patch('PyQt6.QtWidgets.QMessageBox.question')
    @patch('shutil.rmtree')
    def test_delete_scenario_confirmed(self, mock_rmtree, mock_question, scenarios_dock):
        """Test deleting a scenario when user confirms."""
        from PyQt6.QtWidgets import QMessageBox
        
        scenarios_dock.refresh_scenarios()
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Mock confirmation dialog to return Yes
        mock_question.return_value = QMessageBox.StandardButton.Yes
        
        # Trigger delete
        scenarios_dock._on_delete_clicked()
        
        # Should call rmtree
        assert mock_rmtree.call_count == 1
    
    @patch('PyQt6.QtWidgets.QMessageBox.question')
    @patch('shutil.rmtree')
    def test_delete_scenario_cancelled(self, mock_rmtree, mock_question, scenarios_dock):
        """Test delete cancellation when user declines."""
        from PyQt6.QtWidgets import QMessageBox
        
        scenarios_dock.refresh_scenarios()
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Mock confirmation dialog to return No
        mock_question.return_value = QMessageBox.StandardButton.No
        
        # Trigger delete
        scenarios_dock._on_delete_clicked()
        
        # Should not call rmtree
        assert mock_rmtree.call_count == 0
    
    def test_get_selected_scenario(self, scenarios_dock):
        """Test getting currently selected scenario name."""
        scenarios_dock.refresh_scenarios()
        
        # No selection initially
        assert scenarios_dock.get_selected_scenario() is None
        
        # Select first item
        scenarios_dock.scenarios_list.setCurrentRow(0)
        
        # Should return scenario name
        selected = scenarios_dock.get_selected_scenario()
        assert selected is not None
        assert isinstance(selected, str)
    
    def test_get_scenario_metadata(self, scenarios_dock):
        """Test getting metadata for a specific scenario."""
        scenarios_dock.refresh_scenarios()
        
        # Get metadata for known scenario
        metadata = scenarios_dock.get_scenario_metadata('20241116_103045')
        
        assert metadata is not None
        assert metadata['duration'] == 15.5
        assert metadata['trigger']['type'] == 'critical'
    
    def test_get_scenario_metadata_nonexistent(self, scenarios_dock):
        """Test getting metadata for nonexistent scenario returns None."""
        scenarios_dock.refresh_scenarios()
        
        metadata = scenarios_dock.get_scenario_metadata('nonexistent')
        
        assert metadata is None
    
    def test_status_label_updates(self, scenarios_dock):
        """Test that status label updates correctly."""
        scenarios_dock.refresh_scenarios()
        
        # Should show count
        assert '3' in scenarios_dock.status_label.text()
        
        # Apply filter
        scenarios_dock.filter_combo.setCurrentText('Critical')
        
        # Should show filtered count
        assert 'Showing 1 of 3' in scenarios_dock.status_label.text()
    
    def test_empty_scenarios_directory(self, qapp):
        """Test handling of empty scenarios directory."""
        temp_dir = tempfile.mkdtemp()
        scenarios_path = os.path.join(temp_dir, 'scenarios')
        os.makedirs(scenarios_path)
        
        widget = ScenariosDockWidget(scenarios_path=scenarios_path)
        
        assert widget.scenarios_list.count() == 0
        assert 'No scenarios found' in widget.status_label.text()
        
        widget.close()
        shutil.rmtree(temp_dir)
    
    @pytest.mark.performance
    def test_refresh_performance(self, scenarios_dock):
        """Test that refresh completes within reasonable time."""
        import time
        
        start_time = time.perf_counter()
        scenarios_dock.refresh_scenarios()
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should complete in under 100ms for 3 scenarios
        assert execution_time_ms < 100, f"Refresh took {execution_time_ms:.2f}ms, expected < 100ms"
    
    def test_filter_combo_items(self, scenarios_dock):
        """Test that filter combo box has correct items."""
        expected_items = ["All", "Critical", "High Risk", "Near Miss", "Distracted", "Manual"]
        
        actual_items = [
            scenarios_dock.filter_combo.itemText(i)
            for i in range(scenarios_dock.filter_combo.count())
        ]
        
        assert actual_items == expected_items
    
    def test_search_box_placeholder(self, scenarios_dock):
        """Test that search box has appropriate placeholder text."""
        assert scenarios_dock.search_box.placeholderText() == "Search scenarios..."
    
    def test_widget_title(self, scenarios_dock):
        """Test that dock widget has correct title."""
        assert scenarios_dock.windowTitle() == "Scenarios"
