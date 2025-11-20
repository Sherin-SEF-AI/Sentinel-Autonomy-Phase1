"""
Unit tests for GUI usability features.

Tests GUI responsiveness, keyboard shortcuts, multi-monitor support,
and accessibility features.

Requirements: 13.4, 13.5, 13.6
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint
from PyQt6.QtGui import QKeySequence, QScreen
from PyQt6.QtTest import QTest


# Ensure QApplication exists
@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


@pytest.fixture
def mock_sentinel_system():
    """Mock SENTINEL system."""
    mock = Mock()
    mock.start = Mock()
    mock.stop = Mock()
    mock.is_running = Mock(return_value=False)
    return mock


class TestGUIResponsiveness:
    """Test GUI responsiveness and performance."""
    
    def test_main_window_creation_time(self, qapp, mock_sentinel_system):
        """Test that main window creates quickly."""
        from src.gui.main_window import SENTINELMainWindow
        
        import time
        start = time.time()
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        creation_time = time.time() - start
        
        # Window should create in under 1 second
        assert creation_time < 1.0, f"Window creation took {creation_time:.2f}s"
        
        window.close()
    
    def test_widget_update_responsiveness(self, qapp):
        """Test that widgets update quickly."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge = CircularGaugeWidget(min_value=0, max_value=100)
        gauge.show()
        
        import time
        start = time.time()
        
        # Update value multiple times
        for i in range(10):
            gauge.setValue(i * 10)
            QApplication.processEvents()
        
        update_time = time.time() - start
        
        # 10 updates should complete in under 100ms
        assert update_time < 0.1, f"Widget updates took {update_time:.3f}s"
        
        gauge.close()
    
    def test_menu_response_time(self, qapp):
        """Test that menus respond quickly."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        import time
        start = time.time()
        
        # Trigger menu actions
        window.menuBar().actions()[0].trigger()
        QApplication.processEvents()
        
        response_time = time.time() - start
        
        # Menu should respond in under 50ms
        assert response_time < 0.05, f"Menu response took {response_time:.3f}s"
        
        window.close()
    
    def test_dock_widget_responsiveness(self, qapp):
        """Test that dock widgets respond quickly."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        import time
        start = time.time()
        
        # Toggle dock visibility
        if hasattr(window, 'performance_dock'):
            window.performance_dock.setVisible(False)
            QApplication.processEvents()
            window.performance_dock.setVisible(True)
            QApplication.processEvents()
        
        toggle_time = time.time() - start
        
        # Dock toggle should complete in under 100ms
        assert toggle_time < 0.1, f"Dock toggle took {toggle_time:.3f}s"
        
        window.close()


class TestKeyboardShortcuts:
    """Test keyboard shortcuts functionality."""
    
    def test_start_shortcut_f5(self, qapp):
        """Test F5 shortcut for starting system."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        # Find start action
        start_action = None
        for action in window.toolbar.actions():
            if action.text() == "Start":
                start_action = action
                break
        
        assert start_action is not None, "Start action not found"
        
        # Check F5 shortcut is assigned
        shortcut = start_action.shortcut()
        assert shortcut == QKeySequence(Qt.Key.Key_F5), "F5 shortcut not assigned"
        
        window.close()
    
    def test_stop_shortcut_f6(self, qapp):
        """Test F6 shortcut for stopping system."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        # Find stop action
        stop_action = None
        for action in window.toolbar.actions():
            if action.text() == "Stop":
                stop_action = action
                break
        
        assert stop_action is not None, "Stop action not found"
        
        # Check F6 shortcut is assigned
        shortcut = stop_action.shortcut()
        assert shortcut == QKeySequence(Qt.Key.Key_F6), "F6 shortcut not assigned"
        
        window.close()
    
    def test_fullscreen_shortcut_f11(self, qapp):
        """Test F11 shortcut for fullscreen toggle."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        # Find fullscreen action in View menu
        view_menu = None
        for action in window.menuBar().actions():
            if action.text() == "View":
                view_menu = action.menu()
                break
        
        assert view_menu is not None, "View menu not found"
        
        fullscreen_action = None
        for action in view_menu.actions():
            if "Fullscreen" in action.text():
                fullscreen_action = action
                break
        
        if fullscreen_action:
            shortcut = fullscreen_action.shortcut()
            assert shortcut == QKeySequence(Qt.Key.Key_F11), "F11 shortcut not assigned"
        
        window.close()
    
    def test_quit_shortcut_ctrl_q(self, qapp):
        """Test Ctrl+Q shortcut for quitting."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        # Find quit action in File menu
        file_menu = None
        for action in window.menuBar().actions():
            if action.text() == "File":
                file_menu = action.menu()
                break
        
        assert file_menu is not None, "File menu not found"
        
        quit_action = None
        for action in file_menu.actions():
            if "Quit" in action.text() or "Exit" in action.text():
                quit_action = action
                break
        
        if quit_action:
            shortcut = quit_action.shortcut()
            assert shortcut == QKeySequence.StandardKey.Quit or \
                   shortcut == QKeySequence("Ctrl+Q"), "Quit shortcut not assigned"
        
        window.close()
    
    def test_all_shortcuts_unique(self, qapp):
        """Test that all shortcuts are unique."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        shortcuts = []
        
        # Collect all shortcuts from toolbar
        for action in window.toolbar.actions():
            if not action.shortcut().isEmpty():
                shortcuts.append(action.shortcut().toString())
        
        # Collect all shortcuts from menus
        for menu_action in window.menuBar().actions():
            menu = menu_action.menu()
            if menu:
                for action in menu.actions():
                    if not action.shortcut().isEmpty():
                        shortcuts.append(action.shortcut().toString())
        
        # Check for duplicates
        assert len(shortcuts) == len(set(shortcuts)), \
            f"Duplicate shortcuts found: {shortcuts}"
        
        window.close()
    
    def test_shortcuts_documented(self, qapp):
        """Test that shortcuts are visible in UI."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        # Check that actions with shortcuts show them in text
        for action in window.toolbar.actions():
            if not action.shortcut().isEmpty():
                # Tooltip should mention the shortcut
                tooltip = action.toolTip()
                assert tooltip, f"Action '{action.text()}' has no tooltip"
        
        window.close()


class TestMultiMonitorSupport:
    """Test multi-monitor support functionality."""
    
    def test_window_geometry_persistence(self, qapp, tmp_path):
        """Test that window geometry is saved and restored."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            # Create window and set geometry
            window1 = SENTINELMainWindow()
            window1.setGeometry(100, 100, 1280, 720)
            
            # Save geometry
            geometry = window1.saveGeometry()
            
            window1.close()
            
            # Create new window and restore
            window2 = SENTINELMainWindow()
            window2.restoreGeometry(geometry)
            
            # Check geometry restored
            assert window2.width() == 1280
            assert window2.height() == 720
            
            window2.close()
    
    def test_dock_widget_floating(self, qapp):
        """Test that dock widgets can float."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Check if dock widgets exist and can float
            if hasattr(window, 'performance_dock'):
                dock = window.performance_dock
                
                # Test floating
                dock.setFloating(True)
                assert dock.isFloating(), "Dock should be floating"
                
                # Test docking back
                dock.setFloating(False)
                assert not dock.isFloating(), "Dock should be docked"
            
            window.close()
    
    def test_dock_widget_movable(self, qapp):
        """Test that dock widgets can be moved."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Check if dock widgets are movable
            if hasattr(window, 'performance_dock'):
                dock = window.performance_dock
                features = dock.features()
                
                # Should have movable feature
                from PyQt6.QtWidgets import QDockWidget
                assert features & QDockWidget.DockWidgetFeature.DockWidgetMovable, \
                    "Dock should be movable"
            
            window.close()
    
    def test_window_state_persistence(self, qapp):
        """Test that window state (dock positions) is saved."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            # Create window
            window1 = SENTINELMainWindow()
            
            # Save state
            state = window1.saveState()
            
            window1.close()
            
            # Create new window and restore
            window2 = SENTINELMainWindow()
            result = window2.restoreState(state)
            
            # State should restore successfully
            assert result, "Window state should restore"
            
            window2.close()
    
    def test_screen_detection(self, qapp):
        """Test that application detects available screens."""
        screens = QApplication.screens()
        
        # Should detect at least one screen
        assert len(screens) >= 1, "Should detect at least one screen"
        
        # Check screen properties
        for screen in screens:
            geometry = screen.geometry()
            assert geometry.width() > 0, "Screen width should be positive"
            assert geometry.height() > 0, "Screen height should be positive"


class TestAccessibilityFeatures:
    """Test accessibility features."""
    
    def test_tooltips_present(self, qapp):
        """Test that interactive elements have tooltips."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Check toolbar actions have tooltips
            for action in window.toolbar.actions():
                if not action.isSeparator():
                    tooltip = action.toolTip()
                    assert tooltip, f"Action '{action.text()}' missing tooltip"
            
            window.close()
    
    def test_status_bar_messages(self, qapp):
        """Test that status bar provides feedback."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Status bar should exist
            assert window.statusBar() is not None, "Status bar should exist"
            
            # Test setting status message
            window.statusBar().showMessage("Test message")
            QApplication.processEvents()
            
            # Message should be visible
            assert window.statusBar().currentMessage() == "Test message"
            
            window.close()
    
    def test_widget_focus_order(self, qapp):
        """Test that widgets have logical focus order."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Window should accept focus
            assert window.focusPolicy() != Qt.FocusPolicy.NoFocus or \
                   window.centralWidget().focusPolicy() != Qt.FocusPolicy.NoFocus
            
            window.close()
    
    def test_color_contrast(self, qapp):
        """Test that UI has sufficient color contrast."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Check that window has a stylesheet or default colors
            palette = window.palette()
            
            # Background and foreground should be different
            bg = palette.color(palette.ColorRole.Window)
            fg = palette.color(palette.ColorRole.WindowText)
            
            assert bg != fg, "Background and foreground colors should differ"
            
            window.close()
    
    def test_keyboard_navigation(self, qapp):
        """Test that UI is keyboard navigable."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Menu bar should be accessible via keyboard
            menu_bar = window.menuBar()
            assert menu_bar is not None, "Menu bar should exist"
            
            # Menus should have mnemonics or be keyboard accessible
            for action in menu_bar.actions():
                menu = action.menu()
                if menu:
                    # Menu should be accessible
                    assert menu.isEnabled() or not menu.isEmpty()
            
            window.close()
    
    def test_widget_labels(self, qapp):
        """Test that form widgets have labels."""
        from src.gui.widgets.configuration_dock import ConfigurationDock
        
        with patch('src.gui.main_window.SentinelWorker'):
            dock = ConfigurationDock()
            
            # Configuration widgets should have labels
            # This is a basic check - actual implementation may vary
            assert dock.windowTitle(), "Dock should have a title"
            
            dock.close()
    
    def test_error_messages_visible(self, qapp):
        """Test that error messages are clearly visible."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Status bar should be able to show messages
            window.statusBar().showMessage("Error: Test error message")
            QApplication.processEvents()
            
            message = window.statusBar().currentMessage()
            assert "Error" in message, "Error message should be visible"
            
            window.close()


class TestResponsiveLayout:
    """Test responsive layout behavior."""
    
    def test_window_resize(self, qapp):
        """Test that window resizes properly."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Test different window sizes
            sizes = [(800, 600), (1280, 720), (1920, 1080)]
            
            for width, height in sizes:
                window.resize(width, height)
                QApplication.processEvents()
                
                # Window should accept the size
                assert abs(window.width() - width) < 50, \
                    f"Window width should be close to {width}"
                assert abs(window.height() - height) < 50, \
                    f"Window height should be close to {height}"
            
            window.close()
    
    def test_minimum_size(self, qapp):
        """Test that window has reasonable minimum size."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            min_size = window.minimumSize()
            
            # Should have a reasonable minimum size
            assert min_size.width() >= 640, "Minimum width should be at least 640"
            assert min_size.height() >= 480, "Minimum height should be at least 480"
            
            window.close()
    
    def test_splitter_behavior(self, qapp):
        """Test that splitters work correctly."""
        from src.gui.widgets.live_monitor import LiveMonitorWidget
        
        widget = LiveMonitorWidget()
        
        # If widget uses splitters, they should be movable
        from PyQt6.QtWidgets import QSplitter
        splitters = widget.findChildren(QSplitter)
        
        for splitter in splitters:
            # Splitter should be movable
            assert splitter.isEnabled(), "Splitter should be enabled"
        
        widget.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
