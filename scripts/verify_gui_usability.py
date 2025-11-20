#!/usr/bin/env python3
"""
GUI Usability Testing Script

Performs comprehensive usability testing including:
- GUI responsiveness
- Keyboard shortcuts
- Multi-monitor support
- Accessibility features

Requirements: 13.4, 13.5, 13.6
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence
from unittest.mock import patch


class UsabilityTester:
    """Automated usability testing."""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def test(self, name, func):
        """Run a test and record result."""
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print('='*60)
        
        try:
            start = time.time()
            func()
            duration = time.time() - start
            
            self.results.append({
                'name': name,
                'status': 'PASS',
                'duration': duration
            })
            self.passed += 1
            print(f"✓ PASS ({duration:.3f}s)")
            
        except AssertionError as e:
            self.results.append({
                'name': name,
                'status': 'FAIL',
                'error': str(e)
            })
            self.failed += 1
            print(f"✗ FAIL: {e}")
            
        except Exception as e:
            self.results.append({
                'name': name,
                'status': 'ERROR',
                'error': str(e)
            })
            self.failed += 1
            print(f"✗ ERROR: {e}")
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("USABILITY TEST SUMMARY")
        print('='*60)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {self.passed/(self.passed+self.failed)*100:.1f}%")
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for result in self.results:
                if result['status'] != 'PASS':
                    print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")


def test_gui_responsiveness(app, tester):
    """Test GUI responsiveness."""
    
    def test_window_creation():
        """Test window creation time."""
        from src.gui.main_window import SENTINELMainWindow
        
        start = time.time()
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        creation_time = time.time() - start
        
        print(f"  Window creation time: {creation_time:.3f}s")
        assert creation_time < 2.0, f"Window creation too slow: {creation_time:.3f}s"
        
        window.close()
        return window
    
    def test_widget_updates():
        """Test widget update speed."""
        from src.gui.widgets.circular_gauge import CircularGaugeWidget
        
        gauge = CircularGaugeWidget(min_value=0, max_value=100)
        gauge.show()
        
        start = time.time()
        for i in range(20):
            gauge.setValue(i * 5)
            app.processEvents()
        update_time = time.time() - start
        
        print(f"  20 widget updates: {update_time:.3f}s")
        assert update_time < 0.2, f"Widget updates too slow: {update_time:.3f}s"
        
        gauge.close()
    
    def test_menu_response():
        """Test menu response time."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        start = time.time()
        for action in window.menuBar().actions()[:3]:
            action.trigger()
            app.processEvents()
        response_time = time.time() - start
        
        print(f"  Menu response time: {response_time:.3f}s")
        assert response_time < 0.1, f"Menu response too slow: {response_time:.3f}s"
        
        window.close()
    
    tester.test("Window Creation Speed", test_window_creation)
    tester.test("Widget Update Speed", test_widget_updates)
    tester.test("Menu Response Time", test_menu_response)


def test_keyboard_shortcuts(app, tester):
    """Test keyboard shortcuts."""
    
    def test_f5_start():
        """Test F5 start shortcut."""
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
        shortcut = start_action.shortcut()
        print(f"  Start shortcut: {shortcut.toString()}")
        assert shortcut == QKeySequence(Qt.Key.Key_F5), "F5 not assigned to Start"
        
        window.close()
    
    def test_f6_stop():
        """Test F6 stop shortcut."""
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
        shortcut = stop_action.shortcut()
        print(f"  Stop shortcut: {shortcut.toString()}")
        assert shortcut == QKeySequence(Qt.Key.Key_F6), "F6 not assigned to Stop"
        
        window.close()
    
    def test_shortcut_uniqueness():
        """Test that shortcuts are unique."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        shortcuts = []
        
        # Collect shortcuts
        for action in window.toolbar.actions():
            if not action.shortcut().isEmpty():
                shortcuts.append(action.shortcut().toString())
        
        for menu_action in window.menuBar().actions():
            menu = menu_action.menu()
            if menu:
                for action in menu.actions():
                    if not action.shortcut().isEmpty():
                        shortcuts.append(action.shortcut().toString())
        
        print(f"  Total shortcuts: {len(shortcuts)}")
        print(f"  Unique shortcuts: {len(set(shortcuts))}")
        
        duplicates = [s for s in shortcuts if shortcuts.count(s) > 1]
        assert len(shortcuts) == len(set(shortcuts)), \
            f"Duplicate shortcuts: {set(duplicates)}"
        
        window.close()
    
    tester.test("F5 Start Shortcut", test_f5_start)
    tester.test("F6 Stop Shortcut", test_f6_stop)
    tester.test("Shortcut Uniqueness", test_shortcut_uniqueness)


def test_multi_monitor_support(app, tester):
    """Test multi-monitor support."""
    
    def test_screen_detection():
        """Test screen detection."""
        screens = app.screens()
        print(f"  Detected screens: {len(screens)}")
        
        for i, screen in enumerate(screens):
            geometry = screen.geometry()
            print(f"    Screen {i}: {geometry.width()}x{geometry.height()} "
                  f"at ({geometry.x()}, {geometry.y()})")
        
        assert len(screens) >= 1, "No screens detected"
    
    def test_geometry_persistence():
        """Test window geometry persistence."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            # Create and position window
            window1 = SENTINELMainWindow()
            window1.setGeometry(100, 100, 1280, 720)
            geometry = window1.saveGeometry()
            window1.close()
            
            # Restore in new window
            window2 = SENTINELMainWindow()
            window2.restoreGeometry(geometry)
            
            print(f"  Saved geometry: 1280x720")
            print(f"  Restored geometry: {window2.width()}x{window2.height()}")
            
            assert abs(window2.width() - 1280) < 50, "Width not restored"
            assert abs(window2.height() - 720) < 50, "Height not restored"
            
            window2.close()
    
    def test_dock_floating():
        """Test dock widget floating."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
            
            # Test floating capability
            if hasattr(window, 'performance_dock'):
                dock = window.performance_dock
                
                dock.setFloating(True)
                assert dock.isFloating(), "Dock should float"
                print(f"  Dock can float: Yes")
                
                dock.setFloating(False)
                assert not dock.isFloating(), "Dock should dock"
                print(f"  Dock can dock: Yes")
            else:
                print(f"  No docks to test")
            
            window.close()
    
    def test_state_persistence():
        """Test window state persistence."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window1 = SENTINELMainWindow()
            state = window1.saveState()
            window1.close()
            
            window2 = SENTINELMainWindow()
            result = window2.restoreState(state)
            
            print(f"  State persistence: {'Yes' if result else 'No'}")
            assert result, "State should persist"
            
            window2.close()
    
    tester.test("Screen Detection", test_screen_detection)
    tester.test("Geometry Persistence", test_geometry_persistence)
    tester.test("Dock Floating", test_dock_floating)
    tester.test("State Persistence", test_state_persistence)


def test_accessibility_features(app, tester):
    """Test accessibility features."""
    
    def test_tooltips():
        """Test tooltip presence."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        tooltip_count = 0
        missing_tooltips = []
        
        for action in window.toolbar.actions():
            if not action.isSeparator():
                if action.toolTip():
                    tooltip_count += 1
                else:
                    missing_tooltips.append(action.text())
        
        print(f"  Actions with tooltips: {tooltip_count}")
        if missing_tooltips:
            print(f"  Missing tooltips: {missing_tooltips}")
        
        assert len(missing_tooltips) == 0, \
            f"Actions missing tooltips: {missing_tooltips}"
        
        window.close()
    
    def test_status_bar():
        """Test status bar functionality."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        assert window.statusBar() is not None, "Status bar missing"
        
        window.statusBar().showMessage("Test message")
        app.processEvents()
        
        message = window.statusBar().currentMessage()
        print(f"  Status bar message: '{message}'")
        assert message == "Test message", "Status bar not working"
        
        window.close()
    
    def test_color_contrast():
        """Test color contrast."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        palette = window.palette()
        bg = palette.color(palette.ColorRole.Window)
        fg = palette.color(palette.ColorRole.WindowText)
        
        print(f"  Background: {bg.name()}")
        print(f"  Foreground: {fg.name()}")
        
        assert bg != fg, "Background and foreground colors are the same"
        
        window.close()
    
    def test_keyboard_navigation():
        """Test keyboard navigation."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        menu_bar = window.menuBar()
        assert menu_bar is not None, "Menu bar missing"
        
        menu_count = len(menu_bar.actions())
        print(f"  Menus available: {menu_count}")
        assert menu_count > 0, "No menus available"
        
        window.close()
    
    def test_minimum_size():
        """Test minimum window size."""
        from src.gui.main_window import SENTINELMainWindow
        
        with patch('src.gui.main_window.SentinelWorker'):
            window = SENTINELMainWindow()
        
        min_size = window.minimumSize()
        print(f"  Minimum size: {min_size.width()}x{min_size.height()}")
        
        assert min_size.width() >= 640, "Minimum width too small"
        assert min_size.height() >= 480, "Minimum height too small"
        
        window.close()
    
    tester.test("Tooltips Present", test_tooltips)
    tester.test("Status Bar Functional", test_status_bar)
    tester.test("Color Contrast", test_color_contrast)
    tester.test("Keyboard Navigation", test_keyboard_navigation)
    tester.test("Minimum Window Size", test_minimum_size)


def main():
    """Main test execution."""
    print("="*60)
    print("GUI USABILITY TESTING")
    print("="*60)
    print("\nThis script tests:")
    print("  - GUI responsiveness")
    print("  - Keyboard shortcuts")
    print("  - Multi-monitor support")
    print("  - Accessibility features")
    print("\nRequirements: 13.4, 13.5, 13.6")
    
    # Create QApplication
    app = QApplication(sys.argv)
    
    # Create tester
    tester = UsabilityTester()
    
    # Run test suites
    print("\n" + "="*60)
    print("1. GUI RESPONSIVENESS TESTS")
    print("="*60)
    test_gui_responsiveness(app, tester)
    
    print("\n" + "="*60)
    print("2. KEYBOARD SHORTCUTS TESTS")
    print("="*60)
    test_keyboard_shortcuts(app, tester)
    
    print("\n" + "="*60)
    print("3. MULTI-MONITOR SUPPORT TESTS")
    print("="*60)
    test_multi_monitor_support(app, tester)
    
    print("\n" + "="*60)
    print("4. ACCESSIBILITY FEATURES TESTS")
    print("="*60)
    test_accessibility_features(app, tester)
    
    # Print summary
    tester.print_summary()
    
    # Return exit code
    return 0 if tester.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
