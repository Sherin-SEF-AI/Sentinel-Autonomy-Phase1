# Configuration Dock Widget - Integration Example

## Complete Integration with Main Window

This document shows how to integrate the Configuration Dock Widget into the SENTINEL main window.

## Step 1: Import the Widget

```python
# In src/gui/main_window.py
from .widgets import ConfigurationDockWidget
```

## Step 2: Add to Main Window Class

```python
class SENTINELMainWindow(QMainWindow):
    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        
        # ... existing initialization ...
        
        # Create configuration dock
        self.config_dock = None
        self._create_configuration_dock()
        
        # ... rest of initialization ...
    
    def _create_configuration_dock(self):
        """Create and configure the configuration dock widget"""
        # Create dock
        self.config_dock = ConfigurationDockWidget('configs/default.yaml')
        
        # Connect signals
        self.config_dock.config_changed.connect(self._on_config_changed)
        self.config_dock.config_saved.connect(self._on_config_saved)
        
        # Add to window (right side by default)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.config_dock
        )
        
        # Initially hidden (user can show via menu)
        self.config_dock.setVisible(False)
        
        logger.info("Configuration dock created")
    
    def _on_config_changed(self, config: dict):
        """Handle configuration changes"""
        logger.info("Configuration changed - applying updates")
        
        # Update system components with new configuration
        # This is where you'd update your SENTINEL system
        
        # Example: Update detection threshold
        if hasattr(self, 'sentinel_worker'):
            detection_threshold = config.get('models', {}).get('detection', {}).get('confidence_threshold')
            if detection_threshold is not None:
                # Send update to worker thread
                self.sentinel_worker.update_detection_threshold(detection_threshold)
        
        # Show notification in status bar
        self.statusBar().showMessage("Configuration updated", 3000)
    
    def _on_config_saved(self, config_path: str):
        """Handle configuration save"""
        logger.info(f"Configuration saved: {config_path}")
        self.statusBar().showMessage(f"Configuration saved to {config_path}", 5000)
```

## Step 3: Add Menu Actions

```python
def _create_menus(self):
    """Create menu bar with all menus"""
    menubar = self.menuBar()
    
    # ... existing menus ...
    
    # Tools Menu (or create Configuration menu)
    tools_menu = menubar.addMenu("&Tools")
    
    # ... existing actions ...
    
    tools_menu.addSeparator()
    
    # Configuration action
    self.action_show_config = QAction("&Configuration...", self)
    self.action_show_config.setShortcut(QKeySequence("Ctrl+,"))
    self.action_show_config.triggered.connect(self._on_show_configuration)
    tools_menu.addAction(self.action_show_config)
    
    # ... rest of menus ...

def _on_show_configuration(self):
    """Show configuration dock"""
    if self.config_dock:
        self.config_dock.setVisible(True)
        self.config_dock.raise_()
        logger.info("Configuration dock shown")
```

## Step 4: Add to View Menu

```python
def _create_menus(self):
    # ... in View Menu section ...
    
    view_menu = menubar.addMenu("&View")
    
    # ... existing view actions ...
    
    # Dock widgets submenu
    self.docks_menu = view_menu.addMenu("&Dock Widgets")
    
    # Add configuration dock toggle
    if self.config_dock:
        config_dock_action = self.config_dock.toggleViewAction()
        config_dock_action.setText("Configuration")
        self.docks_menu.addAction(config_dock_action)
```

## Step 5: Handle Close Event

```python
def closeEvent(self, event):
    """Handle window close event"""
    # Check for unsaved configuration changes
    if self.config_dock and self.config_dock.has_changes():
        reply = QMessageBox.question(
            self,
            'Unsaved Configuration',
            'Configuration has unsaved changes. Save before closing?',
            QMessageBox.StandardButton.Yes | 
            QMessageBox.StandardButton.No | 
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Trigger save
            self.config_dock._on_save_clicked()
        elif reply == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
    
    # ... rest of close handling ...
    
    self._save_state()
    event.accept()
```

## Step 6: Integrate with SENTINEL Worker

```python
class SentinelWorker(QThread):
    """Worker thread for SENTINEL system"""
    
    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
    
    def update_detection_threshold(self, threshold: float):
        """Update detection threshold in real-time"""
        if hasattr(self, 'detector'):
            self.detector.set_confidence_threshold(threshold)
            logger.info(f"Detection threshold updated: {threshold}")
    
    def update_risk_threshold(self, threshold: float):
        """Update risk assessment threshold"""
        if hasattr(self, 'intelligence_engine'):
            self.intelligence_engine.set_critical_threshold(threshold)
            logger.info(f"Risk threshold updated: {threshold}")
    
    def reload_config(self, config: dict):
        """Reload complete configuration"""
        self.config = config
        # Reinitialize components with new config
        logger.info("Configuration reloaded")
```

## Step 7: Connect Worker to Config Changes

```python
def _on_config_changed(self, config: dict):
    """Handle configuration changes"""
    logger.info("Configuration changed - applying updates")
    
    if not hasattr(self, 'sentinel_worker') or not self.sentinel_worker:
        return
    
    # Update detection parameters
    detection_config = config.get('models', {}).get('detection', {})
    if 'confidence_threshold' in detection_config:
        self.sentinel_worker.update_detection_threshold(
            detection_config['confidence_threshold']
        )
    
    # Update risk parameters
    risk_config = config.get('risk_assessment', {})
    thresholds = risk_config.get('thresholds', {})
    if 'critical' in thresholds:
        self.sentinel_worker.update_risk_threshold(
            thresholds['critical']
        )
    
    # Update alert parameters
    alerts_config = config.get('alerts', {})
    suppression = alerts_config.get('suppression', {})
    if 'duplicate_window' in suppression:
        self.sentinel_worker.update_alert_suppression(
            suppression['duplicate_window']
        )
    
    self.statusBar().showMessage("Configuration updated", 3000)
```

## Complete Example

Here's a complete minimal example:

```python
# src/gui/main_window.py

import logging
from typing import Optional
from PyQt6.QtWidgets import QMainWindow, QMessageBox
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QKeySequence

from .widgets import LiveMonitorWidget, ConfigurationDockWidget
from .themes import ThemeManager

logger = logging.getLogger(__name__)


class SENTINELMainWindow(QMainWindow):
    """Main application window for SENTINEL system"""
    
    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        
        self.settings = QSettings('SENTINEL', 'SentinelGUI')
        self.theme_manager = theme_manager
        self.system_running = False
        
        # Initialize UI
        self._init_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_status_bar()
        self._create_docks()
        
        # Apply theme
        self.theme_manager.apply_theme()
        
        # Restore state
        self._restore_state()
        
        logger.info("SENTINEL Main Window initialized")
    
    def _init_ui(self):
        """Initialize the main UI structure"""
        self.setWindowTitle("SENTINEL - Contextual Safety Intelligence Platform")
        self.setMinimumSize(1280, 720)
        
        # Create central widget
        self.live_monitor = LiveMonitorWidget()
        self.setCentralWidget(self.live_monitor)
    
    def _create_docks(self):
        """Create all dock widgets"""
        # Configuration dock
        self.config_dock = ConfigurationDockWidget('configs/default.yaml')
        self.config_dock.config_changed.connect(self._on_config_changed)
        self.config_dock.config_saved.connect(self._on_config_saved)
        
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.config_dock
        )
        self.config_dock.setVisible(False)
        
        # Add other docks here...
        
        logger.debug("Dock widgets created")
    
    def _create_menus(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        # Configuration action
        self.action_show_config = QAction("&Configuration...", self)
        self.action_show_config.setShortcut(QKeySequence("Ctrl+,"))
        self.action_show_config.triggered.connect(
            lambda: self.config_dock.setVisible(True)
        )
        tools_menu.addAction(self.action_show_config)
        
        # View Menu
        view_menu = menubar.addMenu("&View")
        
        # Dock widgets submenu
        docks_menu = view_menu.addMenu("&Dock Widgets")
        
        # Add configuration dock toggle
        config_dock_action = self.config_dock.toggleViewAction()
        config_dock_action.setText("Configuration")
        docks_menu.addAction(config_dock_action)
    
    def _create_toolbar(self):
        """Create toolbar"""
        # Implementation...
        pass
    
    def _create_status_bar(self):
        """Create status bar"""
        self.statusBar().showMessage("System: Idle")
    
    def _on_config_changed(self, config: dict):
        """Handle configuration changes"""
        logger.info("Configuration changed")
        
        # Apply configuration updates to system
        # ... implementation ...
        
        self.statusBar().showMessage("Configuration updated", 3000)
    
    def _on_config_saved(self, config_path: str):
        """Handle configuration save"""
        logger.info(f"Configuration saved: {config_path}")
        self.statusBar().showMessage(
            f"Configuration saved to {config_path}", 
            5000
        )
    
    def _restore_state(self):
        """Restore window state"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        state = self.settings.value('windowState')
        if state:
            self.restoreState(state)
    
    def _save_state(self):
        """Save window state"""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())
    
    def closeEvent(self, event):
        """Handle window close"""
        # Check for unsaved config changes
        if self.config_dock and self.config_dock.has_changes():
            reply = QMessageBox.question(
                self,
                'Unsaved Configuration',
                'Save configuration changes before closing?',
                QMessageBox.StandardButton.Yes | 
                QMessageBox.StandardButton.No | 
                QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.config_dock._on_save_clicked()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        
        self._save_state()
        event.accept()
```

## Testing the Integration

```python
# test_main_window_with_config.py

import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import SENTINELMainWindow
from gui.themes import ThemeManager

def main():
    app = QApplication(sys.argv)
    
    theme_manager = ThemeManager()
    window = SENTINELMainWindow(theme_manager)
    
    # Show configuration dock for testing
    window.config_dock.setVisible(True)
    
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
```

## Best Practices

1. **Signal Connections**: Always connect config_changed signal to update system
2. **Validation**: Validate configuration before applying to system
3. **Error Handling**: Handle configuration errors gracefully
4. **User Feedback**: Show status messages for config operations
5. **Persistence**: Save window state including dock visibility
6. **Thread Safety**: Use signals for cross-thread config updates
7. **Backup**: Always create backups before saving
8. **Testing**: Test configuration changes in isolation first

## Troubleshooting

### Configuration Not Updating System
- Verify signal connections
- Check worker thread is running
- Enable debug logging

### Dock Not Visible
- Check initial visibility setting
- Verify dock is added to window
- Check View menu toggle action

### Changes Not Persisting
- Verify _save_state() is called on close
- Check QSettings configuration
- Ensure write permissions

## Summary

The Configuration Dock Widget integrates seamlessly with the SENTINEL main window, providing:
- Easy access via menu and keyboard shortcut
- Real-time parameter updates
- Unsaved changes detection
- Profile management
- Professional user experience

Follow this integration pattern for a robust configuration management system.
