# AlertsPanel Integration Guide

## Overview
This guide shows how to integrate the AlertsPanel widget into the SENTINEL main window.

## Quick Integration

### Step 1: Import the Widget
```python
from src.gui.widgets import AlertsPanel
```

### Step 2: Add to Main Window
```python
class SENTINELMainWindow(QMainWindow):
    def __init__(self, theme_manager: ThemeManager):
        super().__init__()
        # ... existing initialization ...
        
        # Create alerts panel as a dock widget
        self.alerts_dock = QDockWidget("Alerts", self)
        self.alerts_panel = AlertsPanel()
        self.alerts_dock.setWidget(self.alerts_panel)
        
        # Add to right side of window
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.alerts_dock)
        
        # Add to View menu
        self.docks_menu.addAction(self.alerts_dock.toggleViewAction())
```

### Step 3: Connect to Alert System
```python
class SENTINELMainWindow(QMainWindow):
    def process_alerts(self, alerts: List[Alert]):
        """Process alerts from the alert system"""
        for alert in alerts:
            self.alerts_panel.add_alert(alert)
```

## Full Example

```python
"""
Example: SENTINEL Main Window with AlertsPanel
"""

from PyQt6.QtWidgets import QMainWindow, QDockWidget
from PyQt6.QtCore import Qt
from src.gui.widgets import AlertsPanel, LiveMonitorWidget
from src.core.data_structures import Alert
import time


class SENTINELMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SENTINEL - Safety Intelligence Platform")
        self.setGeometry(100, 100, 1920, 1080)
        
        # Central widget (live monitor)
        self.live_monitor = LiveMonitorWidget()
        self.setCentralWidget(self.live_monitor)
        
        # Create alerts panel dock
        self._create_alerts_dock()
        
        # Create other docks...
        self._create_driver_state_dock()
        self._create_risk_panel_dock()
    
    def _create_alerts_dock(self):
        """Create alerts panel dock widget"""
        self.alerts_dock = QDockWidget("Alerts", self)
        self.alerts_panel = AlertsPanel()
        self.alerts_dock.setWidget(self.alerts_panel)
        
        # Add to right side
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.alerts_dock
        )
        
        # Configure dock
        self.alerts_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        
        # Set minimum size
        self.alerts_dock.setMinimumWidth(300)
    
    def _create_driver_state_dock(self):
        """Create driver state panel dock"""
        # Implementation...
        pass
    
    def _create_risk_panel_dock(self):
        """Create risk assessment panel dock"""
        # Implementation...
        pass
    
    def on_alerts_received(self, alerts: List[Alert]):
        """
        Handle alerts from the SENTINEL system.
        
        This method should be connected to the alert system's signal.
        """
        for alert in alerts:
            self.alerts_panel.add_alert(alert)
    
    def simulate_alert(self, urgency: str = 'warning'):
        """Simulate an alert (for testing)"""
        alert = Alert(
            timestamp=time.time(),
            urgency=urgency,
            modalities=['visual', 'audio'],
            message=f'Test {urgency} alert',
            hazard_id=1,
            dismissed=False
        )
        self.alerts_panel.add_alert(alert)


# Usage
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = SENTINELMainWindow()
    window.show()
    
    # Simulate some alerts
    window.simulate_alert('critical')
    window.simulate_alert('warning')
    window.simulate_alert('info')
    
    sys.exit(app.exec())
```

## Worker Thread Integration

### Connect Worker Signals
```python
class SENTINELMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... initialization ...
        
        # Create worker thread
        self.sentinel_worker = SentinelWorker()
        
        # Connect alerts signal
        self.sentinel_worker.alerts_ready.connect(
            self.on_alerts_received
        )
        
        # Start worker
        self.sentinel_worker.start()
    
    def on_alerts_received(self, alerts: List[Alert]):
        """Handle alerts from worker thread"""
        for alert in alerts:
            self.alerts_panel.add_alert(alert)
```

## Configuration

### Audio Settings
```python
# Disable audio
alerts_panel.set_audio_enabled(False)

# Set volume to 50%
alerts_panel.set_volume(0.5)

# Mute via button
alerts_panel.mute_button.click()
```

### Statistics Access
```python
# Get current statistics
stats = alerts_panel.get_statistics()
print(f"Total alerts: {stats['total']}")
print(f"Critical: {stats['critical']}")
print(f"Warnings: {stats['warning']}")
print(f"Info: {stats['info']}")
print(f"False positives: {stats['false_positives']}")
```

### History Management
```python
# Clear history
alerts_panel.clear_history()

# Mark false positive
alert_id = 5
alerts_panel.mark_false_positive(alert_id)
```

## Styling

### Theme Integration
The AlertsPanel automatically inherits the application theme. Alert colors are hardcoded for visibility:
- Critical: Red (#ff4444)
- Warning: Orange (#ffaa00)
- Info: Blue (#4488ff)

### Custom Styling
```python
# Apply custom stylesheet to dock
alerts_dock.setStyleSheet("""
    QDockWidget {
        font-size: 11pt;
    }
    QDockWidget::title {
        background-color: #2a2a2a;
        padding: 5px;
    }
""")
```

## Sound Files

### Required Sound Files
Place sound files in the `sounds/` directory:
- `sounds/alarm.wav` - Critical alerts (urgent, attention-grabbing)
- `sounds/beep.wav` - Warning alerts (noticeable but not alarming)
- `sounds/notification.wav` - Info alerts (subtle notification)

### Sound File Format
- Format: WAV (recommended)
- Sample rate: 44.1 kHz or 48 kHz
- Bit depth: 16-bit
- Channels: Mono or Stereo
- Duration: 0.5-2 seconds

### Fallback Behavior
If sound files are not found, the system falls back to `QApplication.beep()`.

## Signals

### Available Signals
```python
# Alert dismissed (future feature)
alerts_panel.alert_dismissed.connect(on_alert_dismissed)

# False positive marked
alerts_panel.false_positive_marked.connect(on_false_positive)
```

### Signal Handlers
```python
def on_alert_dismissed(alert_id: int):
    """Handle alert dismissal"""
    print(f"Alert {alert_id} dismissed")

def on_false_positive(alert_id: int):
    """Handle false positive marking"""
    print(f"Alert {alert_id} marked as false positive")
    # Update ML model, log for analysis, etc.
```

## Best Practices

### 1. Thread Safety
Always add alerts from the GUI thread. If receiving alerts from a worker thread, use signals:
```python
# Worker thread
class SentinelWorker(QThread):
    alerts_ready = pyqtSignal(list)
    
    def run(self):
        alerts = self.process_frame()
        self.alerts_ready.emit(alerts)  # Thread-safe

# Main window
worker.alerts_ready.connect(self.on_alerts_received)
```

### 2. Alert Rate Limiting
Avoid overwhelming the user with too many alerts:
```python
from collections import deque
from time import time

class AlertRateLimiter:
    def __init__(self, max_per_second=2):
        self.max_per_second = max_per_second
        self.recent_alerts = deque()
    
    def should_show_alert(self, alert: Alert) -> bool:
        now = time()
        
        # Remove old alerts
        while self.recent_alerts and self.recent_alerts[0] < now - 1.0:
            self.recent_alerts.popleft()
        
        # Check rate
        if len(self.recent_alerts) >= self.max_per_second:
            return False
        
        self.recent_alerts.append(now)
        return True
```

### 3. Alert Prioritization
Show only the most important alerts:
```python
def add_prioritized_alert(self, alert: Alert):
    """Add alert with priority filtering"""
    # Always show critical
    if alert.urgency == 'critical':
        self.alerts_panel.add_alert(alert)
    # Show warning if not too many recent
    elif alert.urgency == 'warning':
        if self.should_show_warning():
            self.alerts_panel.add_alert(alert)
    # Show info only if explicitly enabled
    elif alert.urgency == 'info' and self.show_info_alerts:
        self.alerts_panel.add_alert(alert)
```

### 4. Periodic Cleanup
Clear old alerts periodically:
```python
def setup_alert_cleanup(self):
    """Setup periodic alert history cleanup"""
    self.cleanup_timer = QTimer()
    self.cleanup_timer.timeout.connect(self.cleanup_old_alerts)
    self.cleanup_timer.start(300000)  # Every 5 minutes

def cleanup_old_alerts(self):
    """Remove alerts older than 1 hour"""
    cutoff_time = time() - 3600
    self.alerts_panel.alert_history = [
        entry for entry in self.alerts_panel.alert_history
        if entry['alert'].timestamp > cutoff_time
    ]
```

## Testing

### Manual Testing
Use the standalone test application:
```bash
python3 test_alerts_panel.py
```

### Unit Testing
Run the unit tests:
```bash
python3 -m pytest tests/unit/test_alerts_panel.py -v
```

### Integration Testing
Test with the full SENTINEL system:
```python
# In main window
def test_alert_integration(self):
    """Test alert integration"""
    # Create test alert
    alert = Alert(
        timestamp=time.time(),
        urgency='critical',
        modalities=['visual', 'audio'],
        message='Test integration',
        hazard_id=999,
        dismissed=False
    )
    
    # Add to panel
    self.alerts_panel.add_alert(alert)
    
    # Verify
    assert self.alerts_panel.total_alerts > 0
    assert len(self.alerts_panel.alert_history) > 0
```

## Troubleshooting

### Audio Not Playing
1. Check sound files exist in `sounds/` directory
2. Verify audio is not muted
3. Check volume level
4. Verify QMediaPlayer is properly initialized
5. Check system audio settings

### Alerts Not Displaying
1. Verify `add_alert()` is called from GUI thread
2. Check alert object is properly formatted
3. Verify panel is visible (not hidden by dock)
4. Check filter settings

### Performance Issues
1. Limit alert rate (max 2-3 per second)
2. Clear old alerts periodically
3. Use filter to reduce displayed alerts
4. Disable auto-scroll if needed

## Summary

The AlertsPanel is a fully-featured widget for displaying and managing safety alerts in the SENTINEL system. It provides:
- ✓ Visual display with color coding
- ✓ Audio feedback with controls
- ✓ Statistics tracking
- ✓ History management
- ✓ Export functionality
- ✓ Critical alert effects

Integration is straightforward and the widget is ready for production use.
