"""
Standalone test for AlertsPanel widget

This script creates a test window with the AlertsPanel and simulates alerts.
"""

import sys
import time
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import QTimer

from src.gui.widgets.alerts_panel import AlertsPanel
from src.core.data_structures import Alert


class TestWindow(QMainWindow):
    """Test window for AlertsPanel"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("AlertsPanel Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create alerts panel
        self.alerts_panel = AlertsPanel()
        layout.addWidget(self.alerts_panel)
        
        # Create test buttons
        button_layout = QHBoxLayout()
        
        self.critical_btn = QPushButton("Add Critical Alert")
        self.critical_btn.clicked.connect(self.add_critical_alert)
        button_layout.addWidget(self.critical_btn)
        
        self.warning_btn = QPushButton("Add Warning Alert")
        self.warning_btn.clicked.connect(self.add_warning_alert)
        button_layout.addWidget(self.warning_btn)
        
        self.info_btn = QPushButton("Add Info Alert")
        self.info_btn.clicked.connect(self.add_info_alert)
        button_layout.addWidget(self.info_btn)
        
        self.auto_btn = QPushButton("Start Auto Alerts")
        self.auto_btn.setCheckable(True)
        self.auto_btn.clicked.connect(self.toggle_auto_alerts)
        button_layout.addWidget(self.auto_btn)
        
        layout.addLayout(button_layout)
        
        # Auto alert timer
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.add_random_alert)
        self.alert_counter = 0
        
        print("AlertsPanel test window initialized")
        print("Click buttons to add alerts or enable auto mode")
    
    def add_critical_alert(self):
        """Add a critical alert"""
        alert = Alert(
            timestamp=time.time(),
            urgency='critical',
            modalities=['visual', 'audio', 'haptic'],
            message='COLLISION IMMINENT! Vehicle ahead braking hard.',
            hazard_id=self.alert_counter,
            dismissed=False
        )
        self.alerts_panel.add_alert(alert)
        self.alert_counter += 1
        print(f"Added critical alert #{self.alert_counter}")
    
    def add_warning_alert(self):
        """Add a warning alert"""
        messages = [
            'Pedestrian detected in blind spot.',
            'Vehicle approaching from left.',
            'Driver distraction detected.',
            'Lane departure warning.',
            'Following distance too close.'
        ]
        
        alert = Alert(
            timestamp=time.time(),
            urgency='warning',
            modalities=['visual', 'audio'],
            message=messages[self.alert_counter % len(messages)],
            hazard_id=self.alert_counter,
            dismissed=False
        )
        self.alerts_panel.add_alert(alert)
        self.alert_counter += 1
        print(f"Added warning alert #{self.alert_counter}")
    
    def add_info_alert(self):
        """Add an info alert"""
        messages = [
            'Speed limit changed to 50 km/h.',
            'Traffic light ahead turning yellow.',
            'Parking space detected on right.',
            'Navigation: Turn right in 200m.',
            'Weather conditions: Light rain.'
        ]
        
        alert = Alert(
            timestamp=time.time(),
            urgency='info',
            modalities=['visual'],
            message=messages[self.alert_counter % len(messages)],
            hazard_id=self.alert_counter,
            dismissed=False
        )
        self.alerts_panel.add_alert(alert)
        self.alert_counter += 1
        print(f"Added info alert #{self.alert_counter}")
    
    def add_random_alert(self):
        """Add a random alert"""
        import random
        
        alert_type = random.choice(['critical', 'warning', 'info'])
        
        if alert_type == 'critical':
            self.add_critical_alert()
        elif alert_type == 'warning':
            self.add_warning_alert()
        else:
            self.add_info_alert()
    
    def toggle_auto_alerts(self, checked):
        """Toggle automatic alert generation"""
        if checked:
            self.auto_timer.start(2000)  # Add alert every 2 seconds
            self.auto_btn.setText("Stop Auto Alerts")
            print("Auto alerts started")
        else:
            self.auto_timer.stop()
            self.auto_btn.setText("Start Auto Alerts")
            print("Auto alerts stopped")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Create and show test window
    window = TestWindow()
    window.show()
    
    # Add some initial alerts
    print("\nAdding initial test alerts...")
    
    # Critical alert
    alert1 = Alert(
        timestamp=time.time(),
        urgency='critical',
        modalities=['visual', 'audio', 'haptic'],
        message='Emergency braking required! Obstacle detected.',
        hazard_id=1,
        dismissed=False
    )
    window.alerts_panel.add_alert(alert1)
    
    # Warning alert
    alert2 = Alert(
        timestamp=time.time(),
        urgency='warning',
        modalities=['visual', 'audio'],
        message='Cyclist detected in adjacent lane.',
        hazard_id=2,
        dismissed=False
    )
    window.alerts_panel.add_alert(alert2)
    
    # Info alert
    alert3 = Alert(
        timestamp=time.time(),
        urgency='info',
        modalities=['visual'],
        message='Approaching intersection in 100 meters.',
        hazard_id=3,
        dismissed=False
    )
    window.alerts_panel.add_alert(alert3)
    
    print("\nTest window ready!")
    print("Statistics:", window.alerts_panel.get_statistics())
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
