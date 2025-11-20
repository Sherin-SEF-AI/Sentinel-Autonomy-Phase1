#!/usr/bin/env python3
"""
Test script for Risk Assessment Panel

Tests all components of the risk assessment panel.
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import QTimer
import random

# Add src to path
sys.path.insert(0, 'src')

from gui.widgets.risk_panel import RiskAssessmentPanel

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TestWindow(QMainWindow):
    """Test window for risk assessment panel"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Risk Assessment Panel Test")
        self.setGeometry(100, 100, 1000, 800)
        
        # Central widget
        central = QWidget()
        layout = QVBoxLayout()
        
        # Control buttons
        controls = QHBoxLayout()
        
        self.auto_update_btn = QPushButton("Start Auto Update")
        self.auto_update_btn.clicked.connect(self.toggle_auto_update)
        controls.addWidget(self.auto_update_btn)
        
        update_btn = QPushButton("Update Once")
        update_btn.clicked.connect(self.update_once)
        controls.addWidget(update_btn)
        
        add_alert_btn = QPushButton("Add Alert")
        add_alert_btn.clicked.connect(self.add_alert)
        controls.addWidget(add_alert_btn)
        
        layout.addLayout(controls)
        
        # Risk assessment panel
        self.risk_panel = RiskAssessmentPanel()
        layout.addWidget(self.risk_panel)
        
        central.setLayout(layout)
        self.setCentralWidget(central)
        
        # Auto-update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_once)
        self.auto_updating = False
        
        logger.info("Test window initialized")
    
    def toggle_auto_update(self):
        """Toggle auto-update mode"""
        if self.auto_updating:
            self.update_timer.stop()
            self.auto_update_btn.setText("Start Auto Update")
            self.auto_updating = False
            logger.info("Auto-update stopped")
        else:
            self.update_timer.start(100)  # 10 Hz
            self.auto_update_btn.setText("Stop Auto Update")
            self.auto_updating = True
            logger.info("Auto-update started")
    
    def update_once(self):
        """Update risk panel with random data"""
        # Random risk score
        risk_score = random.random()
        self.risk_panel.update_risk_score(risk_score)
        
        # Random hazards
        hazard_types = ['vehicle', 'pedestrian', 'cyclist', 'motorcycle', 'truck']
        zones = ['front', 'front-right', 'right', 'rear-right', 'rear', 'rear-left', 'left', 'front-left']
        
        num_hazards = random.randint(0, 5)
        hazards = []
        for _ in range(num_hazards):
            hazard = {
                'type': random.choice(hazard_types),
                'zone': random.choice(zones),
                'ttc': random.uniform(0.5, 5.0),
                'risk_score': random.random(),
                'attended': random.choice([True, False])
            }
            hazards.append(hazard)
        
        # Sort by risk score
        hazards.sort(key=lambda h: h['risk_score'], reverse=True)
        self.risk_panel.update_hazards(hazards)
        
        # Random zone risks
        zone_risks = [random.random() * 0.8 for _ in range(8)]
        self.risk_panel.update_zone_risks(zone_risks)
        
        # Random TTC
        min_ttc = random.uniform(0.5, 10.0)
        self.risk_panel.update_ttc(min_ttc)
    
    def add_alert(self):
        """Add random alert event"""
        urgency = random.choice(['info', 'warning', 'critical'])
        self.risk_panel.add_alert_event(urgency)
        logger.info(f"Added alert: {urgency}")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Apply dark theme
    app.setStyle('Fusion')
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(35, 35, 35))
    app.setPalette(palette)
    
    window = TestWindow()
    window.show()
    
    logger.info("Application started")
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
