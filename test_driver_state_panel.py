#!/usr/bin/env python3
"""
Test script for Driver State Panel

Tests the driver state panel with simulated data.
"""

import sys
import logging
import random
import math
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QTimer

# Add src to path
sys.path.insert(0, 'src')

from gui.widgets import DriverStatePanel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class TestWindow(QMainWindow):
    """Test window for driver state panel"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Driver State Panel Test")
        self.setGeometry(100, 100, 1200, 900)
        
        # Create driver state panel
        self.driver_panel = DriverStatePanel()
        self.setCentralWidget(self.driver_panel)
        
        # Simulation state
        self.time = 0.0
        self.drowsiness_base = 20.0
        self.distraction_active = False
        
        # Timer for updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_simulation)
        self.update_timer.start(100)  # 10 Hz updates
        
        logger.info("Test window initialized")
    
    def update_simulation(self):
        """Update with simulated driver state data"""
        self.time += 0.1
        
        # Simulate varying drowsiness
        drowsiness_score = self.drowsiness_base + 15 * math.sin(self.time * 0.2)
        drowsiness_score = max(0, min(100, drowsiness_score))
        
        # Simulate gaze movement
        gaze_yaw = 30 * math.sin(self.time * 0.3)
        gaze_pitch = 10 * math.sin(self.time * 0.5)
        
        # Determine attention zone from yaw
        if -30 <= gaze_yaw <= 30:
            attention_zone = 'front'
        elif 30 < gaze_yaw <= 75:
            attention_zone = 'front_left'
        elif gaze_yaw > 75:
            attention_zone = 'left'
        elif -75 <= gaze_yaw < -30:
            attention_zone = 'front_right'
        else:
            attention_zone = 'right'
        
        # Simulate distraction
        if random.random() < 0.01:  # 1% chance to toggle distraction
            self.distraction_active = not self.distraction_active
        
        distraction_type = 'phone' if self.distraction_active else 'none'
        distraction_duration = random.uniform(0, 3) if self.distraction_active else 0
        distraction_confidence = 0.8 if self.distraction_active else 0.1
        
        # Calculate readiness score
        alertness = 100 - drowsiness_score
        attention = 100 - (distraction_confidence * 100)
        readiness_score = 0.4 * alertness + 0.3 * attention + 0.3 * 80  # Base 80
        
        # Simulate eye state
        blink_rate = 15 + 10 * (drowsiness_score / 100)
        
        # Create driver state dictionary
        driver_state = {
            'face_detected': True,
            'readiness_score': readiness_score,
            'gaze': {
                'pitch': gaze_pitch,
                'yaw': gaze_yaw,
                'attention_zone': attention_zone
            },
            'head_pose': {
                'pitch': gaze_pitch,
                'yaw': gaze_yaw,
                'roll': 5 * math.sin(self.time * 0.4)
            },
            'drowsiness': {
                'score': drowsiness_score,
                'yawn_detected': drowsiness_score > 60,
                'micro_sleep': drowsiness_score > 80
            },
            'distraction': {
                'type': distraction_type,
                'confidence': distraction_confidence,
                'duration': distraction_duration
            },
            'eye_state': {
                'blink_rate': blink_rate,
                'left_ear': 0.3 if drowsiness_score > 50 else 0.2,
                'right_ear': 0.3 if drowsiness_score > 50 else 0.2,
                'perclos': drowsiness_score / 100
            }
        }
        
        # Update panel
        self.driver_panel.update_driver_state(driver_state)
        
        # Occasionally trigger threshold crossings for testing
        if int(self.time) % 20 == 0 and self.time > 1:
            # Simulate sudden drowsiness spike
            self.drowsiness_base = random.uniform(40, 70)
            logger.info(f"Simulated drowsiness change to {self.drowsiness_base:.1f}")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show test window
    window = TestWindow()
    window.show()
    
    logger.info("Starting driver state panel test")
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
