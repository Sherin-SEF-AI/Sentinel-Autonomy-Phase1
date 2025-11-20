#!/usr/bin/env python3
"""
Test script for BEV Canvas widget

This script demonstrates the interactive BEV canvas with:
- BEV image display
- Object overlays
- Trajectory visualization
- Attention zones
- Distance grid
- Screenshot and recording
"""

import sys
import numpy as np
import math
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt6.QtCore import QTimer
from src.gui.widgets.bev_canvas import BEVCanvas


class BEVCanvasTestWindow(QMainWindow):
    """Test window for BEV Canvas"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("BEV Canvas Test")
        self.setGeometry(100, 100, 1200, 900)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create BEV canvas
        self.bev_canvas = BEVCanvas()
        layout.addWidget(self.bev_canvas)
        
        # Connect object click signal
        self.bev_canvas.object_clicked.connect(self.on_object_clicked)
        
        # Create control buttons
        button_layout = QHBoxLayout()
        
        self.btn_update_bev = QPushButton("Update BEV")
        self.btn_update_bev.clicked.connect(self.update_bev_image)
        button_layout.addWidget(self.btn_update_bev)
        
        self.btn_update_objects = QPushButton("Update Objects")
        self.btn_update_objects.clicked.connect(self.update_objects)
        button_layout.addWidget(self.btn_update_objects)
        
        self.btn_update_trajectories = QPushButton("Update Trajectories")
        self.btn_update_trajectories.clicked.connect(self.update_trajectories)
        button_layout.addWidget(self.btn_update_trajectories)
        
        self.btn_update_zones = QPushButton("Update Zones")
        self.btn_update_zones.clicked.connect(self.update_zones)
        button_layout.addWidget(self.btn_update_zones)
        
        self.btn_screenshot = QPushButton("Screenshot")
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        button_layout.addWidget(self.btn_screenshot)
        
        self.btn_toggle_grid = QPushButton("Toggle Grid")
        self.btn_toggle_grid.clicked.connect(self.toggle_grid)
        button_layout.addWidget(self.btn_toggle_grid)
        
        layout.addLayout(button_layout)
        
        # Animation state
        self.animation_frame = 0
        
        # Start animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(100)  # 10 Hz
        
        # Initial setup
        self.update_bev_image()
        self.update_objects()
        self.update_trajectories()
        self.update_zones()
    
    def update_bev_image(self):
        """Update BEV image with test pattern"""
        # Create a test BEV image (640x640)
        bev_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Draw a simple road pattern
        # Road (gray)
        bev_image[200:440, :] = [80, 80, 80]
        
        # Lane markings (white dashed lines)
        for y in range(0, 640, 40):
            bev_image[y:y+20, 310:330] = [255, 255, 255]
        
        # Curbs (yellow)
        bev_image[195:205, :] = [0, 255, 255]
        bev_image[435:445, :] = [0, 255, 255]
        
        self.bev_canvas.update_bev_image(bev_image)
    
    def update_objects(self):
        """Update object overlays with test data"""
        detections = [
            {
                'object_id': 1,
                'bbox_3d': (15, -2, 0, 2, 1.5, 4.5, 0),  # Vehicle ahead
                'class_name': 'vehicle',
                'confidence': 0.95,
                'track_id': 101
            },
            {
                'object_id': 2,
                'bbox_3d': (8, 3, 0, 0.6, 1.8, 0.6, math.pi/4),  # Pedestrian on right
                'class_name': 'pedestrian',
                'confidence': 0.88,
                'track_id': 102
            },
            {
                'object_id': 3,
                'bbox_3d': (20, 5, 0, 1.8, 1.5, 4, -math.pi/6),  # Vehicle on right
                'class_name': 'vehicle',
                'confidence': 0.92,
                'track_id': 103
            }
        ]
        
        self.bev_canvas.update_objects(detections)
    
    def update_trajectories(self):
        """Update trajectory overlays with test data"""
        trajectories = [
            {
                'object_id': 1,
                'points': [(15 + i*0.5, -2) for i in range(20)],  # Straight ahead
                'collision_probability': 0.2
            },
            {
                'object_id': 2,
                'points': [(8 + i*0.3, 3 - i*0.1) for i in range(15)],  # Moving left
                'collision_probability': 0.7
            },
            {
                'object_id': 3,
                'points': [(20 + i*0.4, 5 + i*0.05) for i in range(18)],  # Slight right
                'collision_probability': 0.1
            }
        ]
        
        self.bev_canvas.update_trajectories(trajectories)
    
    def update_zones(self):
        """Update attention zones with test data"""
        # Rotate attended zone based on animation frame
        zone_names = ['front', 'front_right', 'right', 'rear_right', 
                     'rear', 'rear_left', 'left', 'front_left']
        attended_idx = (self.animation_frame // 10) % len(zone_names)
        
        zones_data = {
            'attended_zone': zone_names[attended_idx],
            'zone_risks': {
                'front': 0.3,
                'front_right': 0.5,
                'right': 0.7,
                'rear_right': 0.2,
                'rear': 0.1,
                'rear_left': 0.2,
                'left': 0.4,
                'front_left': 0.6
            }
        }
        
        self.bev_canvas.update_attention_zones(zones_data)
    
    def animate(self):
        """Animation loop"""
        self.animation_frame += 1
        
        # Update zones every 10 frames
        if self.animation_frame % 10 == 0:
            self.update_zones()
    
    def on_object_clicked(self, object_id: int):
        """Handle object click"""
        print(f"Object clicked: {object_id}")
        self.statusBar().showMessage(f"Selected object: {object_id}", 3000)
    
    def take_screenshot(self):
        """Take screenshot"""
        success = self.bev_canvas.capture_screenshot()
        if success:
            self.statusBar().showMessage("Screenshot saved", 3000)
        else:
            self.statusBar().showMessage("Screenshot cancelled", 3000)
    
    def toggle_grid(self):
        """Toggle distance grid visibility"""
        self.bev_canvas.set_show_grid(not self.bev_canvas.show_grid)


def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show window
    window = BEVCanvasTestWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
