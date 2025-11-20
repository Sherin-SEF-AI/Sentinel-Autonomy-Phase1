"""Map View Dock Widget for HD map visualization."""

import logging
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QListWidget, QListWidgetItem,
    QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QTransform, QPolygonF
)
import numpy as np

logger = logging.getLogger(__name__)


class MapGraphicsView(QGraphicsView):
    """Graphics view for map display with zoom and pan."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Zoom settings
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        delta = event.angleDelta().y()
        
        if delta > 0:
            zoom_change = 1.15
        else:
            zoom_change = 1.0 / 1.15
        
        new_zoom = self.zoom_factor * zoom_change
        
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_factor = new_zoom
            self.scale(zoom_change, zoom_change)


class MapViewDock(QWidget):
    """
    Map View Dock Widget.
    
    Displays HD map with vehicle position and upcoming features.
    
    Features:
    - Map visualization with lanes and features
    - Vehicle position and heading indicator
    - Current lane highlighting
    - Upcoming features list
    - Zoom and pan controls
    """
    
    # Signals
    feature_selected = pyqtSignal(str)  # feature_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.logger = logging.getLogger(__name__ + '.MapViewDock')
        
        # Map data
        self.lanes: Dict = {}
        self.features: List = []
        self.vehicle_position: Optional[tuple] = None
        self.vehicle_heading: float = 0.0
        self.current_lane_id: Optional[str] = None
        
        # Graphics items
        self.lane_items = []
        self.feature_items = []
        self.vehicle_item = None
        
        # Setup UI
        self._setup_ui()
        
        self.logger.info("MapViewDock initialized")
    
    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title
        title_label = QLabel("HD Map View")
        title_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 8px;
                font-size: 12px;
                font-weight: bold;
                border-bottom: 2px solid #00aaff;
            }
        """)
        layout.addWidget(title_label)
        
        # Map view
        self.scene = QGraphicsScene()
        self.view = MapGraphicsView()
        self.view.setScene(self.scene)
        self.view.setMinimumHeight(300)
        
        # Set background
        self.scene.setBackgroundBrush(QBrush(QColor(40, 40, 40)))
        
        layout.addWidget(self.view, stretch=3)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.show_lanes_cb = QCheckBox("Show Lanes")
        self.show_lanes_cb.setChecked(True)
        self.show_lanes_cb.stateChanged.connect(self._on_show_lanes_changed)
        controls_layout.addWidget(self.show_lanes_cb)
        
        self.show_features_cb = QCheckBox("Show Features")
        self.show_features_cb.setChecked(True)
        self.show_features_cb.stateChanged.connect(self._on_show_features_changed)
        controls_layout.addWidget(self.show_features_cb)
        
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        controls_layout.addWidget(reset_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Upcoming features list
        features_group = QGroupBox("Upcoming Features")
        features_layout = QVBoxLayout()
        
        self.features_list = QListWidget()
        self.features_list.setMaximumHeight(150)
        self.features_list.itemClicked.connect(self._on_feature_clicked)
        features_layout.addWidget(self.features_list)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group, stretch=1)
        
        # Status label
        self.status_label = QLabel("No map loaded")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #aaaaaa;
                padding: 4px;
                font-size: 10px;
                border-top: 1px solid #333333;
            }
        """)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def update_map(self, lanes: Dict, features: List):
        """Update map data.
        
        Args:
            lanes: Dictionary of lane_id -> Lane objects
            features: List of MapFeature objects
        """
        self.lanes = lanes
        self.features = features
        
        # Redraw map
        self._draw_map()
        
        # Update status
        self.status_label.setText(
            f"Map loaded: {len(lanes)} lanes, {len(features)} features"
        )
        
        self.logger.info(f"Map updated: {len(lanes)} lanes, {len(features)} features")
    
    def update_vehicle_position(self, position: tuple, heading: float,
                               current_lane_id: Optional[str] = None):
        """Update vehicle position and heading.
        
        Args:
            position: Vehicle position (x, y) in map frame
            heading: Vehicle heading in radians
            current_lane_id: Current lane ID (optional)
        """
        self.vehicle_position = position
        self.vehicle_heading = heading
        self.current_lane_id = current_lane_id
        
        # Redraw vehicle
        self._draw_vehicle()
        
        # Center view on vehicle
        if position:
            self.view.centerOn(position[0], -position[1])
    
    def update_upcoming_features(self, upcoming: List[Dict]):
        """Update list of upcoming features.
        
        Args:
            upcoming: List of dicts with 'feature' and 'distance'
        """
        self.features_list.clear()
        
        for item in upcoming:
            feature = item['feature']
            distance = item['distance']
            
            # Format feature info
            feature_type = feature.type.capitalize()
            feature_info = f"{feature_type} - {distance:.1f}m"
            
            # Add attributes if available
            if 'speed_limit' in feature.attributes:
                feature_info += f" (Speed: {feature.attributes['speed_limit']} km/h)"
            
            list_item = QListWidgetItem(feature_info)
            list_item.setData(Qt.ItemDataRole.UserRole, feature.feature_id)
            self.features_list.addItem(list_item)
    
    def _draw_map(self):
        """Draw the map on the scene."""
        # Clear existing items
        self._clear_map_items()
        
        if not self.lanes:
            return
        
        # Calculate scene bounds
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        
        for lane in self.lanes.values():
            for x, y, _ in lane.centerline:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
        
        # Set scene rect with padding
        padding = 50
        self.scene.setSceneRect(
            min_x - padding, -(max_y + padding),
            (max_x - min_x) + 2*padding, (max_y - min_y) + 2*padding
        )
        
        # Draw lanes
        if self.show_lanes_cb.isChecked():
            for lane_id, lane in self.lanes.items():
                self._draw_lane(lane, lane_id == self.current_lane_id)
        
        # Draw features
        if self.show_features_cb.isChecked():
            for feature in self.features:
                self._draw_feature(feature)
        
        # Draw vehicle
        self._draw_vehicle()
    
    def _draw_lane(self, lane, is_current: bool = False):
        """Draw a lane on the scene.
        
        Args:
            lane: Lane object
            is_current: Whether this is the current lane
        """
        # Choose color
        if is_current:
            color = QColor(0, 200, 255, 200)  # Bright blue for current lane
            width = 3
        else:
            color = QColor(100, 100, 150, 100)
            width = 1
        
        # Draw centerline
        if lane.centerline and len(lane.centerline) >= 2:
            pen = QPen(color, width)
            
            for i in range(len(lane.centerline) - 1):
                x1, y1, _ = lane.centerline[i]
                x2, y2, _ = lane.centerline[i + 1]
                
                # Flip y coordinate for Qt coordinate system
                line_item = self.scene.addLine(x1, -y1, x2, -y2, pen)
                self.lane_items.append(line_item)
        
        # Draw boundaries (lighter)
        boundary_color = QColor(color.red(), color.green(), color.blue(), 50)
        boundary_pen = QPen(boundary_color, 1, Qt.PenStyle.DashLine)
        
        for boundary in [lane.left_boundary, lane.right_boundary]:
            if boundary and len(boundary) >= 2:
                for i in range(len(boundary) - 1):
                    x1, y1, _ = boundary[i]
                    x2, y2, _ = boundary[i + 1]
                    
                    line_item = self.scene.addLine(x1, -y1, x2, -y2, boundary_pen)
                    self.lane_items.append(line_item)
    
    def _draw_feature(self, feature):
        """Draw a map feature on the scene.
        
        Args:
            feature: MapFeature object
        """
        x, y, _ = feature.position
        y = -y  # Flip y coordinate
        
        # Draw based on type
        if feature.type == 'sign':
            # Triangle for signs
            size = 5
            points = [
                QPointF(x, y - size),
                QPointF(x - size, y + size),
                QPointF(x + size, y + size)
            ]
            polygon = QPolygonF(points)
            item = self.scene.addPolygon(
                polygon,
                QPen(QColor(255, 200, 0), 2),
                QBrush(QColor(255, 200, 0, 150))
            )
            self.feature_items.append(item)
            
        elif feature.type == 'light':
            # Circle for traffic lights
            size = 4
            item = self.scene.addEllipse(
                x - size/2, y - size/2, size, size,
                QPen(QColor(255, 100, 100), 2),
                QBrush(QColor(255, 100, 100, 150))
            )
            self.feature_items.append(item)
            
        elif feature.type == 'crosswalk':
            # Lines for crosswalks
            if feature.geometry and len(feature.geometry) >= 2:
                pen = QPen(QColor(255, 255, 255), 2)
                for i in range(len(feature.geometry) - 1):
                    x1, y1 = feature.geometry[i]
                    x2, y2 = feature.geometry[i + 1]
                    
                    line_item = self.scene.addLine(x1, -y1, x2, -y2, pen)
                    self.feature_items.append(line_item)
    
    def _draw_vehicle(self):
        """Draw vehicle position and heading indicator."""
        # Remove old vehicle item
        if self.vehicle_item:
            self.scene.removeItem(self.vehicle_item)
            self.vehicle_item = None
        
        if not self.vehicle_position:
            return
        
        x, y = self.vehicle_position
        y = -y  # Flip y coordinate
        
        # Draw vehicle as arrow
        size = 10
        heading = self.vehicle_heading
        
        # Arrow points
        front_x = x + size * np.cos(heading)
        front_y = y - size * np.sin(heading)  # Negative because y is flipped
        
        left_x = x + size/2 * np.cos(heading + 2.5)
        left_y = y - size/2 * np.sin(heading + 2.5)
        
        right_x = x + size/2 * np.cos(heading - 2.5)
        right_y = y - size/2 * np.sin(heading - 2.5)
        
        points = [
            QPointF(front_x, front_y),
            QPointF(left_x, left_y),
            QPointF(x, y),
            QPointF(right_x, right_y)
        ]
        
        polygon = QPolygonF(points)
        self.vehicle_item = self.scene.addPolygon(
            polygon,
            QPen(QColor(0, 255, 0), 2),
            QBrush(QColor(0, 255, 0, 200))
        )
    
    def _clear_map_items(self):
        """Clear all map graphics items."""
        for item in self.lane_items:
            self.scene.removeItem(item)
        self.lane_items.clear()
        
        for item in self.feature_items:
            self.scene.removeItem(item)
        self.feature_items.clear()
    
    def _reset_view(self):
        """Reset view to show entire map."""
        self.view.resetTransform()
        self.view.zoom_factor = 1.0
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    
    def _on_show_lanes_changed(self, state):
        """Handle show lanes checkbox change."""
        show = state == Qt.CheckState.Checked.value
        for item in self.lane_items:
            item.setVisible(show)
    
    def _on_show_features_changed(self, state):
        """Handle show features checkbox change."""
        show = state == Qt.CheckState.Checked.value
        for item in self.feature_items:
            item.setVisible(show)
    
    def _on_feature_clicked(self, item: QListWidgetItem):
        """Handle feature list item click."""
        feature_id = item.data(Qt.ItemDataRole.UserRole)
        if feature_id:
            self.feature_selected.emit(feature_id)
            self.logger.debug(f"Feature selected: {feature_id}")
