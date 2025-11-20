"""
BEV Canvas Widget

Interactive Bird's Eye View canvas with zoom, pan, and object overlay capabilities.
Implements QGraphicsScene for rendering BEV with interactive controls.
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QLabel, QGraphicsRectItem, QGraphicsTextItem,
    QGraphicsItemGroup, QGraphicsEllipseItem, QGraphicsPolygonItem,
    QFileDialog
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent, QMouseEvent,
    QTransform, QPen, QColor, QBrush, QFont, QPolygonF
)

logger = logging.getLogger(__name__)


class BEVGraphicsView(QGraphicsView):
    """
    Custom QGraphicsView with zoom and pan controls.
    
    Features:
    - Mouse wheel zoom
    - Drag to pan
    - Smooth transformations
    - Click detection
    """
    
    # Signal for item clicks
    item_clicked = pyqtSignal(QPointF)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Zoom settings
        self.zoom_factor = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 5.0
        self.zoom_step = 1.15
        
        # Pan settings
        self.panning = False
        self.pan_start_pos = QPointF()
        
        logger.debug("BEVGraphicsView initialized")
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming"""
        # Get wheel delta
        delta = event.angleDelta().y()
        
        if delta > 0:
            # Zoom in
            zoom_change = self.zoom_step
        else:
            # Zoom out
            zoom_change = 1.0 / self.zoom_step
        
        # Calculate new zoom factor
        new_zoom = self.zoom_factor * zoom_change
        
        # Clamp zoom factor
        if self.min_zoom <= new_zoom <= self.max_zoom:
            self.zoom_factor = new_zoom
            self.scale(zoom_change, zoom_change)
            logger.debug(f"Zoom factor: {self.zoom_factor:.2f}")
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press for panning and clicking"""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning with middle mouse button
            self.panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            # Emit click signal with scene position
            scene_pos = self.mapToScene(event.pos())
            self.item_clicked.emit(scene_pos)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for panning"""
        if self.panning:
            # Calculate pan delta
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            
            # Pan the view
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.resetTransform()
        self.zoom_factor = 1.0
        logger.debug("View reset to default")


class BEVCanvas(QWidget):
    """
    Interactive Bird's Eye View canvas widget.
    
    Features:
    - Display BEV image as background
    - Zoom and pan controls
    - Object overlay support
    - Trajectory visualization
    - Attention zone overlay
    - Distance grid
    
    Signals:
    - object_clicked: Emitted when an object is clicked (object_id)
    """
    
    object_clicked = pyqtSignal(int)  # object_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Scene and view
        self.scene = QGraphicsScene()
        self.view = BEVGraphicsView()
        self.view.setScene(self.scene)
        
        # BEV image item
        self.bev_image_item: Optional[QGraphicsPixmapItem] = None
        self.current_bev_frame: Optional[np.ndarray] = None
        
        # BEV parameters (from design: 640x640, 0.1m per pixel)
        self.bev_size = 640
        self.meters_per_pixel = 0.1
        self.vehicle_position = (320, 480)  # Vehicle position in BEV coordinates
        
        # Overlay items
        self.object_items: Dict[int, Any] = {}  # object_id -> graphics items
        self.trajectory_items: Dict[int, Any] = {}  # object_id -> trajectory items
        self.zone_items: List[Any] = []
        self.grid_items: List[Any] = []
        
        # Visibility flags
        self.show_objects = True
        self.show_trajectories = True
        self.show_map = False  # Map overlay disabled by default
        self.show_zones = True
        self.show_grid = True
        
        # Selected object
        self.selected_object_id: Optional[int] = None
        
        self._init_ui()
        self._setup_scene()
        self._connect_signals()
        
        # Create distance grid
        self.create_distance_grid()
        
        logger.info("BEVCanvas initialized")
    
    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add view to layout
        layout.addWidget(self.view)
        
        # Add info label
        self.info_label = QLabel("BEV Canvas - Use mouse wheel to zoom, drag to pan")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #2a2a2a;
                color: #ffffff;
                padding: 4px;
                font-size: 10px;
                border-top: 1px solid #333333;
            }
        """)
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
    
    def _setup_scene(self):
        """Setup the graphics scene"""
        # Set scene rect to BEV size
        self.scene.setSceneRect(0, 0, self.bev_size, self.bev_size)
        
        # Set background color
        self.scene.setBackgroundBrush(QBrush(QColor(26, 26, 26)))
        
        logger.debug("Graphics scene configured")
    
    def _connect_signals(self):
        """Connect internal signals"""
        # Connect view click to object selection
        self.view.item_clicked.connect(self._on_item_clicked)
    
    def _on_item_clicked(self, scene_pos: QPointF):
        """
        Handle item click in the scene.
        
        Args:
            scene_pos: Click position in scene coordinates
        """
        # Get object at position
        object_id = self.get_object_at_position(scene_pos)
        
        if object_id is not None:
            # Unhighlight previous selection
            if self.selected_object_id is not None:
                self.unhighlight_all_objects()
            
            # Highlight new selection
            self.selected_object_id = object_id
            self.highlight_object(object_id)
            
            # Emit signal
            self.object_clicked.emit(object_id)
            
            logger.info(f"Object {object_id} clicked")
        else:
            # Clear selection
            if self.selected_object_id is not None:
                self.unhighlight_all_objects()
                self.selected_object_id = None
    
    def update_bev_image(self, bev_frame: np.ndarray):
        """
        Update the BEV background image.
        
        Args:
            bev_frame: Numpy array containing BEV image (640, 640, 3)
        """
        if bev_frame is None or bev_frame.size == 0:
            logger.warning("Invalid BEV frame")
            return
        
        try:
            self.current_bev_frame = bev_frame
            
            # Convert numpy array to QPixmap
            pixmap = self._numpy_to_pixmap(bev_frame)
            
            # Update or create BEV image item
            if self.bev_image_item is None:
                self.bev_image_item = QGraphicsPixmapItem(pixmap)
                self.bev_image_item.setZValue(0)  # Background layer
                self.scene.addItem(self.bev_image_item)
            else:
                self.bev_image_item.setPixmap(pixmap)
            
            logger.debug("BEV image updated")
            
        except Exception as e:
            logger.error(f"Error updating BEV image: {e}")
    
    def clear_bev_image(self):
        """Clear the BEV image"""
        if self.bev_image_item is not None:
            self.scene.removeItem(self.bev_image_item)
            self.bev_image_item = None
        self.current_bev_frame = None
        logger.debug("BEV image cleared")
    
    def _numpy_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """
        Convert numpy array to QPixmap.
        
        Args:
            frame: Numpy array in BGR or RGB format (H, W, 3)
            
        Returns:
            QPixmap object
        """
        height, width, channels = frame.shape
        
        # Ensure frame is in RGB format (OpenCV uses BGR)
        if channels == 3:
            # Assume BGR from OpenCV, convert to RGB
            rgb_frame = frame[:, :, ::-1].copy()
        else:
            rgb_frame = frame.copy()
        
        # Create QImage from numpy array
        bytes_per_line = channels * width
        q_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        # Convert to QPixmap
        return QPixmap.fromImage(q_image)
    
    def reset_view(self):
        """Reset zoom and pan to default"""
        self.view.reset_view()
    
    def set_show_objects(self, show: bool):
        """Toggle object overlay visibility"""
        self.show_objects = show
        for items in self.object_items.values():
            if isinstance(items, list):
                for item in items:
                    item.setVisible(show)
            else:
                items.setVisible(show)
        logger.debug(f"Object overlay visibility: {show}")
    
    def set_show_trajectories(self, show: bool):
        """Toggle trajectory overlay visibility"""
        self.show_trajectories = show
        for items in self.trajectory_items.values():
            if isinstance(items, list):
                for item in items:
                    item.setVisible(show)
            else:
                items.setVisible(show)
        logger.debug(f"Trajectory overlay visibility: {show}")
    
    def set_show_zones(self, show: bool):
        """Toggle attention zone overlay visibility"""
        self.show_zones = show
        for item in self.zone_items:
            item.setVisible(show)
        logger.debug(f"Zone overlay visibility: {show}")
    
    def set_show_grid(self, show: bool):
        """Toggle distance grid visibility"""
        self.show_grid = show
        for item in self.grid_items:
            item.setVisible(show)
        logger.debug(f"Grid overlay visibility: {show}")
    
    def clear_all_overlays(self):
        """Clear all overlay items"""
        # Clear objects
        for items in self.object_items.values():
            if isinstance(items, list):
                for item in items:
                    self.scene.removeItem(item)
            else:
                self.scene.removeItem(items)
        self.object_items.clear()
        
        # Clear trajectories
        for items in self.trajectory_items.values():
            if isinstance(items, list):
                for item in items:
                    self.scene.removeItem(item)
            else:
                self.scene.removeItem(items)
        self.trajectory_items.clear()
        
        # Clear zones
        for item in self.zone_items:
            self.scene.removeItem(item)
        self.zone_items.clear()
        
        # Clear grid
        for item in self.grid_items:
            self.scene.removeItem(item)
        self.grid_items.clear()
        
        logger.debug("All overlays cleared")
    
    def get_scene_coordinates(self, vehicle_x: float, vehicle_y: float) -> tuple:
        """
        Convert vehicle coordinates (meters) to scene coordinates (pixels).
        
        Args:
            vehicle_x: X coordinate in vehicle frame (meters, forward)
            vehicle_y: Y coordinate in vehicle frame (meters, left)
            
        Returns:
            Tuple of (scene_x, scene_y) in pixels
        """
        # Vehicle frame: X forward, Y left
        # BEV frame: X right, Y up (forward)
        # Scene coordinates: origin at top-left
        
        # Convert to BEV coordinates
        bev_x = self.vehicle_position[0] - (vehicle_y / self.meters_per_pixel)
        bev_y = self.vehicle_position[1] - (vehicle_x / self.meters_per_pixel)
        
        return (bev_x, bev_y)
    
    def get_vehicle_coordinates(self, scene_x: float, scene_y: float) -> tuple:
        """
        Convert scene coordinates (pixels) to vehicle coordinates (meters).
        
        Args:
            scene_x: X coordinate in scene (pixels)
            scene_y: Y coordinate in scene (pixels)
            
        Returns:
            Tuple of (vehicle_x, vehicle_y) in meters
        """
        # Inverse of get_scene_coordinates
        vehicle_y = -(scene_x - self.vehicle_position[0]) * self.meters_per_pixel
        vehicle_x = -(scene_y - self.vehicle_position[1]) * self.meters_per_pixel
        
        return (vehicle_x, vehicle_y)
    
    # Object overlay methods
    
    def update_objects(self, detections: List[Dict[str, Any]]):
        """
        Update object overlays with detected objects.
        
        Args:
            detections: List of detection dictionaries with keys:
                - object_id: Unique object ID
                - bbox_3d: (x, y, z, w, h, l, theta) in vehicle frame
                - class_name: Object class name
                - confidence: Detection confidence (0-1)
                - track_id: Tracking ID
        """
        # Clear existing objects
        for items in self.object_items.values():
            if isinstance(items, list):
                for item in items:
                    self.scene.removeItem(item)
            else:
                self.scene.removeItem(items)
        self.object_items.clear()
        
        # Add new objects
        for detection in detections:
            self._add_object_overlay(detection)
        
        logger.debug(f"Updated {len(detections)} object overlays")
    
    def _add_object_overlay(self, detection: Dict[str, Any]):
        """
        Add a single object overlay to the scene.
        
        Args:
            detection: Detection dictionary
        """
        object_id = detection.get('object_id', -1)
        bbox_3d = detection.get('bbox_3d', None)
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0.0)
        track_id = detection.get('track_id', -1)
        
        if bbox_3d is None or len(bbox_3d) < 7:
            logger.warning(f"Invalid bbox_3d for object {object_id}")
            return
        
        # Extract bbox parameters
        x, y, z, w, h, l, theta = bbox_3d
        
        # Get color for class
        color = self._get_class_color(class_name)
        
        # Create group for all object items
        object_group = QGraphicsItemGroup()
        
        # Draw 3D bounding box (top-down view)
        bbox_item = self._create_bbox_item(x, y, w, l, theta, color)
        object_group.addToGroup(bbox_item)
        
        # Add center marker
        center_x, center_y = self.get_scene_coordinates(x, y)
        center_marker = QGraphicsEllipseItem(center_x - 3, center_y - 3, 6, 6)
        center_marker.setBrush(QBrush(color))
        center_marker.setPen(QPen(Qt.PenStyle.NoPen))
        object_group.addToGroup(center_marker)
        
        # Add text label with track ID and confidence
        label_text = f"ID:{track_id}\n{class_name}\n{confidence:.2f}"
        text_item = QGraphicsTextItem(label_text)
        text_item.setDefaultTextColor(color)
        text_item.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        text_item.setPos(center_x + 5, center_y - 20)
        object_group.addToGroup(text_item)
        
        # Set z-value for layering (objects above BEV image)
        object_group.setZValue(10)
        
        # Make clickable
        object_group.setFlag(QGraphicsItemGroup.GraphicsItemFlag.ItemIsSelectable)
        object_group.setData(0, object_id)  # Store object_id
        
        # Add to scene
        self.scene.addItem(object_group)
        
        # Store reference
        self.object_items[object_id] = object_group
        
        # Set visibility
        object_group.setVisible(self.show_objects)
    
    def _create_bbox_item(self, x: float, y: float, w: float, l: float, 
                          theta: float, color: QColor) -> QGraphicsPolygonItem:
        """
        Create a 3D bounding box item (top-down view).
        
        Args:
            x, y: Center position in vehicle frame (meters)
            w: Width (meters)
            l: Length (meters)
            theta: Orientation angle (radians)
            color: Box color
            
        Returns:
            QGraphicsPolygonItem representing the bounding box
        """
        import math
        
        # Calculate corner points in vehicle frame
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Half dimensions
        half_l = l / 2.0
        half_w = w / 2.0
        
        # Corner offsets (in vehicle frame)
        corners = [
            (-half_l, -half_w),  # Rear left
            (half_l, -half_w),   # Front left
            (half_l, half_w),    # Front right
            (-half_l, half_w),   # Rear right
        ]
        
        # Rotate and translate corners
        polygon_points = []
        for dx, dy in corners:
            # Rotate
            rx = dx * cos_theta - dy * sin_theta
            ry = dx * sin_theta + dy * cos_theta
            
            # Translate
            px = x + rx
            py = y + ry
            
            # Convert to scene coordinates
            sx, sy = self.get_scene_coordinates(px, py)
            polygon_points.append(QPointF(sx, sy))
        
        # Create polygon
        polygon = QPolygonF(polygon_points)
        
        # Create graphics item
        bbox_item = QGraphicsPolygonItem(polygon)
        bbox_item.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
        bbox_item.setBrush(QBrush(color.lighter(150), Qt.BrushStyle.Dense7Pattern))
        
        return bbox_item
    
    def _get_class_color(self, class_name: str) -> QColor:
        """
        Get color for object class.
        
        Args:
            class_name: Object class name
            
        Returns:
            QColor for the class
        """
        # Color mapping for different classes
        color_map = {
            'vehicle': QColor(0, 170, 255),      # Blue
            'car': QColor(0, 170, 255),          # Blue
            'truck': QColor(0, 120, 200),        # Dark blue
            'bus': QColor(0, 100, 180),          # Darker blue
            'pedestrian': QColor(255, 170, 0),   # Orange
            'person': QColor(255, 170, 0),       # Orange
            'cyclist': QColor(255, 85, 0),       # Red-orange
            'bicycle': QColor(255, 85, 0),       # Red-orange
            'motorcycle': QColor(255, 0, 127),   # Pink
            'traffic_sign': QColor(255, 255, 0), # Yellow
            'traffic_light': QColor(0, 255, 0),  # Green
            'obstacle': QColor(255, 0, 0),       # Red
        }
        
        # Return color or default
        return color_map.get(class_name.lower(), QColor(150, 150, 150))
    
    def get_object_at_position(self, scene_pos: QPointF) -> Optional[int]:
        """
        Get object ID at a given scene position.
        
        Args:
            scene_pos: Position in scene coordinates
            
        Returns:
            Object ID if found, None otherwise
        """
        items = self.scene.items(scene_pos)
        
        for item in items:
            # Check if item is part of an object group
            if isinstance(item, QGraphicsItemGroup):
                object_id = item.data(0)
                if object_id is not None:
                    return object_id
            
            # Check parent group
            parent = item.parentItem()
            if isinstance(parent, QGraphicsItemGroup):
                object_id = parent.data(0)
                if object_id is not None:
                    return object_id
        
        return None
    
    def highlight_object(self, object_id: int):
        """
        Highlight a specific object.
        
        Args:
            object_id: Object ID to highlight
        """
        if object_id in self.object_items:
            group = self.object_items[object_id]
            # Add highlight effect (make brighter)
            for item in group.childItems():
                if isinstance(item, QGraphicsPolygonItem):
                    pen = item.pen()
                    pen.setWidth(4)
                    item.setPen(pen)
            logger.debug(f"Highlighted object {object_id}")
    
    def unhighlight_all_objects(self):
        """Remove highlight from all objects"""
        for group in self.object_items.values():
            for item in group.childItems():
                if isinstance(item, QGraphicsPolygonItem):
                    pen = item.pen()
                    pen.setWidth(2)
                    item.setPen(pen)
    
    # Trajectory visualization methods
    
    def update_trajectories(self, trajectories: List[Dict[str, Any]]):
        """
        Update trajectory overlays for tracked objects.
        
        Args:
            trajectories: List of trajectory dictionaries with keys:
                - object_id: Object ID
                - points: List of (x, y) positions in vehicle frame (meters)
                - uncertainty: Optional uncertainty bounds
                - collision_probability: Collision probability (0-1)
        """
        # Clear existing trajectories
        for items in self.trajectory_items.values():
            if isinstance(items, list):
                for item in items:
                    self.scene.removeItem(item)
            else:
                self.scene.removeItem(items)
        self.trajectory_items.clear()
        
        # Add new trajectories
        for trajectory in trajectories:
            self._add_trajectory_overlay(trajectory)
        
        logger.debug(f"Updated {len(trajectories)} trajectory overlays")
    
    def _add_trajectory_overlay(self, trajectory: Dict[str, Any]):
        """
        Add a single trajectory overlay to the scene.
        
        Args:
            trajectory: Trajectory dictionary
        """
        object_id = trajectory.get('object_id', -1)
        points = trajectory.get('points', [])
        uncertainty = trajectory.get('uncertainty', None)
        collision_prob = trajectory.get('collision_probability', 0.0)
        
        if not points or len(points) < 2:
            return
        
        # Get color based on collision probability
        color = self._get_trajectory_color(collision_prob)
        
        # Create group for trajectory items
        trajectory_group = QGraphicsItemGroup()
        
        # Draw trajectory line
        trajectory_line = self._create_trajectory_line(points, color)
        trajectory_group.addToGroup(trajectory_line)
        
        # Draw uncertainty bounds if available
        if uncertainty is not None:
            uncertainty_item = self._create_uncertainty_bounds(points, uncertainty, color)
            if uncertainty_item:
                trajectory_group.addToGroup(uncertainty_item)
        
        # Draw waypoint markers
        for point in points[::3]:  # Every 3rd point to avoid clutter
            marker = self._create_waypoint_marker(point, color)
            trajectory_group.addToGroup(marker)
        
        # Set z-value for layering (trajectories above BEV, below objects)
        trajectory_group.setZValue(5)
        
        # Add to scene
        self.scene.addItem(trajectory_group)
        
        # Store reference
        self.trajectory_items[object_id] = trajectory_group
        
        # Set visibility
        trajectory_group.setVisible(self.show_trajectories)
    
    def _create_trajectory_line(self, points: List[tuple], color: QColor) -> QGraphicsItemGroup:
        """
        Create trajectory line from points.
        
        Args:
            points: List of (x, y) positions in vehicle frame
            color: Line color
            
        Returns:
            QGraphicsItemGroup containing line segments
        """
        from PyQt6.QtWidgets import QGraphicsLineItem
        
        line_group = QGraphicsItemGroup()
        
        # Convert points to scene coordinates
        scene_points = [self.get_scene_coordinates(x, y) for x, y in points]
        
        # Create line segments
        for i in range(len(scene_points) - 1):
            x1, y1 = scene_points[i]
            x2, y2 = scene_points[i + 1]
            
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            line_group.addToGroup(line)
        
        return line_group
    
    def _create_uncertainty_bounds(self, points: List[tuple], 
                                   uncertainty: List[float], 
                                   color: QColor) -> Optional[QGraphicsPolygonItem]:
        """
        Create uncertainty bounds visualization.
        
        Args:
            points: List of (x, y) positions
            uncertainty: List of uncertainty values (standard deviations)
            color: Base color
            
        Returns:
            QGraphicsPolygonItem or None
        """
        if len(points) != len(uncertainty):
            return None
        
        # Create upper and lower bounds
        upper_points = []
        lower_points = []
        
        for (x, y), std in zip(points, uncertainty):
            # Simple circular uncertainty (could be improved with covariance)
            sx, sy = self.get_scene_coordinates(x, y)
            
            # Convert std from meters to pixels
            std_pixels = std / self.meters_per_pixel
            
            upper_points.append(QPointF(sx, sy - std_pixels))
            lower_points.append(QPointF(sx, sy + std_pixels))
        
        # Create polygon from upper and reversed lower bounds
        polygon_points = upper_points + list(reversed(lower_points))
        polygon = QPolygonF(polygon_points)
        
        # Create graphics item
        uncertainty_item = QGraphicsPolygonItem(polygon)
        uncertainty_item.setPen(QPen(Qt.PenStyle.NoPen))
        
        # Semi-transparent fill
        fill_color = QColor(color)
        fill_color.setAlpha(50)
        uncertainty_item.setBrush(QBrush(fill_color))
        
        return uncertainty_item
    
    def _create_waypoint_marker(self, point: tuple, color: QColor) -> QGraphicsEllipseItem:
        """
        Create a waypoint marker.
        
        Args:
            point: (x, y) position in vehicle frame
            color: Marker color
            
        Returns:
            QGraphicsEllipseItem
        """
        x, y = point
        sx, sy = self.get_scene_coordinates(x, y)
        
        marker = QGraphicsEllipseItem(sx - 2, sy - 2, 4, 4)
        marker.setBrush(QBrush(color))
        marker.setPen(QPen(Qt.PenStyle.NoPen))
        
        return marker
    
    def _get_trajectory_color(self, collision_prob: float) -> QColor:
        """
        Get color for trajectory based on collision probability.
        
        Args:
            collision_prob: Collision probability (0-1)
            
        Returns:
            QColor
        """
        # Color gradient from green (safe) to red (dangerous)
        if collision_prob < 0.3:
            # Green
            return QColor(0, 255, 0)
        elif collision_prob < 0.6:
            # Yellow
            return QColor(255, 255, 0)
        elif collision_prob < 0.8:
            # Orange
            return QColor(255, 170, 0)
        else:
            # Red
            return QColor(255, 0, 0)
    
    # Attention zone overlay methods
    
    def update_attention_zones(self, zones_data: Dict[str, Any]):
        """
        Update attention zone overlays.
        
        Args:
            zones_data: Dictionary with keys:
                - attended_zone: Name of zone driver is looking at
                - zone_risks: Dict mapping zone names to risk scores (0-1)
        """
        # Clear existing zones
        for item in self.zone_items:
            self.scene.removeItem(item)
        self.zone_items.clear()
        
        attended_zone = zones_data.get('attended_zone', None)
        zone_risks = zones_data.get('zone_risks', {})
        
        # Define 8 zones (from design: front, front-left, left, rear-left, rear, rear-right, right, front-right)
        zone_names = [
            'front', 'front_right', 'right', 'rear_right',
            'rear', 'rear_left', 'left', 'front_left'
        ]
        
        # Create zone overlays
        for i, zone_name in enumerate(zone_names):
            risk_score = zone_risks.get(zone_name, 0.0)
            is_attended = (zone_name == attended_zone)
            
            zone_item = self._create_zone_sector(i, zone_name, risk_score, is_attended)
            self.zone_items.append(zone_item)
            self.scene.addItem(zone_item)
            zone_item.setVisible(self.show_zones)
        
        logger.debug(f"Updated attention zones, attended: {attended_zone}")
    
    def _create_zone_sector(self, zone_index: int, zone_name: str, 
                           risk_score: float, is_attended: bool) -> QGraphicsItemGroup:
        """
        Create a zone sector visualization.
        
        Args:
            zone_index: Zone index (0-7)
            zone_name: Zone name
            risk_score: Risk score for the zone (0-1)
            is_attended: Whether driver is looking at this zone
            
        Returns:
            QGraphicsItemGroup containing zone visualization
        """
        import math
        from PyQt6.QtWidgets import QGraphicsPathItem
        from PyQt6.QtGui import QPainterPath
        
        zone_group = QGraphicsItemGroup()
        
        # Get vehicle position in scene coordinates
        veh_x, veh_y = self.vehicle_position
        
        # Calculate zone sector angles (45 degrees each)
        # Zone 0 (front) is centered at 0 degrees (up in BEV)
        angle_per_zone = 360.0 / 8
        start_angle = zone_index * angle_per_zone - angle_per_zone / 2
        span_angle = angle_per_zone
        
        # Zone radius (in pixels)
        inner_radius = 30  # Don't draw too close to vehicle
        outer_radius = 200  # Zone extends 20 meters (200 pixels)
        
        # Create sector path
        path = QPainterPath()
        
        # Move to inner arc start
        start_rad = math.radians(start_angle - 90)  # Qt angles start from 3 o'clock
        path.moveTo(
            veh_x + inner_radius * math.cos(start_rad),
            veh_y + inner_radius * math.sin(start_rad)
        )
        
        # Arc to outer radius
        end_rad = math.radians(start_angle - 90)
        path.lineTo(
            veh_x + outer_radius * math.cos(start_rad),
            veh_y + outer_radius * math.sin(start_rad)
        )
        
        # Outer arc
        rect = QRectF(
            veh_x - outer_radius, veh_y - outer_radius,
            outer_radius * 2, outer_radius * 2
        )
        path.arcTo(rect, start_angle, span_angle)
        
        # Line back to inner arc
        end_rad = math.radians(start_angle + span_angle - 90)
        path.lineTo(
            veh_x + inner_radius * math.cos(end_rad),
            veh_y + inner_radius * math.sin(end_rad)
        )
        
        # Inner arc back to start
        inner_rect = QRectF(
            veh_x - inner_radius, veh_y - inner_radius,
            inner_radius * 2, inner_radius * 2
        )
        path.arcTo(inner_rect, start_angle + span_angle, -span_angle)
        
        path.closeSubpath()
        
        # Create graphics item
        sector_item = QGraphicsPathItem(path)
        
        # Set color based on risk and attention
        color = self._get_zone_color(risk_score, is_attended)
        
        # Set pen and brush
        if is_attended:
            sector_item.setPen(QPen(color, 3, Qt.PenStyle.SolidLine))
        else:
            sector_item.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
        
        fill_color = QColor(color)
        fill_color.setAlpha(30 if not is_attended else 60)
        sector_item.setBrush(QBrush(fill_color))
        
        zone_group.addToGroup(sector_item)
        
        # Add zone label
        label_angle = start_angle + span_angle / 2
        label_radius = (inner_radius + outer_radius) / 2
        label_rad = math.radians(label_angle - 90)
        label_x = veh_x + label_radius * math.cos(label_rad)
        label_y = veh_y + label_radius * math.sin(label_rad)
        
        label_text = f"{zone_name.replace('_', '-')}\n{risk_score:.2f}"
        text_item = QGraphicsTextItem(label_text)
        text_item.setDefaultTextColor(color)
        text_item.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        
        # Center text
        text_rect = text_item.boundingRect()
        text_item.setPos(label_x - text_rect.width() / 2, label_y - text_rect.height() / 2)
        
        zone_group.addToGroup(text_item)
        
        # Set z-value (zones below trajectories)
        zone_group.setZValue(3)
        
        return zone_group
    
    def _get_zone_color(self, risk_score: float, is_attended: bool) -> QColor:
        """
        Get color for attention zone.
        
        Args:
            risk_score: Risk score (0-1)
            is_attended: Whether zone is attended
            
        Returns:
            QColor
        """
        if is_attended:
            # Attended zones are blue-ish
            if risk_score < 0.3:
                return QColor(0, 200, 255)  # Cyan
            elif risk_score < 0.6:
                return QColor(0, 150, 255)  # Light blue
            else:
                return QColor(100, 100, 255)  # Blue
        else:
            # Unattended zones use risk-based colors
            if risk_score < 0.3:
                return QColor(100, 100, 100)  # Gray (low risk)
            elif risk_score < 0.6:
                return QColor(255, 200, 0)  # Yellow (medium risk)
            else:
                return QColor(255, 50, 50)  # Red (high risk)
    
    # Distance grid methods
    
    def create_distance_grid(self):
        """
        Create distance grid with concentric circles and radial lines.
        
        Grid features:
        - Concentric circles at 5-meter intervals
        - Radial lines for angular reference (every 45 degrees)
        - Distance labels
        """
        # Clear existing grid
        for item in self.grid_items:
            self.scene.removeItem(item)
        self.grid_items.clear()
        
        # Get vehicle position
        veh_x, veh_y = self.vehicle_position
        
        # Grid color
        grid_color = QColor(80, 80, 80)
        grid_pen = QPen(grid_color, 1, Qt.PenStyle.DotLine)
        
        # Create concentric circles at 5-meter intervals
        max_distance = 30  # meters
        interval = 5  # meters
        
        for distance in range(interval, max_distance + 1, interval):
            # Convert to pixels
            radius_pixels = distance / self.meters_per_pixel
            
            # Create circle
            circle = QGraphicsEllipseItem(
                veh_x - radius_pixels,
                veh_y - radius_pixels,
                radius_pixels * 2,
                radius_pixels * 2
            )
            circle.setPen(grid_pen)
            circle.setZValue(1)  # Below zones
            self.grid_items.append(circle)
            self.scene.addItem(circle)
            circle.setVisible(self.show_grid)
            
            # Add distance label at top of circle
            label = QGraphicsTextItem(f"{distance}m")
            label.setDefaultTextColor(grid_color)
            label.setFont(QFont("Arial", 8))
            label_rect = label.boundingRect()
            label.setPos(veh_x - label_rect.width() / 2, veh_y - radius_pixels - label_rect.height())
            label.setZValue(1)
            self.grid_items.append(label)
            self.scene.addItem(label)
            label.setVisible(self.show_grid)
        
        # Create radial lines (every 45 degrees for 8 directions)
        import math
        from PyQt6.QtWidgets import QGraphicsLineItem
        
        max_radius_pixels = max_distance / self.meters_per_pixel
        
        for angle_deg in range(0, 360, 45):
            angle_rad = math.radians(angle_deg - 90)  # Qt angles start from 3 o'clock
            
            # Calculate end point
            end_x = veh_x + max_radius_pixels * math.cos(angle_rad)
            end_y = veh_y + max_radius_pixels * math.sin(angle_rad)
            
            # Create line
            line = QGraphicsLineItem(veh_x, veh_y, end_x, end_y)
            line.setPen(grid_pen)
            line.setZValue(1)
            self.grid_items.append(line)
            self.scene.addItem(line)
            line.setVisible(self.show_grid)
        
        # Add vehicle marker at center
        vehicle_marker = QGraphicsEllipseItem(veh_x - 5, veh_y - 5, 10, 10)
        vehicle_marker.setBrush(QBrush(QColor(255, 255, 255)))
        vehicle_marker.setPen(QPen(QColor(0, 0, 0), 2))
        vehicle_marker.setZValue(2)
        self.grid_items.append(vehicle_marker)
        self.scene.addItem(vehicle_marker)
        vehicle_marker.setVisible(self.show_grid)
        
        # Add vehicle orientation indicator (arrow pointing forward)
        from PyQt6.QtWidgets import QGraphicsLineItem
        arrow_length = 15
        arrow = QGraphicsLineItem(veh_x, veh_y, veh_x, veh_y - arrow_length)
        arrow.setPen(QPen(QColor(255, 255, 255), 3))
        arrow.setZValue(2)
        self.grid_items.append(arrow)
        self.scene.addItem(arrow)
        arrow.setVisible(self.show_grid)
        
        logger.debug("Distance grid created")
    
    # Screenshot and recording methods
    
    def capture_screenshot(self, filename: Optional[str] = None) -> bool:
        """
        Capture screenshot of the BEV canvas.
        
        Args:
            filename: Optional filename to save to. If None, shows save dialog.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get filename if not provided
            if filename is None:
                # Generate default filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"bev_screenshot_{timestamp}.png"
                
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Screenshot",
                    default_name,
                    "PNG Images (*.png);;All Files (*)"
                )
                
                if not filename:
                    return False
            
            # Render scene to image
            scene_rect = self.scene.sceneRect()
            image = QImage(
                int(scene_rect.width()),
                int(scene_rect.height()),
                QImage.Format.Format_ARGB32
            )
            image.fill(Qt.GlobalColor.transparent)
            
            # Create painter and render
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.scene.render(painter)
            painter.end()
            
            # Add timestamp annotation
            painter = QPainter(image)
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            painter.drawText(10, 20, f"SENTINEL BEV - {timestamp_text}")
            painter.end()
            
            # Save image
            success = image.save(filename)
            
            if success:
                logger.info(f"Screenshot saved to: {filename}")
            else:
                logger.error(f"Failed to save screenshot to: {filename}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error capturing screenshot: {e}")
            return False
    
    def start_recording(self, filename: Optional[str] = None, fps: int = 30) -> bool:
        """
        Start recording BEV canvas to video.
        
        Args:
            filename: Optional filename to save to. If None, shows save dialog.
            fps: Frames per second for recording
            
        Returns:
            True if recording started, False otherwise
        """
        try:
            # Check if already recording
            if hasattr(self, 'recording') and self.recording:
                logger.warning("Recording already in progress")
                return False
            
            # Get filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                default_name = f"bev_recording_{timestamp}.mp4"
                
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Save Recording",
                    default_name,
                    "MP4 Videos (*.mp4);;All Files (*)"
                )
                
                if not filename:
                    return False
            
            # Initialize recording state
            self.recording = True
            self.recording_filename = filename
            self.recording_fps = fps
            self.recording_frames = []
            
            # Start recording timer
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self._capture_recording_frame)
            self.recording_timer.start(int(1000 / fps))
            
            logger.info(f"Started recording to: {filename} at {fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return False
    
    def _capture_recording_frame(self):
        """Capture a single frame for recording"""
        try:
            # Render scene to image
            scene_rect = self.scene.sceneRect()
            image = QImage(
                int(scene_rect.width()),
                int(scene_rect.height()),
                QImage.Format.Format_RGB888
            )
            image.fill(QColor(26, 26, 26))
            
            # Create painter and render
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.scene.render(painter)
            painter.end()
            
            # Convert to numpy array
            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(height * width * 3)
            frame = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3))
            
            # Store frame
            self.recording_frames.append(frame.copy())
            
        except Exception as e:
            logger.error(f"Error capturing recording frame: {e}")
    
    def stop_recording(self) -> bool:
        """
        Stop recording and save video.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not hasattr(self, 'recording') or not self.recording:
                logger.warning("No recording in progress")
                return False
            
            # Stop timer
            self.recording_timer.stop()
            self.recording = False
            
            # Check if we have frames
            if not self.recording_frames:
                logger.warning("No frames captured")
                return False
            
            # Save video using OpenCV
            try:
                import cv2
                
                height, width = self.recording_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    self.recording_filename,
                    fourcc,
                    self.recording_fps,
                    (width, height)
                )
                
                # Write frames
                for frame in self.recording_frames:
                    # Convert RGB to BGR for OpenCV
                    bgr_frame = frame[:, :, ::-1]
                    out.write(bgr_frame)
                
                out.release()
                
                logger.info(f"Recording saved: {self.recording_filename} ({len(self.recording_frames)} frames)")
                
                # Clean up
                self.recording_frames = []
                
                return True
                
            except ImportError:
                logger.error("OpenCV not available for video recording")
                return False
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return False
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return hasattr(self, 'recording') and self.recording
    
    # ==================== Map Overlay Methods ====================
    
    def set_show_map(self, show: bool):
        """Toggle map overlay visibility.
        
        Args:
            show: Whether to show map overlay
        """
        self.show_map = show
        
        # Toggle visibility of all map items
        if hasattr(self, 'map_items'):
            for item in self.map_items:
                item.setVisible(show)
        
        logger.debug(f"Map overlay visibility: {show}")
    
    def update_map_overlay(self, lanes: Dict, features: List, vehicle_position: tuple):
        """Update map overlay with lanes and features.
        
        Args:
            lanes: Dictionary of lane_id -> Lane objects
            features: List of MapFeature objects
            vehicle_position: Vehicle position (x, y) in map frame
        """
        # Clear existing map items
        if hasattr(self, 'map_items'):
            for item in self.map_items:
                self.scene.removeItem(item)
        
        self.map_items = []
        
        if not self.show_map:
            return
        
        try:
            # Draw lanes
            for lane_id, lane in lanes.items():
                self._draw_lane(lane, vehicle_position)
            
            # Draw features
            for feature in features:
                self._draw_feature(feature, vehicle_position)
            
            logger.debug(f"Map overlay updated: {len(lanes)} lanes, {len(features)} features")
            
        except Exception as e:
            logger.error(f"Error updating map overlay: {e}")
    
    def _draw_lane(self, lane, vehicle_position: tuple):
        """Draw a lane on the BEV canvas.
        
        Args:
            lane: Lane object with centerline and boundaries
            vehicle_position: Vehicle position (x, y) in map frame
        """
        vx, vy = vehicle_position
        
        # Draw lane boundaries
        for boundary, color in [(lane.left_boundary, QColor(100, 100, 255, 100)),
                                (lane.right_boundary, QColor(100, 100, 255, 100))]:
            if not boundary:
                continue
            
            # Convert to BEV coordinates
            points = []
            for x, y, _ in boundary:
                # Transform to vehicle frame
                bev_x, bev_y = self._map_to_bev_coords(x - vx, y - vy)
                if bev_x is not None:
                    points.append(QPointF(bev_x, bev_y))
            
            if len(points) >= 2:
                # Draw polyline
                pen = QPen(color, 2, Qt.PenStyle.DashLine)
                for i in range(len(points) - 1):
                    line_item = self.scene.addLine(
                        points[i].x(), points[i].y(),
                        points[i+1].x(), points[i+1].y(),
                        pen
                    )
                    line_item.setZValue(1)  # Above BEV image
                    self.map_items.append(line_item)
        
        # Draw centerline
        if lane.centerline:
            points = []
            for x, y, _ in lane.centerline:
                bev_x, bev_y = self._map_to_bev_coords(x - vx, y - vy)
                if bev_x is not None:
                    points.append(QPointF(bev_x, bev_y))
            
            if len(points) >= 2:
                pen = QPen(QColor(150, 150, 255, 150), 1, Qt.PenStyle.DotLine)
                for i in range(len(points) - 1):
                    line_item = self.scene.addLine(
                        points[i].x(), points[i].y(),
                        points[i+1].x(), points[i+1].y(),
                        pen
                    )
                    line_item.setZValue(1)
                    self.map_items.append(line_item)
    
    def _draw_feature(self, feature, vehicle_position: tuple):
        """Draw a map feature on the BEV canvas.
        
        Args:
            feature: MapFeature object
            vehicle_position: Vehicle position (x, y) in map frame
        """
        vx, vy = vehicle_position
        fx, fy, _ = feature.position
        
        # Transform to vehicle frame
        bev_x, bev_y = self._map_to_bev_coords(fx - vx, fy - vy)
        
        if bev_x is None:
            return
        
        # Draw based on feature type
        if feature.type == 'sign':
            # Draw traffic sign as triangle
            size = 8
            points = [
                QPointF(bev_x, bev_y - size),
                QPointF(bev_x - size, bev_y + size),
                QPointF(bev_x + size, bev_y + size)
            ]
            polygon = QPolygonF(points)
            item = self.scene.addPolygon(
                polygon,
                QPen(QColor(255, 200, 0), 2),
                QBrush(QColor(255, 200, 0, 100))
            )
            item.setZValue(2)
            self.map_items.append(item)
            
        elif feature.type == 'light':
            # Draw traffic light as circle
            size = 6
            item = self.scene.addEllipse(
                bev_x - size/2, bev_y - size/2, size, size,
                QPen(QColor(255, 100, 100), 2),
                QBrush(QColor(255, 100, 100, 100))
            )
            item.setZValue(2)
            self.map_items.append(item)
            
        elif feature.type == 'crosswalk':
            # Draw crosswalk as rectangle
            if feature.geometry and len(feature.geometry) >= 2:
                points = []
                for x, y in feature.geometry:
                    bx, by = self._map_to_bev_coords(x - vx, y - vy)
                    if bx is not None:
                        points.append(QPointF(bx, by))
                
                if len(points) >= 2:
                    pen = QPen(QColor(255, 255, 255), 3, Qt.PenStyle.SolidLine)
                    for i in range(len(points) - 1):
                        line_item = self.scene.addLine(
                            points[i].x(), points[i].y(),
                            points[i+1].x(), points[i+1].y(),
                            pen
                        )
                        line_item.setZValue(2)
                        self.map_items.append(line_item)
    
    def _map_to_bev_coords(self, x: float, y: float) -> tuple:
        """Convert map coordinates to BEV pixel coordinates.
        
        Args:
            x: X coordinate in vehicle frame (meters, forward)
            y: Y coordinate in vehicle frame (meters, left)
            
        Returns:
            Tuple of (bev_x, bev_y) in pixels, or (None, None) if out of bounds
        """
        # BEV scale: 0.1 meters per pixel
        # BEV origin: center bottom (320, 480)
        scale = 10.0  # pixels per meter
        origin_x = 320
        origin_y = 480
        
        # Transform: x (forward) -> -y in BEV, y (left) -> -x in BEV
        bev_x = origin_x - y * scale
        bev_y = origin_y - x * scale
        
        # Check bounds
        if 0 <= bev_x < self.bev_size and 0 <= bev_y < self.bev_size:
            return (bev_x, bev_y)
        else:
            return (None, None)
