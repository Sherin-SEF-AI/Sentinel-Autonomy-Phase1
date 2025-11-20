"""2D object detector using YOLOv8."""

import logging
from typing import List, Dict
import numpy as np
from pathlib import Path

from src.core.data_structures import Detection2D


class Detector2D:
    """2D object detector for per-camera detection using YOLOv8."""
    
    def __init__(self, config: Dict):
        """
        Initialize 2D detector.
        
        Args:
            config: Detection configuration containing:
                - architecture: Model architecture (e.g., 'YOLOv8')
                - variant: Model variant (e.g., 'yolov8m')
                - weights: Path to model weights
                - confidence_threshold: Minimum confidence for detections
                - nms_threshold: NMS IoU threshold
                - device: Device for inference ('cuda' or 'cpu')
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.device = config.get('device', 'cuda')
        
        # Load YOLOv8 model
        self._load_model()
        
        self.logger.info(f"Detector2D initialized with {config.get('variant', 'yolov8m')}")
    
    def _load_model(self):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            weights_path = self.config.get('weights', 'models/yolov8m_automotive.pt')
            
            # Check if weights exist
            if not Path(weights_path).exists():
                self.logger.warning(f"Weights not found at {weights_path}, using default YOLOv8m")
                weights_path = 'yolov8m.pt'
            
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            
            self.logger.info(f"Loaded YOLOv8 model from {weights_path}")
            
        except ImportError:
            self.logger.error("ultralytics package not installed. Install with: pip install ultralytics")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame: np.ndarray, camera_id: int) -> List[Detection2D]:
        """
        Detect objects in a single camera frame.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            camera_id: Camera identifier
        
        Returns:
            List of Detection2D objects
        """
        try:
            # Run inference
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                iou=self.nms_threshold,
                verbose=False,
                device=self.device
            )
            
            detections = []
            
            # Parse results
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get bounding box coordinates (x1, y1, x2, y2)
                    bbox = boxes.xyxy[i].cpu().numpy()
                    
                    # Get class and confidence
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class name
                    class_name = result.names[cls_id]
                    
                    # Filter by automotive classes
                    if class_name in ['car', 'truck', 'bus', 'person', 'bicycle', 
                                     'motorcycle', 'traffic light', 'stop sign']:
                        # Map to SENTINEL class names
                        mapped_class = self._map_class_name(class_name)
                        
                        detection = Detection2D(
                            bbox=(float(bbox[0]), float(bbox[1]), 
                                 float(bbox[2]), float(bbox[3])),
                            class_name=mapped_class,
                            confidence=confidence,
                            camera_id=camera_id
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed for camera {camera_id}: {e}")
            return []
    
    def _map_class_name(self, yolo_class: str) -> str:
        """
        Map YOLO class names to SENTINEL class names.
        
        Args:
            yolo_class: YOLO class name
        
        Returns:
            SENTINEL class name
        """
        mapping = {
            'car': 'vehicle',
            'truck': 'vehicle',
            'bus': 'vehicle',
            'person': 'pedestrian',
            'bicycle': 'cyclist',
            'motorcycle': 'cyclist',
            'traffic light': 'traffic_light',
            'stop sign': 'traffic_sign'
        }
        return mapping.get(yolo_class, yolo_class)
    
    def detect_batch(self, frames: Dict[int, np.ndarray]) -> Dict[int, List[Detection2D]]:
        """
        Detect objects in multiple camera frames.
        
        Args:
            frames: Dictionary mapping camera_id to frame
        
        Returns:
            Dictionary mapping camera_id to list of Detection2D objects
        """
        detections = {}
        for camera_id, frame in frames.items():
            detections[camera_id] = self.detect(frame, camera_id)
        
        return detections
