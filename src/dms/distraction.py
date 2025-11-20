"""Distraction classification using MobileNetV3 and behavioral analysis."""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import time

logger = logging.getLogger(__name__)


class DistractionClassifier:
    """Distraction classification using MobileNetV3 and gaze analysis."""
    
    DISTRACTION_TYPES = [
        'safe_driving',
        'phone_usage',
        'looking_at_passenger',
        'adjusting_controls',
        'eyes_off_road',
        'hands_off_wheel'
    ]
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize distraction classifier.
        
        Args:
            model_path: Path to MobileNetV3 model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # Distraction tracking
        self.current_distraction = 'safe_driving'
        self.distraction_start_time = None
        self.eyes_off_road_start = None
        self.eyes_off_road_threshold = 2.0  # seconds
        
        if model_path and Path(model_path).exists():
            try:
                self.model = self._load_model(model_path)
                logger.info(f"DistractionClassifier initialized with model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load distraction model: {e}. Using rule-based classification.")
        else:
            logger.warning(f"Distraction model not found at {model_path}. Using rule-based classification.")
    
    def _load_model(self, model_path: str):
        """Load MobileNetV3 distraction classifier."""
        # Placeholder for actual MobileNetV3 model
        model = SimplifiedDistractionModel(num_classes=len(self.DISTRACTION_TYPES))
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def classify_distraction(self, frame: np.ndarray, gaze: Dict[str, Any], 
                           head_pose: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify driver distraction type.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            gaze: Gaze dictionary with pitch, yaw, attention_zone
            head_pose: Head pose dictionary with roll, pitch, yaw
            
        Returns:
            Dictionary with:
                - type: Distraction type
                - confidence: Classification confidence
                - duration: Duration of current distraction in seconds
                - eyes_off_road: Boolean flag
        """
        if frame is None:
            return self._default_distraction()
        
        try:
            current_time = time.time()
            
            # Determine distraction type
            if self.model is not None:
                distraction_type, confidence = self._model_based_classification(frame)
            else:
                distraction_type, confidence = self._rule_based_classification(gaze, head_pose)
            
            # Check eyes off road condition
            eyes_off_road = self._check_eyes_off_road(gaze, current_time)
            
            # Update distraction tracking
            if distraction_type != self.current_distraction:
                self.current_distraction = distraction_type
                self.distraction_start_time = current_time
                duration = 0.0
            else:
                if self.distraction_start_time is not None:
                    duration = current_time - self.distraction_start_time
                else:
                    duration = 0.0
            
            return {
                'type': distraction_type,
                'confidence': float(confidence),
                'duration': float(duration),
                'eyes_off_road': eyes_off_road
            }
        
        except Exception as e:
            logger.error(f"Distraction classification failed: {e}")
            return self._default_distraction()
    
    def _model_based_classification(self, frame: np.ndarray) -> tuple:
        """
        Classify distraction using neural network model.
        
        Args:
            frame: Input image
            
        Returns:
            Tuple of (distraction_type, confidence)
        """
        # Preprocess frame
        input_tensor = self._preprocess_frame(frame)
        
        with torch.no_grad():
            # Run inference
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            
            confidence, pred_idx = torch.max(probs, dim=1)
            distraction_type = self.DISTRACTION_TYPES[pred_idx.item()]
            
        return distraction_type, confidence.item()
    
    def _rule_based_classification(self, gaze: Dict[str, Any], 
                                   head_pose: Dict[str, float]) -> tuple:
        """
        Classify distraction using rule-based approach.
        
        Args:
            gaze: Gaze dictionary
            head_pose: Head pose dictionary
            
        Returns:
            Tuple of (distraction_type, confidence)
        """
        attention_zone = gaze.get('attention_zone', 'front')
        head_yaw = head_pose.get('yaw', 0.0)
        head_pitch = head_pose.get('pitch', 0.0)
        
        # Rule-based classification
        if attention_zone == 'front':
            return 'safe_driving', 0.9
        
        elif attention_zone in ['left', 'right']:
            # Looking to the side - could be passenger or controls
            if abs(head_yaw) > 45:
                return 'looking_at_passenger', 0.7
            else:
                return 'adjusting_controls', 0.6
        
        elif head_pitch < -20:
            # Looking down - likely phone or controls
            return 'phone_usage', 0.7
        
        elif attention_zone in ['rear_left', 'rear_right', 'rear']:
            # Looking back
            return 'eyes_off_road', 0.8
        
        else:
            return 'eyes_off_road', 0.6
    
    def _check_eyes_off_road(self, gaze: Dict[str, Any], current_time: float) -> bool:
        """
        Check if eyes have been off road for more than threshold.
        
        Args:
            gaze: Gaze dictionary
            current_time: Current timestamp
            
        Returns:
            True if eyes off road for > 2 seconds
        """
        attention_zone = gaze.get('attention_zone', 'front')
        
        # Define "on road" zones
        on_road_zones = ['front', 'front_left', 'front_right']
        
        if attention_zone not in on_road_zones:
            # Eyes are off road
            if self.eyes_off_road_start is None:
                self.eyes_off_road_start = current_time
            else:
                duration = current_time - self.eyes_off_road_start
                if duration > self.eyes_off_road_threshold:
                    return True
        else:
            # Eyes are on road
            self.eyes_off_road_start = None
        
        return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for model input."""
        # Resize to model input size
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_normalized = (frame_normalized - mean) / std
        
        # Convert to tensor (C, H, W)
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor
    
    def _default_distraction(self) -> Dict[str, Any]:
        """Return default distraction state."""
        return {
            'type': 'safe_driving',
            'confidence': 0.5,
            'duration': 0.0,
            'eyes_off_road': False
        }


class SimplifiedDistractionModel(nn.Module):
    """Simplified distraction classification model (placeholder for MobileNetV3)."""
    
    def __init__(self, num_classes: int = 6):
        super().__init__()
        
        # Simplified architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
