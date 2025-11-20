"""Gaze estimation using L2CS-Net model."""

import logging
from typing import Dict, Any, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path

logger = logging.getLogger(__name__)


class GazeEstimator:
    """Gaze estimation using L2CS-Net model."""
    
    # Define 8 attention zones around vehicle
    ATTENTION_ZONES = {
        'front': (-30, 30),           # -30° to 30° yaw
        'front_left': (30, 75),       # 30° to 75° yaw
        'left': (75, 105),            # 75° to 105° yaw
        'rear_left': (105, 150),      # 105° to 150° yaw
        'rear': (150, 180),           # 150° to 180° yaw (and -180° to -150°)
        'rear_right': (-150, -105),   # -150° to -105° yaw
        'right': (-105, -75),         # -105° to -75° yaw
        'front_right': (-75, -30),    # -75° to -30° yaw
    }
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        """
        Initialize gaze estimator.
        
        Args:
            model_path: Path to L2CS-Net model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if model_path and Path(model_path).exists():
            try:
                self.model = self._load_model(model_path)
                logger.info(f"GazeEstimator initialized with model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load gaze model: {e}. Using fallback estimation.")
        else:
            logger.warning(f"Gaze model not found at {model_path}. Using fallback estimation.")
    
    def _load_model(self, model_path: str):
        """Load L2CS-Net model."""
        # Placeholder for actual L2CS-Net model loading
        # In production, this would load the actual model architecture and weights
        model = SimplifiedGazeModel()
        
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def estimate_gaze(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, Any]:
        """
        Estimate gaze direction from face image and landmarks.
        
        Args:
            frame: Input image (H, W, 3) in BGR format
            landmarks: Facial landmarks array (68, 2)
            
        Returns:
            Dictionary with:
                - pitch: Gaze pitch angle in degrees
                - yaw: Gaze yaw angle in degrees
                - attention_zone: One of 8 spatial zones
        """
        if frame is None or landmarks is None or len(landmarks) < 68:
            return self._default_gaze()
        
        try:
            if self.model is not None:
                # Use model-based estimation
                pitch, yaw = self._model_based_estimation(frame, landmarks)
            else:
                # Use geometric fallback estimation
                pitch, yaw = self._geometric_estimation(landmarks)
            
            # Map gaze to attention zone
            attention_zone = self._map_to_attention_zone(yaw)
            
            return {
                'pitch': float(pitch),
                'yaw': float(yaw),
                'attention_zone': attention_zone
            }
        
        except Exception as e:
            logger.error(f"Gaze estimation failed: {e}")
            return self._default_gaze()
    
    def _model_based_estimation(self, frame: np.ndarray, landmarks: np.ndarray) -> tuple:
        """
        Estimate gaze using neural network model.
        
        Args:
            frame: Input image
            landmarks: Facial landmarks
            
        Returns:
            Tuple of (pitch, yaw) in degrees
        """
        # Extract eye regions
        left_eye_region = self._extract_eye_region(frame, landmarks, eye='left')
        right_eye_region = self._extract_eye_region(frame, landmarks, eye='right')
        
        # Preprocess for model
        left_eye_tensor = self._preprocess_eye(left_eye_region)
        right_eye_tensor = self._preprocess_eye(right_eye_region)
        
        with torch.no_grad():
            # Run inference
            pitch_left, yaw_left = self.model(left_eye_tensor)
            pitch_right, yaw_right = self.model(right_eye_tensor)
            
            # Average both eyes
            pitch = (pitch_left.item() + pitch_right.item()) / 2
            yaw = (yaw_left.item() + yaw_right.item()) / 2
        
        return pitch, yaw
    
    def _geometric_estimation(self, landmarks: np.ndarray) -> tuple:
        """
        Estimate gaze using geometric approach based on eye landmarks.
        
        Args:
            landmarks: Facial landmarks (68, 2)
            
        Returns:
            Tuple of (pitch, yaw) in degrees
        """
        # Right eye landmarks (36-41)
        right_eye = landmarks[36:42]
        # Left eye landmarks (42-47)
        left_eye = landmarks[42:48]
        
        # Calculate eye centers
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye_center = np.mean(left_eye, axis=0)
        
        # Nose tip (landmark 30)
        nose_tip = landmarks[30]
        
        # Estimate yaw from eye-nose geometry
        eye_center = (right_eye_center + left_eye_center) / 2
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        
        # Horizontal offset from nose to eye center
        horizontal_offset = nose_tip[0] - eye_center[0]
        yaw = np.degrees(np.arctan2(horizontal_offset, eye_distance)) * 2
        
        # Estimate pitch from vertical eye position
        vertical_offset = nose_tip[1] - eye_center[1]
        pitch = np.degrees(np.arctan2(vertical_offset, eye_distance)) * 1.5
        
        # Clamp to reasonable ranges
        yaw = np.clip(yaw, -90, 90)
        pitch = np.clip(pitch, -60, 60)
        
        return float(pitch), float(yaw)
    
    def _extract_eye_region(self, frame: np.ndarray, landmarks: np.ndarray, eye: str) -> np.ndarray:
        """Extract eye region from frame."""
        if eye == 'left':
            eye_landmarks = landmarks[42:48]  # Left eye
        else:
            eye_landmarks = landmarks[36:42]  # Right eye
        
        # Get bounding box
        x_min = int(np.min(eye_landmarks[:, 0])) - 10
        x_max = int(np.max(eye_landmarks[:, 0])) + 10
        y_min = int(np.min(eye_landmarks[:, 1])) - 10
        y_max = int(np.max(eye_landmarks[:, 1])) + 10
        
        # Ensure bounds are valid
        h, w = frame.shape[:2]
        x_min = max(0, x_min)
        x_max = min(w, x_max)
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        # Resize to standard size
        if eye_region.size > 0:
            eye_region = cv2.resize(eye_region, (60, 36))
        else:
            eye_region = np.zeros((36, 60, 3), dtype=np.uint8)
        
        return eye_region
    
    def _preprocess_eye(self, eye_region: np.ndarray) -> torch.Tensor:
        """Preprocess eye region for model input."""
        # Convert to RGB
        eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
        
        # Normalize
        eye_normalized = eye_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor (C, H, W)
        eye_tensor = torch.from_numpy(eye_normalized).permute(2, 0, 1).unsqueeze(0)
        eye_tensor = eye_tensor.to(self.device)
        
        return eye_tensor
    
    def _map_to_attention_zone(self, yaw: float) -> str:
        """
        Map gaze yaw angle to one of 8 attention zones.
        
        Args:
            yaw: Gaze yaw angle in degrees
            
        Returns:
            Attention zone name
        """
        # Normalize yaw to [-180, 180]
        yaw = ((yaw + 180) % 360) - 180
        
        for zone, (min_angle, max_angle) in self.ATTENTION_ZONES.items():
            if zone == 'rear':
                # Special case for rear (wraps around)
                if yaw >= 150 or yaw <= -150:
                    return zone
            else:
                if min_angle <= yaw < max_angle:
                    return zone
        
        # Default to front if no match
        return 'front'
    
    def _default_gaze(self) -> Dict[str, Any]:
        """Return default gaze when estimation fails."""
        return {
            'pitch': 0.0,
            'yaw': 0.0,
            'attention_zone': 'front'
        }


class SimplifiedGazeModel(nn.Module):
    """Simplified gaze estimation model (placeholder for L2CS-Net)."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # Calculate flattened size: (60/4) * (36/4) * 64 = 15 * 9 * 64 = 8640
        self.pitch_fc = nn.Linear(8640, 1)
        self.yaw_fc = nn.Linear(8640, 1)
    
    def forward(self, x):
        features = self.features(x)
        pitch = self.pitch_fc(features) * 60  # Scale to [-60, 60]
        yaw = self.yaw_fc(features) * 90      # Scale to [-90, 90]
        return pitch, yaw
