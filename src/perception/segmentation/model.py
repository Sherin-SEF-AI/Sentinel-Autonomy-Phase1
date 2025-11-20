"""Model wrapper for BEV semantic segmentation."""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class BEVSegmentationModel:
    """Wrapper for BEV semantic segmentation model with FP16 precision."""
    
    # Class mapping for semantic segmentation
    CLASSES = [
        'road',
        'lane_marking',
        'vehicle',
        'pedestrian',
        'cyclist',
        'obstacle',
        'parking_space',
        'curb',
        'vegetation'
    ]
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        use_fp16: bool = True,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize BEV segmentation model.
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run inference on ('cuda' or 'cpu')
            use_fp16: Whether to use FP16 mixed precision
            input_size: Expected input image size (H, W)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'
        self.input_size = input_size
        self.num_classes = len(self.CLASSES)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable FP16 if requested
        if self.use_fp16:
            self.model.half()
            logger.info("FP16 precision enabled for segmentation model")

        # Enable CUDA optimizations if available
        if self.device.type == 'cuda':
            # Enable cudnn auto-tuner for optimal performance
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled")

        # Pre-allocate tensors for efficiency
        self._preallocate_tensors()

        logger.info(f"BEV segmentation model loaded on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """
        Load pretrained segmentation model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded PyTorch model
        """
        model_file = Path(model_path)
        
        if not model_file.exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.info("Creating placeholder model for development")
            # Create a simple placeholder model for development/testing
            return self._create_placeholder_model()
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model architecture and weights
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model = checkpoint['model']
            else:
                # Assume checkpoint is the model itself
                model = checkpoint
            
            logger.info(f"Loaded model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Creating placeholder model for development")
            return self._create_placeholder_model()
    
    def _create_placeholder_model(self) -> nn.Module:
        """
        Create a simple placeholder model for development/testing.
        
        Returns:
            Simple segmentation model
        """
        class PlaceholderSegmentationModel(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                # Simple encoder-decoder architecture
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, num_classes, 1),
                )
            
            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x
        
        return PlaceholderSegmentationModel(self.num_classes)
    
    def _preallocate_tensors(self):
        """Pre-allocate tensors to avoid dynamic allocation during inference."""
        dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Pre-allocate input tensor
        self.input_tensor = torch.zeros(
            (1, 3, self.input_size[0], self.input_size[1]),
            dtype=dtype,
            device=self.device
        )
        
        logger.debug("Pre-allocated inference tensors")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.

        Args:
            image: Input BEV image (H, W, 3) in BGR format, uint8

        Returns:
            Preprocessed tensor (1, 3, H, W)
        """
        # Convert BGR to RGB and normalize to [0, 1] in one step (more efficient)
        # Use cv2.cvtColor if available, otherwise use numpy slicing with copy
        image_rgb = np.ascontiguousarray(image[:, :, ::-1])
        image_float = image_rgb.astype(np.float32) / 255.0

        # Convert to tensor and rearrange dimensions
        # Use the pre-allocated tensor to avoid memory allocation overhead
        tensor = torch.from_numpy(image_float).permute(2, 0, 1).unsqueeze(0)

        # Copy to pre-allocated device tensor to avoid allocation
        self.input_tensor.copy_(tensor, non_blocking=True)

        return self.input_tensor
    
    def postprocess(
        self,
        output: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model output to class map and confidence.
        
        Args:
            output: Model output tensor (1, num_classes, H, W)
            
        Returns:
            Tuple of (class_map, confidence_map)
            - class_map: (H, W) int8 array of class indices
            - confidence_map: (H, W) float32 array of confidence scores
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)
        
        # Get class predictions and confidence
        confidence, class_indices = torch.max(probs, dim=1)
        
        # Convert to numpy
        class_map = class_indices.squeeze(0).cpu().numpy().astype(np.int8)
        confidence_map = confidence.squeeze(0).cpu().numpy().astype(np.float32)
        
        return class_map, confidence_map
    
    @torch.no_grad()
    def infer(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on BEV image.
        
        Args:
            image: Input BEV image (H, W, 3) in BGR format, uint8
            
        Returns:
            Tuple of (class_map, confidence_map)
            - class_map: (H, W) int8 array of class indices
            - confidence_map: (H, W) float32 array of confidence scores
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Inference
        output = self.model(input_tensor)
        
        # Postprocess
        class_map, confidence_map = self.postprocess(output)
        
        return class_map, confidence_map
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("Cleared GPU cache")
