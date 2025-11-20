"""Tests for semantic segmentation module."""

import pytest
import numpy as np
import time
from pathlib import Path

from src.perception.segmentation import (
    BEVSegmentationModel,
    TemporalSmoother,
    SemanticSegmentor
)
from src.core.data_structures import SegmentationOutput


class TestBEVSegmentationModel:
    """Test BEV segmentation model wrapper."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32'
        }
        
        model = BEVSegmentationModel(
            model_path=config['weights'],
            device=config['device'],
            use_fp16=False
        )
        
        assert model is not None
        assert model.num_classes == 9
        assert len(model.CLASSES) == 9
    
    def test_model_inference(self):
        """Test model inference produces valid output."""
        model = BEVSegmentationModel(
            model_path='models/bev_segmentation.pth',
            device='cpu',
            use_fp16=False
        )
        
        # Create dummy BEV image
        bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        class_map, confidence = model.infer(bev_image)
        
        # Validate output shapes
        assert class_map.shape == (640, 640)
        assert confidence.shape == (640, 640)
        
        # Validate output types
        assert class_map.dtype == np.int8
        assert confidence.dtype == np.float32
        
        # Validate output ranges
        assert np.all(class_map >= 0)
        assert np.all(class_map < 9)
        assert np.all(confidence >= 0)
        assert np.all(confidence <= 1)


class TestTemporalSmoother:
    """Test temporal smoothing."""
    
    def test_smoother_initialization(self):
        """Test smoother can be initialized."""
        smoother = TemporalSmoother(alpha=0.7)
        assert smoother.alpha == 0.7
        assert smoother.frame_count == 0
    
    def test_smoother_first_frame(self):
        """Test first frame passes through unchanged."""
        smoother = TemporalSmoother(alpha=0.7)
        
        class_map = np.random.randint(0, 9, (640, 640), dtype=np.int8)
        confidence = np.random.rand(640, 640).astype(np.float32)
        
        smoothed_class, smoothed_conf = smoother.smooth(class_map, confidence)
        
        assert np.array_equal(smoothed_class, class_map)
        assert np.array_equal(smoothed_conf, confidence)
        assert smoother.frame_count == 1
    
    def test_smoother_temporal_consistency(self):
        """Test smoother reduces temporal variation."""
        smoother = TemporalSmoother(alpha=0.7)
        
        # First frame
        class_map1 = np.zeros((640, 640), dtype=np.int8)
        confidence1 = np.ones((640, 640), dtype=np.float32) * 0.9
        smoother.smooth(class_map1, confidence1)
        
        # Second frame with different values
        class_map2 = np.ones((640, 640), dtype=np.int8)
        confidence2 = np.ones((640, 640), dtype=np.float32) * 0.5
        _, smoothed_conf2 = smoother.smooth(class_map2, confidence2)
        
        # Smoothed confidence should be between the two values
        assert np.all(smoothed_conf2 > confidence2)
        assert np.all(smoothed_conf2 < confidence1)
    
    def test_smoother_reset(self):
        """Test smoother can be reset."""
        smoother = TemporalSmoother(alpha=0.7)
        
        class_map = np.zeros((640, 640), dtype=np.int8)
        confidence = np.ones((640, 640), dtype=np.float32)
        smoother.smooth(class_map, confidence)
        
        assert smoother.frame_count == 1
        
        smoother.reset()
        assert smoother.frame_count == 0
        assert smoother.smoothed_confidence is None


class TestSemanticSegmentor:
    """Test semantic segmentor."""
    
    def test_segmentor_initialization(self):
        """Test segmentor can be initialized."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': True,
            'smoothing_alpha': 0.7
        }
        
        segmentor = SemanticSegmentor(config)
        assert segmentor is not None
        assert segmentor.use_temporal_smoothing is True
    
    def test_segmentor_segment(self):
        """Test segmentation produces valid output."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': False
        }
        
        segmentor = SemanticSegmentor(config)
        
        # Create dummy BEV image
        bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run segmentation
        output = segmentor.segment(bev_image)
        
        # Validate output
        assert isinstance(output, SegmentationOutput)
        assert output.class_map.shape == (640, 640)
        assert output.confidence.shape == (640, 640)
        assert output.class_map.dtype == np.int8
        assert output.confidence.dtype == np.float32
    
    def test_segmentor_with_temporal_smoothing(self):
        """Test segmentation with temporal smoothing."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': True,
            'smoothing_alpha': 0.7
        }
        
        segmentor = SemanticSegmentor(config)
        
        # Process multiple frames
        for _ in range(5):
            bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            output = segmentor.segment(bev_image)
            assert isinstance(output, SegmentationOutput)
    
    def test_segmentor_error_recovery(self):
        """Test segmentor recovers from errors."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': False
        }
        
        segmentor = SemanticSegmentor(config)
        
        # First, get a valid output
        bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        output1 = segmentor.segment(bev_image)
        assert segmentor.last_valid_output is not None
        
        # Try with invalid input
        output2 = segmentor.segment(None)
        
        # Should return fallback output
        assert isinstance(output2, SegmentationOutput)
        assert output2.class_map.shape == (640, 640)
    
    def test_segmentor_performance_tracking(self):
        """Test performance statistics tracking."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': False
        }
        
        segmentor = SemanticSegmentor(config)
        
        # Process a few frames
        for _ in range(3):
            bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            segmentor.segment(bev_image)
        
        # Get performance stats
        stats = segmentor.get_performance_stats()
        
        assert 'mean_inference_time' in stats
        assert 'p95_inference_time' in stats
        assert 'fps' in stats
        assert stats['mean_inference_time'] > 0
    
    def test_segmentor_reset(self):
        """Test segmentor can be reset."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': True
        }
        
        segmentor = SemanticSegmentor(config)
        
        # Process a frame
        bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        segmentor.segment(bev_image)
        
        assert segmentor.last_valid_output is not None
        assert len(segmentor.inference_times) > 0
        
        # Reset
        segmentor.reset()
        
        assert segmentor.last_valid_output is None
        assert len(segmentor.inference_times) == 0


def calculate_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 9) -> float:
    """
    Calculate mean Intersection over Union.
    
    Args:
        pred: Predicted class map (H, W)
        gt: Ground truth class map (H, W)
        num_classes: Number of classes
        
    Returns:
        Mean IoU score
    """
    ious = []
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        gt_mask = gt == cls
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            # Class not present in ground truth or prediction
            continue
        
        iou = intersection / union
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


class TestSegmentationValidation:
    """Test segmentation accuracy validation."""
    
    def test_miou_calculation(self):
        """Test mIoU calculation."""
        # Perfect prediction
        pred = np.array([[0, 1], [2, 3]], dtype=np.int8)
        gt = np.array([[0, 1], [2, 3]], dtype=np.int8)
        
        miou = calculate_miou(pred, gt, num_classes=4)
        assert miou == 1.0
        
        # Partial overlap
        pred = np.array([[0, 0], [0, 0]], dtype=np.int8)
        gt = np.array([[0, 1], [0, 1]], dtype=np.int8)
        
        miou = calculate_miou(pred, gt, num_classes=2)
        assert 0 < miou < 1
    
    def test_segmentation_on_synthetic_data(self):
        """Test segmentation on synthetic validation data."""
        config = {
            'weights': 'models/bev_segmentation.pth',
            'device': 'cpu',
            'precision': 'fp32',
            'temporal_smoothing': False
        }
        
        segmentor = SemanticSegmentor(config)
        
        # Create synthetic validation set
        num_samples = 10
        ious = []
        
        for _ in range(num_samples):
            # Create synthetic BEV image
            bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run segmentation
            output = segmentor.segment(bev_image)
            
            # Create synthetic ground truth (for testing purposes)
            # In real validation, this would come from labeled dataset
            gt = np.random.randint(0, 9, (640, 640), dtype=np.int8)
            
            # Calculate IoU
            iou = calculate_miou(output.class_map, gt, num_classes=9)
            ious.append(iou)
        
        mean_iou = np.mean(ious)
        
        # Note: With random ground truth, we don't expect high mIoU
        # This test validates the calculation pipeline
        assert 0 <= mean_iou <= 1
        
        print(f"\nSynthetic validation mIoU: {mean_iou:.3f}")
        print("Note: Real validation requires labeled dataset")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
