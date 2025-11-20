"""Standalone tests for temporal smoother (no dependencies)."""

import pytest
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Load smoother module directly without going through __init__.py
smoother_path = Path(__file__).parent.parent / 'src' / 'perception' / 'segmentation' / 'smoother.py'
spec = importlib.util.spec_from_file_location("smoother", smoother_path)
smoother_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoother_module)

TemporalSmoother = smoother_module.TemporalSmoother


class TestTemporalSmoother:
    """Test temporal smoothing."""
    
    def test_smoother_initialization(self):
        """Test smoother can be initialized."""
        smoother = TemporalSmoother(alpha=0.7)
        assert smoother.alpha == 0.7
        assert smoother.frame_count == 0
    
    def test_smoother_invalid_alpha(self):
        """Test smoother rejects invalid alpha values."""
        with pytest.raises(ValueError):
            TemporalSmoother(alpha=1.5)
        
        with pytest.raises(ValueError):
            TemporalSmoother(alpha=-0.1)
    
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
        # alpha=0.7: smoothed = 0.7 * 0.5 + 0.3 * 0.9 = 0.35 + 0.27 = 0.62
        expected = 0.7 * 0.5 + 0.3 * 0.9
        assert np.allclose(smoothed_conf2, expected)
        assert np.all(smoothed_conf2 > confidence2)
        assert np.all(smoothed_conf2 < confidence1)
    
    def test_smoother_multiple_frames(self):
        """Test smoother over multiple frames."""
        smoother = TemporalSmoother(alpha=0.7)
        
        # Process 10 frames
        for i in range(10):
            class_map = np.full((640, 640), i % 9, dtype=np.int8)
            confidence = np.ones((640, 640), dtype=np.float32) * (0.5 + i * 0.05)
            smoothed_class, smoothed_conf = smoother.smooth(class_map, confidence)
            
            assert smoothed_class.shape == (640, 640)
            assert smoothed_conf.shape == (640, 640)
        
        assert smoother.frame_count == 10
    
    def test_smoother_reset(self):
        """Test smoother can be reset."""
        smoother = TemporalSmoother(alpha=0.7)
        
        class_map = np.zeros((640, 640), dtype=np.int8)
        confidence = np.ones((640, 640), dtype=np.float32)
        smoother.smooth(class_map, confidence)
        
        assert smoother.frame_count == 1
        assert smoother.smoothed_confidence is not None
        
        smoother.reset()
        assert smoother.frame_count == 0
        assert smoother.smoothed_confidence is None
    
    def test_smoother_different_alpha_values(self):
        """Test smoother behavior with different alpha values."""
        # High alpha (more weight on current frame)
        smoother_high = TemporalSmoother(alpha=0.9)
        
        # Low alpha (more weight on history)
        smoother_low = TemporalSmoother(alpha=0.3)
        
        # First frame
        class_map1 = np.zeros((640, 640), dtype=np.int8)
        confidence1 = np.ones((640, 640), dtype=np.float32) * 0.9
        smoother_high.smooth(class_map1, confidence1.copy())
        smoother_low.smooth(class_map1, confidence1.copy())
        
        # Second frame
        class_map2 = np.ones((640, 640), dtype=np.int8)
        confidence2 = np.ones((640, 640), dtype=np.float32) * 0.3
        _, smoothed_high = smoother_high.smooth(class_map2, confidence2.copy())
        _, smoothed_low = smoother_low.smooth(class_map2, confidence2.copy())
        
        # High alpha should be closer to current frame
        # Low alpha should be closer to previous frame
        assert np.mean(smoothed_high) < np.mean(smoothed_low)


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


class TestMIoUCalculation:
    """Test mIoU calculation utility."""
    
    def test_miou_perfect_prediction(self):
        """Test mIoU with perfect prediction."""
        pred = np.array([[0, 1], [2, 3]], dtype=np.int8)
        gt = np.array([[0, 1], [2, 3]], dtype=np.int8)
        
        miou = calculate_miou(pred, gt, num_classes=4)
        assert miou == 1.0
    
    def test_miou_no_overlap(self):
        """Test mIoU with no overlap."""
        pred = np.array([[0, 0], [0, 0]], dtype=np.int8)
        gt = np.array([[1, 1], [1, 1]], dtype=np.int8)
        
        miou = calculate_miou(pred, gt, num_classes=2)
        assert miou == 0.0
    
    def test_miou_partial_overlap(self):
        """Test mIoU with partial overlap."""
        pred = np.array([[0, 0], [0, 0]], dtype=np.int8)
        gt = np.array([[0, 1], [0, 1]], dtype=np.int8)
        
        miou = calculate_miou(pred, gt, num_classes=2)
        
        # Class 0: pred has 4 pixels, gt has 2 pixels
        #          intersection=2, union=4, iou=2/4=0.5
        # Class 1: pred has 0 pixels, gt has 2 pixels
        #          intersection=0, union=2, iou=0
        # Mean: (0.5 + 0) / 2 = 0.25
        expected = 0.25
        assert np.isclose(miou, expected)
    
    def test_miou_multiclass(self):
        """Test mIoU with multiple classes."""
        # Create 4x4 image with 4 classes
        pred = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ], dtype=np.int8)
        
        gt = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3]
        ], dtype=np.int8)
        
        miou = calculate_miou(pred, gt, num_classes=4)
        assert miou == 1.0
    
    def test_miou_with_missing_classes(self):
        """Test mIoU when some classes are not present."""
        # Only classes 0 and 1 present
        pred = np.array([[0, 0], [1, 1]], dtype=np.int8)
        gt = np.array([[0, 0], [1, 1]], dtype=np.int8)
        
        # Calculate for 9 classes (but only 2 present)
        miou = calculate_miou(pred, gt, num_classes=9)
        
        # Should only average over present classes
        assert miou == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
