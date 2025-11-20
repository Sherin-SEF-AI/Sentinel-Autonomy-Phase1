#!/usr/bin/env python3
"""
Example usage of semantic segmentation module.

This script demonstrates how to use the SemanticSegmentor to classify
BEV images into semantic classes.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from perception.segmentation import SemanticSegmentor


# Class colors for visualization (BGR format)
CLASS_COLORS = {
    0: (128, 64, 128),   # road - purple
    1: (255, 255, 255),  # lane_marking - white
    2: (0, 0, 142),      # vehicle - dark blue
    3: (220, 20, 60),    # pedestrian - red
    4: (119, 11, 32),    # cyclist - dark red
    5: (0, 0, 70),       # obstacle - very dark blue
    6: (250, 170, 160),  # parking_space - light pink
    7: (244, 35, 232),   # curb - pink
    8: (107, 142, 35),   # vegetation - olive green
}


def colorize_segmentation(class_map: np.ndarray) -> np.ndarray:
    """
    Convert class map to colored visualization.
    
    Args:
        class_map: Class indices (H, W)
        
    Returns:
        Colored image (H, W, 3) in BGR format
    """
    h, w = class_map.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls, color in CLASS_COLORS.items():
        mask = class_map == cls
        colored[mask] = color
    
    return colored


def overlay_segmentation(
    image: np.ndarray,
    class_map: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay segmentation on original image.
    
    Args:
        image: Original image (H, W, 3)
        class_map: Class indices (H, W)
        alpha: Blending factor (0-1)
        
    Returns:
        Blended image (H, W, 3)
    """
    colored = colorize_segmentation(class_map)
    blended = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    return blended


def main():
    print("Semantic Segmentation Example")
    print("=" * 60)
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    print(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    seg_config = config.get('models', {}).get('segmentation', {})
    
    # Override for CPU if no GPU available
    seg_config['device'] = 'cpu'
    seg_config['precision'] = 'fp32'
    
    # Initialize segmentor
    print("Initializing semantic segmentor...")
    segmentor = SemanticSegmentor(seg_config)
    
    # Create synthetic BEV image for demonstration
    print("\nGenerating synthetic BEV image...")
    bev_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Add some structure to make it more interesting
    # Road area (center)
    cv2.rectangle(bev_image, (100, 0), (540, 640), (128, 128, 128), -1)
    
    # Lane markings
    cv2.line(bev_image, (320, 0), (320, 640), (255, 255, 255), 3)
    
    # Vehicles
    cv2.rectangle(bev_image, (200, 300), (250, 350), (0, 0, 200), -1)
    cv2.rectangle(bev_image, (400, 400), (450, 450), (0, 0, 200), -1)
    
    # Run segmentation
    print("Running segmentation...")
    output = segmentor.segment(bev_image)
    
    print(f"\nSegmentation complete!")
    print(f"  Timestamp: {output.timestamp:.3f}")
    print(f"  Class map shape: {output.class_map.shape}")
    print(f"  Confidence shape: {output.confidence.shape}")
    print(f"  Unique classes: {np.unique(output.class_map)}")
    
    # Get performance stats
    stats = segmentor.get_performance_stats()
    print(f"\nPerformance:")
    print(f"  Inference time: {stats['mean_inference_time']*1000:.1f}ms")
    print(f"  FPS: {stats['fps']:.1f}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Colorized segmentation
    colored = colorize_segmentation(output.class_map)
    
    # Overlay on original
    overlay = overlay_segmentation(bev_image, output.class_map, alpha=0.6)
    
    # Confidence heatmap
    confidence_vis = (output.confidence * 255).astype(np.uint8)
    confidence_colored = cv2.applyColorMap(confidence_vis, cv2.COLORMAP_JET)
    
    # Save outputs
    output_dir = Path(__file__).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / 'seg_input.png'), bev_image)
    cv2.imwrite(str(output_dir / 'seg_colored.png'), colored)
    cv2.imwrite(str(output_dir / 'seg_overlay.png'), overlay)
    cv2.imwrite(str(output_dir / 'seg_confidence.png'), confidence_colored)
    
    print(f"\nOutputs saved to {output_dir}/")
    print("  - seg_input.png: Original BEV image")
    print("  - seg_colored.png: Colorized segmentation")
    print("  - seg_overlay.png: Segmentation overlay")
    print("  - seg_confidence.png: Confidence heatmap")
    
    # Test temporal smoothing
    if seg_config.get('temporal_smoothing', False):
        print("\nTesting temporal smoothing...")
        
        for i in range(5):
            # Add some noise to simulate frame changes
            noisy_image = bev_image.copy()
            noise = np.random.randint(-20, 20, bev_image.shape, dtype=np.int16)
            noisy_image = np.clip(noisy_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            output = segmentor.segment(noisy_image)
            print(f"  Frame {i+1}: {len(np.unique(output.class_map))} classes detected")
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == '__main__':
    main()
