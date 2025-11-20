"""Example usage of BEV generation module."""

import numpy as np
import yaml
from pathlib import Path

# This example demonstrates how to use the BEV generation module
# Note: Requires opencv-python to be installed

def load_calibration(calibration_path: str) -> dict:
    """Load calibration from YAML file."""
    with open(calibration_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Demonstrate BEV generation."""
    try:
        from src.perception.bev import BEVGenerator
        from src.core.config import ConfigManager
        
        # Load configuration
        config_manager = ConfigManager('configs/default.yaml')
        bev_config = config_manager.get('bev')
        
        # Load calibrations for external cameras
        calibrations = {
            'front_left': load_calibration('configs/calibration/front_left.yaml'),
            'front_right': load_calibration('configs/calibration/front_right.yaml')
        }
        
        # Initialize BEV generator
        generator = BEVGenerator(bev_config, calibrations)
        print(f"BEV Generator initialized with output size: {generator.output_size}")
        
        # Simulate camera frames (in real usage, these come from CameraManager)
        # Front left camera: 1280x720
        front_left_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        # Front right camera: 1280x720
        front_right_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        frames = [front_left_frame, front_right_frame]
        
        # Generate BEV
        bev_output = generator.generate(frames)
        
        print(f"BEV Output:")
        print(f"  - Image shape: {bev_output.image.shape}")
        print(f"  - Mask shape: {bev_output.mask.shape}")
        print(f"  - Valid pixels: {np.sum(bev_output.mask)}/{bev_output.mask.size}")
        print(f"  - Timestamp: {bev_output.timestamp}")
        
        # Get performance statistics
        stats = generator.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"  - Average: {stats['avg_ms']:.2f}ms")
        print(f"  - Min: {stats['min_ms']:.2f}ms")
        print(f"  - Max: {stats['max_ms']:.2f}ms")
        print(f"  - P95: {stats['p95_ms']:.2f}ms")
        print(f"  - Target: 15.0ms")
        
        if stats['avg_ms'] < 15.0:
            print("\n✓ Performance target met!")
        else:
            print(f"\n⚠ Performance target exceeded by {stats['avg_ms'] - 15.0:.2f}ms")
        
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Please install required packages: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
