"""Example usage of Camera Management Module."""

import time
import logging
from src.camera import CameraManager
from src.core.config import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate camera manager usage."""
    # Load configuration
    config_manager = ConfigManager('configs/default.yaml')
    config = config_manager.config
    
    # Create camera manager
    camera_manager = CameraManager(config)
    
    try:
        # Start cameras
        print("Starting cameras...")
        camera_manager.start()
        
        # Check health
        if camera_manager.is_healthy():
            print("✓ All cameras healthy")
        else:
            print("⚠ Some cameras unhealthy")
        
        # Capture frames for 10 seconds
        print("\nCapturing frames for 10 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 10.0:
            # Get synchronized frame bundle
            bundle = camera_manager.get_frame_bundle()
            
            if bundle is not None:
                frame_count += 1
                
                # Print frame info every second
                if frame_count % 30 == 0:
                    print(f"Captured {frame_count} frames")
                    print(f"  Interior: {bundle.interior.shape}")
                    print(f"  Front Left: {bundle.front_left.shape}")
                    print(f"  Front Right: {bundle.front_right.shape}")
                    print(f"  Timestamp: {bundle.timestamp:.3f}")
            
            time.sleep(0.033)  # ~30 FPS
        
        # Print statistics
        print("\n" + "="*50)
        print("Camera Manager Statistics:")
        print("="*50)
        stats = camera_manager.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Get calibration info
        print("\n" + "="*50)
        print("Calibration Information:")
        print("="*50)
        for camera_id in [0, 1, 2]:
            calib = camera_manager.get_calibration(camera_id)
            if calib:
                print(f"\nCamera {camera_id}:")
                print(f"  Focal length: ({calib.intrinsics.fx:.1f}, {calib.intrinsics.fy:.1f})")
                print(f"  Principal point: ({calib.intrinsics.cx:.1f}, {calib.intrinsics.cy:.1f})")
                print(f"  Translation: {calib.extrinsics.translation}")
                print(f"  Rotation: {calib.extrinsics.rotation}")
        
    finally:
        # Stop cameras
        print("\nStopping cameras...")
        camera_manager.stop()
        print("✓ Cameras stopped")

if __name__ == '__main__':
    main()
