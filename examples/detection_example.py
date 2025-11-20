"""Example demonstrating multi-view object detection."""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.core.config import ConfigManager
from src.perception.detection import ObjectDetector


def main():
    """Run object detection example."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting object detection example...")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
    config_manager = ConfigManager(str(config_path))
    
    # Prepare detection configuration
    detection_config = {
        'detection': config_manager.get('models.detection', {}),
        'fusion': config_manager.get('fusion', {}),
        'tracking': config_manager.get('tracking', {})
    }
    
    # Load calibration data for external cameras
    calibration_data = {}
    for camera_name, camera_id in [('front_left', 1), ('front_right', 2)]:
        calib_path = config_manager.get(f'cameras.{camera_name}.calibration')
        if calib_path:
            try:
                calibration_data[camera_id] = config_manager.load_calibration(calib_path)
                logger.info(f"Loaded calibration for {camera_name} (ID: {camera_id})")
            except Exception as e:
                logger.warning(f"Could not load calibration for {camera_name}: {e}")
                # Use default calibration
                calibration_data[camera_id] = {
                    'intrinsics': {
                        'fx': 800.0, 'fy': 800.0,
                        'cx': 640.0, 'cy': 360.0,
                        'distortion': [0, 0, 0, 0, 0]
                    },
                    'extrinsics': {
                        'translation': [2.0, -0.5 if camera_id == 1 else 0.5, 1.2],
                        'rotation': [0, 0.1, 0]
                    }
                }
    
    # Initialize detector
    logger.info("Initializing object detector...")
    detector = ObjectDetector(detection_config, calibration_data)
    
    # Initialize cameras
    logger.info("Initializing cameras...")
    camera_ids = {
        1: config_manager.get('cameras.front_left.device', 1),
        2: config_manager.get('cameras.front_right.device', 2)
    }
    
    caps = {}
    for cam_id, device_id in camera_ids.items():
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            caps[cam_id] = cap
            logger.info(f"Camera {cam_id} opened successfully")
        else:
            logger.warning(f"Could not open camera {cam_id} (device {device_id})")
    
    if not caps:
        logger.error("No cameras available. Using test images instead.")
        use_test_images = True
    else:
        use_test_images = False
    
    logger.info("Starting detection loop. Press 'q' to quit.")
    
    frame_count = 0
    fps_start = time.time()
    
    try:
        while True:
            frames = {}
            
            if use_test_images:
                # Create synthetic test frames
                for cam_id in [1, 2]:
                    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    # Add some random rectangles to simulate objects
                    for _ in range(3):
                        x1 = np.random.randint(0, 1000)
                        y1 = np.random.randint(0, 500)
                        x2 = x1 + np.random.randint(100, 300)
                        y2 = y1 + np.random.randint(100, 300)
                        color = tuple(np.random.randint(0, 255, 3).tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                    frames[cam_id] = frame
            else:
                # Capture from real cameras
                for cam_id, cap in caps.items():
                    ret, frame = cap.read()
                    if ret:
                        frames[cam_id] = frame
            
            if not frames:
                logger.warning("No frames captured")
                break
            
            # Run detection
            start_time = time.time()
            detections_2d, detections_3d = detector.detect(frames)
            detection_time = (time.time() - start_time) * 1000
            
            # Display results
            logger.info(f"Frame {frame_count}: {len(detections_3d)} objects tracked "
                       f"({detection_time:.1f}ms)")
            
            for det_3d in detections_3d:
                x, y, z, w, h, l, theta = det_3d.bbox_3d
                vx, vy, vz = det_3d.velocity
                logger.info(f"  Track {det_3d.track_id}: {det_3d.class_name} "
                          f"at ({x:.1f}, {y:.1f}, {z:.1f})m, "
                          f"vel ({vx:.1f}, {vy:.1f}, {vz:.1f})m/s, "
                          f"conf {det_3d.confidence:.2f}")
            
            # Visualize detections on frames
            for cam_id, frame in frames.items():
                vis_frame = frame.copy()
                
                # Draw 2D detections
                if cam_id in detections_2d:
                    for det_2d in detections_2d[cam_id]:
                        x1, y1, x2, y2 = det_2d.bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Draw bounding box
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{det_2d.class_name} {det_2d.confidence:.2f}"
                        cv2.putText(vis_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add info text
                info_text = f"Camera {cam_id} | {detection_time:.1f}ms | {len(detections_3d)} tracks"
                cv2.putText(vis_frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow(f'Camera {cam_id}', vis_frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                logger.info(f"FPS: {fps:.1f}")
                fps_start = time.time()
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Limit to reasonable rate for test images
            if use_test_images:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Cleanup
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        stats = detector.get_statistics()
        logger.info(f"Detection statistics:")
        logger.info(f"  Active tracks: {stats['active_tracks']}")
        logger.info(f"  Confirmed tracks: {stats['confirmed_tracks']}")
        logger.info(f"  Total frames: {stats['frame_count']}")
        logger.info(f"  Error count: {stats['error_count']}")
    
    logger.info("Object detection example completed")


if __name__ == '__main__':
    main()
