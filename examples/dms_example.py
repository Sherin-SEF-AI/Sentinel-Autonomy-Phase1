"""Example demonstrating Driver Monitoring System (DMS) functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import logging
from src.core.config import ConfigManager
from src.dms import DriverMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on frame."""
    if landmarks is None or len(landmarks) == 0:
        return frame
    
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    
    return frame


def draw_gaze_arrow(frame, landmarks, gaze):
    """Draw gaze direction arrow."""
    if landmarks is None or len(landmarks) < 68:
        return frame
    
    # Get eye centers
    left_eye = landmarks[42:48]
    right_eye = landmarks[36:42]
    eye_center = ((left_eye.mean(axis=0) + right_eye.mean(axis=0)) / 2).astype(int)
    
    # Calculate arrow endpoint based on gaze angles
    pitch = gaze.get('pitch', 0.0)
    yaw = gaze.get('yaw', 0.0)
    
    arrow_length = 50
    dx = int(arrow_length * np.sin(np.radians(yaw)))
    dy = int(arrow_length * np.sin(np.radians(pitch)))
    
    end_point = (eye_center[0] + dx, eye_center[1] + dy)
    
    cv2.arrowedLine(frame, tuple(eye_center), end_point, (255, 0, 0), 2, tipLength=0.3)
    
    return frame


def draw_info_panel(frame, driver_state):
    """Draw information panel with driver state."""
    h, w = frame.shape[:2]
    panel_height = 200
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    
    # Text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    y_offset = 20
    line_height = 25
    
    # Face detection status
    face_status = "DETECTED" if driver_state.face_detected else "NOT DETECTED"
    face_color = (0, 255, 0) if driver_state.face_detected else (0, 0, 255)
    cv2.putText(panel, f"Face: {face_status}", (10, y_offset), 
                font, font_scale, face_color, thickness)
    
    # Head pose
    y_offset += line_height
    head_pose = driver_state.head_pose
    cv2.putText(panel, 
                f"Head Pose - Roll: {head_pose['roll']:.1f} Pitch: {head_pose['pitch']:.1f} Yaw: {head_pose['yaw']:.1f}",
                (10, y_offset), font, font_scale, color, thickness)
    
    # Gaze
    y_offset += line_height
    gaze = driver_state.gaze
    cv2.putText(panel,
                f"Gaze - Pitch: {gaze['pitch']:.1f} Yaw: {gaze['yaw']:.1f} Zone: {gaze['attention_zone']}",
                (10, y_offset), font, font_scale, color, thickness)
    
    # Eye state
    y_offset += line_height
    eye_state = driver_state.eye_state
    cv2.putText(panel,
                f"Eyes - L_EAR: {eye_state['left_ear']:.2f} R_EAR: {eye_state['right_ear']:.2f} PERCLOS: {eye_state['perclos']:.2f}",
                (10, y_offset), font, font_scale, color, thickness)
    
    # Drowsiness
    y_offset += line_height
    drowsiness = driver_state.drowsiness
    drowsiness_color = (0, 255, 0) if drowsiness['score'] < 0.3 else (0, 165, 255) if drowsiness['score'] < 0.7 else (0, 0, 255)
    cv2.putText(panel,
                f"Drowsiness: {drowsiness['score']:.2f} Yawn: {drowsiness['yawn_detected']} MicroSleep: {drowsiness['micro_sleep']}",
                (10, y_offset), font, font_scale, drowsiness_color, thickness)
    
    # Distraction
    y_offset += line_height
    distraction = driver_state.distraction
    distraction_color = (0, 255, 0) if distraction['type'] == 'safe_driving' else (0, 165, 255)
    cv2.putText(panel,
                f"Distraction: {distraction['type']} ({distraction['confidence']:.2f}) Duration: {distraction['duration']:.1f}s",
                (10, y_offset), font, font_scale, distraction_color, thickness)
    
    # Readiness score
    y_offset += line_height
    readiness = driver_state.readiness_score
    readiness_color = (0, 255, 0) if readiness > 70 else (0, 165, 255) if readiness > 40 else (0, 0, 255)
    cv2.putText(panel,
                f"READINESS SCORE: {readiness:.1f}/100",
                (10, y_offset), font, 0.7, readiness_color, 2)
    
    # Draw readiness bar
    bar_width = w - 20
    bar_height = 20
    bar_x = 10
    bar_y = y_offset + 15
    
    # Background bar
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Readiness bar
    fill_width = int(bar_width * (readiness / 100))
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), readiness_color, -1)
    
    # Combine frame and panel
    combined = np.vstack([frame, panel])
    
    return combined


def main():
    """Main function to run DMS example."""
    logger.info("Starting DMS example")
    
    # Load configuration
    config_manager = ConfigManager('configs/default.yaml')
    config = config_manager.config
    
    # Initialize DMS
    logger.info("Initializing Driver Monitor...")
    dms = DriverMonitor(config)
    
    # Open camera (use interior camera)
    camera_config = config.get('cameras', {}).get('interior', {})
    camera_id = camera_config.get('device', 0)
    
    logger.info(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_id}")
        logger.info("Using test pattern instead")
        use_test_pattern = True
    else:
        use_test_pattern = False
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.get('resolution', [640, 480])[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.get('resolution', [640, 480])[1])
        cap.set(cv2.CAP_PROP_FPS, camera_config.get('fps', 30))
    
    logger.info("Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            if use_test_pattern:
                # Create test pattern
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
            
            frame_count += 1
            
            # Analyze driver state
            driver_state = dms.analyze(frame)
            
            # Visualize results
            vis_frame = frame.copy()
            
            if driver_state.face_detected:
                # Draw landmarks
                vis_frame = draw_landmarks(vis_frame, driver_state.landmarks)
                
                # Draw gaze arrow
                vis_frame = draw_gaze_arrow(vis_frame, driver_state.landmarks, driver_state.gaze)
            
            # Draw info panel
            output = draw_info_panel(vis_frame, driver_state)
            
            # Display
            cv2.imshow('Driver Monitoring System', output)
            
            # Log every 30 frames
            if frame_count % 30 == 0:
                logger.info(f"Frame {frame_count}: Readiness={driver_state.readiness_score:.1f}, "
                          f"Drowsiness={driver_state.drowsiness['score']:.2f}, "
                          f"Distraction={driver_state.distraction['type']}")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        if not use_test_pattern:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("DMS example completed")


if __name__ == '__main__':
    main()
