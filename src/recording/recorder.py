"""Frame recording for SENTINEL system."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, asdict

from ..core.data_structures import (
    CameraBundle, BEVOutput, Detection3D, DriverState, RiskAssessment, Alert
)


@dataclass
class RecordedFrame:
    """Single recorded frame with all system outputs."""
    timestamp: float
    camera_frames: Dict[str, np.ndarray]  # interior, front_left, front_right
    bev_output: Optional[np.ndarray]  # BEV image
    detections_3d: List[Dict[str, Any]]  # Serializable detection data
    driver_state: Dict[str, Any]  # Serializable driver state
    risk_assessment: Dict[str, Any]  # Serializable risk data
    alerts: List[Dict[str, Any]]  # Serializable alert data


class FrameRecorder:
    """
    Records frames and system outputs during triggered scenarios.
    
    Maintains a circular buffer for pre-trigger context and records
    all data during active recording sessions.
    """
    
    def __init__(self, config: Dict[str, Any], buffer_size: int = 90):
        """
        Initialize frame recorder.
        
        Args:
            config: Recording configuration
            buffer_size: Number of frames to keep in pre-trigger buffer (default: 90 = 3s at 30fps)
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Circular buffer for pre-trigger context
        self.buffer_size = buffer_size
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Active recording storage
        self.is_recording = False
        self.recorded_frames: List[RecordedFrame] = []
        self.recording_start_time: Optional[float] = None
        self.max_duration = config.get('max_duration', 30.0)
        
        self.logger.info(
            f"FrameRecorder initialized - "
            f"buffer_size={buffer_size}, max_duration={self.max_duration}s"
        )
    
    def save_frame(
        self,
        timestamp: float,
        camera_bundle: CameraBundle,
        bev_output: Optional[BEVOutput],
        detections_3d: List[Detection3D],
        driver_state: DriverState,
        risk_assessment: RiskAssessment,
        alerts: List[Alert]
    ) -> None:
        """
        Save a frame to the buffer or recording.
        
        Args:
            timestamp: Frame timestamp
            camera_bundle: Camera frames
            bev_output: BEV output (optional)
            detections_3d: 3D object detections
            driver_state: Driver state
            risk_assessment: Risk assessment
            alerts: Generated alerts
        """
        # Create recorded frame
        frame = self._create_recorded_frame(
            timestamp, camera_bundle, bev_output, detections_3d,
            driver_state, risk_assessment, alerts
        )
        
        if self.is_recording:
            # Add to active recording
            self.recorded_frames.append(frame)
            
            # Check duration limit
            duration = timestamp - self.recording_start_time
            if duration >= self.max_duration:
                self.logger.warning(
                    f"Recording reached max duration ({self.max_duration}s), "
                    f"stopping automatically"
                )
                self.stop_recording()
        else:
            # Add to circular buffer for pre-trigger context
            self.frame_buffer.append(frame)
    
    def start_recording(self, timestamp: float) -> None:
        """
        Start recording session.
        
        Args:
            timestamp: Recording start timestamp
        """
        if self.is_recording:
            self.logger.warning("Recording already in progress")
            return
        
        self.is_recording = True
        self.recording_start_time = timestamp
        self.recorded_frames = []
        
        # Copy pre-trigger buffer to recording
        self.recorded_frames.extend(list(self.frame_buffer))
        
        self.logger.info(
            f"Recording started at t={timestamp:.3f}, "
            f"included {len(self.frame_buffer)} pre-trigger frames"
        )
    
    def stop_recording(self) -> None:
        """Stop recording session."""
        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return
        
        self.is_recording = False
        duration = self.recorded_frames[-1].timestamp - self.recorded_frames[0].timestamp
        
        self.logger.info(
            f"Recording stopped - "
            f"captured {len(self.recorded_frames)} frames, "
            f"duration={duration:.2f}s"
        )
    
    def get_recorded_frames(self) -> List[RecordedFrame]:
        """
        Get all recorded frames.
        
        Returns:
            List of recorded frames
        """
        return self.recorded_frames.copy()
    
    def clear_recording(self) -> None:
        """Clear recorded frames."""
        self.recorded_frames = []
        self.recording_start_time = None
        self.logger.debug("Recording cleared")
    
    def _create_recorded_frame(
        self,
        timestamp: float,
        camera_bundle: CameraBundle,
        bev_output: Optional[BEVOutput],
        detections_3d: List[Detection3D],
        driver_state: DriverState,
        risk_assessment: RiskAssessment,
        alerts: List[Alert]
    ) -> RecordedFrame:
        """Create a recorded frame from system outputs."""
        # Store camera frames
        camera_frames = {
            'interior': camera_bundle.interior.copy(),
            'front_left': camera_bundle.front_left.copy(),
            'front_right': camera_bundle.front_right.copy()
        }
        
        # Store BEV output
        bev_image = bev_output.image.copy() if bev_output else None
        
        # Serialize detections
        detections_data = [
            {
                'bbox_3d': det.bbox_3d,
                'class_name': det.class_name,
                'confidence': det.confidence,
                'velocity': det.velocity,
                'track_id': det.track_id
            }
            for det in detections_3d
        ]
        
        # Serialize driver state (handle numpy arrays)
        driver_data = {
            'face_detected': driver_state.face_detected,
            'landmarks': driver_state.landmarks.tolist() if driver_state.face_detected else [],
            'head_pose': driver_state.head_pose,
            'gaze': driver_state.gaze,
            'eye_state': driver_state.eye_state,
            'drowsiness': driver_state.drowsiness,
            'distraction': driver_state.distraction,
            'readiness_score': driver_state.readiness_score
        }
        
        # Serialize risk assessment
        risk_data = {
            'scene_graph': risk_assessment.scene_graph,
            'hazards': [
                {
                    'object_id': h.object_id,
                    'type': h.type,
                    'position': h.position,
                    'velocity': h.velocity,
                    'trajectory': h.trajectory,
                    'ttc': h.ttc,
                    'zone': h.zone,
                    'base_risk': h.base_risk
                }
                for h in risk_assessment.hazards
            ],
            'attention_map': risk_assessment.attention_map,
            'top_risks': [
                {
                    'hazard_id': r.hazard.object_id,
                    'contextual_score': r.contextual_score,
                    'driver_aware': r.driver_aware,
                    'urgency': r.urgency,
                    'intervention_needed': r.intervention_needed
                }
                for r in risk_assessment.top_risks
            ]
        }
        
        # Serialize alerts
        alerts_data = [
            {
                'timestamp': alert.timestamp,
                'urgency': alert.urgency,
                'modalities': alert.modalities,
                'message': alert.message,
                'hazard_id': alert.hazard_id,
                'dismissed': alert.dismissed
            }
            for alert in alerts
        ]
        
        return RecordedFrame(
            timestamp=timestamp,
            camera_frames=camera_frames,
            bev_output=bev_image,
            detections_3d=detections_data,
            driver_state=driver_data,
            risk_assessment=risk_data,
            alerts=alerts_data
        )
