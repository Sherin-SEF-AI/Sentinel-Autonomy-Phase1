"""
Data serialization utilities for streaming SENTINEL data over WebSocket.

Converts dataclasses and numpy arrays to JSON-serializable formats.
"""

import numpy as np
import base64
import cv2
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert, Hazard, Risk
)


def encode_image(image: np.ndarray, format: str = '.jpg', quality: int = 85) -> str:
    """
    Encode numpy image to base64 string.
    
    Args:
        image: Image array (H, W, C)
        format: Image format ('.jpg' or '.png')
        quality: JPEG quality (1-100)
    
    Returns:
        Base64 encoded image string
    """
    if format == '.jpg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    else:
        encode_param = []
    
    success, buffer = cv2.imencode(format, image, encode_param)
    if not success:
        raise ValueError("Failed to encode image")
    
    return base64.b64encode(buffer).decode('utf-8')


def serialize_bev_output(bev: BEVOutput) -> Dict[str, Any]:
    """Serialize BEV output for streaming."""
    return {
        'timestamp': bev.timestamp,
        'image': encode_image(bev.image),
        'mask': bev.mask.tolist() if bev.mask is not None else None
    }


def serialize_segmentation_output(seg: SegmentationOutput) -> Dict[str, Any]:
    """Serialize segmentation output for streaming."""
    # Create colored overlay for visualization
    # Map class indices to colors
    color_map = {
        0: [0, 0, 0],        # background
        1: [128, 64, 128],   # road
        2: [255, 255, 255],  # lane_marking
        3: [0, 0, 142],      # vehicle
        4: [220, 20, 60],    # pedestrian
        5: [119, 11, 32],    # cyclist
        6: [190, 153, 153],  # obstacle
        7: [250, 170, 160],  # parking_space
        8: [128, 128, 128],  # curb
        9: [107, 142, 35],   # vegetation
    }
    
    # Create RGB overlay
    h, w = seg.class_map.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        mask = seg.class_map == class_id
        overlay[mask] = color
    
    return {
        'timestamp': seg.timestamp,
        'class_map': seg.class_map.tolist(),
        'confidence': seg.confidence.tolist() if seg.confidence is not None else None,
        'overlay': encode_image(overlay)
    }


def serialize_detection_3d(detection: Detection3D) -> Dict[str, Any]:
    """Serialize 3D detection for streaming."""
    return {
        'bbox_3d': {
            'x': float(detection.bbox_3d[0]),
            'y': float(detection.bbox_3d[1]),
            'z': float(detection.bbox_3d[2]),
            'w': float(detection.bbox_3d[3]),
            'h': float(detection.bbox_3d[4]),
            'l': float(detection.bbox_3d[5]),
            'theta': float(detection.bbox_3d[6])
        },
        'class_name': detection.class_name,
        'confidence': float(detection.confidence),
        'velocity': {
            'vx': float(detection.velocity[0]),
            'vy': float(detection.velocity[1]),
            'vz': float(detection.velocity[2])
        },
        'track_id': int(detection.track_id)
    }


def serialize_driver_state(driver: DriverState) -> Dict[str, Any]:
    """Serialize driver state for streaming."""
    return {
        'face_detected': driver.face_detected,
        'landmarks': driver.landmarks.tolist() if driver.landmarks is not None else None,
        'head_pose': driver.head_pose,
        'gaze': driver.gaze,
        'eye_state': driver.eye_state,
        'drowsiness': driver.drowsiness,
        'distraction': driver.distraction,
        'readiness_score': float(driver.readiness_score)
    }


def serialize_hazard(hazard: Hazard) -> Dict[str, Any]:
    """Serialize hazard for streaming."""
    return {
        'object_id': int(hazard.object_id),
        'type': hazard.type,
        'position': {
            'x': float(hazard.position[0]),
            'y': float(hazard.position[1]),
            'z': float(hazard.position[2])
        },
        'velocity': {
            'vx': float(hazard.velocity[0]),
            'vy': float(hazard.velocity[1]),
            'vz': float(hazard.velocity[2])
        },
        'trajectory': [
            {'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])}
            for p in hazard.trajectory
        ],
        'ttc': float(hazard.ttc) if hazard.ttc is not None else None,
        'zone': hazard.zone,
        'base_risk': float(hazard.base_risk)
    }


def serialize_risk(risk: Risk) -> Dict[str, Any]:
    """Serialize risk assessment for streaming."""
    return {
        'hazard': serialize_hazard(risk.hazard),
        'contextual_score': float(risk.contextual_score),
        'driver_aware': risk.driver_aware,
        'urgency': risk.urgency,
        'intervention_needed': risk.intervention_needed
    }


def serialize_risk_assessment(assessment: RiskAssessment) -> Dict[str, Any]:
    """Serialize complete risk assessment for streaming."""
    return {
        'scene_graph': assessment.scene_graph,
        'hazards': [serialize_hazard(h) for h in assessment.hazards],
        'attention_map': assessment.attention_map,
        'top_risks': [serialize_risk(r) for r in assessment.top_risks]
    }


def serialize_alert(alert: Alert) -> Dict[str, Any]:
    """Serialize alert for streaming."""
    return {
        'timestamp': alert.timestamp,
        'urgency': alert.urgency,
        'modalities': alert.modalities,
        'message': alert.message,
        'hazard_id': int(alert.hazard_id),
        'dismissed': alert.dismissed
    }


def serialize_frame_data(
    timestamp: float,
    bev: Optional[BEVOutput] = None,
    segmentation: Optional[SegmentationOutput] = None,
    detections: Optional[List[Detection3D]] = None,
    driver_state: Optional[DriverState] = None,
    risk_assessment: Optional[RiskAssessment] = None,
    alerts: Optional[List[Alert]] = None,
    performance: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Serialize complete frame data for streaming.
    
    Args:
        timestamp: Frame timestamp
        bev: BEV output
        segmentation: Segmentation output
        detections: List of 3D detections
        driver_state: Driver state
        risk_assessment: Risk assessment
        alerts: List of alerts
        performance: Performance metrics
    
    Returns:
        JSON-serializable dictionary
    """
    data = {
        'type': 'frame_data',
        'timestamp': timestamp,
        'server_time': datetime.now().isoformat()
    }
    
    if bev is not None:
        data['bev'] = serialize_bev_output(bev)
    
    if segmentation is not None:
        data['segmentation'] = serialize_segmentation_output(segmentation)
    
    if detections is not None:
        data['detections'] = [serialize_detection_3d(d) for d in detections]
    
    if driver_state is not None:
        data['driver_state'] = serialize_driver_state(driver_state)
    
    if risk_assessment is not None:
        data['risk_assessment'] = serialize_risk_assessment(risk_assessment)
    
    if alerts is not None:
        data['alerts'] = [serialize_alert(a) for a in alerts]
    
    if performance is not None:
        data['performance'] = performance
    
    return data
