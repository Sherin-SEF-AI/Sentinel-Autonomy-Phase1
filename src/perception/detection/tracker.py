"""Object tracking using DeepSORT-inspired algorithm."""

import logging
from typing import List, Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.data_structures import Detection3D


class Track:
    """Represents a tracked object."""
    
    def __init__(self, track_id: int, detection: Detection3D):
        """
        Initialize track.
        
        Args:
            track_id: Unique track identifier
            detection: Initial detection
        """
        self.track_id = track_id
        self.detections = [detection]
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = 'tentative'  # 'tentative', 'confirmed', 'deleted'
        
        # Kalman filter state (simplified)
        x, y, z, w, h, l, theta = detection.bbox_3d
        self.mean = np.array([x, y, z, 0, 0, 0])  # [x, y, z, vx, vy, vz]
        self.covariance = np.eye(6) * 10.0
    
    def predict(self, dt: float = 0.033):
        """
        Predict next state using constant velocity model.
        
        Args:
            dt: Time step in seconds
        """
        # State transition matrix
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise
        Q = np.eye(6) * 0.1
        
        # Predict
        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: Detection3D):
        """
        Update track with new detection.
        
        Args:
            detection: New detection
        """
        x, y, z, w, h, l, theta = detection.bbox_3d
        measurement = np.array([x, y, z])
        
        # Measurement matrix
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        R = np.eye(3) * 1.0
        
        # Innovation
        y_innov = measurement - H @ self.mean
        S = H @ self.covariance @ H.T + R
        
        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Update
        self.mean = self.mean + K @ y_innov
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
        
        self.detections.append(detection)
        self.hits += 1
        self.time_since_update = 0
    
    def get_state(self) -> Detection3D:
        """
        Get current track state as Detection3D.
        
        Returns:
            Detection3D with current track state
        """
        # Get latest detection for dimensions and class
        latest_det = self.detections[-1]
        _, _, _, w, h, l, theta = latest_det.bbox_3d
        
        # Use Kalman filter state for position and velocity
        x, y, z, vx, vy, vz = self.mean
        
        return Detection3D(
            bbox_3d=(float(x), float(y), float(z), 
                    float(w), float(h), float(l), float(theta)),
            class_name=latest_det.class_name,
            confidence=latest_det.confidence,
            velocity=(float(vx), float(vy), float(vz)),
            track_id=self.track_id
        )


class ObjectTracker:
    """Multi-object tracker using DeepSORT-inspired algorithm."""
    
    def __init__(self, config: Dict):
        """
        Initialize object tracker.
        
        Args:
            config: Tracking configuration containing:
                - max_age: Maximum frames to keep track without update (default: 30)
                - min_hits: Minimum hits before track is confirmed (default: 3)
                - iou_threshold: IoU threshold for matching (default: 0.3)
        """
        self.logger = logging.getLogger(__name__)
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.frame_count = 0
        
        self.logger.info(f"ObjectTracker initialized with max_age={self.max_age}, "
                        f"min_hits={self.min_hits}, iou_threshold={self.iou_threshold}")
    
    def update(self, detections: List[Detection3D]) -> List[Detection3D]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection3D objects
        
        Returns:
            List of tracked Detection3D objects with track IDs
        """
        self.frame_count += 1
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_tracks = self._match(detections)
        
        # Update matched tracks
        for det_idx, track_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._initiate_track(detections[det_idx])
        
        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
        
        # Update track states
        for track in self.tracks:
            if track.hits >= self.min_hits:
                track.state = 'confirmed'
            if track.time_since_update > self.max_age:
                track.state = 'deleted'
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if t.state != 'deleted']
        
        # Return confirmed tracks
        tracked_detections = []
        for track in self.tracks:
            if track.state == 'confirmed':
                tracked_detections.append(track.get_state())
        
        return tracked_detections
    
    def _match(self, detections: List[Detection3D]) -> Tuple[List[Tuple[int, int]], 
                                                              List[int], List[int]]:
        """
        Match detections to tracks using Hungarian algorithm.
        
        Args:
            detections: List of detections
        
        Returns:
            Tuple of (matched pairs, unmatched detection indices, unmatched track indices)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Build cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(detections), len(self.tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                track_det = track.get_state()
                iou = self._compute_iou_3d(det, track_det)
                cost_matrix[i, j] = 1 - iou
        
        # Solve assignment problem
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches by IoU threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        for det_idx, track_idx in zip(det_indices, track_indices):
            if cost_matrix[det_idx, track_idx] <= (1 - self.iou_threshold):
                matched.append((det_idx, track_idx))
                unmatched_dets.remove(det_idx)
                unmatched_tracks.remove(track_idx)
        
        return matched, unmatched_dets, unmatched_tracks
    
    def _compute_iou_3d(self, det1: Detection3D, det2: Detection3D) -> float:
        """
        Compute 3D IoU between two detections (simplified BEV IoU).
        
        Args:
            det1: First detection
            det2: Second detection
        
        Returns:
            IoU value between 0 and 1
        """
        # Extract positions and dimensions
        x1, y1, z1, w1, h1, l1, theta1 = det1.bbox_3d
        x2, y2, z2, w2, h2, l2, theta2 = det2.bbox_3d
        
        # Simplified 2D IoU in bird's eye view
        left1 = x1 - l1 / 2
        right1 = x1 + l1 / 2
        top1 = y1 + w1 / 2
        bottom1 = y1 - w1 / 2
        
        left2 = x2 - l2 / 2
        right2 = x2 + l2 / 2
        top2 = y2 + w2 / 2
        bottom2 = y2 - w2 / 2
        
        # Compute intersection
        inter_left = max(left1, left2)
        inter_right = min(right1, right2)
        inter_top = min(top1, top2)
        inter_bottom = max(bottom1, bottom2)
        
        if inter_right < inter_left or inter_top < inter_bottom:
            return 0.0
        
        inter_area = (inter_right - inter_left) * (inter_top - inter_bottom)
        
        # Compute union
        area1 = l1 * w1
        area2 = l2 * w2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        iou = inter_area / union_area
        return float(iou)
    
    def _initiate_track(self, detection: Detection3D):
        """
        Create new track from detection.
        
        Args:
            detection: Detection to track
        """
        track = Track(self.next_track_id, detection)
        self.tracks.append(track)
        self.next_track_id += 1
