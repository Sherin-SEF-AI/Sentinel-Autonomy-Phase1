"""Multi-view detection fusion using Hungarian algorithm."""

import logging
from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.data_structures import Detection3D


class MultiViewFusion:
    """Fuses 3D detections from multiple camera views."""
    
    def __init__(self, config: Dict):
        """
        Initialize multi-view fusion.
        
        Args:
            config: Fusion configuration containing:
                - iou_threshold_3d: IoU threshold for matching (default: 0.3)
                - confidence_weighting: Whether to weight by confidence (default: True)
        """
        self.logger = logging.getLogger(__name__)
        self.iou_threshold = config.get('iou_threshold_3d', 0.3)
        self.confidence_weighting = config.get('confidence_weighting', True)
        
        self.logger.info(f"MultiViewFusion initialized with IoU threshold {self.iou_threshold}")
    
    def fuse(self, detections: List[Detection3D]) -> List[Detection3D]:
        """
        Fuse detections from multiple views into unified list.
        
        Args:
            detections: List of Detection3D objects from all cameras
        
        Returns:
            Fused list of Detection3D objects
        """
        if len(detections) == 0:
            return []
        
        if len(detections) == 1:
            return detections
        
        # Build IoU matrix
        n = len(detections)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                iou = self._compute_iou_3d(detections[i], detections[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # Find clusters of overlapping detections
        clusters = self._find_clusters(iou_matrix, self.iou_threshold)
        
        # Merge detections in each cluster
        fused_detections = []
        for cluster in clusters:
            cluster_dets = [detections[i] for i in cluster]
            merged = self._merge_detections(cluster_dets)
            fused_detections.append(merged)
        
        return fused_detections
    
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
        
        # Simplified 2D IoU in bird's eye view (ignoring rotation)
        # Compute bounding rectangles
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
    
    def _find_clusters(self, iou_matrix: np.ndarray, threshold: float) -> List[List[int]]:
        """
        Find clusters of overlapping detections.
        
        Args:
            iou_matrix: NxN matrix of IoU values
            threshold: IoU threshold for clustering
        
        Returns:
            List of clusters, where each cluster is a list of detection indices
        """
        n = iou_matrix.shape[0]
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster = [i]
            visited[i] = True
            
            # Find all detections that overlap with this cluster
            queue = [i]
            while queue:
                current = queue.pop(0)
                
                for j in range(n):
                    if not visited[j] and iou_matrix[current, j] >= threshold:
                        cluster.append(j)
                        visited[j] = True
                        queue.append(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _merge_detections(self, detections: List[Detection3D]) -> Detection3D:
        """
        Merge multiple detections into a single detection.
        
        Args:
            detections: List of detections to merge
        
        Returns:
            Merged Detection3D
        """
        if len(detections) == 1:
            return detections[0]
        
        # Weight by confidence if enabled
        if self.confidence_weighting:
            weights = np.array([d.confidence for d in detections])
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(detections)) / len(detections)
        
        # Weighted average of positions and dimensions
        x = sum(w * d.bbox_3d[0] for w, d in zip(weights, detections))
        y = sum(w * d.bbox_3d[1] for w, d in zip(weights, detections))
        z = sum(w * d.bbox_3d[2] for w, d in zip(weights, detections))
        w_dim = sum(w * d.bbox_3d[3] for w, d in zip(weights, detections))
        h_dim = sum(w * d.bbox_3d[4] for w, d in zip(weights, detections))
        l_dim = sum(w * d.bbox_3d[5] for w, d in zip(weights, detections))
        theta = sum(w * d.bbox_3d[6] for w, d in zip(weights, detections))
        
        # Maximum confidence
        confidence = max(d.confidence for d in detections)
        
        # Most common class
        class_counts = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        class_name = max(class_counts, key=class_counts.get)
        
        # Velocity (will be updated by tracker)
        velocity = detections[0].velocity
        
        # Track ID (will be assigned by tracker)
        track_id = detections[0].track_id
        
        merged = Detection3D(
            bbox_3d=(float(x), float(y), float(z), 
                    float(w_dim), float(h_dim), float(l_dim), float(theta)),
            class_name=class_name,
            confidence=float(confidence),
            velocity=velocity,
            track_id=track_id
        )
        
        return merged
