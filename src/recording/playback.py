"""Scenario playback for SENTINEL system."""

import os
import json
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class ScenarioPlayback:
    """
    Loads and plays back recorded scenarios.
    
    Supports:
    - Loading scenarios from disk
    - Frame-by-frame navigation
    - Access to all recorded data
    """
    
    def __init__(self, storage_path: str = 'scenarios/'):
        """
        Initialize scenario playback.
        
        Args:
            storage_path: Path to scenarios directory
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = storage_path
        
        # Current scenario state
        self.scenario_dir: Optional[str] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.annotations: Optional[Dict[str, Any]] = None
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
        self.current_frame_index: int = 0
        self.num_frames: int = 0
        
        self.logger.info(f"ScenarioPlayback initialized - storage_path={storage_path}")
    
    def list_scenarios(self) -> List[str]:
        """
        List all available scenarios.
        
        Returns:
            List of scenario directory names
        """
        if not os.path.exists(self.storage_path):
            return []
        
        scenarios = [
            d for d in os.listdir(self.storage_path)
            if os.path.isdir(os.path.join(self.storage_path, d))
        ]
        scenarios.sort(reverse=True)  # Most recent first
        
        return scenarios
    
    def load_scenario(self, scenario_name: str) -> bool:
        """
        Load a scenario from disk.
        
        Args:
            scenario_name: Name of scenario directory
            
        Returns:
            True if loaded successfully
        """
        scenario_dir = os.path.join(self.storage_path, scenario_name)
        
        if not os.path.exists(scenario_dir):
            self.logger.error(f"Scenario not found: {scenario_dir}")
            return False
        
        # Close any previously loaded scenario
        self.close()
        
        self.scenario_dir = scenario_dir
        
        # Load metadata
        metadata_path = os.path.join(scenario_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            self.logger.error(f"Metadata not found: {metadata_path}")
            return False
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Load annotations
        annotations_path = os.path.join(scenario_dir, 'annotations.json')
        if not os.path.exists(annotations_path):
            self.logger.error(f"Annotations not found: {annotations_path}")
            return False
        
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        
        # Open video captures
        for camera_name, video_file in self.metadata['files'].items():
            video_path = os.path.join(scenario_dir, video_file)
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    self.video_captures[camera_name] = cap
                else:
                    self.logger.warning(f"Failed to open video: {video_path}")
        
        self.num_frames = self.metadata.get('num_frames', len(self.annotations['frames']))
        self.current_frame_index = 0
        
        self.logger.info(
            f"Loaded scenario: {scenario_name}, "
            f"frames={self.num_frames}, duration={self.metadata['duration']:.2f}s"
        )
        
        return True
    
    def get_frame(self, frame_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific frame from the loaded scenario.
        
        Args:
            frame_index: Frame index (None for current frame)
            
        Returns:
            Dictionary with camera frames and annotations, or None if invalid
        """
        if not self.scenario_dir:
            self.logger.error("No scenario loaded")
            return None
        
        if frame_index is None:
            frame_index = self.current_frame_index
        
        if frame_index < 0 or frame_index >= self.num_frames:
            self.logger.error(f"Invalid frame index: {frame_index}")
            return None
        
        # Get camera frames
        camera_frames = {}
        for camera_name, cap in self.video_captures.items():
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB
                camera_frames[camera_name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.logger.warning(
                    f"Failed to read frame {frame_index} from {camera_name}"
                )
        
        # Get annotations
        annotations = self.annotations['frames'][frame_index]
        
        return {
            'frame_index': frame_index,
            'camera_frames': camera_frames,
            'annotations': annotations
        }
    
    def next_frame(self) -> Optional[Dict[str, Any]]:
        """
        Get next frame.
        
        Returns:
            Frame data or None if at end
        """
        if self.current_frame_index >= self.num_frames - 1:
            return None
        
        self.current_frame_index += 1
        return self.get_frame()
    
    def previous_frame(self) -> Optional[Dict[str, Any]]:
        """
        Get previous frame.
        
        Returns:
            Frame data or None if at beginning
        """
        if self.current_frame_index <= 0:
            return None
        
        self.current_frame_index -= 1
        return self.get_frame()
    
    def seek(self, frame_index: int) -> bool:
        """
        Seek to specific frame.
        
        Args:
            frame_index: Target frame index
            
        Returns:
            True if successful
        """
        if frame_index < 0 or frame_index >= self.num_frames:
            return False
        
        self.current_frame_index = frame_index
        return True
    
    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get scenario metadata.
        
        Returns:
            Metadata dictionary or None if no scenario loaded
        """
        return self.metadata
    
    def get_annotations(self) -> Optional[Dict[str, Any]]:
        """
        Get all annotations.
        
        Returns:
            Annotations dictionary or None if no scenario loaded
        """
        return self.annotations
    
    def close(self) -> None:
        """Close loaded scenario and release resources."""
        for cap in self.video_captures.values():
            cap.release()
        
        self.video_captures = {}
        self.scenario_dir = None
        self.metadata = None
        self.annotations = None
        self.current_frame_index = 0
        self.num_frames = 0
        
        self.logger.debug("Scenario closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
