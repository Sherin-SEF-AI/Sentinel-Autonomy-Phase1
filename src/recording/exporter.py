"""Scenario export for SENTINEL system."""

import os
import json
import logging
import cv2
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from .recorder import RecordedFrame


class ScenarioExporter:
    """
    Exports recorded scenarios to disk.
    
    Creates directory structure with:
    - MP4 videos for each camera view
    - JSON metadata file
    - JSON annotations file with frame-by-frame data
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scenario exporter.
        
        Args:
            config: Recording configuration with storage path
        """
        self.logger = logging.getLogger(__name__)
        self.storage_path = config.get('storage_path', 'scenarios/')
        
        # Create storage directory if it doesn't exist
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ScenarioExporter initialized - storage_path={self.storage_path}")
    
    def export_scenario(
        self,
        frames: List[RecordedFrame],
        trigger_type: str,
        trigger_reason: str,
        location: str = None
    ) -> str:
        """
        Export recorded scenario to disk.
        
        Args:
            frames: List of recorded frames
            trigger_type: Type of trigger that initiated recording
            trigger_reason: Reason for recording trigger
            location: Optional GPS location
            
        Returns:
            Path to exported scenario directory
        """
        if not frames:
            self.logger.warning("No frames to export")
            return None
        
        # Create scenario directory with timestamp
        timestamp = datetime.fromtimestamp(frames[0].timestamp)
        scenario_name = timestamp.strftime("%Y%m%d_%H%M%S")
        scenario_dir = os.path.join(self.storage_path, scenario_name)
        Path(scenario_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting scenario to {scenario_dir}")
        
        # Export camera videos
        video_files = self._export_videos(frames, scenario_dir)
        
        # Export metadata
        metadata = self._create_metadata(
            frames, trigger_type, trigger_reason, location, video_files
        )
        metadata_path = os.path.join(scenario_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export annotations
        annotations = self._create_annotations(frames)
        annotations_path = os.path.join(scenario_dir, 'annotations.json')
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        self.logger.info(
            f"Scenario exported successfully - "
            f"{len(frames)} frames, duration={metadata['duration']:.2f}s"
        )
        
        return scenario_dir
    
    def _export_videos(
        self,
        frames: List[RecordedFrame],
        scenario_dir: str
    ) -> Dict[str, str]:
        """
        Export camera frames as MP4 videos.
        
        Args:
            frames: List of recorded frames
            scenario_dir: Scenario directory path
            
        Returns:
            Dictionary mapping camera names to video filenames
        """
        video_files = {}
        
        # Define camera configurations
        cameras = {
            'interior': {'resolution': (640, 480), 'fps': 30},
            'front_left': {'resolution': (1280, 720), 'fps': 30},
            'front_right': {'resolution': (1280, 720), 'fps': 30}
        }
        
        # Export BEV if available
        if frames[0].bev_output is not None:
            cameras['bev'] = {'resolution': (640, 640), 'fps': 30}
        
        for camera_name, config in cameras.items():
            video_filename = f"{camera_name}.mp4"
            video_path = os.path.join(scenario_dir, video_filename)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                video_path,
                fourcc,
                config['fps'],
                config['resolution']
            )
            
            # Write frames
            for frame in frames:
                if camera_name == 'bev':
                    if frame.bev_output is not None:
                        # Convert RGB to BGR for OpenCV
                        img = cv2.cvtColor(frame.bev_output, cv2.COLOR_RGB2BGR)
                        writer.write(img)
                else:
                    img = frame.camera_frames.get(camera_name)
                    if img is not None:
                        # Convert RGB to BGR for OpenCV
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        writer.write(img)
            
            writer.release()
            video_files[camera_name] = video_filename
            
            self.logger.debug(f"Exported {camera_name} video: {video_filename}")
        
        return video_files
    
    def _create_metadata(
        self,
        frames: List[RecordedFrame],
        trigger_type: str,
        trigger_reason: str,
        location: str,
        video_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create metadata JSON."""
        start_time = frames[0].timestamp
        end_time = frames[-1].timestamp
        duration = end_time - start_time
        
        metadata = {
            'timestamp': datetime.fromtimestamp(start_time).isoformat(),
            'duration': duration,
            'num_frames': len(frames),
            'trigger': {
                'type': trigger_type,
                'reason': trigger_reason
            },
            'files': video_files
        }
        
        if location:
            metadata['location'] = location
        
        return metadata
    
    def _create_annotations(self, frames: List[RecordedFrame]) -> Dict[str, Any]:
        """Create annotations JSON with frame-by-frame data."""
        annotations = {
            'frames': []
        }
        
        for frame in frames:
            frame_data = {
                'timestamp': frame.timestamp,
                'detections_3d': frame.detections_3d,
                'driver_state': frame.driver_state,
                'risk_assessment': frame.risk_assessment,
                'alerts': frame.alerts
            }
            annotations['frames'].append(frame_data)
        
        return annotations
