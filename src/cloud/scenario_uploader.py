"""
Scenario uploader for cloud synchronization.

Uploads high-risk recorded scenarios to cloud backend with compression
and user consent management.
"""

import os
import logging
import threading
import subprocess
from typing import Dict, Any, Optional, List
from queue import Queue
from pathlib import Path
import json
import shutil

from .api_client import CloudAPIClient


class ScenarioUploader:
    """
    Uploads recorded scenarios to cloud backend.
    
    Features:
    - User consent checking
    - Video compression
    - Background upload
    - Upload status tracking
    """
    
    def __init__(
        self,
        api_client: CloudAPIClient,
        scenarios_dir: str = "scenarios",
        upload_consent: bool = False,
        compression_quality: int = 23  # CRF value for ffmpeg (lower = better quality)
    ):
        """
        Initialize scenario uploader.
        
        Args:
            api_client: Cloud API client
            scenarios_dir: Directory containing recorded scenarios
            upload_consent: Whether user has consented to scenario upload
            compression_quality: Video compression quality (18-28 recommended)
        """
        self.api_client = api_client
        self.scenarios_dir = scenarios_dir
        self.upload_consent = upload_consent
        self.compression_quality = compression_quality
        
        self.logger = logging.getLogger(__name__)
        
        # Upload queue
        self.upload_queue: Queue = Queue()
        
        # Upload status tracking
        self.upload_status: Dict[str, str] = {}  # scenario_id -> status
        
        # Background upload thread
        self.running = False
        self.upload_thread: Optional[threading.Thread] = None
        
        self.logger.info(f"ScenarioUploader initialized (consent: {upload_consent})")
    
    def set_consent(self, consent: bool) -> None:
        """
        Set user consent for scenario upload.
        
        Args:
            consent: True to enable uploads, False to disable
        """
        self.upload_consent = consent
        self.logger.info(f"Scenario upload consent set to {consent}")
    
    def queue_scenario(self, scenario_path: str, priority: str = 'normal') -> bool:
        """
        Queue a scenario for upload.
        
        Args:
            scenario_path: Path to scenario directory
            priority: Upload priority ('high', 'normal', 'low')
        
        Returns:
            True if queued successfully
        """
        if not self.upload_consent:
            self.logger.info(f"Scenario upload disabled by user consent")
            return False
        
        if not os.path.exists(scenario_path):
            self.logger.error(f"Scenario path does not exist: {scenario_path}")
            return False
        
        # Extract scenario ID from path
        scenario_id = os.path.basename(scenario_path)
        
        # Check if already queued or uploaded
        if scenario_id in self.upload_status:
            status = self.upload_status[scenario_id]
            if status in ['queued', 'uploading', 'completed']:
                self.logger.debug(f"Scenario {scenario_id} already {status}")
                return False
        
        # Queue for upload
        self.upload_queue.put({
            'scenario_id': scenario_id,
            'path': scenario_path,
            'priority': priority
        })
        
        self.upload_status[scenario_id] = 'queued'
        self.logger.info(f"Queued scenario {scenario_id} for upload (priority: {priority})")
        
        return True
    
    def start_background_upload(self) -> None:
        """Start background upload thread"""
        if self.running:
            return
        
        self.running = True
        self.upload_thread = threading.Thread(target=self._upload_loop, daemon=True)
        self.upload_thread.start()
        self.logger.info("Background scenario upload thread started")
    
    def stop_background_upload(self) -> None:
        """Stop background upload thread"""
        if not self.running:
            return
        
        self.running = False
        if self.upload_thread:
            self.upload_thread.join(timeout=10.0)
        self.logger.info("Background scenario upload thread stopped")
    
    def _upload_loop(self) -> None:
        """Background upload loop"""
        while self.running:
            try:
                if not self.upload_queue.empty():
                    # Check connectivity
                    if self.api_client.check_connectivity():
                        self._process_next_upload()
                    else:
                        self.logger.warning("No connectivity, waiting...")
                        threading.Event().wait(10.0)
                else:
                    # Wait for new scenarios
                    threading.Event().wait(5.0)
            
            except Exception as e:
                self.logger.error(f"Error in upload loop: {e}", exc_info=True)
                threading.Event().wait(5.0)
    
    def _process_next_upload(self) -> None:
        """Process next scenario in upload queue"""
        try:
            item = self.upload_queue.get_nowait()
            scenario_id = item['scenario_id']
            scenario_path = item['path']
            
            self.upload_status[scenario_id] = 'uploading'
            self.logger.info(f"Uploading scenario {scenario_id}")
            
            # Compress and upload
            success = self._compress_and_upload(scenario_id, scenario_path)
            
            if success:
                self.upload_status[scenario_id] = 'completed'
                self.logger.info(f"Successfully uploaded scenario {scenario_id}")
            else:
                self.upload_status[scenario_id] = 'failed'
                self.logger.error(f"Failed to upload scenario {scenario_id}")
        
        except Exception as e:
            self.logger.error(f"Error processing upload: {e}", exc_info=True)
    
    def _compress_and_upload(self, scenario_id: str, scenario_path: str) -> bool:
        """
        Compress scenario videos and upload to cloud.
        
        Args:
            scenario_id: Scenario identifier
            scenario_path: Path to scenario directory
        
        Returns:
            True if upload successful
        """
        try:
            # Create temporary directory for compressed files
            temp_dir = os.path.join(scenario_path, 'compressed')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Load metadata
            metadata_path = os.path.join(scenario_path, 'metadata.json')
            if not os.path.exists(metadata_path):
                self.logger.error(f"Metadata not found: {metadata_path}")
                return False
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Compress video files
            compressed_files = {}
            video_files = ['interior.mp4', 'front_left.mp4', 'front_right.mp4', 'bev.mp4']
            
            for video_file in video_files:
                video_path = os.path.join(scenario_path, video_file)
                if os.path.exists(video_path):
                    compressed_path = os.path.join(temp_dir, video_file)
                    
                    if self._compress_video(video_path, compressed_path):
                        compressed_files[video_file] = compressed_path
                    else:
                        self.logger.warning(f"Failed to compress {video_file}, using original")
                        compressed_files[video_file] = video_path
            
            # Upload metadata
            response = self.api_client.post(
                '/scenarios',
                data={
                    'scenario_id': scenario_id,
                    'metadata': metadata
                }
            )
            
            if not response.success:
                self.logger.error(f"Failed to upload metadata: {response.error}")
                return False
            
            # Upload compressed videos
            for video_name, video_path in compressed_files.items():
                with open(video_path, 'rb') as f:
                    files = {video_name: f}
                    response = self.api_client.post(
                        f'/scenarios/{scenario_id}/videos',
                        files=files
                    )
                
                if not response.success:
                    self.logger.error(f"Failed to upload {video_name}: {response.error}")
                    return False
            
            # Upload annotations
            annotations_path = os.path.join(scenario_path, 'annotations.json')
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
                
                response = self.api_client.post(
                    f'/scenarios/{scenario_id}/annotations',
                    data=annotations
                )
                
                if not response.success:
                    self.logger.warning(f"Failed to upload annotations: {response.error}")
            
            # Cleanup compressed files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error compressing and uploading: {e}", exc_info=True)
            return False
    
    def _compress_video(self, input_path: str, output_path: str) -> bool:
        """
        Compress video using ffmpeg.
        
        Args:
            input_path: Input video file
            output_path: Output compressed video file
        
        Returns:
            True if compression successful
        """
        try:
            # Check if ffmpeg is available
            if not shutil.which('ffmpeg'):
                self.logger.warning("ffmpeg not found, skipping compression")
                return False
            
            # Compress with H.264 codec
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-crf', str(self.compression_quality),
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-y',  # Overwrite output
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Check file size reduction
                original_size = os.path.getsize(input_path)
                compressed_size = os.path.getsize(output_path)
                reduction = (1 - compressed_size / original_size) * 100
                
                self.logger.debug(f"Compressed {os.path.basename(input_path)}: "
                                f"{original_size / 1024 / 1024:.1f}MB -> "
                                f"{compressed_size / 1024 / 1024:.1f}MB "
                                f"({reduction:.1f}% reduction)")
                return True
            else:
                self.logger.error(f"ffmpeg failed: {result.stderr.decode()}")
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error("Video compression timeout")
            return False
        
        except Exception as e:
            self.logger.error(f"Error compressing video: {e}", exc_info=True)
            return False
    
    def get_upload_status(self, scenario_id: str) -> Optional[str]:
        """
        Get upload status for a scenario.
        
        Args:
            scenario_id: Scenario identifier
        
        Returns:
            Status string or None if not found
        """
        return self.upload_status.get(scenario_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get upload statistics.
        
        Returns:
            Dictionary with statistics
        """
        status_counts = {}
        for status in self.upload_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'consent_enabled': self.upload_consent,
            'queued_scenarios': self.upload_queue.qsize(),
            'status_counts': status_counts
        }
