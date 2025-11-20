"""Main scenario recorder for SENTINEL system."""

import logging
from typing import Dict, Any, List, Optional

from ..core.data_structures import (
    CameraBundle, BEVOutput, Detection3D, DriverState, RiskAssessment, Alert
)
from .trigger import RecordingTrigger, TriggerEvent
from .recorder import FrameRecorder
from .exporter import ScenarioExporter
from .playback import ScenarioPlayback


class ScenarioRecorder:
    """
    Main scenario recorder integrating trigger logic, recording, export, and playback.
    
    Provides high-level API for:
    - Automatic trigger-based recording
    - Manual recording control
    - Scenario export
    - Scenario playback
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scenario recorder.
        
        Args:
            config: Recording configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            self.logger.info("ScenarioRecorder disabled by configuration")
            return
        
        # Initialize components
        self.trigger = RecordingTrigger(config)
        self.recorder = FrameRecorder(config)
        self.exporter = ScenarioExporter(config)
        self.playback = ScenarioPlayback(config.get('storage_path', 'scenarios/'))
        
        # Recording state
        self.trigger_events: List[TriggerEvent] = []
        self.last_export_path: Optional[str] = None
        
        self.logger.info("ScenarioRecorder initialized")
    
    def process_frame(
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
        Process a frame and handle recording logic.
        
        This should be called for every frame in the main processing loop.
        
        Args:
            timestamp: Frame timestamp
            camera_bundle: Camera frames
            bev_output: BEV output (optional)
            detections_3d: 3D object detections
            driver_state: Driver state
            risk_assessment: Risk assessment
            alerts: Generated alerts
        """
        if not self.enabled:
            return
        
        # Save frame to buffer/recording
        self.recorder.save_frame(
            timestamp, camera_bundle, bev_output, detections_3d,
            driver_state, risk_assessment, alerts
        )
        
        # Check triggers
        triggers = self.trigger.check_triggers(
            timestamp, risk_assessment, driver_state, alerts
        )
        
        if triggers and not self.recorder.is_recording:
            # Start recording
            self.trigger_events = triggers
            self.start_recording(timestamp)
        
        # Auto-stop recording after max duration (handled in FrameRecorder)
    
    def start_recording(self, timestamp: float) -> None:
        """
        Start recording manually or via trigger.
        
        Args:
            timestamp: Recording start timestamp
        """
        if not self.enabled:
            self.logger.warning("Recording disabled")
            return
        
        self.recorder.start_recording(timestamp)
        self.logger.info(f"Recording started at t={timestamp:.3f}")
    
    def stop_recording(self) -> None:
        """Stop recording."""
        if not self.enabled:
            return
        
        if not self.recorder.is_recording:
            self.logger.warning("No recording in progress")
            return
        
        self.recorder.stop_recording()
        self.logger.info("Recording stopped")
    
    def export_scenario(self, location: str = None) -> Optional[str]:
        """
        Export the current recording as a scenario.
        
        Args:
            location: Optional GPS location
            
        Returns:
            Path to exported scenario directory, or None if failed
        """
        if not self.enabled:
            self.logger.warning("Recording disabled")
            return None
        
        frames = self.recorder.get_recorded_frames()
        
        if not frames:
            self.logger.warning("No frames to export")
            return None
        
        # Determine trigger info
        if self.trigger_events:
            trigger_type = self.trigger_events[0].trigger_type
            trigger_reason = self.trigger_events[0].reason
        else:
            trigger_type = 'manual'
            trigger_reason = 'Manual recording'
        
        # Export scenario
        scenario_path = self.exporter.export_scenario(
            frames, trigger_type, trigger_reason, location
        )
        
        if scenario_path:
            self.last_export_path = scenario_path
            self.logger.info(f"Scenario exported to {scenario_path}")
            
            # Clear recording
            self.recorder.clear_recording()
            self.trigger_events = []
        
        return scenario_path
    
    def is_recording(self) -> bool:
        """
        Check if currently recording.
        
        Returns:
            True if recording in progress
        """
        return self.enabled and self.recorder.is_recording
    
    def get_recording_duration(self) -> float:
        """
        Get current recording duration.
        
        Returns:
            Duration in seconds, or 0 if not recording
        """
        if not self.is_recording():
            return 0.0
        
        frames = self.recorder.get_recorded_frames()
        if not frames:
            return 0.0
        
        return frames[-1].timestamp - frames[0].timestamp
    
    def get_num_recorded_frames(self) -> int:
        """
        Get number of recorded frames.
        
        Returns:
            Number of frames in current recording
        """
        return len(self.recorder.get_recorded_frames())
    
    # Playback API
    
    def list_scenarios(self) -> List[str]:
        """
        List all available scenarios.
        
        Returns:
            List of scenario directory names
        """
        return self.playback.list_scenarios()
    
    def load_scenario(self, scenario_name: str) -> bool:
        """
        Load a scenario for playback.
        
        Args:
            scenario_name: Name of scenario directory
            
        Returns:
            True if loaded successfully
        """
        return self.playback.load_scenario(scenario_name)
    
    def get_playback_frame(self, frame_index: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get a frame from loaded scenario.
        
        Args:
            frame_index: Frame index (None for current frame)
            
        Returns:
            Frame data or None if invalid
        """
        return self.playback.get_frame(frame_index)
    
    def next_playback_frame(self) -> Optional[Dict[str, Any]]:
        """Get next playback frame."""
        return self.playback.next_frame()
    
    def previous_playback_frame(self) -> Optional[Dict[str, Any]]:
        """Get previous playback frame."""
        return self.playback.previous_frame()
    
    def seek_playback(self, frame_index: int) -> bool:
        """
        Seek to specific frame in playback.
        
        Args:
            frame_index: Target frame index
            
        Returns:
            True if successful
        """
        return self.playback.seek(frame_index)
    
    def get_scenario_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata for loaded scenario."""
        return self.playback.get_metadata()
    
    def get_scenario_annotations(self) -> Optional[Dict[str, Any]]:
        """Get annotations for loaded scenario."""
        return self.playback.get_annotations()
    
    def close_playback(self) -> None:
        """Close loaded scenario."""
        self.playback.close()
