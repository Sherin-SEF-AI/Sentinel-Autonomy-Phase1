"""
SENTINEL Worker Thread

QThread subclass that runs the SENTINEL system processing loop in the background,
emitting signals to update the GUI with processed data.
"""

import logging
import time
import traceback
from typing import Optional, Dict, List
import copy
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker

from src.core.config import ConfigManager
from src.core.data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput,
    Detection3D, DriverState, RiskAssessment, Alert
)

# Import system modules
from src.camera import CameraManager
from src.perception.bev import BEVGenerator
from src.perception.segmentation import SemanticSegmentor
from src.perception.detection import ObjectDetector
from src.dms import DriverMonitor
from src.intelligence import ContextualIntelligence
from src.alerts import AlertSystem
from src.recording import ScenarioRecorder


logger = logging.getLogger(__name__)


class SentinelWorker(QThread):
    """
    Worker thread for SENTINEL system processing.
    
    Runs the main processing loop in the background and emits signals
    with processed data to update the GUI without blocking.
    
    Signals:
        frame_ready: Emitted when camera frames are captured
        bev_ready: Emitted when BEV image is generated
        detections_ready: Emitted when objects are detected
        driver_state_ready: Emitted when driver state is analyzed
        risks_ready: Emitted when risk assessment is complete
        alerts_ready: Emitted when alerts are generated
        performance_ready: Emitted with performance metrics
        error_occurred: Emitted when an error occurs
        status_changed: Emitted when system status changes
    """
    
    # Define signals for all data outputs
    frame_ready = pyqtSignal(dict)  # {camera_id: frame}
    bev_ready = pyqtSignal(np.ndarray, np.ndarray)  # (bev_image, mask)
    detections_ready = pyqtSignal(list)  # List[Detection3D]
    driver_state_ready = pyqtSignal(object)  # DriverState
    risks_ready = pyqtSignal(object)  # RiskAssessment
    alerts_ready = pyqtSignal(list)  # List[Alert]
    performance_ready = pyqtSignal(dict)  # Performance metrics
    error_occurred = pyqtSignal(str, str)  # (error_type, error_message)
    status_changed = pyqtSignal(str)  # Status message
    
    def __init__(self, config: ConfigManager, parent=None):
        """
        Initialize the SENTINEL worker thread.
        
        Args:
            config: Configuration manager instance
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self.config = config
        self.logger = logging.getLogger('sentinel.worker')
        
        # Thread control
        self._running = False
        self._stop_requested = False
        self._mutex = QMutex()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.module_latencies = {
            'camera': [],
            'bev': [],
            'segmentation': [],
            'detection': [],
            'dms': [],
            'intelligence': [],
            'alerts': []
        }
        
        # Module instances (initialized in run())
        self.camera_manager: Optional[CameraManager] = None
        self.bev_generator: Optional[BEVGenerator] = None
        self.segmentor: Optional[SemanticSegmentor] = None
        self.detector: Optional[ObjectDetector] = None
        self.dms: Optional[DriverMonitor] = None
        self.intelligence: Optional[ContextualIntelligence] = None
        self.alert_system: Optional[AlertSystem] = None
        self.recorder: Optional[ScenarioRecorder] = None
        
        self.logger.info("SentinelWorker initialized")
    
    def run(self):
        """
        Main thread execution method.
        
        Initializes all modules and runs the processing loop until stopped.
        """
        try:
            self.logger.info("SentinelWorker thread starting...")
            self.status_changed.emit("Initializing modules...")
            
            # Initialize all modules
            if not self._initialize_modules():
                self.error_occurred.emit("Initialization", "Failed to initialize system modules")
                return
            
            self.status_changed.emit("Starting system...")
            
            # Start camera capture
            self.camera_manager.start()
            
            self._running = True
            self.start_time = time.time()
            
            self.status_changed.emit("System running")
            self.logger.info("SentinelWorker processing loop started")
            
            # Main processing loop
            self._processing_loop()
            
        except Exception as e:
            self.logger.error(f"Fatal error in worker thread: {e}")
            self.logger.error(traceback.format_exc())
            self.error_occurred.emit("Fatal", f"Worker thread crashed: {str(e)}")
        
        finally:
            self._cleanup()
            self.logger.info("SentinelWorker thread stopped")
    
    def _initialize_modules(self) -> bool:
        """
        Initialize all SENTINEL system modules.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing system modules...")
            
            # Initialize camera manager
            self.logger.info("Initializing Camera Manager...")
            self.camera_manager = CameraManager(self.config)

            # Get calibrations for BEV generator
            camera_name_to_id = {'interior': 0, 'front_left': 1, 'front_right': 2}
            calibrations = {}
            for camera_name, camera_id in camera_name_to_id.items():
                calib = self.camera_manager.get_calibration(camera_id)
                if calib is not None:
                    # Convert CameraCalibration to dict format expected by BEVGenerator
                    calibrations[camera_name] = {
                        'intrinsics': calib.intrinsics.to_matrix(),
                        'extrinsics': calib.extrinsics.to_transform_matrix(),
                        'homography': calib.homography
                    }

            # Initialize BEV generator
            self.logger.info("Initializing BEV Generator...")
            bev_config = self.config.get('bev', {})
            self.bev_generator = BEVGenerator(bev_config, calibrations)
            
            # Initialize semantic segmentor
            self.logger.info("Initializing Semantic Segmentor...")
            self.segmentor = SemanticSegmentor(self.config)
            
            # Initialize object detector
            self.logger.info("Initializing Object Detector...")
            detection_config = self.config.get('detection', {})
            self.detector = ObjectDetector(detection_config, calibrations)
            
            # Initialize DMS
            self.logger.info("Initializing Driver Monitoring System...")
            self.dms = DriverMonitor(self.config)
            
            # Initialize contextual intelligence
            self.logger.info("Initializing Contextual Intelligence Engine...")
            self.intelligence = ContextualIntelligence(self.config)
            
            # Initialize alert system
            self.logger.info("Initializing Alert System...")
            self.alert_system = AlertSystem(self.config)
            
            # Initialize scenario recorder
            self.logger.info("Initializing Scenario Recorder...")
            self.recorder = ScenarioRecorder(self.config)
            
            self.logger.info("All modules initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize modules: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _processing_loop(self):
        """
        Main processing loop that runs continuously until stopped.
        
        Processes camera frames through all pipeline stages and emits
        signals with the results.
        """
        self.logger.info("Starting main processing loop...")
        
        while self._running and not self._stop_requested:
            try:
                loop_start = time.time()
                
                # Get synchronized camera bundle
                t0 = time.time()
                camera_bundle = self.camera_manager.get_frame_bundle()
                camera_latency = time.time() - t0
                self.module_latencies['camera'].append(camera_latency)
                
                if camera_bundle is None:
                    # No frames available, skip this iteration
                    self.msleep(1)  # Sleep for 1ms
                    continue

                # Extract timestamp from bundle
                timestamp = camera_bundle.timestamp

                # Deep copy frames for thread-safe emission
                frames_dict = self._copy_camera_bundle(camera_bundle)
                self.frame_ready.emit(frames_dict)
                
                # Process DMS and perception pipelines in parallel
                dms_result = [None]
                perception_result = [None]
                
                def process_dms():
                    """Process DMS pipeline."""
                    try:
                        t0 = time.time()
                        driver_state = self.dms.analyze(camera_bundle.interior)
                        self.module_latencies['dms'].append(time.time() - t0)
                        
                        # Deep copy for thread-safe emission
                        dms_result[0] = self._copy_driver_state(driver_state)
                        
                    except Exception as e:
                        self.logger.error(f"DMS processing error: {e}")
                        self.error_occurred.emit("DMS", str(e))
                
                def process_perception():
                    """Process perception pipeline."""
                    try:
                        # Generate BEV from external cameras
                        t0 = time.time()
                        frames = [camera_bundle.front_left, camera_bundle.front_right]
                        bev_output = self.bev_generator.generate(frames)
                        self.module_latencies['bev'].append(time.time() - t0)
                        
                        # Run semantic segmentation on BEV
                        t0 = time.time()
                        seg_output = self.segmentor.segment(bev_output.image)
                        self.module_latencies['segmentation'].append(time.time() - t0)
                        
                        # Detect and track objects
                        t0 = time.time()
                        camera_frames = {
                            0: camera_bundle.front_left,
                            1: camera_bundle.front_right
                        }
                        detections_2d, detections_3d = self.detector.detect(camera_frames)
                        self.module_latencies['detection'].append(time.time() - t0)
                        
                        perception_result[0] = (bev_output, seg_output, detections_3d)
                        
                    except Exception as e:
                        self.logger.error(f"Perception processing error: {e}")
                        self.error_occurred.emit("Perception", str(e))
                
                # Run DMS and perception in parallel using Python threads
                import threading
                dms_thread = threading.Thread(target=process_dms)
                perception_thread = threading.Thread(target=process_perception)
                
                dms_thread.start()
                perception_thread.start()
                
                dms_thread.join()
                perception_thread.join()
                
                # Check if both pipelines succeeded
                driver_state = dms_result[0]
                if perception_result[0] is None or driver_state is None:
                    continue
                
                bev_output, seg_output, detections_3d = perception_result[0]
                
                # Emit BEV data (deep copy for thread safety)
                bev_image_copy = bev_output.image.copy()
                bev_mask_copy = bev_output.mask.copy()
                self.bev_ready.emit(bev_image_copy, bev_mask_copy)
                
                # Emit driver state
                self.driver_state_ready.emit(driver_state)
                
                # Emit detections (deep copy)
                detections_copy = self._copy_detections(detections_3d)
                self.detections_ready.emit(detections_copy)
                
                # Assess contextual risks
                t0 = time.time()
                risk_assessment = self.intelligence.assess(
                    detections_3d,
                    driver_state,
                    seg_output
                )
                self.module_latencies['intelligence'].append(time.time() - t0)
                
                # Emit risk assessment (deep copy)
                risk_copy = self._copy_risk_assessment(risk_assessment)
                self.risks_ready.emit(risk_copy)
                
                # Generate and dispatch alerts
                t0 = time.time()
                alerts = self.alert_system.process(risk_assessment, driver_state)
                self.module_latencies['alerts'].append(time.time() - t0)
                
                # Emit alerts (deep copy)
                alerts_copy = self._copy_alerts(alerts)
                self.alerts_ready.emit(alerts_copy)

                # Process frame for recording (automatic trigger-based recording)
                self.recorder.process_frame(
                    timestamp=timestamp,
                    camera_bundle=camera_bundle,
                    bev_output=bev_output,
                    detections_3d=detections_3d,
                    driver_state=driver_state,
                    risk_assessment=risk_assessment,
                    alerts=alerts
                )

                # Update frame count and emit performance metrics
                self.frame_count += 1
                loop_time = time.time() - loop_start

                if self.frame_count % 10 == 0:  # Emit every 10 frames
                    perf_metrics = self._calculate_performance_metrics(loop_time)
                    self.performance_ready.emit(perf_metrics)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                self.logger.error(traceback.format_exc())
                self.error_occurred.emit("Processing", str(e))
                self.msleep(100)  # Prevent tight error loop
        
        self.logger.info("Processing loop stopped")
    
    def stop(self):
        """
        Request the worker thread to stop.
        
        This method is thread-safe and can be called from the GUI thread.
        """
        with QMutexLocker(self._mutex):
            self.logger.info("Stop requested for SentinelWorker")
            self._stop_requested = True
            self._running = False
    
    def _cleanup(self):
        """
        Clean up resources and stop all modules.
        """
        try:
            self.logger.info("Cleaning up SentinelWorker resources...")
            self.status_changed.emit("Stopping system...")
            
            # Stop recording if active
            if self.recorder and self.recorder.is_recording:
                self.logger.info("Stopping active recording...")
                self.recorder.stop_recording()
                scenario_path = self.recorder.export_scenario()
                if scenario_path:
                    self.logger.info(f"Final scenario exported to: {scenario_path}")
            
            # Stop camera capture
            if self.camera_manager:
                self.logger.info("Stopping camera capture...")
                self.camera_manager.stop()
            
            # Clear GPU cache if using PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("GPU cache cleared")
            except ImportError:
                pass
            
            self.status_changed.emit("System stopped")
            self.logger.info("Cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            self.logger.error(traceback.format_exc())
    
    # Thread-safe data copying methods
    
    def _copy_camera_bundle(self, bundle: CameraBundle) -> Dict:
        """
        Deep copy camera bundle for thread-safe emission.
        
        Args:
            bundle: Camera bundle to copy
            
        Returns:
            Dictionary with copied frames
        """
        return {
            'timestamp': bundle.timestamp,
            'interior': bundle.interior.copy() if bundle.interior is not None else None,
            'front_left': bundle.front_left.copy() if bundle.front_left is not None else None,
            'front_right': bundle.front_right.copy() if bundle.front_right is not None else None
        }
    
    def _copy_driver_state(self, state: DriverState) -> DriverState:
        """
        Deep copy driver state for thread-safe emission.
        
        Args:
            state: Driver state to copy
            
        Returns:
            Copied driver state
        """
        # Use copy.deepcopy for complex nested structures
        return copy.deepcopy(state)
    
    def _copy_detections(self, detections: List[Detection3D]) -> List[Detection3D]:
        """
        Deep copy detections list for thread-safe emission.
        
        Args:
            detections: List of detections to copy
            
        Returns:
            Copied detections list
        """
        return copy.deepcopy(detections)
    
    def _copy_risk_assessment(self, assessment: RiskAssessment) -> RiskAssessment:
        """
        Deep copy risk assessment for thread-safe emission.
        
        Args:
            assessment: Risk assessment to copy
            
        Returns:
            Copied risk assessment
        """
        return copy.deepcopy(assessment)
    
    def _copy_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        Deep copy alerts list for thread-safe emission.
        
        Args:
            alerts: List of alerts to copy
            
        Returns:
            Copied alerts list
        """
        return copy.deepcopy(alerts)
    
    def _calculate_performance_metrics(self, loop_time: float) -> Dict:
        """
        Calculate current performance metrics.
        
        Args:
            loop_time: Time taken for current loop iteration
            
        Returns:
            Dictionary with performance metrics
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        current_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Calculate average latencies (last 100 samples)
        avg_latencies = {}
        for module, latencies in self.module_latencies.items():
            if latencies:
                recent = latencies[-100:]
                avg_latencies[module] = np.mean(recent) * 1000  # Convert to ms
                
                # Keep only last 100 samples to prevent memory growth
                if len(latencies) > 100:
                    self.module_latencies[module] = recent
        
        return {
            'fps': current_fps,
            'frame_count': self.frame_count,
            'loop_time_ms': loop_time * 1000,
            'module_latencies': avg_latencies,
            'total_latency_ms': sum(avg_latencies.values())
        }
