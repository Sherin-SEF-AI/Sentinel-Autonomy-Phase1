"""Main entry point for SENTINEL system."""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List
import threading
import traceback
import psutil
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import ConfigManager
from src.core.logging import LoggerSetup
from src.core.data_structures import (
    CameraBundle, BEVOutput, SegmentationOutput, 
    Detection3D, DriverState, RiskAssessment, Alert
)

# Import all modules
from src.camera import CameraManager
from src.perception.bev import BEVGenerator
from src.perception.segmentation import SemanticSegmentor
from src.perception.detection import ObjectDetector
from src.dms import DriverMonitor
from src.intelligence import ContextualIntelligence
from src.alerts import AlertSystem
from src.recording import ScenarioRecorder
from visualization.backend import VisualizationServer, StreamingManager


class SentinelSystem:
    """Main SENTINEL system orchestrator."""
    
    def __init__(self, config: ConfigManager):
        """Initialize SENTINEL system with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = logging.getLogger('sentinel.system')
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = None
        self.module_latencies = {
            'camera': [],
            'bev': [],
            'segmentation': [],
            'detection': [],
            'dms': [],
            'intelligence': [],
            'alerts': [],
            'recording': [],
            'visualization': []
        }
        
        # System resource monitoring
        self.process = psutil.Process()
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.gpu_memory_history = []
        
        # Performance logger
        self.perf_logger = logging.getLogger('sentinel.performance')
        
        # Start performance monitoring thread
        self.perf_monitor_thread = None
        
        # Module instances
        self.camera_manager: Optional[CameraManager] = None
        self.bev_generator: Optional[BEVGenerator] = None
        self.segmentor: Optional[SemanticSegmentor] = None
        self.detector: Optional[ObjectDetector] = None
        self.dms: Optional[DriverMonitor] = None
        self.intelligence: Optional[ContextualIntelligence] = None
        self.alert_system: Optional[AlertSystem] = None
        self.recorder: Optional[ScenarioRecorder] = None
        self.viz_server: Optional[VisualizationServer] = None
        self.streaming_manager: Optional[StreamingManager] = None
        
        self.logger.info("Initializing SENTINEL system modules...")
        
        # Restore previous state if available
        self._restore_system_state()
        
        self._initialize_modules()
        
    def _initialize_modules(self):
        """Initialize all system modules."""
        try:
            # Initialize camera manager
            self.logger.info("Initializing Camera Manager...")
            self.camera_manager = CameraManager(self.config)
            
            # Initialize BEV generator
            self.logger.info("Initializing BEV Generator...")
            self.bev_generator = BEVGenerator(self.config)
            
            # Initialize semantic segmentor
            self.logger.info("Initializing Semantic Segmentor...")
            self.segmentor = SemanticSegmentor(self.config)
            
            # Initialize object detector
            self.logger.info("Initializing Object Detector...")
            self.detector = ObjectDetector(self.config)
            
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
            
            # Initialize visualization server if enabled
            if self.config.get('visualization.enabled', True):
                self.logger.info("Initializing Visualization Server...")
                port = self.config.get('visualization.port', 8080)
                self.viz_server = VisualizationServer(port=port)
                self.streaming_manager = StreamingManager(self.viz_server)
            
            self.logger.info("All modules initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize modules: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def start(self):
        """Start the SENTINEL system."""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting SENTINEL system...")
        self.running = True
        self.start_time = time.time()
        
        try:
            # Start camera capture
            self.logger.info("Starting camera capture...")
            self.camera_manager.start()
            
            # Start visualization server if enabled
            if self.viz_server:
                self.logger.info("Starting visualization server...")
                self.viz_server.start()
            
            self.logger.info("SENTINEL system started successfully")
            self.logger.info("=" * 60)
            
            # Start performance monitoring thread
            self.perf_monitor_thread = threading.Thread(
                target=self._performance_monitoring_loop, 
                daemon=True
            )
            self.perf_monitor_thread.start()
            
            # Start main processing loop in separate thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.running = False
            raise
    
    def _performance_monitoring_loop(self):
        """Monitor system performance metrics continuously."""
        self.perf_logger.info("Starting performance monitoring...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Monitor CPU usage
                cpu_percent = self.process.cpu_percent(interval=1.0)
                self.cpu_usage_history.append(cpu_percent)
                
                # Monitor memory usage
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.memory_usage_history.append(memory_mb)
                
                # Monitor GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                        self.gpu_memory_history.append(gpu_memory_mb)
                except ImportError:
                    pass
                
                # Keep only last 60 samples (1 minute of data)
                if len(self.cpu_usage_history) > 60:
                    self.cpu_usage_history = self.cpu_usage_history[-60:]
                if len(self.memory_usage_history) > 60:
                    self.memory_usage_history = self.memory_usage_history[-60:]
                if len(self.gpu_memory_history) > 60:
                    self.gpu_memory_history = self.gpu_memory_history[-60:]
                
                # Log performance metrics every 10 seconds
                if len(self.cpu_usage_history) % 10 == 0:
                    self._log_performance_metrics()
                
            except Exception as e:
                self.perf_logger.error(f"Performance monitoring error: {e}")
                time.sleep(1.0)
        
        self.perf_logger.info("Performance monitoring stopped")
    
    def _log_performance_metrics(self):
        """Log current performance metrics."""
        if not self.cpu_usage_history:
            return
        
        # Calculate FPS
        elapsed = time.time() - self.start_time if self.start_time else 0
        current_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Calculate average latencies
        avg_latencies = {}
        for module, latencies in self.module_latencies.items():
            if latencies:
                # Keep only last 100 samples
                recent_latencies = latencies[-100:]
                avg_latencies[module] = np.mean(recent_latencies) * 1000  # Convert to ms
        
        # Calculate resource usage
        avg_cpu = np.mean(self.cpu_usage_history[-10:]) if self.cpu_usage_history else 0
        avg_memory = np.mean(self.memory_usage_history[-10:]) if self.memory_usage_history else 0
        avg_gpu_memory = np.mean(self.gpu_memory_history[-10:]) if self.gpu_memory_history else 0
        
        # Log metrics
        self.perf_logger.info("=" * 60)
        self.perf_logger.info("PERFORMANCE METRICS")
        self.perf_logger.info("=" * 60)
        self.perf_logger.info(f"FPS: {current_fps:.2f}")
        self.perf_logger.info(f"Frames processed: {self.frame_count}")
        self.perf_logger.info(f"CPU usage: {avg_cpu:.1f}%")
        self.perf_logger.info(f"Memory usage: {avg_memory:.1f} MB")
        
        if avg_gpu_memory > 0:
            self.perf_logger.info(f"GPU memory: {avg_gpu_memory:.1f} MB")
        
        if avg_latencies:
            self.perf_logger.info("Module latencies (avg):")
            for module, latency in sorted(avg_latencies.items()):
                self.perf_logger.info(f"  {module}: {latency:.2f}ms")
        
        # Calculate total pipeline latency
        total_latency = sum(avg_latencies.values())
        self.perf_logger.info(f"Total pipeline latency: {total_latency:.2f}ms")
        self.perf_logger.info("=" * 60)
    
    def _processing_loop(self):
        """Main processing loop that runs continuously."""
        self.logger.info("Starting main processing loop...")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                loop_start = time.time()
                
                # Get synchronized camera bundle
                t0 = time.time()
                camera_bundle = self.camera_manager.get_frame_bundle()
                self.module_latencies['camera'].append(time.time() - t0)
                
                if camera_bundle is None:
                    # No frames available, skip this iteration
                    time.sleep(0.001)
                    continue
                
                # Process DMS and perception pipelines in parallel
                dms_result = [None]
                perception_result = [None]
                
                def process_dms():
                    """Process DMS pipeline."""
                    try:
                        t0 = time.time()
                        dms_result[0] = self.dms.analyze(camera_bundle.interior)
                        self.module_latencies['dms'].append(time.time() - t0)
                    except Exception as e:
                        self.logger.error(f"DMS processing error: {e}")
                
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
                
                # Run DMS and perception in parallel
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
                
                # Assess contextual risks
                t0 = time.time()
                risk_assessment = self.intelligence.assess(
                    detections_3d, 
                    driver_state, 
                    seg_output
                )
                self.module_latencies['intelligence'].append(time.time() - t0)
                
                # Generate and dispatch alerts
                t0 = time.time()
                alerts = self.alert_system.process(risk_assessment, driver_state)
                self.module_latencies['alerts'].append(time.time() - t0)
                
                # Record scenarios when triggered
                t0 = time.time()
                if self.recorder.should_record(risk_assessment, driver_state):
                    if not self.recorder.is_recording:
                        self.recorder.start_recording()
                    
                    self.recorder.save_frame(
                        camera_bundle=camera_bundle,
                        bev_output=bev_output,
                        detections=detections_3d,
                        driver_state=driver_state,
                        risk_assessment=risk_assessment,
                        alerts=alerts
                    )
                elif self.recorder.is_recording:
                    # Stop recording if no longer triggered
                    self.recorder.stop_recording()
                    scenario_path = self.recorder.export_scenario()
                    if scenario_path:
                        self.logger.info(f"Scenario exported to: {scenario_path}")
                
                self.module_latencies['recording'].append(time.time() - t0)
                
                # Stream data to visualization dashboard
                if self.streaming_manager:
                    t0 = time.time()
                    try:
                        self.streaming_manager.stream_frame_data(
                            camera_bundle=camera_bundle,
                            bev_output=bev_output,
                            seg_output=seg_output,
                            detections_3d=detections_3d,
                            driver_state=driver_state,
                            risk_assessment=risk_assessment,
                            alerts=alerts
                        )
                    except Exception as e:
                        self.logger.debug(f"Visualization streaming error: {e}")
                    self.module_latencies['visualization'].append(time.time() - t0)
                
                # Update frame count and log performance
                self.frame_count += 1
                loop_time = time.time() - loop_start
                
                if self.frame_count % 30 == 0:  # Log every 30 frames (~1 second)
                    fps = 1.0 / loop_time if loop_time > 0 else 0
                    self.logger.info(
                        f"Frame {self.frame_count}: {loop_time*1000:.1f}ms "
                        f"({fps:.1f} FPS)"
                    )
                
                # Periodic state save
                self._periodic_state_save()
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                self.logger.error(traceback.format_exc())
                time.sleep(0.1)  # Prevent tight error loop
        
        self.logger.info("Processing loop stopped")
    
    def stop(self):
        """Stop the SENTINEL system gracefully."""
        if not self.running:
            return
        
        self.logger.info("Stopping SENTINEL system...")
        self.running = False
        self.shutdown_event.set()
        
        try:
            # Wait for processing thread to finish current frame
            if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
                self.logger.info("Waiting for processing thread to finish...")
                self.processing_thread.join(timeout=2.0)
            
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
            
            # Stop visualization server
            if self.viz_server:
                self.logger.info("Stopping visualization server...")
                self.viz_server.stop()
            
            # Save system state
            self._save_system_state()
            
            # Close all resources
            self._close_resources()
            
            # Log final statistics
            self._log_final_statistics()
            
            self.logger.info("SENTINEL system stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.logger.error(traceback.format_exc())
    
    def _restore_system_state(self):
        """Restore system state from previous session."""
        try:
            import pickle
            from pathlib import Path
            
            state_file = Path("state") / "system_state.pkl"
            
            if not state_file.exists():
                self.logger.info("No previous state found, starting fresh")
                return
            
            recovery_start = time.time()
            
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            # Check if state is recent (within last hour)
            state_age = time.time() - state.get('timestamp', 0)
            if state_age > 3600:  # 1 hour
                self.logger.info(
                    f"Previous state is too old ({state_age/60:.1f} minutes), "
                    "starting fresh"
                )
                return
            
            # Restore state
            prev_frame_count = state.get('frame_count', 0)
            prev_runtime = state.get('total_runtime', 0)
            
            recovery_time = time.time() - recovery_start
            
            self.logger.info("=" * 60)
            self.logger.info("STATE RECOVERY")
            self.logger.info("=" * 60)
            self.logger.info(f"Previous session: {prev_frame_count} frames, "
                           f"{prev_runtime:.1f}s runtime")
            self.logger.info(f"Recovery time: {recovery_time*1000:.1f}ms")
            self.logger.info("=" * 60)
            
            # Verify recovery time requirement (<2 seconds)
            if recovery_time < 2.0:
                self.logger.info("✓ Recovery time requirement met (<2s)")
            else:
                self.logger.warning(f"✗ Recovery time exceeded 2s: {recovery_time:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"Failed to restore system state: {e}")
            self.logger.info("Starting with fresh state")
    
    def _save_system_state(self):
        """Save current system state for recovery."""
        try:
            import pickle
            from pathlib import Path
            
            state_dir = Path("state")
            state_dir.mkdir(exist_ok=True)
            
            state = {
                'frame_count': self.frame_count,
                'total_runtime': time.time() - self.start_time if self.start_time else 0,
                'timestamp': time.time()
            }
            
            state_file = state_dir / "system_state.pkl"
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"System state saved to {state_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save system state: {e}")
    
    def _periodic_state_save(self):
        """Periodically save system state (called from processing loop)."""
        # Save state every 100 frames
        if self.frame_count % 100 == 0:
            try:
                self._save_system_state()
            except Exception as e:
                self.logger.debug(f"Periodic state save failed: {e}")
    
    def _close_resources(self):
        """Close all system resources."""
        try:
            # Clear GPU cache if using PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.info("GPU cache cleared")
            except ImportError:
                pass
            
            # Close any open file handles
            # (Individual modules should handle their own cleanup)
            
            self.logger.info("All resources closed")
            
        except Exception as e:
            self.logger.warning(f"Error closing resources: {e}")
    
    def _log_final_statistics(self):
        """Log final performance statistics."""
        if self.start_time and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            self.logger.info("=" * 60)
            self.logger.info("FINAL STATISTICS")
            self.logger.info("=" * 60)
            self.logger.info(f"Total frames processed: {self.frame_count}")
            self.logger.info(f"Total runtime: {elapsed:.2f} seconds")
            self.logger.info(f"Average FPS: {avg_fps:.2f}")
            
            # Log average and p95 latencies per module
            for module, latencies in self.module_latencies.items():
                if latencies:
                    avg_latency = np.mean(latencies) * 1000
                    p95_latency = np.percentile(latencies, 95) * 1000
                    self.logger.info(
                        f"{module}: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms"
                    )
            
            # Log resource usage statistics
            if self.cpu_usage_history:
                avg_cpu = np.mean(self.cpu_usage_history)
                max_cpu = np.max(self.cpu_usage_history)
                self.logger.info(f"CPU usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%")
            
            if self.memory_usage_history:
                avg_mem = np.mean(self.memory_usage_history)
                max_mem = np.max(self.memory_usage_history)
                self.logger.info(f"Memory: avg={avg_mem:.1f}MB, max={max_mem:.1f}MB")
            
            if self.gpu_memory_history:
                avg_gpu = np.mean(self.gpu_memory_history)
                max_gpu = np.max(self.gpu_memory_history)
                self.logger.info(f"GPU memory: avg={avg_gpu:.1f}MB, max={max_gpu:.1f}MB")
            
            # Check performance requirements
            self.logger.info("=" * 60)
            self.logger.info("REQUIREMENT VALIDATION")
            self.logger.info("=" * 60)
            
            # Check FPS requirement (≥30 FPS)
            fps_ok = avg_fps >= 30.0
            self.logger.info(f"FPS ≥ 30: {'✓ PASS' if fps_ok else '✗ FAIL'} ({avg_fps:.2f})")
            
            # Check latency requirement (<100ms at p95)
            total_latencies = []
            for latencies in self.module_latencies.values():
                total_latencies.extend(latencies)
            
            if total_latencies:
                p95_total = np.percentile(total_latencies, 95) * 1000
                latency_ok = p95_total < 100.0
                self.logger.info(
                    f"Latency < 100ms (p95): {'✓ PASS' if latency_ok else '✗ FAIL'} "
                    f"({p95_total:.2f}ms)"
                )
            
            # Check CPU requirement (≤60% on 8-core)
            if self.cpu_usage_history:
                cpu_ok = avg_cpu <= 60.0
                self.logger.info(
                    f"CPU ≤ 60%: {'✓ PASS' if cpu_ok else '✗ FAIL'} ({avg_cpu:.1f}%)"
                )
            
            # Check GPU memory requirement (≤8GB)
            if self.gpu_memory_history:
                max_gpu_gb = max_gpu / 1024
                gpu_ok = max_gpu_gb <= 8.0
                self.logger.info(
                    f"GPU memory ≤ 8GB: {'✓ PASS' if gpu_ok else '✗ FAIL'} "
                    f"({max_gpu_gb:.2f}GB)"
                )


def main():
    """Main function to start SENTINEL system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SENTINEL Contextual Safety Intelligence Platform')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Override logging level from config'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = ConfigManager(args.config)
        
        if not config.validate():
            print("ERROR: Invalid configuration file")
            return 1
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}")
        return 1
    
    # Set up logging
    log_level = args.log_level or config.get('system.log_level', 'INFO')
    logger = LoggerSetup.setup(log_level=log_level)
    
    logger.info("=" * 60)
    logger.info(f"SENTINEL System v{config.get('system.version', '1.0')}")
    logger.info("=" * 60)
    logger.info(f"Configuration loaded from: {args.config}")
    logger.info(f"Log level: {log_level}")
    
    # Initialize SENTINEL system
    try:
        system = SentinelSystem(config)
    except Exception as e:
        logger.error(f"Failed to initialize SENTINEL system: {e}")
        return 1
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        system.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the system
    try:
        system.start()
        logger.info("System ready. Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while system.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"System error: {e}")
        logger.error(traceback.format_exc())
        return 1
    finally:
        system.stop()
    
    return 0


if __name__ == '__main__':
    exit(main())
