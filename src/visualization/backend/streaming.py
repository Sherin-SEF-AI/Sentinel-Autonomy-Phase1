"""
Real-time data streaming manager for SENTINEL visualization.

Handles streaming of:
- BEV images with semantic overlay
- 3D bounding boxes
- Driver state metrics
- Risk scores
- System performance metrics (FPS, latency)

Updates at 30 Hz for real-time visualization.
"""

import asyncio
import time
import logging
from typing import Optional, Dict, Any, List
from collections import deque
import numpy as np

from src.core.data_structures import (
    BEVOutput, SegmentationOutput, Detection3D, DriverState,
    RiskAssessment, Alert
)
from src.visualization.backend.data_serializer import serialize_frame_data

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitors system performance metrics."""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of frames to average over
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.module_latencies = {
            'camera': deque(maxlen=window_size),
            'bev': deque(maxlen=window_size),
            'segmentation': deque(maxlen=window_size),
            'detection': deque(maxlen=window_size),
            'dms': deque(maxlen=window_size),
            'intelligence': deque(maxlen=window_size),
            'alerts': deque(maxlen=window_size),
            'total': deque(maxlen=window_size)
        }
        self.last_frame_time = None
    
    def record_frame(self):
        """Record frame timestamp for FPS calculation."""
        current_time = time.time()
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        self.last_frame_time = current_time
    
    def record_latency(self, module: str, latency_ms: float):
        """
        Record module latency.
        
        Args:
            module: Module name
            latency_ms: Latency in milliseconds
        """
        if module in self.module_latencies:
            self.module_latencies[module].append(latency_ms)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.frame_times) == 0:
            return 0.0
        avg_frame_time = np.mean(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_latencies(self) -> Dict[str, float]:
        """Get average latencies for all modules."""
        latencies = {}
        for module, times in self.module_latencies.items():
            if len(times) > 0:
                latencies[module] = float(np.mean(times))
            else:
                latencies[module] = 0.0
        return latencies
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get complete performance metrics."""
        return {
            'fps': round(self.get_fps(), 2),
            'latency': self.get_latencies(),
            'frame_count': len(self.frame_times)
        }


class StreamingManager:
    """Manages real-time data streaming to visualization clients."""
    
    def __init__(self, server, target_fps: int = 30):
        """
        Initialize streaming manager.
        
        Args:
            server: VisualizationServer instance
            target_fps: Target streaming frame rate
        """
        self.server = server
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        self.performance_monitor = PerformanceMonitor()
        self.is_streaming = False
        self.stream_task = None
        
        # Latest data cache
        self.latest_bev: Optional[BEVOutput] = None
        self.latest_segmentation: Optional[SegmentationOutput] = None
        self.latest_detections: Optional[List[Detection3D]] = None
        self.latest_driver_state: Optional[DriverState] = None
        self.latest_risk_assessment: Optional[RiskAssessment] = None
        self.latest_alerts: Optional[List[Alert]] = None
        
        # Statistics
        self.frames_streamed = 0
        self.start_time = None
        
        logger.info(f"Streaming manager initialized (target: {target_fps} Hz)")
    
    def update_bev(self, bev: BEVOutput):
        """Update BEV data."""
        self.latest_bev = bev
    
    def update_segmentation(self, segmentation: SegmentationOutput):
        """Update segmentation data."""
        self.latest_segmentation = segmentation
    
    def update_detections(self, detections: List[Detection3D]):
        """Update detection data."""
        self.latest_detections = detections
    
    def update_driver_state(self, driver_state: DriverState):
        """Update driver state data."""
        self.latest_driver_state = driver_state
    
    def update_risk_assessment(self, risk_assessment: RiskAssessment):
        """Update risk assessment data."""
        self.latest_risk_assessment = risk_assessment
    
    def update_alerts(self, alerts: List[Alert]):
        """Update alerts data."""
        self.latest_alerts = alerts
    
    def record_module_latency(self, module: str, latency_ms: float):
        """
        Record latency for a processing module.
        
        Args:
            module: Module name (camera, bev, segmentation, etc.)
            latency_ms: Latency in milliseconds
        """
        self.performance_monitor.record_latency(module, latency_ms)
    
    async def stream_frame(self, timestamp: float):
        """
        Stream current frame data to all connected clients.
        
        Args:
            timestamp: Frame timestamp
        """
        # Record frame for FPS calculation
        self.performance_monitor.record_frame()
        
        # Get performance metrics
        performance = self.performance_monitor.get_metrics()
        
        # Add GPU and CPU metrics if available
        try:
            import psutil
            performance['cpu_percent'] = psutil.cpu_percent(interval=None)
            
            # Try to get GPU memory usage
            try:
                import torch
                if torch.cuda.is_available():
                    performance['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            except ImportError:
                pass
        except ImportError:
            pass
        
        # Serialize frame data
        data = serialize_frame_data(
            timestamp=timestamp,
            bev=self.latest_bev,
            segmentation=self.latest_segmentation,
            detections=self.latest_detections,
            driver_state=self.latest_driver_state,
            risk_assessment=self.latest_risk_assessment,
            alerts=self.latest_alerts,
            performance=performance
        )
        
        # Stream to all connected clients
        await self.server.stream_data(data)
        
        self.frames_streamed += 1
        
        # Log statistics periodically
        if self.frames_streamed % (self.target_fps * 10) == 0:  # Every 10 seconds
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(
                f"Streaming stats: {self.frames_streamed} frames, "
                f"{performance['fps']:.1f} FPS, "
                f"{len(self.server.connection_manager.active_connections)} clients, "
                f"uptime: {elapsed:.1f}s"
            )
    
    async def start_streaming(self):
        """Start streaming loop."""
        if self.is_streaming:
            logger.warning("Streaming already active")
            return
        
        self.is_streaming = True
        self.start_time = time.time()
        logger.info(f"Starting streaming at {self.target_fps} Hz")
        
        frame_count = 0
        while self.is_streaming:
            loop_start = time.time()
            
            # Generate timestamp
            timestamp = frame_count / self.target_fps
            
            # Stream frame
            try:
                await self.stream_frame(timestamp)
            except Exception as e:
                logger.error(f"Error streaming frame: {e}")
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.frame_interval - elapsed)
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                logger.warning(
                    f"Streaming falling behind: {elapsed*1000:.1f}ms > "
                    f"{self.frame_interval*1000:.1f}ms target"
                )
            
            frame_count += 1
    
    def stop_streaming(self):
        """Stop streaming loop."""
        if not self.is_streaming:
            logger.warning("Streaming not active")
            return
        
        self.is_streaming = False
        logger.info(f"Stopping streaming (streamed {self.frames_streamed} frames)")
    
    async def run(self):
        """Run streaming manager (convenience method)."""
        await self.start_streaming()


class StreamingIntegration:
    """
    Integration helper for connecting SENTINEL system to streaming manager.
    
    This class provides a simple interface for the main system to push
    data to the visualization backend.
    """
    
    def __init__(self, streaming_manager: StreamingManager):
        """
        Initialize streaming integration.
        
        Args:
            streaming_manager: StreamingManager instance
        """
        self.streaming = streaming_manager
    
    def push_frame_data(
        self,
        timestamp: float,
        bev: Optional[BEVOutput] = None,
        segmentation: Optional[SegmentationOutput] = None,
        detections: Optional[List[Detection3D]] = None,
        driver_state: Optional[DriverState] = None,
        risk_assessment: Optional[RiskAssessment] = None,
        alerts: Optional[List[Alert]] = None,
        latencies: Optional[Dict[str, float]] = None
    ):
        """
        Push complete frame data to streaming manager.
        
        Args:
            timestamp: Frame timestamp
            bev: BEV output
            segmentation: Segmentation output
            detections: List of 3D detections
            driver_state: Driver state
            risk_assessment: Risk assessment
            alerts: List of alerts
            latencies: Module latencies in milliseconds
        """
        # Update data
        if bev is not None:
            self.streaming.update_bev(bev)
        
        if segmentation is not None:
            self.streaming.update_segmentation(segmentation)
        
        if detections is not None:
            self.streaming.update_detections(detections)
        
        if driver_state is not None:
            self.streaming.update_driver_state(driver_state)
        
        if risk_assessment is not None:
            self.streaming.update_risk_assessment(risk_assessment)
        
        if alerts is not None:
            self.streaming.update_alerts(alerts)
        
        # Record latencies
        if latencies is not None:
            for module, latency in latencies.items():
                self.streaming.record_module_latency(module, latency)
    
    async def stream_current_frame(self, timestamp: float):
        """
        Stream current frame data.
        
        Args:
            timestamp: Frame timestamp
        """
        await self.streaming.stream_frame(timestamp)


def create_streaming_manager(server, target_fps: int = 30) -> StreamingManager:
    """
    Factory function to create streaming manager.
    
    Args:
        server: VisualizationServer instance
        target_fps: Target streaming frame rate
    
    Returns:
        StreamingManager instance
    """
    return StreamingManager(server, target_fps)
