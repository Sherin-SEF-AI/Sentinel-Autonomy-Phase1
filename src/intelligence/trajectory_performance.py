"""Performance optimization and profiling for trajectory prediction."""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for trajectory prediction."""
    total_time: float  # Total processing time (ms)
    lstm_time: float  # LSTM inference time (ms)
    physics_time: float  # Physics model time (ms)
    collision_time: float  # Collision calculation time (ms)
    num_objects: int  # Number of objects processed
    time_per_object: float  # Average time per object (ms)


class TrajectoryPerformanceOptimizer:
    """Optimize trajectory prediction performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Performance targets
        self.target_time_per_object = config.get('target_time_per_object', 5.0)  # ms
        self.enable_profiling = config.get('enable_profiling', False)
        self.enable_caching = config.get('enable_caching', True)
        self.enable_parallel = config.get('enable_parallel', True)
        
        # Caches
        self.trajectory_cache: Dict[int, Any] = {}
        self.cache_ttl = 0.1  # seconds
        self.cache_timestamps: Dict[int, float] = {}
        
        # Performance history
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_length = 100
        
        # Profiling data
        self.profiling_data: Dict[str, List[float]] = {
            'lstm': [],
            'physics': [],
            'collision': [],
            'total': []
        }
        
        self.logger.info(
            f"Trajectory Performance Optimizer initialized: "
            f"target={self.target_time_per_object}ms/obj, "
            f"caching={self.enable_caching}, parallel={self.enable_parallel}"
        )
    
    def profile_prediction(
        self,
        predictor_func,
        *args,
        **kwargs
    ) -> tuple:
        """
        Profile a prediction function.
        
        Args:
            predictor_func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tuple of (result, elapsed_time_ms)
        """
        start_time = time.perf_counter()
        result = predictor_func(*args, **kwargs)
        elapsed_time = (time.perf_counter() - start_time) * 1000.0  # Convert to ms
        
        return result, elapsed_time
    
    def optimize_lstm_inference(self, model, device: str = 'cuda'):
        """
        Optimize LSTM model for inference.
        
        Args:
            model: LSTM model
            device: Device to use
            
        Returns:
            Optimized model
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            # Move to device
            model = model.to(device)
            
            # Set to eval mode
            model.eval()
            
            # Enable inference optimizations
            if hasattr(torch, 'jit'):
                # Try to use TorchScript for optimization
                try:
                    # Create example input
                    example_input = torch.randn(1, 10, 6).to(device)
                    
                    # Trace model
                    traced_model = torch.jit.trace(model, example_input)
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                    
                    self.logger.info("LSTM model optimized with TorchScript")
                    return traced_model
                except Exception as e:
                    self.logger.warning(f"TorchScript optimization failed: {e}")
            
            # Enable cudnn benchmarking for consistent input sizes
            if device == 'cuda' and torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
                self.logger.info("Enabled cuDNN benchmarking")
            
            return model
        
        except Exception as e:
            self.logger.error(f"LSTM optimization failed: {e}")
            return model
    
    def batch_predictions(
        self,
        predictor,
        detections: List,
        batch_size: int = 8
    ) -> List:
        """
        Batch predictions for better GPU utilization.
        
        Args:
            predictor: Predictor function
            detections: List of detections
            batch_size: Batch size
            
        Returns:
            List of predictions
        """
        predictions = []
        
        # Process in batches
        for i in range(0, len(detections), batch_size):
            batch = detections[i:i + batch_size]
            batch_predictions = predictor(batch)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def cache_trajectory(
        self,
        track_id: int,
        trajectory: Any,
        timestamp: float
    ):
        """
        Cache trajectory prediction.
        
        Args:
            track_id: Object track ID
            trajectory: Trajectory prediction
            timestamp: Current timestamp
        """
        if not self.enable_caching:
            return
        
        self.trajectory_cache[track_id] = trajectory
        self.cache_timestamps[track_id] = timestamp
    
    def get_cached_trajectory(
        self,
        track_id: int,
        current_timestamp: float
    ) -> Optional[Any]:
        """
        Get cached trajectory if still valid.
        
        Args:
            track_id: Object track ID
            current_timestamp: Current timestamp
            
        Returns:
            Cached trajectory or None
        """
        if not self.enable_caching:
            return None
        
        if track_id not in self.trajectory_cache:
            return None
        
        # Check if cache is still valid
        cache_age = current_timestamp - self.cache_timestamps.get(track_id, 0)
        
        if cache_age > self.cache_ttl:
            # Cache expired
            del self.trajectory_cache[track_id]
            del self.cache_timestamps[track_id]
            return None
        
        return self.trajectory_cache[track_id]
    
    def clear_cache(self):
        """Clear trajectory cache."""
        self.trajectory_cache.clear()
        self.cache_timestamps.clear()
    
    def parallel_physics_models(
        self,
        physics_predictor,
        detections: List,
        history: Dict
    ) -> Dict:
        """
        Run physics models in parallel.
        
        Args:
            physics_predictor: Physics predictor
            detections: List of detections
            history: Motion history
            
        Returns:
            Dictionary of predictions
        """
        if not self.enable_parallel:
            # Sequential processing
            predictions = {}
            for detection in detections:
                det_history = history.get(detection.track_id, [])
                pred = physics_predictor.predict_physics(detection, det_history)
                predictions[detection.track_id] = pred
            return predictions
        
        # Parallel processing using threading
        from concurrent.futures import ThreadPoolExecutor
        
        predictions = {}
        
        def predict_single(detection):
            det_history = history.get(detection.track_id, [])
            return detection.track_id, physics_predictor.predict_physics(detection, det_history)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_single, det) for det in detections]
            
            for future in futures:
                track_id, pred = future.result()
                predictions[track_id] = pred
        
        return predictions
    
    def record_performance(self, metrics: PerformanceMetrics):
        """
        Record performance metrics.
        
        Args:
            metrics: Performance metrics
        """
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.max_history_length:
            self.performance_history.pop(0)
        
        # Update profiling data
        if self.enable_profiling:
            self.profiling_data['lstm'].append(metrics.lstm_time)
            self.profiling_data['physics'].append(metrics.physics_time)
            self.profiling_data['collision'].append(metrics.collision_time)
            self.profiling_data['total'].append(metrics.total_time)
            
            # Keep only recent profiling data
            for key in self.profiling_data:
                if len(self.profiling_data[key]) > self.max_history_length:
                    self.profiling_data[key].pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary statistics.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.performance_history:
            return {}
        
        total_times = [m.total_time for m in self.performance_history]
        per_object_times = [m.time_per_object for m in self.performance_history]
        
        summary = {
            'avg_total_time': np.mean(total_times),
            'p50_total_time': np.percentile(total_times, 50),
            'p95_total_time': np.percentile(total_times, 95),
            'p99_total_time': np.percentile(total_times, 99),
            'avg_time_per_object': np.mean(per_object_times),
            'p95_time_per_object': np.percentile(per_object_times, 95),
            'target_met': np.percentile(per_object_times, 95) < self.target_time_per_object,
            'num_samples': len(self.performance_history)
        }
        
        # Add component breakdown if profiling enabled
        if self.enable_profiling and self.profiling_data['lstm']:
            summary['avg_lstm_time'] = np.mean(self.profiling_data['lstm'])
            summary['avg_physics_time'] = np.mean(self.profiling_data['physics'])
            summary['avg_collision_time'] = np.mean(self.profiling_data['collision'])
        
        return summary
    
    def check_performance_target(self) -> bool:
        """
        Check if performance target is being met.
        
        Returns:
            True if target is met
        """
        if not self.performance_history:
            return True
        
        per_object_times = [m.time_per_object for m in self.performance_history[-10:]]
        p95_time = np.percentile(per_object_times, 95)
        
        return p95_time < self.target_time_per_object
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get recommendations for performance optimization.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        summary = self.get_performance_summary()
        
        # Check if target is not met
        if not summary.get('target_met', True):
            recommendations.append(
                f"Performance target not met: {summary['p95_time_per_object']:.2f}ms > "
                f"{self.target_time_per_object}ms per object"
            )
        
        # Check component times
        if self.enable_profiling:
            avg_lstm = summary.get('avg_lstm_time', 0)
            avg_physics = summary.get('avg_physics_time', 0)
            avg_collision = summary.get('avg_collision_time', 0)
            
            total_component = avg_lstm + avg_physics + avg_collision
            
            if avg_lstm > total_component * 0.5:
                recommendations.append(
                    "LSTM inference is bottleneck. Consider: "
                    "1) Model quantization, 2) Reduce sequence length, 3) Use smaller model"
                )
            
            if avg_physics > total_component * 0.4:
                recommendations.append(
                    "Physics models are slow. Consider: "
                    "1) Enable parallel processing, 2) Reduce number of hypotheses"
                )
            
            if avg_collision > total_component * 0.3:
                recommendations.append(
                    "Collision calculation is slow. Consider: "
                    "1) Reduce trajectory resolution, 2) Use spatial indexing"
                )
        
        # Check caching
        if not self.enable_caching:
            recommendations.append("Enable caching to improve performance")
        
        # Check parallelization
        if not self.enable_parallel:
            recommendations.append("Enable parallel processing for physics models")
        
        return recommendations
    
    def log_performance_report(self):
        """Log performance report."""
        summary = self.get_performance_summary()
        
        if not summary:
            return
        
        self.logger.info("=== Trajectory Prediction Performance Report ===")
        self.logger.info(f"Average total time: {summary['avg_total_time']:.2f}ms")
        self.logger.info(f"P95 total time: {summary['p95_total_time']:.2f}ms")
        self.logger.info(f"Average time per object: {summary['avg_time_per_object']:.2f}ms")
        self.logger.info(f"P95 time per object: {summary['p95_time_per_object']:.2f}ms")
        self.logger.info(f"Target ({self.target_time_per_object}ms): {'MET' if summary['target_met'] else 'NOT MET'}")
        
        if self.enable_profiling:
            self.logger.info(f"  LSTM: {summary.get('avg_lstm_time', 0):.2f}ms")
            self.logger.info(f"  Physics: {summary.get('avg_physics_time', 0):.2f}ms")
            self.logger.info(f"  Collision: {summary.get('avg_collision_time', 0):.2f}ms")
        
        # Log recommendations
        recommendations = self.get_optimization_recommendations()
        if recommendations:
            self.logger.info("Optimization recommendations:")
            for i, rec in enumerate(recommendations, 1):
                self.logger.info(f"  {i}. {rec}")
