"""
Example: Advanced Trajectory Prediction

Demonstrates the advanced trajectory prediction system with LSTM and physics-based models.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core.data_structures import Detection3D
from src.intelligence.advanced_trajectory import (
    AdvancedTrajectoryPredictor,
    CollisionProbabilityCalculator,
    UncertaintyEstimator,
    Trajectory
)
from src.intelligence.trajectory_visualization import TrajectoryVisualizer
from src.intelligence.trajectory_performance import (
    TrajectoryPerformanceOptimizer,
    PerformanceMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_sample_detections():
    """Create sample detections for testing."""
    detections = []
    
    # Vehicle ahead, moving forward
    detections.append(Detection3D(
        bbox_3d=(10.0, 0.0, 0.0, 4.5, 2.0, 1.5, 0.0),
        class_name='vehicle',
        confidence=0.95,
        velocity=(5.0, 0.0, 0.0),
        track_id=1
    ))
    
    # Pedestrian crossing from left
    detections.append(Detection3D(
        bbox_3d=(15.0, -5.0, 0.0, 0.5, 0.5, 1.7, 0.0),
        class_name='pedestrian',
        confidence=0.90,
        velocity=(0.0, 1.5, 0.0),
        track_id=2
    ))
    
    # Cyclist on right, turning
    detections.append(Detection3D(
        bbox_3d=(8.0, 3.0, 0.0, 1.8, 0.6, 1.2, 0.3),
        class_name='cyclist',
        confidence=0.88,
        velocity=(3.0, -0.5, 0.0),
        track_id=3
    ))
    
    return detections


def create_ego_trajectory():
    """Create sample ego vehicle trajectory."""
    # Ego vehicle moving forward at constant speed
    points = []
    timestamps = []
    uncertainties = []
    
    for i in range(50):
        t = i * 0.1
        x = 5.0 * t  # 5 m/s forward
        y = 0.0
        z = 0.0
        
        points.append((x, y, z))
        timestamps.append(t)
        
        # Increasing uncertainty over time
        cov = np.eye(3) * (0.1 + 0.05 * t) ** 2
        uncertainties.append(cov)
    
    return Trajectory(
        points=points,
        timestamps=timestamps,
        uncertainty=uncertainties,
        confidence=0.9,
        model='cv'
    )


def main():
    """Main example function."""
    logger.info("=== Advanced Trajectory Prediction Example ===")
    
    # Configuration
    config = {
        'enabled': True,
        'horizon': 5.0,
        'dt': 0.1,
        'num_hypotheses': 3,
        'use_lstm': False,  # Disable LSTM for this example (no trained model)
        'uncertainty_estimation': True
    }
    
    # Initialize components
    logger.info("Initializing trajectory predictor...")
    predictor = AdvancedTrajectoryPredictor(config)
    
    logger.info("Initializing collision calculator...")
    collision_calc = CollisionProbabilityCalculator()
    
    logger.info("Initializing uncertainty estimator...")
    uncertainty_est = UncertaintyEstimator()
    
    logger.info("Initializing visualizer...")
    visualizer = TrajectoryVisualizer()
    
    logger.info("Initializing performance optimizer...")
    perf_config = {
        'target_time_per_object': 5.0,
        'enable_profiling': True,
        'enable_caching': True,
        'enable_parallel': True
    }
    perf_optimizer = TrajectoryPerformanceOptimizer(perf_config)
    
    # Create sample data
    logger.info("\nCreating sample detections...")
    detections = create_sample_detections()
    logger.info(f"Created {len(detections)} detections")
    
    logger.info("\nCreating ego vehicle trajectory...")
    ego_trajectory = create_ego_trajectory()
    logger.info(f"Ego trajectory: {len(ego_trajectory.points)} points over {ego_trajectory.timestamps[-1]:.1f}s")
    
    # Predict trajectories
    logger.info("\n=== Predicting Trajectories ===")
    import time
    start_time = time.perf_counter()
    
    object_trajectories = predictor.predict_all(detections)
    
    elapsed_time = (time.perf_counter() - start_time) * 1000.0
    logger.info(f"Prediction completed in {elapsed_time:.2f}ms")
    
    # Display predictions
    for track_id, trajectories in object_trajectories.items():
        detection = next(d for d in detections if d.track_id == track_id)
        logger.info(f"\nObject {track_id} ({detection.class_name}):")
        logger.info(f"  Generated {len(trajectories)} trajectory hypotheses")
        
        for i, traj in enumerate(trajectories):
            logger.info(f"  Hypothesis {i+1}:")
            logger.info(f"    Model: {traj.model}")
            logger.info(f"    Confidence: {traj.confidence:.3f}")
            logger.info(f"    Points: {len(traj.points)}")
            logger.info(f"    Duration: {traj.timestamps[-1]:.1f}s")
            
            # Show first and last points
            first_point = traj.points[0]
            last_point = traj.points[-1]
            logger.info(f"    Start: ({first_point[0]:.2f}, {first_point[1]:.2f}, {first_point[2]:.2f})")
            logger.info(f"    End: ({last_point[0]:.2f}, {last_point[1]:.2f}, {last_point[2]:.2f})")
    
    # Calculate collision probabilities
    logger.info("\n=== Calculating Collision Probabilities ===")
    
    object_sizes = {
        1: 4.5,  # Vehicle
        2: 0.5,  # Pedestrian
        3: 1.8   # Cyclist
    }
    
    collision_probs = collision_calc.calculate_all_collision_probabilities(
        ego_trajectory,
        object_trajectories,
        object_sizes
    )
    
    logger.info(f"Calculated collision probabilities for {len(collision_probs)} objects")
    
    for track_id, (prob, time_step, hyp_idx) in collision_probs.items():
        detection = next(d for d in detections if d.track_id == track_id)
        time_to_collision = object_trajectories[track_id][hyp_idx].timestamps[time_step]
        
        logger.info(f"\nObject {track_id} ({detection.class_name}):")
        logger.info(f"  Collision probability: {prob:.3f}")
        logger.info(f"  Time to closest approach: {time_to_collision:.2f}s")
        logger.info(f"  Best hypothesis: {hyp_idx}")
        
        if prob > 0.5:
            logger.warning(f"  ⚠️  HIGH COLLISION RISK!")
    
    # Prepare for visualization
    logger.info("\n=== Preparing Visualization Data ===")
    
    display_trajectories = visualizer.prepare_trajectories_for_display(
        object_trajectories,
        collision_probs,
        show_all_hypotheses=False
    )
    
    logger.info(f"Prepared {len(display_trajectories)} trajectories for display")
    
    # Show visualization data sample
    if display_trajectories:
        sample = display_trajectories[0]
        logger.info(f"\nSample visualization data:")
        logger.info(f"  Object ID: {sample['object_id']}")
        logger.info(f"  Model: {sample['model']}")
        logger.info(f"  Confidence: {sample['confidence']:.3f}")
        logger.info(f"  Collision probability: {sample['collision_probability']:.3f}")
        logger.info(f"  Number of points: {len(sample['points'])}")
    
    # Create legend
    legend = visualizer.create_trajectory_legend(object_trajectories, collision_probs)
    logger.info(f"\nLegend entries: {len(legend)}")
    for entry in legend:
        logger.info(f"  - {entry['label']}: {entry['description']}")
    
    # Performance metrics
    logger.info("\n=== Performance Metrics ===")
    
    metrics = PerformanceMetrics(
        total_time=elapsed_time,
        lstm_time=0.0,
        physics_time=elapsed_time,
        collision_time=0.0,
        num_objects=len(detections),
        time_per_object=elapsed_time / len(detections) if detections else 0.0
    )
    
    perf_optimizer.record_performance(metrics)
    
    logger.info(f"Total time: {metrics.total_time:.2f}ms")
    logger.info(f"Time per object: {metrics.time_per_object:.2f}ms")
    logger.info(f"Target: {perf_optimizer.target_time_per_object:.2f}ms")
    logger.info(f"Target met: {metrics.time_per_object < perf_optimizer.target_time_per_object}")
    
    # Get performance summary
    summary = perf_optimizer.get_performance_summary()
    if summary:
        logger.info(f"\nPerformance summary:")
        logger.info(f"  Average time per object: {summary['avg_time_per_object']:.2f}ms")
        logger.info(f"  P95 time per object: {summary['p95_time_per_object']:.2f}ms")
    
    # Get recommendations
    recommendations = perf_optimizer.get_optimization_recommendations()
    if recommendations:
        logger.info(f"\nOptimization recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
    
    logger.info("\n=== Example Complete ===")


if __name__ == '__main__':
    main()
