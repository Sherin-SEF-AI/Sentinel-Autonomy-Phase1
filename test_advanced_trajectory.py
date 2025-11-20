"""
Standalone test for advanced trajectory prediction.
Tests the core functionality without requiring full SENTINEL dependencies.
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# Mock Detection3D for testing
@dataclass
class Detection3D:
    bbox_3d: Tuple[float, float, float, float, float, float, float]
    class_name: str
    confidence: float
    velocity: Tuple[float, float, float]
    track_id: int


def test_physics_models():
    """Test physics-based trajectory prediction."""
    print("Testing physics-based models...")
    
    from src.intelligence.advanced_trajectory import PhysicsBasedPredictor
    
    predictor = PhysicsBasedPredictor(dt=0.1)
    
    # Test constant velocity
    position = np.array([10.0, 0.0, 0.0])
    velocity = np.array([5.0, 0.0, 0.0])
    
    positions, uncertainties = predictor.predict_constant_velocity(
        position, velocity, num_steps=10
    )
    
    assert len(positions) == 10
    assert len(uncertainties) == 10
    assert positions[0][0] > position[0]  # Moving forward
    
    print(f"  ✓ Constant velocity: {len(positions)} steps")
    print(f"    Start: {position}")
    print(f"    End: {positions[-1]}")
    
    # Test constant acceleration
    acceleration = np.array([1.0, 0.0, 0.0])
    positions, uncertainties = predictor.predict_constant_acceleration(
        position, velocity, acceleration, num_steps=10
    )
    
    assert len(positions) == 10
    print(f"  ✓ Constant acceleration: {len(positions)} steps")
    
    # Test constant turn rate
    yaw = 0.0
    yaw_rate = 0.1  # radians/second
    positions, uncertainties = predictor.predict_constant_turn_rate(
        position, velocity, yaw, yaw_rate, num_steps=10
    )
    
    assert len(positions) == 10
    print(f"  ✓ Constant turn rate: {len(positions)} steps")
    
    print("Physics models test passed!\n")


def test_trajectory_predictor():
    """Test advanced trajectory predictor."""
    print("Testing advanced trajectory predictor...")
    
    from src.intelligence.advanced_trajectory import AdvancedTrajectoryPredictor
    
    config = {
        'enabled': True,
        'horizon': 3.0,
        'dt': 0.1,
        'num_hypotheses': 3,
        'use_lstm': False,  # Disable LSTM
        'uncertainty_estimation': True
    }
    
    predictor = AdvancedTrajectoryPredictor(config)
    
    # Create test detections
    detections = [
        Detection3D(
            bbox_3d=(10.0, 0.0, 0.0, 4.5, 2.0, 1.5, 0.0),
            class_name='vehicle',
            confidence=0.95,
            velocity=(5.0, 0.0, 0.0),
            track_id=1
        ),
        Detection3D(
            bbox_3d=(15.0, -5.0, 0.0, 0.5, 0.5, 1.7, 0.0),
            class_name='pedestrian',
            confidence=0.90,
            velocity=(0.0, 1.5, 0.0),
            track_id=2
        )
    ]
    
    # Predict trajectories
    trajectories = predictor.predict_all(detections)
    
    assert len(trajectories) > 0
    print(f"  ✓ Predicted trajectories for {len(trajectories)} objects")
    
    for track_id, trajs in trajectories.items():
        print(f"    Object {track_id}: {len(trajs)} hypotheses")
        for i, traj in enumerate(trajs):
            print(f"      Hypothesis {i+1}: model={traj.model}, confidence={traj.confidence:.3f}, points={len(traj.points)}")
    
    print("Trajectory predictor test passed!\n")


def test_collision_probability():
    """Test collision probability calculator."""
    print("Testing collision probability calculator...")
    
    from src.intelligence.advanced_trajectory import (
        CollisionProbabilityCalculator,
        Trajectory
    )
    
    calculator = CollisionProbabilityCalculator()
    
    # Create test trajectories
    ego_points = [(i * 0.5, 0.0, 0.0) for i in range(30)]
    ego_timestamps = [i * 0.1 for i in range(30)]
    ego_uncertainties = [np.eye(3) * 0.1 for _ in range(30)]
    
    ego_trajectory = Trajectory(
        points=ego_points,
        timestamps=ego_timestamps,
        uncertainty=ego_uncertainties,
        confidence=0.9,
        model='cv'
    )
    
    # Object trajectory crossing path
    obj_points = [(10.0, i * 0.3 - 5.0, 0.0) for i in range(30)]
    obj_timestamps = [i * 0.1 for i in range(30)]
    obj_uncertainties = [np.eye(3) * 0.2 for _ in range(30)]
    
    obj_trajectory = Trajectory(
        points=obj_points,
        timestamps=obj_timestamps,
        uncertainty=obj_uncertainties,
        confidence=0.8,
        model='cv'
    )
    
    # Calculate collision probability
    prob, time_step = calculator.calculate_trajectory_collision_probability(
        ego_trajectory,
        obj_trajectory,
        object_size=2.0
    )
    
    print(f"  ✓ Collision probability: {prob:.3f}")
    print(f"    Time of closest approach: {obj_timestamps[time_step]:.2f}s")
    
    assert 0.0 <= prob <= 1.0
    assert 0 <= time_step < len(obj_points)
    
    print("Collision probability test passed!\n")


def test_uncertainty_estimation():
    """Test uncertainty estimator."""
    print("Testing uncertainty estimator...")
    
    from src.intelligence.advanced_trajectory import (
        UncertaintyEstimator,
        Trajectory
    )
    
    estimator = UncertaintyEstimator()
    
    # Create test trajectory
    points = [(i * 1.0, 0.0, 0.0) for i in range(20)]
    timestamps = [i * 0.1 for i in range(20)]
    uncertainties = [np.eye(3) * 0.1 for _ in range(20)]
    
    trajectory = Trajectory(
        points=points,
        timestamps=timestamps,
        uncertainty=uncertainties,
        confidence=0.8,
        model='cv'
    )
    
    # Test covariance calculation
    detection = Detection3D(
        bbox_3d=(0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0),
        class_name='vehicle',
        confidence=0.9,
        velocity=(5.0, 0.0, 0.0),
        track_id=1
    )
    
    covariances = estimator.calculate_covariance(trajectory, detection, 'cv')
    
    assert len(covariances) == len(points)
    print(f"  ✓ Calculated {len(covariances)} covariance matrices")
    
    # Test confidence estimation
    confidence = estimator.estimate_confidence(trajectory, history_length=15, scene_complexity=0.3)
    
    assert 0.0 <= confidence <= 1.0
    print(f"  ✓ Estimated confidence: {confidence:.3f}")
    
    # Test uncertainty propagation
    initial_cov = np.eye(3) * 0.1
    velocity = np.array([5.0, 0.0, 0.0])
    
    propagated = estimator.propagate_uncertainty(initial_cov, velocity, num_steps=10, dt=0.1)
    
    assert len(propagated) == 10
    print(f"  ✓ Propagated uncertainty for {len(propagated)} steps")
    
    print("Uncertainty estimation test passed!\n")


def test_visualization():
    """Test trajectory visualizer."""
    print("Testing trajectory visualizer...")
    
    from src.intelligence.trajectory_visualization import TrajectoryVisualizer
    from src.intelligence.advanced_trajectory import Trajectory
    
    visualizer = TrajectoryVisualizer()
    
    # Create test trajectories
    trajectories = {
        1: [
            Trajectory(
                points=[(i * 1.0, 0.0, 0.0) for i in range(20)],
                timestamps=[i * 0.1 for i in range(20)],
                uncertainty=[np.eye(3) * 0.1 for _ in range(20)],
                confidence=0.9,
                model='cv'
            )
        ],
        2: [
            Trajectory(
                points=[(10.0, i * 0.5, 0.0) for i in range(20)],
                timestamps=[i * 0.1 for i in range(20)],
                uncertainty=[np.eye(3) * 0.2 for _ in range(20)],
                confidence=0.7,
                model='lstm'
            )
        ]
    }
    
    collision_probs = {
        1: (0.3, 10, 0),
        2: (0.8, 15, 0)
    }
    
    # Prepare for display
    display_trajectories = visualizer.prepare_trajectories_for_display(
        trajectories,
        collision_probs,
        show_all_hypotheses=False
    )
    
    assert len(display_trajectories) == 2
    print(f"  ✓ Prepared {len(display_trajectories)} trajectories for display")
    
    for traj in display_trajectories:
        print(f"    Object {traj['object_id']}: {len(traj['points'])} points, collision_prob={traj['collision_probability']:.3f}")
    
    # Test filtering
    filtered = visualizer.filter_trajectories_by_distance(display_trajectories, max_distance=15.0)
    print(f"  ✓ Filtered by distance: {len(filtered)} trajectories")
    
    filtered = visualizer.filter_trajectories_by_collision_risk(display_trajectories, min_collision_prob=0.5)
    print(f"  ✓ Filtered by collision risk: {len(filtered)} trajectories")
    
    print("Visualization test passed!\n")


def test_performance_optimizer():
    """Test performance optimizer."""
    print("Testing performance optimizer...")
    
    from src.intelligence.trajectory_performance import (
        TrajectoryPerformanceOptimizer,
        PerformanceMetrics
    )
    
    config = {
        'target_time_per_object': 5.0,
        'enable_profiling': True,
        'enable_caching': True,
        'enable_parallel': True
    }
    
    optimizer = TrajectoryPerformanceOptimizer(config)
    
    # Record some metrics
    for i in range(10):
        metrics = PerformanceMetrics(
            total_time=20.0 + i,
            lstm_time=5.0,
            physics_time=10.0,
            collision_time=5.0,
            num_objects=5,
            time_per_object=4.0 + i * 0.1
        )
        optimizer.record_performance(metrics)
    
    # Get summary
    summary = optimizer.get_performance_summary()
    
    assert 'avg_total_time' in summary
    assert 'p95_time_per_object' in summary
    
    print(f"  ✓ Performance summary:")
    print(f"    Average total time: {summary['avg_total_time']:.2f}ms")
    print(f"    P95 time per object: {summary['p95_time_per_object']:.2f}ms")
    print(f"    Target met: {summary['target_met']}")
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    print(f"  ✓ Generated {len(recommendations)} recommendations")
    
    print("Performance optimizer test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Advanced Trajectory Prediction Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_physics_models()
        test_trajectory_predictor()
        test_collision_probability()
        test_uncertainty_estimation()
        test_visualization()
        test_performance_optimizer()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
