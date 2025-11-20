"""
Comprehensive validation tests for trajectory prediction system.
Tests LSTM model accuracy, physics models, uncertainty estimation, and performance.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from dataclasses import dataclass

from src.intelligence.advanced_trajectory import (
    Trajectory,
    LSTMTrajectoryModel,
    ConstantVelocityModel,
    ConstantAccelerationModel,
    ConstantTurnRateModel,
    TrajectoryPredictor,
    calculate_collision_probability,
    TORCH_AVAILABLE
)
from src.core.data_structures import Detection3D


@pytest.fixture
def sample_detections():
    """Create sample detection history for testing."""
    detections = []
    for i in range(30):
        det = Detection3D(
            bbox_3d=(10.0 + i * 0.5, 5.0, 0.0, 2.0, 1.5, 4.5, 0.0),
            class_name='car',
            confidence=0.9,
            velocity=(5.0, 0.0, 0.0),
            track_id=1
        )
        detections.append(det)
    return detections


@pytest.fixture
def trajectory_predictor():
    """Create trajectory predictor instance."""
    config = {
        'trajectory_prediction': {
            'horizon': 5.0,
            'dt': 0.1,
            'num_hypotheses': 3,
            'lstm_model': None,  # Use physics models only
            'uncertainty_estimation': True
        }
    }
    return TrajectoryPredictor(config)


class TestPhysicsModels:
    """Test physics-based trajectory prediction models."""
    
    def test_constant_velocity_model(self):
        """Test constant velocity model predictions."""
        model = ConstantVelocityModel()
        
        # Initial state: position (10, 5, 0), velocity (5, 0, 0)
        state = np.array([10.0, 5.0, 0.0, 5.0, 0.0, 0.0])
        
        # Predict 1 second ahead
        trajectory = model.predict(state, horizon=1.0, dt=0.1)
        
        assert trajectory is not None
        assert len(trajectory.points) == 10  # 1.0 / 0.1
        
        # Check final position (should be 10 + 5*1 = 15)
        final_point = trajectory.points[-1]
        assert abs(final_point[0] - 15.0) < 0.1
        assert abs(final_point[1] - 5.0) < 0.1
    
    def test_constant_acceleration_model(self):
        """Test constant acceleration model predictions."""
        model = ConstantAccelerationModel()
        
        # Initial state with acceleration
        state = np.array([10.0, 5.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        
        trajectory = model.predict(state, horizon=1.0, dt=0.1)
        
        assert trajectory is not None
        assert len(trajectory.points) == 10
        
        # With constant acceleration, position should follow s = s0 + v*t + 0.5*a*t^2
        # x = 10 + 5*1 + 0.5*1*1^2 = 15.5
        final_point = trajectory.points[-1]
        assert abs(final_point[0] - 15.5) < 0.2
    
    def test_constant_turn_rate_model(self):
        """Test constant turn rate model predictions."""
        model = ConstantTurnRateModel()
        
        # Initial state with turn rate
        state = np.array([10.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.1])  # 0.1 rad/s turn rate
        
        trajectory = model.predict(state, horizon=1.0, dt=0.1)
        
        assert trajectory is not None
        assert len(trajectory.points) == 10
        
        # Vehicle should follow circular arc
        final_point = trajectory.points[-1]
        # Position should have changed in both x and y
        assert abs(final_point[0] - 10.0) > 1.0
        assert abs(final_point[1] - 5.0) > 0.1


class TestUncertaintyEstimation:
    """Test uncertainty estimation in trajectory predictions."""
    
    def test_uncertainty_propagation(self):
        """Test that uncertainty increases over time."""
        model = ConstantVelocityModel()
        state = np.array([10.0, 5.0, 0.0, 5.0, 0.0, 0.0])
        
        trajectory = model.predict(state, horizon=3.0, dt=0.1)
        
        # Uncertainty should increase with time
        uncertainties = [np.trace(cov) for cov in trajectory.uncertainty]
        
        # Check that uncertainty is monotonically increasing
        for i in range(1, len(uncertainties)):
            assert uncertainties[i] >= uncertainties[i-1], \
                f"Uncertainty decreased at step {i}"
    
    def test_uncertainty_bounds(self):
        """Test that uncertainty values are reasonable."""
        model = ConstantVelocityModel()
        state = np.array([10.0, 5.0, 0.0, 5.0, 0.0, 0.0])
        
        trajectory = model.predict(state, horizon=5.0, dt=0.1)
        
        for cov_matrix in trajectory.uncertainty:
            # Covariance matrix should be positive semi-definite
            eigenvalues = np.linalg.eigvals(cov_matrix)
            assert np.all(eigenvalues >= -1e-10), "Covariance matrix not PSD"
            
            # Uncertainty should be reasonable (not too large)
            assert np.trace(cov_matrix) < 100.0, "Uncertainty too large"
    
    def test_confidence_calculation(self):
        """Test trajectory confidence calculation."""
        model = ConstantVelocityModel()
        state = np.array([10.0, 5.0, 0.0, 5.0, 0.0, 0.0])
        
        trajectory = model.predict(state, horizon=3.0, dt=0.1)
        
        # Confidence should be between 0 and 1
        assert 0.0 <= trajectory.confidence <= 1.0
        
        # Confidence should decrease with longer horizons
        trajectory_long = model.predict(state, horizon=10.0, dt=0.1)
        assert trajectory_long.confidence <= trajectory.confidence


class TestCollisionProbability:
    """Test collision probability calculation."""
    
    def test_collision_probability_no_overlap(self):
        """Test collision probability when trajectories don't overlap."""
        # Two trajectories far apart
        traj1_points = [(i * 1.0, 0.0, 0.0) for i in range(10)]
        traj2_points = [(i * 1.0, 10.0, 0.0) for i in range(10)]
        
        traj1 = Trajectory(
            points=traj1_points,
            timestamps=[i * 0.1 for i in range(10)],
            uncertainty=[np.eye(3) * 0.1 for _ in range(10)],
            confidence=0.9,
            model='cv'
        )
        
        traj2 = Trajectory(
            points=traj2_points,
            timestamps=[i * 0.1 for i in range(10)],
            uncertainty=[np.eye(3) * 0.1 for _ in range(10)],
            confidence=0.9,
            model='cv'
        )
        
        prob = calculate_collision_probability(traj1, traj2)
        
        # Should be very low probability
        assert prob < 0.01
    
    def test_collision_probability_overlap(self):
        """Test collision probability when trajectories overlap."""
        # Two trajectories that intersect
        traj1_points = [(i * 1.0, 0.0, 0.0) for i in range(10)]
        traj2_points = [(5.0, i * 1.0 - 5.0, 0.0) for i in range(10)]
        
        traj1 = Trajectory(
            points=traj1_points,
            timestamps=[i * 0.1 for i in range(10)],
            uncertainty=[np.eye(3) * 0.5 for _ in range(10)],
            confidence=0.9,
            model='cv'
        )
        
        traj2 = Trajectory(
            points=traj2_points,
            timestamps=[i * 0.1 for i in range(10)],
            uncertainty=[np.eye(3) * 0.5 for _ in range(10)],
            confidence=0.9,
            model='cv'
        )
        
        prob = calculate_collision_probability(traj1, traj2)
        
        # Should have significant probability
        assert prob > 0.1
    
    def test_collision_probability_bounds(self):
        """Test that collision probability is always between 0 and 1."""
        # Random trajectories
        for _ in range(10):
            traj1_points = [(np.random.rand() * 20, np.random.rand() * 20, 0.0) 
                           for _ in range(10)]
            traj2_points = [(np.random.rand() * 20, np.random.rand() * 20, 0.0) 
                           for _ in range(10)]
            
            traj1 = Trajectory(
                points=traj1_points,
                timestamps=[i * 0.1 for i in range(10)],
                uncertainty=[np.eye(3) * 0.5 for _ in range(10)],
                confidence=0.9,
                model='cv'
            )
            
            traj2 = Trajectory(
                points=traj2_points,
                timestamps=[i * 0.1 for i in range(10)],
                uncertainty=[np.eye(3) * 0.5 for _ in range(10)],
                confidence=0.9,
                model='cv'
            )
            
            prob = calculate_collision_probability(traj1, traj2)
            assert 0.0 <= prob <= 1.0


class TestTrajectoryPredictor:
    """Test TrajectoryPredictor integration."""
    
    def test_predictor_initialization(self, trajectory_predictor):
        """Test predictor initializes correctly."""
        assert trajectory_predictor is not None
        assert trajectory_predictor.cv_model is not None
        assert trajectory_predictor.ca_model is not None
        assert trajectory_predictor.ct_model is not None
    
    def test_predict_single_object(self, trajectory_predictor, sample_detections):
        """Test prediction for single object."""
        trajectories = trajectory_predictor.predict(sample_detections[-1], sample_detections)
        
        assert trajectories is not None
        assert len(trajectories) > 0
        assert len(trajectories) <= 3  # Max 3 hypotheses
        
        # Check trajectory properties
        for traj in trajectories:
            assert len(traj.points) > 0
            assert len(traj.timestamps) == len(traj.points)
            assert len(traj.uncertainty) == len(traj.points)
            assert 0.0 <= traj.confidence <= 1.0
    
    def test_predict_multiple_hypotheses(self, trajectory_predictor, sample_detections):
        """Test that multiple hypotheses are generated."""
        trajectories = trajectory_predictor.predict(sample_detections[-1], sample_detections)
        
        # Should generate multiple hypotheses
        assert len(trajectories) >= 2
        
        # Hypotheses should be different
        for i in range(len(trajectories) - 1):
            points1 = np.array(trajectories[i].points)
            points2 = np.array(trajectories[i+1].points)
            
            # At least some points should differ
            assert not np.allclose(points1, points2)
    
    def test_predict_with_insufficient_history(self, trajectory_predictor):
        """Test prediction with insufficient history."""
        # Only 2 detections
        detections = [
            Detection3D(
                bbox_3d=(10.0, 5.0, 0.0, 2.0, 1.5, 4.5, 0.0),
                class_name='car',
                confidence=0.9,
                velocity=(5.0, 0.0, 0.0),
                track_id=1
            )
            for _ in range(2)
        ]
        
        trajectories = trajectory_predictor.predict(detections[-1], detections)
        
        # Should still return predictions (using velocity)
        assert trajectories is not None
        assert len(trajectories) > 0


class TestPerformance:
    """Test trajectory prediction performance."""
    
    def test_prediction_latency(self, trajectory_predictor, sample_detections):
        """Test that prediction completes within 5ms."""
        # Warm up
        trajectory_predictor.predict(sample_detections[-1], sample_detections)
        
        # Measure performance
        iterations = 100
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            trajectory_predictor.predict(sample_detections[-1], sample_detections)
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should complete within 5ms
        assert avg_time_ms < 5.0, f"Average prediction time {avg_time_ms:.2f}ms exceeds 5ms target"
    
    def test_collision_probability_performance(self):
        """Test collision probability calculation performance."""
        # Create test trajectories
        traj1_points = [(i * 1.0, 0.0, 0.0) for i in range(50)]
        traj2_points = [(i * 1.0, 5.0, 0.0) for i in range(50)]
        
        traj1 = Trajectory(
            points=traj1_points,
            timestamps=[i * 0.1 for i in range(50)],
            uncertainty=[np.eye(3) * 0.1 for _ in range(50)],
            confidence=0.9,
            model='cv'
        )
        
        traj2 = Trajectory(
            points=traj2_points,
            timestamps=[i * 0.1 for i in range(50)],
            uncertainty=[np.eye(3) * 0.1 for _ in range(50)],
            confidence=0.9,
            model='cv'
        )
        
        # Warm up
        calculate_collision_probability(traj1, traj2)
        
        # Measure performance
        iterations = 1000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            calculate_collision_probability(traj1, traj2)
        
        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000
        
        # Should be very fast
        assert avg_time_ms < 1.0, f"Average collision calc time {avg_time_ms:.2f}ms exceeds 1ms"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLSTMModel:
    """Test LSTM model accuracy (requires PyTorch)."""
    
    def test_lstm_model_prediction(self):
        """Test LSTM model can make predictions."""
        model = LSTMTrajectoryModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            output_size=3
        )
        
        # Create sample input (batch_size=1, seq_len=10, features=6)
        history = np.random.randn(1, 10, 6).astype(np.float32)
        
        # Predict
        prediction = model.predict(history, horizon=5, dt=0.1)
        
        assert prediction is not None
        assert prediction.shape == (1, 5, 3)  # 5 future steps, 3 coordinates
    
    def test_lstm_model_training(self):
        """Test LSTM model can be trained."""
        from src.intelligence.advanced_trajectory import train_lstm_model
        
        # Create synthetic training data
        train_data = []
        for _ in range(100):
            history = np.random.randn(10, 6).astype(np.float32)
            future = np.random.randn(5, 3).astype(np.float32)
            train_data.append((history, future))
        
        # Train model
        model = train_lstm_model(train_data, epochs=2, batch_size=16)
        
        assert model is not None
        
        # Test prediction
        test_history = np.random.randn(1, 10, 6).astype(np.float32)
        prediction = model.predict(test_history, horizon=5, dt=0.1)
        
        assert prediction is not None


class TestAccuracy:
    """Test trajectory prediction accuracy."""
    
    def test_straight_line_accuracy(self):
        """Test accuracy for straight-line motion."""
        model = ConstantVelocityModel()
        
        # Object moving in straight line
        state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
        
        trajectory = model.predict(state, horizon=5.0, dt=0.5)
        
        # Check predictions are accurate
        for i, point in enumerate(trajectory.points):
            expected_x = (i + 1) * 0.5 * 10.0  # v * t
            assert abs(point[0] - expected_x) < 0.1, \
                f"Position error at step {i}: {abs(point[0] - expected_x)}"
    
    def test_accelerating_motion_accuracy(self):
        """Test accuracy for accelerating motion."""
        model = ConstantAccelerationModel()
        
        # Object with constant acceleration
        state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 2.0, 0.0, 0.0])
        
        trajectory = model.predict(state, horizon=5.0, dt=0.5)
        
        # Check predictions follow kinematic equations
        for i, point in enumerate(trajectory.points):
            t = (i + 1) * 0.5
            expected_x = 10.0 * t + 0.5 * 2.0 * t * t
            assert abs(point[0] - expected_x) < 0.2, \
                f"Position error at step {i}: {abs(point[0] - expected_x)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
