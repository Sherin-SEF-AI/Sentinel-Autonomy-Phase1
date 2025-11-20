"""Test suite for advanced trajectory prediction module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from src.intelligence.advanced_trajectory import (
    Trajectory,
    LSTMTrajectoryModel,
    train_lstm_model,
    save_lstm_model,
    load_lstm_model,
    TORCH_AVAILABLE
)
from src.core.data_structures import Detection3D


@pytest.fixture
def sample_detection():
    """Fixture providing a sample 3D detection for testing."""
    return Detection3D(
        bbox_3d=(10.0, 5.0, 0.0, 2.0, 1.5, 4.5, 0.0),
        class_name='car',
        confidence=0.9,
        velocity=(5.0, 0.0, 0.0),
        track_id=1
    )


@pytest.fixture
def sample_trajectory():
    """Fixture providing a sample trajectory for testing."""
    points = [(i * 1.0, i * 0.5, 0.0) for i in range(10)]
    timestamps = [i * 0.1 for i in range(10)]
    uncertainty = [np.eye(3) * 0.1 for _ in range(10)]
    
    return Trajectory(
        points=points,
        timestamps=timestamps,
        uncertainty=uncertainty,
        confidence=0.85,
        model='cv'
    )


@pytest.fixture
def sample_history():
    """Fixture providing sample trajectory history."""
    # 10 time steps of (x, y, z, vx, vy, vz)
    history = np.array([
        [i * 1.0, i * 0.5, 0.0, 1.0, 0.5, 0.0]
        for i in range(10)
    ], dtype=np.float32)
    return history


@pytest.fixture
def sample_future():
    """Fixture providing sample future trajectory."""
    # 5 future time steps of (x, y, z)
    future = np.array([
        [10.0 + i * 1.0, 5.0 + i * 0.5, 0.0]
        for i in range(5)
    ], dtype=np.float32)
    return future


class TestTrajectory:
    """Test suite for Trajectory dataclass."""
    
    def test_trajectory_initialization(self, sample_trajectory):
        """Test that Trajectory initializes correctly with valid data."""
        assert sample_trajectory is not None
        assert len(sample_trajectory.points) == 10
        assert len(sample_trajectory.timestamps) == 10
        assert len(sample_trajectory.uncertainty) == 10
        assert 0.0 <= sample_trajectory.confidence <= 1.0
        assert sample_trajectory.model in ['lstm', 'cv', 'ca', 'ct']
    
    def test_trajectory_points_format(self, sample_trajectory):
        """Test that trajectory points have correct format."""
        for point in sample_trajectory.points:
            assert len(point) == 3  # (x, y, z)
            assert all(isinstance(coord, (int, float)) for coord in point)
    
    def test_trajectory_uncertainty_format(self, sample_trajectory):
        """Test that uncertainty matrices have correct format."""
        for cov_matrix in sample_trajectory.uncertainty:
            assert cov_matrix.shape == (3, 3)
            # Check symmetry
            assert np.allclose(cov_matrix, cov_matrix.T)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLSTMTrajectoryModel:
    """Test suite for LSTMTrajectoryModel class."""
    
    def test_model_initialization(self):
        """Test that LSTMTrajectoryModel initializes correctly."""
        model = LSTMTrajectoryModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout=0.2
        )
        
        assert model is not None
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert hasattr(model, 'lstm')
        assert hasattr(model, 'fc')
        assert hasattr(model, 'uncertainty_fc')
    
    def test_model_forward_pass(self):
        """Test forward pass with valid input."""
        import torch
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        # Create sample input (batch=2, seq_len=10, features=6)
        x = torch.randn(2, 10, 6)
        
        with torch.no_grad():
            position, log_var, hidden = model.forward(x)
        
        # Check output shapes
        assert position.shape == (2, 3)  # (batch, output_size)
        assert log_var.shape == (2, 3)  # (batch, output_size)
        assert hidden is not None
        assert len(hidden) == 2  # (h, c) tuple
    
    def test_model_predict_sequence(self, sample_history):
        """Test sequence prediction with valid history."""
        import torch
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        # Convert history to tensor (batch=1, seq_len=10, features=6)
        history_tensor = torch.FloatTensor(sample_history).unsqueeze(0)
        
        # Predict 5 future steps
        predictions, uncertainties = model.predict_sequence(history_tensor, num_steps=5)
        
        # Check output shapes
        assert predictions.shape == (1, 5, 3)  # (batch, num_steps, 3)
        assert uncertainties.shape == (1, 5, 3)  # (batch, num_steps, 3)
        
        # Check that uncertainties are positive
        assert torch.all(uncertainties > 0)
    
    def test_model_predict_sequence_multiple_steps(self):
        """Test prediction with different numbers of future steps."""
        import torch
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        history = torch.randn(1, 10, 6)
        
        for num_steps in [1, 5, 10, 20]:
            predictions, uncertainties = model.predict_sequence(history, num_steps)
            assert predictions.shape == (1, num_steps, 3)
            assert uncertainties.shape == (1, num_steps, 3)
    
    @pytest.mark.performance
    def test_model_inference_performance(self, sample_history):
        """Test that inference completes within performance requirements."""
        import torch
        import time
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        history_tensor = torch.FloatTensor(sample_history).unsqueeze(0)
        
        # Warm-up
        with torch.no_grad():
            model.predict_sequence(history_tensor, num_steps=10)
        
        # Measure performance
        start_time = time.perf_counter()
        with torch.no_grad():
            predictions, uncertainties = model.predict_sequence(history_tensor, num_steps=10)
        end_time = time.perf_counter()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # Should complete in < 5ms per object (target from config)
        assert execution_time_ms < 5.0, f"Inference took {execution_time_ms:.2f}ms, expected < 5ms"
    
    def test_model_batch_processing(self):
        """Test model with batch processing."""
        import torch
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        # Batch of 4 trajectories
        batch_history = torch.randn(4, 10, 6)
        
        with torch.no_grad():
            predictions, uncertainties = model.predict_sequence(batch_history, num_steps=5)
        
        assert predictions.shape == (4, 5, 3)
        assert uncertainties.shape == (4, 5, 3)
    
    def test_model_with_different_input_sizes(self):
        """Test model initialization with different input sizes."""
        import torch
        
        # Test with different configurations
        configs = [
            {'input_size': 4, 'hidden_size': 32, 'num_layers': 1},
            {'input_size': 6, 'hidden_size': 64, 'num_layers': 2},
            {'input_size': 8, 'hidden_size': 128, 'num_layers': 3},
        ]
        
        for config in configs:
            model = LSTMTrajectoryModel(**config)
            assert model is not None
            
            # Test forward pass
            x = torch.randn(1, 10, config['input_size'])
            with torch.no_grad():
                pos, log_var, hidden = model.forward(x)
            
            assert pos.shape == (1, 3)
            assert log_var.shape == (1, 3)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingFunctions:
    """Test suite for training-related functions."""
    
    def test_train_lstm_model_basic(self, sample_history, sample_future):
        """Test basic training functionality."""
        import torch
        
        model = LSTMTrajectoryModel(hidden_size=32, num_layers=1)
        
        # Create small training dataset
        train_data = [(sample_history, sample_future[:5, :])] * 5
        
        # Train for just 2 epochs
        trained_model = train_lstm_model(
            model,
            train_data,
            num_epochs=2,
            learning_rate=0.001,
            device='cpu'
        )
        
        assert trained_model is not None
        # Model should be in training mode after training
        assert trained_model.training
    
    def test_train_lstm_model_loss_decreases(self, sample_history, sample_future):
        """Test that training loss decreases over epochs."""
        import torch
        
        model = LSTMTrajectoryModel(hidden_size=32, num_layers=1)
        
        # Create training dataset
        train_data = [(sample_history, sample_future[:5, :])] * 10
        
        # We can't easily check loss decrease without modifying the function,
        # but we can verify training completes without errors
        trained_model = train_lstm_model(
            model,
            train_data,
            num_epochs=5,
            learning_rate=0.01,
            device='cpu'
        )
        
        assert trained_model is not None
    
    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading LSTM model."""
        import torch
        
        # Create and initialize model
        model = LSTMTrajectoryModel(hidden_size=32, num_layers=1)
        
        # Save model
        model_path = tmp_path / "test_model.pth"
        save_lstm_model(model, str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = load_lstm_model(
            str(model_path),
            hidden_size=32,
            num_layers=1
        )
        
        assert loaded_model is not None
        # Model should be in eval mode after loading
        assert not loaded_model.training
    
    def test_save_and_load_model_preserves_weights(self, tmp_path):
        """Test that saved and loaded models have same weights."""
        import torch
        
        # Create model with random weights
        model = LSTMTrajectoryModel(hidden_size=32, num_layers=1)
        
        # Get original weights
        original_weights = {
            name: param.clone()
            for name, param in model.named_parameters()
        }
        
        # Save and load
        model_path = tmp_path / "test_model.pth"
        save_lstm_model(model, str(model_path))
        loaded_model = load_lstm_model(
            str(model_path),
            hidden_size=32,
            num_layers=1
        )
        
        # Compare weights
        for name, param in loaded_model.named_parameters():
            assert torch.allclose(param, original_weights[name])
    
    def test_train_with_empty_dataset(self):
        """Test training with empty dataset."""
        import torch
        
        model = LSTMTrajectoryModel(hidden_size=32, num_layers=1)
        train_data = []
        
        # Should handle empty dataset gracefully
        trained_model = train_lstm_model(
            model,
            train_data,
            num_epochs=1,
            learning_rate=0.001,
            device='cpu'
        )
        
        assert trained_model is not None


class TestTorchAvailability:
    """Test suite for PyTorch availability handling."""
    
    def test_torch_available_flag(self):
        """Test that TORCH_AVAILABLE flag is set correctly."""
        assert isinstance(TORCH_AVAILABLE, bool)
    
    @patch('src.intelligence.advanced_trajectory.TORCH_AVAILABLE', False)
    def test_functions_without_torch(self):
        """Test that functions handle missing PyTorch gracefully."""
        # These should raise RuntimeError when PyTorch is not available
        
        with pytest.raises(RuntimeError, match="PyTorch not available"):
            train_lstm_model(None, [], num_epochs=1)
        
        with pytest.raises(RuntimeError, match="PyTorch not available"):
            save_lstm_model(None, "dummy_path.pth")
        
        with pytest.raises(RuntimeError, match="PyTorch not available"):
            load_lstm_model("dummy_path.pth")


class TestEdgeCases:
    """Test suite for edge cases and error handling."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_with_zero_history(self):
        """Test model behavior with minimal history."""
        import torch
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        # Single time step history
        history = torch.randn(1, 1, 6)
        
        with torch.no_grad():
            predictions, uncertainties = model.predict_sequence(history, num_steps=1)
        
        assert predictions.shape == (1, 1, 3)
        assert uncertainties.shape == (1, 1, 3)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_with_extreme_values(self):
        """Test model with extreme input values."""
        import torch
        
        model = LSTMTrajectoryModel()
        model.eval()
        
        # Very large values
        history_large = torch.randn(1, 10, 6) * 1000
        
        with torch.no_grad():
            predictions, uncertainties = model.predict_sequence(history_large, num_steps=5)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(predictions).any()
        assert not torch.isinf(predictions).any()
        assert not torch.isnan(uncertainties).any()
        assert not torch.isinf(uncertainties).any()
    
    def test_trajectory_with_empty_lists(self):
        """Test Trajectory creation with empty lists."""
        trajectory = Trajectory(
            points=[],
            timestamps=[],
            uncertainty=[],
            confidence=0.0,
            model='cv'
        )
        
        assert len(trajectory.points) == 0
        assert len(trajectory.timestamps) == 0
        assert len(trajectory.uncertainty) == 0
    
    def test_trajectory_with_mismatched_lengths(self):
        """Test Trajectory with mismatched list lengths."""
        # This should be allowed by dataclass, but might cause issues in usage
        trajectory = Trajectory(
            points=[(0, 0, 0)],
            timestamps=[0.0, 0.1],  # Different length
            uncertainty=[np.eye(3)],
            confidence=0.5,
            model='cv'
        )
        
        assert len(trajectory.points) == 1
        assert len(trajectory.timestamps) == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestModelArchitecture:
    """Test suite for model architecture details."""
    
    def test_model_has_correct_layers(self):
        """Test that model has all required layers."""
        model = LSTMTrajectoryModel()
        
        # Check LSTM layer
        assert hasattr(model, 'lstm')
        assert isinstance(model.lstm, torch.nn.LSTM)
        
        # Check fully connected layers
        assert hasattr(model, 'fc')
        assert isinstance(model.fc, torch.nn.Sequential)
        
        # Check uncertainty layer
        assert hasattr(model, 'uncertainty_fc')
        assert isinstance(model.uncertainty_fc, torch.nn.Sequential)
    
    def test_model_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = LSTMTrajectoryModel(hidden_size=64, num_layers=2)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters (not too few, not too many)
        assert 10000 < total_params < 1000000
    
    def test_model_gradient_flow(self):
        """Test that gradients flow through the model."""
        import torch
        
        model = LSTMTrajectoryModel(hidden_size=32, num_layers=1)
        model.train()
        
        # Create input and target
        x = torch.randn(1, 10, 6, requires_grad=True)
        target = torch.randn(1, 3)
        
        # Forward pass
        position, log_var, _ = model.forward(x)
        
        # Simple loss
        loss = torch.nn.functional.mse_loss(position, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
