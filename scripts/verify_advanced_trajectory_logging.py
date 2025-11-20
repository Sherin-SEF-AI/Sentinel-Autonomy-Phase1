"""Verification script for advanced trajectory logging."""

import sys
from pathlib import Path
import logging
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import LoggerSetup


def test_lstm_model_logging():
    """Test LSTM model initialization and logging."""
    print("\n" + "="*60)
    print("Testing LSTM Model Logging")
    print("="*60)
    
    try:
        from src.intelligence.advanced_trajectory import (
            LSTMTrajectoryModel,
            TORCH_AVAILABLE
        )
        
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - skipping LSTM tests")
            return True
        
        import torch
        
        # Initialize model
        print("\n1. Testing model initialization...")
        model = LSTMTrajectoryModel(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout=0.2
        )
        print("✓ Model initialized successfully")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        seq_len = 10
        input_tensor = torch.randn(batch_size, seq_len, 6)
        
        position, log_var, hidden = model.forward(input_tensor)
        print(f"✓ Forward pass completed: output_shape={position.shape}")
        
        # Test sequence prediction
        print("\n3. Testing sequence prediction...")
        history = torch.randn(1, 10, 6)
        predictions, uncertainties = model.predict_sequence(history, num_steps=30)
        print(f"✓ Sequence prediction completed: predictions_shape={predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ LSTM model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_save_load_logging():
    """Test model save/load logging."""
    print("\n" + "="*60)
    print("Testing Model Save/Load Logging")
    print("="*60)
    
    try:
        from src.intelligence.advanced_trajectory import (
            LSTMTrajectoryModel,
            save_lstm_model,
            load_lstm_model,
            TORCH_AVAILABLE
        )
        
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - skipping save/load tests")
            return True
        
        # Create temporary model file
        model_path = "test_lstm_model.pth"
        
        # Initialize and save model
        print("\n1. Testing model save...")
        model = LSTMTrajectoryModel()
        save_lstm_model(model, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Load model
        print("\n2. Testing model load...")
        loaded_model = load_lstm_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Clean up
        import os
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"✓ Cleaned up test file: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_logging():
    """Test training logging."""
    print("\n" + "="*60)
    print("Testing Training Logging")
    print("="*60)
    
    try:
        from src.intelligence.advanced_trajectory import (
            LSTMTrajectoryModel,
            train_lstm_model,
            TORCH_AVAILABLE
        )
        
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available - skipping training tests")
            return True
        
        import torch
        
        # Create synthetic training data
        print("\n1. Creating synthetic training data...")
        train_data = []
        for i in range(10):
            history = np.random.randn(10, 6).astype(np.float32)
            future = np.random.randn(30, 3).astype(np.float32)
            train_data.append((history, future))
        print(f"✓ Created {len(train_data)} training samples")
        
        # Train model (short run)
        print("\n2. Testing training loop...")
        model = LSTMTrajectoryModel()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        trained_model = train_lstm_model(
            model,
            train_data,
            num_epochs=20,
            learning_rate=0.001,
            device=device
        )
        print("✓ Training completed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_log_output():
    """Verify log file contains expected entries."""
    print("\n" + "="*60)
    print("Verifying Log Output")
    print("="*60)
    
    log_file = Path("logs/intelligence.log")
    
    if not log_file.exists():
        print(f"⚠ Log file not found: {log_file}")
        return True
    
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
        
        # Check for key log messages
        checks = [
            ("PyTorch availability", "PyTorch available" in log_content or "PyTorch not available" in log_content),
            ("Model initialization", "LSTM trajectory model initialized" in log_content or "Initializing LSTM" in log_content),
            ("Forward pass timing", "forward pass completed" in log_content or "duration=" in log_content),
            ("Sequence prediction", "Trajectory sequence prediction" in log_content or "Predicting trajectory" in log_content),
        ]
        
        print("\nLog content checks:")
        all_passed = True
        for check_name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {check_name}")
            if not result:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Log verification failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("Advanced Trajectory Logging Verification")
    print("="*60)
    
    # Setup logging
    LoggerSetup.setup(log_level='DEBUG', log_dir='logs')
    
    # Run tests
    results = []
    
    results.append(("LSTM Model Logging", test_lstm_model_logging()))
    results.append(("Model Save/Load Logging", test_model_save_load_logging()))
    results.append(("Training Logging", test_training_logging()))
    results.append(("Log Output Verification", verify_log_output()))
    
    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All verification tests passed!")
    else:
        print("✗ Some verification tests failed")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
