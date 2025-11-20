# Advanced Trajectory Prediction

This module implements advanced trajectory prediction with LSTM-based learning and physics-based models for the SENTINEL safety system.

## Overview

The advanced trajectory prediction system generates multiple trajectory hypotheses for detected objects, estimates uncertainty bounds, and calculates collision probabilities with the ego vehicle.

## Components

### 1. LSTM Trajectory Model (`advanced_trajectory.py`)

**LSTMTrajectoryModel**: Deep learning model for trajectory prediction
- Architecture: 2-layer LSTM with 64 hidden units
- Input: Historical motion (position + velocity)
- Output: Future positions with uncertainty estimates
- Training: Negative log-likelihood loss with uncertainty

**Features:**
- Learns from historical trajectory patterns
- Outputs uncertainty estimates (log variance)
- Supports sequence prediction for multiple time steps
- Can be optimized with TorchScript for inference

### 2. Physics-Based Models (`advanced_trajectory.py`)

**PhysicsBasedPredictor**: Classical motion models
- **Constant Velocity (CV)**: Linear motion prediction
- **Constant Acceleration (CA)**: Accounts for acceleration/deceleration
- **Constant Turn Rate (CT)**: Models turning vehicles

**Model Selection:**
- Automatic selection based on object type and motion history
- Vehicles/cyclists: Can use CT model if turning detected
- Pedestrians: Typically use CV model
- Acceleration detection: Switches to CA model

### 3. Advanced Trajectory Predictor (`advanced_trajectory.py`)

**AdvancedTrajectoryPredictor**: Main prediction engine
- Maintains motion history buffer (30 frames)
- Generates multiple trajectory hypotheses (default: 3)
- Combines LSTM and physics-based predictions
- Extracts scene context features
- Ranks hypotheses by confidence

**Configuration:**
```yaml
trajectory_prediction:
  enabled: true
  horizon: 5.0  # seconds
  dt: 0.1  # time step
  num_hypotheses: 3
  use_lstm: true
  lstm_model: "models/trajectory_lstm.pth"
  uncertainty_estimation: true
```

### 4. Uncertainty Estimation (`advanced_trajectory.py`)

**UncertaintyEstimator**: Quantifies prediction uncertainty
- Calculates covariance matrices for each time step
- Propagates uncertainty through time
- Estimates confidence scores based on:
  - Motion history length
  - Scene complexity
  - Model uncertainty
- Merges multiple hypotheses with combined uncertainty

### 5. Collision Probability Calculator (`advanced_trajectory.py`)

**CollisionProbabilityCalculator**: Assesses collision risk
- Mahalanobis distance computation with uncertainty ellipses
- Considers vehicle and object dimensions
- Outputs probability scores (0-1)
- Identifies time and location of maximum risk

**Features:**
- Uncertainty-aware distance calculation
- Chi-squared based ellipse overlap detection
- Per-hypothesis collision probability
- Time-to-collision estimation from trajectories

### 6. Advanced Risk Assessment (`advanced_risk.py`)

**AdvancedRiskAssessor**: Integrates trajectories with risk assessment
- Enhances base risk with collision probabilities
- Blends traditional risk factors with trajectory predictions
- Creates hazards with trajectory information
- Supports contextual risk calculation

**Risk Enhancement:**
```
enhanced_risk = (1 - w) * base_risk + w * collision_probability
```
where `w` is the collision probability weight (default: 0.4)

### 7. Trajectory Visualization (`trajectory_visualization.py`)

**TrajectoryVisualizer**: Prepares data for GUI display
- Converts Trajectory objects to BEV canvas format
- Extracts 2D points and uncertainty bounds
- Color codes by collision probability
- Supports multi-hypothesis display
- Filters by distance and collision risk

**Display Format:**
```python
{
    'object_id': int,
    'points': [(x, y), ...],  # 2D positions
    'uncertainty': [std, ...],  # Standard deviations
    'collision_probability': float,
    'confidence': float,
    'model': str,  # 'lstm', 'cv', 'ca', 'ct'
    'timestamps': [t, ...]
}
```

### 8. Performance Optimization (`trajectory_performance.py`)

**TrajectoryPerformanceOptimizer**: Monitors and optimizes performance
- Profiling of LSTM, physics, and collision components
- Trajectory caching with TTL
- Parallel physics model execution
- LSTM model optimization (TorchScript, cuDNN)
- Performance metrics tracking

**Performance Targets:**
- < 5ms per object (P95)
- Total latency budget: ~50ms for 10 objects

**Optimization Features:**
- Model quantization support
- Batch inference for GPU efficiency
- Cache management
- Parallel threading for physics models
- Performance recommendations

## Usage

### Basic Usage

```python
from src.intelligence.advanced_trajectory import AdvancedTrajectoryPredictor
from src.core.data_structures import Detection3D

# Initialize predictor
config = {
    'enabled': True,
    'horizon': 5.0,
    'dt': 0.1,
    'num_hypotheses': 3,
    'use_lstm': False  # Set True when model is trained
}
predictor = AdvancedTrajectoryPredictor(config)

# Predict trajectories
detections = [...]  # List of Detection3D objects
trajectories = predictor.predict_all(detections)

# trajectories: Dict[track_id, List[Trajectory]]
```

### With Collision Probability

```python
from src.intelligence.advanced_trajectory import CollisionProbabilityCalculator

# Initialize calculator
collision_calc = CollisionProbabilityCalculator()

# Calculate collision probabilities
ego_trajectory = ...  # Ego vehicle trajectory
object_sizes = {track_id: size_in_meters}

collision_probs = collision_calc.calculate_all_collision_probabilities(
    ego_trajectory,
    trajectories,
    object_sizes
)

# collision_probs: Dict[track_id, (probability, time_step, hypothesis_idx)]
```

### With Risk Assessment

```python
from src.intelligence.advanced_risk import AdvancedRiskAssessor

# Initialize assessor
assessor = AdvancedRiskAssessor(config)

# Assess hazards with trajectories
hazards, trajectories, collision_probs = assessor.assess_hazards_with_trajectories(
    detections,
    ego_trajectory,
    driver_state
)

# Create risks
risks = assessor.create_risks_with_trajectories(
    hazards,
    driver_state,
    attention_map
)
```

### Visualization

```python
from src.intelligence.trajectory_visualization import TrajectoryVisualizer

# Initialize visualizer
visualizer = TrajectoryVisualizer()

# Prepare for display
display_trajectories = visualizer.prepare_trajectories_for_display(
    trajectories,
    collision_probs,
    show_all_hypotheses=False
)

# Update BEV canvas
bev_canvas.update_trajectories(display_trajectories)
```

## Training LSTM Model

To train the LSTM trajectory model:

```python
from src.intelligence.advanced_trajectory import (
    LSTMTrajectoryModel,
    train_lstm_model,
    save_lstm_model
)

# Create model
model = LSTMTrajectoryModel(
    input_size=6,  # x, y, z, vx, vy, vz
    hidden_size=64,
    num_layers=2,
    output_size=3  # x, y, z
)

# Prepare training data
# train_data: List[(history, future)] where
#   history: (seq_len, 6) array
#   future: (num_steps, 3) array
train_data = [...]

# Train model
trained_model = train_lstm_model(
    model,
    train_data,
    num_epochs=100,
    learning_rate=0.001,
    device='cuda'
)

# Save model
save_lstm_model(trained_model, 'models/trajectory_lstm.pth')
```

## Performance Monitoring

```python
from src.intelligence.trajectory_performance import TrajectoryPerformanceOptimizer

# Initialize optimizer
perf_config = {
    'target_time_per_object': 5.0,
    'enable_profiling': True,
    'enable_caching': True,
    'enable_parallel': True
}
optimizer = TrajectoryPerformanceOptimizer(perf_config)

# Profile prediction
result, elapsed_time = optimizer.profile_prediction(
    predictor.predict_all,
    detections
)

# Get performance summary
summary = optimizer.get_performance_summary()
print(f"P95 time per object: {summary['p95_time_per_object']:.2f}ms")

# Get recommendations
recommendations = optimizer.get_optimization_recommendations()
for rec in recommendations:
    print(f"- {rec}")
```

## Integration with SENTINEL System

The advanced trajectory prediction integrates with the main SENTINEL system through the contextual intelligence engine:

1. **Detection Phase**: Objects detected and tracked
2. **Trajectory Prediction**: Multiple hypotheses generated
3. **Collision Assessment**: Probabilities calculated
4. **Risk Enhancement**: Traditional risk enhanced with collision probability
5. **Visualization**: Trajectories displayed on BEV canvas
6. **Alert Generation**: High-risk trajectories trigger alerts

## Configuration

See `configs/default.yaml` for full configuration options:

```yaml
risk_assessment:
  trajectory_prediction:
    enabled: true
    horizon: 5.0
    dt: 0.1
    num_hypotheses: 3
    use_lstm: false
    lstm_model: "models/trajectory_lstm.pth"
    uncertainty_estimation: true
    use_collision_probability: true
    collision_probability_weight: 0.4
    target_time_per_object: 5.0
    enable_profiling: false
    enable_caching: true
    enable_parallel: true
```

## Example

Run the example script:

```bash
python examples/advanced_trajectory_example.py
```

This demonstrates:
- Creating sample detections
- Predicting trajectories with multiple hypotheses
- Calculating collision probabilities
- Preparing visualization data
- Performance monitoring

## Requirements

- PyTorch (optional, for LSTM model)
- NumPy
- SciPy (for chi-squared distribution)

## Performance Characteristics

**Typical Performance (without LSTM):**
- Physics models: ~2-3ms per object
- Collision calculation: ~1-2ms per object
- Total: ~3-5ms per object

**With LSTM:**
- LSTM inference: ~3-5ms per object (GPU)
- Total: ~5-8ms per object

**Optimization Tips:**
1. Disable LSTM if not needed (use physics models only)
2. Enable caching for static objects
3. Reduce number of hypotheses (1-2 instead of 3)
4. Reduce trajectory horizon (3s instead of 5s)
5. Use parallel processing for multiple objects

## Future Enhancements

- [ ] Interaction-aware prediction (consider other objects)
- [ ] Map-aware prediction (use lane geometry)
- [ ] Maneuver classification (lane change, turn, stop)
- [ ] Ensemble learning (combine multiple LSTM models)
- [ ] Attention mechanism for scene context
- [ ] Real-time model adaptation
