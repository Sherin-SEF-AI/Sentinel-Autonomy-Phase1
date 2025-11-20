# Advanced Trajectory Logging - Quick Reference

## Logger Configuration

**Module**: `src.intelligence.advanced_trajectory`
**Log Level**: DEBUG (development), INFO (production)
**Log Files**: 
- `logs/intelligence.log` (module-specific)
- `logs/sentinel.log` (system-wide)

## Key Log Messages

### Initialization
```
INFO - PyTorch available: LSTM trajectory prediction enabled
INFO - LSTM trajectory model initialized: parameters=45123
```

### Inference
```
DEBUG - Predicting trajectory sequence: batch_size=1, history_len=10, num_steps=30
DEBUG - LSTM forward pass completed: duration=3.45ms, batch_size=1
DEBUG - Trajectory sequence prediction completed: duration=4.23ms
```

### Training
```
INFO - Training LSTM model started: num_epochs=100, learning_rate=0.001
INFO - Epoch 10/100 completed: loss=0.0234, best_loss=0.0234, duration=1.45s
INFO - Training completed: total_duration=90.23s, final_loss=0.0089
```

### Model Save/Load
```
INFO - Saving LSTM model to: models/trajectory_lstm.pth
INFO - LSTM model saved successfully: models/trajectory_lstm.pth
INFO - Loading LSTM model from: models/trajectory_lstm.pth
INFO - LSTM model loaded successfully: duration=45.67ms, parameters=45123
```

### Errors
```
ERROR - Training failed: PyTorch not available
ERROR - Failed to load LSTM model from {path}: {error}
```

## Performance Metrics

| Operation | Target | Logged Metric |
|-----------|--------|---------------|
| Forward Pass | <5ms | duration=X.XXms |
| Sequence Prediction | <10ms | duration=X.XXms |
| Model Load | <100ms | duration=X.XXms |
| Training Epoch | Variable | duration=X.XXs |

## Changing Log Level

### Runtime (Python)
```python
from src.core.logging import LoggerSetup
LoggerSetup.set_level('DEBUG')  # or 'INFO', 'WARNING', 'ERROR'
```

### Configuration (YAML)
Edit `configs/logging.yaml`:
```yaml
src.intelligence.advanced_trajectory:
  level: INFO  # Change to DEBUG for detailed logging
```

## Viewing Logs

### Tail intelligence logs
```bash
tail -f logs/intelligence.log
```

### Filter for trajectory logs
```bash
grep "advanced_trajectory" logs/intelligence.log
```

### View performance metrics
```bash
grep "duration=" logs/intelligence.log | grep "advanced_trajectory"
```

## Verification

Run verification script:
```bash
python3 scripts/verify_advanced_trajectory_logging_simple.py
```

Expected output:
```
✓ PASS: Logging Configuration
✓ PASS: Code Structure
```

## Common Issues

### No logs appearing
- Check log level in `configs/logging.yaml`
- Verify `logs/` directory exists
- Check file permissions

### Too much logging
- Change level from DEBUG to INFO in config
- Restart application to apply changes

### Performance impact
- Use INFO level in production
- DEBUG level adds ~1% overhead
- Disable file logging for maximum performance

## Integration Example

```python
from src.intelligence.advanced_trajectory import (
    LSTMTrajectoryModel,
    load_lstm_model
)

# Logs: "Loading LSTM model from: models/trajectory_lstm.pth"
model = load_lstm_model('models/trajectory_lstm.pth')

# Logs: "Predicting trajectory sequence: batch_size=1, history_len=10, num_steps=30"
predictions, uncertainties = model.predict_sequence(history, num_steps=30)
```

## Related Modules

- `src.intelligence.trajectory_performance` - Performance profiling
- `src.intelligence.trajectory_visualization` - Visualization utilities
- `src.intelligence.advanced_risk` - Risk assessment with advanced trajectories
