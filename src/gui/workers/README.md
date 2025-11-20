# SENTINEL Worker Threads

## Overview

The worker threads module provides background processing capabilities for the SENTINEL GUI application. The main `SentinelWorker` class runs the complete SENTINEL system pipeline in a separate thread, preventing GUI blocking and ensuring responsive user interaction.

## Architecture

### Thread Model

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Thread (GUI)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Video    │  │   Driver   │  │    Risk    │            │
│  │  Displays  │  │   Panel    │  │   Panel    │            │
│  └─────▲──────┘  └─────▲──────┘  └─────▲──────┘            │
│        │                │                │                   │
│        │ Signals        │                │                   │
└────────┼────────────────┼────────────────┼───────────────────┘
         │                │                │
         │                │                │
┌────────┼────────────────┼────────────────┼───────────────────┐
│        │                │                │                   │
│  ┌─────┴──────┐  ┌──────┴─────┐  ┌──────┴─────┐            │
│  │   Camera   │  │    DMS     │  │    Risk    │            │
│  │  Capture   │  │  Analysis  │  │ Assessment │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                              │
│              Worker Thread (Background)                      │
└──────────────────────────────────────────────────────────────┘
```

## SentinelWorker Class

### Initialization

```python
from gui.workers import SentinelWorker
from core.config import ConfigManager

# Load configuration
config = ConfigManager('configs/default.yaml')

# Create worker
worker = SentinelWorker(config)
```

### Signals

The worker emits the following signals:

| Signal | Type | Description |
|--------|------|-------------|
| `frame_ready` | `dict` | Camera frames ready |
| `bev_ready` | `(np.ndarray, np.ndarray)` | BEV image and mask |
| `detections_ready` | `list` | 3D object detections |
| `driver_state_ready` | `DriverState` | Driver monitoring state |
| `risks_ready` | `RiskAssessment` | Risk assessment results |
| `alerts_ready` | `list` | Generated alerts |
| `performance_ready` | `dict` | Performance metrics |
| `error_occurred` | `(str, str)` | Error type and message |
| `status_changed` | `str` | Status update |

### Usage Example

```python
# Create worker
worker = SentinelWorker(config)

# Connect signals
worker.frame_ready.connect(on_frames_ready)
worker.driver_state_ready.connect(on_driver_state)
worker.performance_ready.connect(on_performance)
worker.error_occurred.connect(on_error)

# Start processing
worker.start()

# ... application runs ...

# Stop processing
worker.stop()
worker.wait(5000)  # Wait up to 5 seconds
```

## Thread Safety

### Data Copying

All data is deep copied before emission to ensure thread safety:

```python
# Camera frames
frames_dict = self._copy_camera_bundle(camera_bundle)
self.frame_ready.emit(frames_dict)

# Driver state
driver_state_copy = self._copy_driver_state(driver_state)
self.driver_state_ready.emit(driver_state_copy)

# Detections
detections_copy = self._copy_detections(detections_3d)
self.detections_ready.emit(detections_copy)
```

### Synchronization

- `QMutex` used for worker control flags
- Qt signals handle cross-thread communication
- No shared mutable state between threads
- All GUI updates happen in main thread

## Processing Pipeline

### Initialization Phase

1. Create all system modules:
   - Camera Manager
   - BEV Generator
   - Semantic Segmentor
   - Object Detector
   - Driver Monitor
   - Contextual Intelligence
   - Alert System
   - Scenario Recorder

2. Start camera capture
3. Begin processing loop

### Processing Loop

```python
while running:
    # 1. Capture frames
    camera_bundle = camera_manager.get_frame_bundle()
    
    # 2. Parallel processing
    #    - DMS analysis (interior camera)
    #    - Perception pipeline (external cameras)
    
    # 3. Risk assessment
    risk_assessment = intelligence.assess(...)
    
    # 4. Alert generation
    alerts = alert_system.process(...)
    
    # 5. Recording (if triggered)
    recorder.save_frame(...)
    
    # 6. Emit signals
    self.frame_ready.emit(...)
    self.driver_state_ready.emit(...)
    self.risks_ready.emit(...)
    # ... etc
```

### Cleanup Phase

1. Stop recording if active
2. Stop camera capture
3. Clear GPU cache
4. Close all resources

## Error Handling

### Error Types

- **Fatal**: Critical errors requiring system shutdown
- **Initialization**: Module initialization failures
- **DMS**: Driver monitoring errors
- **Perception**: Perception pipeline errors
- **Processing**: General processing errors

### Error Recovery

```python
try:
    # Processing code
    result = process_data()
except Exception as e:
    # Log error
    logger.error(f"Error: {e}")
    
    # Emit error signal
    self.error_occurred.emit(error_type, str(e))
    
    # Continue processing (for non-fatal errors)
```

### Handling Errors in GUI

```python
def on_error(error_type: str, error_message: str):
    """Handle worker errors."""
    if error_type == "Fatal":
        # Show critical error and stop
        QMessageBox.critical(self, "Error", error_message)
        worker.stop()
    else:
        # Show warning
        statusBar().showMessage(f"Warning: {error_message}")
```

## Performance Monitoring

### Metrics Collection

The worker tracks:
- Frames per second (FPS)
- Total frames processed
- Per-module latencies
- Total pipeline latency

### Metrics Structure

```python
{
    'fps': 30.5,
    'frame_count': 1000,
    'loop_time_ms': 32.5,
    'module_latencies': {
        'camera': 5.2,
        'bev': 14.8,
        'segmentation': 15.1,
        'detection': 19.5,
        'dms': 24.3,
        'intelligence': 9.8,
        'alerts': 2.1
    },
    'total_latency_ms': 90.8
}
```

### Using Performance Data

```python
def on_performance(metrics: dict):
    """Update performance displays."""
    fps = metrics['fps']
    latency = metrics['total_latency_ms']
    
    fps_label.setText(f"FPS: {fps:.2f}")
    latency_label.setText(f"Latency: {latency:.2f}ms")
    
    # Check requirements
    if fps < 30.0:
        logger.warning(f"FPS below target: {fps:.2f}")
    if latency > 100.0:
        logger.warning(f"Latency above target: {latency:.2f}ms")
```

## Best Practices

### DO ✓

1. **Always deep copy data** before emitting signals
2. **Use try-except blocks** in processing code
3. **Emit error signals** for recoverable errors
4. **Log all significant events** for debugging
5. **Clean up resources** in cleanup method
6. **Wait for worker** to finish before destroying
7. **Handle all error types** appropriately

### DON'T ✗

1. **Don't access GUI widgets** from worker thread
2. **Don't share mutable objects** between threads
3. **Don't block worker thread** with long operations
4. **Don't ignore error signals**
5. **Don't terminate** without trying graceful stop
6. **Don't create GUI objects** in worker thread
7. **Don't use global variables** for communication

## Debugging

### Enable Debug Logging

```python
import logging

# Set worker logger to DEBUG
logging.getLogger('sentinel.worker').setLevel(logging.DEBUG)
```

### Monitor Signals

```python
from PyQt6.QtTest import QSignalSpy

# Create spy
spy = QSignalSpy(worker.frame_ready)

# Check emissions
print(f"Signals emitted: {len(spy)}")
```

### Check Thread State

```python
# Is running?
if worker.isRunning():
    print("Worker is running")

# Is finished?
if worker.isFinished():
    print("Worker has finished")
```

## Examples

### Basic Usage

See `examples/worker_thread_example.py` for a complete working example.

### Integration with Main Window

See `src/gui/main_window.py` for full integration example with all signals connected.

## Testing

### Unit Tests

Run unit tests:
```bash
pytest tests/unit/test_sentinel_worker.py -v
```

### Test Coverage

- Worker initialization
- Signal definitions
- Stop method
- Thread-safe data copying
- Performance metrics calculation
- Module initialization
- Signal emission

## Requirements Satisfied

- **Requirement 13.1**: PyQt6 GUI with background processing
- **Requirement 13.3**: Real-time updates without GUI blocking
- **Requirement 11.3**: Error recovery and automatic handling

## Related Documentation

- [Worker Thread Integration Guide](../../../WORKER_THREAD_INTEGRATION_GUIDE.md)
- [GUI Architecture](../GUI_ARCHITECTURE.md)
- [Task 24 Summary](../../../.kiro/specs/sentinel-safety-system/TASK_24_SUMMARY.md)

## Support

For issues or questions:
1. Check the integration guide
2. Review example code
3. Enable debug logging
4. Check unit tests for usage patterns
