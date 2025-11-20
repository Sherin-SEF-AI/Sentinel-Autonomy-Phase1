# SENTINEL Worker Thread Integration Guide

## Quick Start

### Starting the System with Worker Thread

```python
from PyQt6.QtWidgets import QApplication
from gui.main_window import SENTINELMainWindow
from gui.themes import ThemeManager
from core.config import ConfigManager

# Create application
app = QApplication([])

# Load configuration
config = ConfigManager('configs/default.yaml')

# Create theme manager
theme_manager = ThemeManager(app)

# Create main window (worker thread integrated)
main_window = SENTINELMainWindow(theme_manager, config)
main_window.show()

# Run application
app.exec()
```

### Using the Worker Thread

The worker thread is automatically managed by the main window:

```python
# Start system (creates and starts worker)
main_window._on_start_system()

# Stop system (stops worker gracefully)
main_window._on_stop_system()
```

## Worker Thread Signals

### Available Signals

| Signal | Parameters | Description |
|--------|-----------|-------------|
| `frame_ready` | `dict` | Camera frames (interior, front_left, front_right) |
| `bev_ready` | `np.ndarray, np.ndarray` | BEV image and mask |
| `detections_ready` | `list` | List of Detection3D objects |
| `driver_state_ready` | `DriverState` | Driver monitoring state |
| `risks_ready` | `RiskAssessment` | Risk assessment results |
| `alerts_ready` | `list` | List of Alert objects |
| `performance_ready` | `dict` | Performance metrics |
| `error_occurred` | `str, str` | Error type and message |
| `status_changed` | `str` | Status message |

### Connecting Custom Slots

```python
# Create worker
worker = SentinelWorker(config)

# Connect to custom slot
worker.frame_ready.connect(my_custom_handler)

def my_custom_handler(frames: dict):
    """Handle camera frames."""
    interior = frames.get('interior')
    if interior is not None:
        # Process interior camera frame
        pass
```

## Thread-Safe Data Access

### Deep Copying

All data is automatically deep copied before emission:

```python
# In worker thread
camera_bundle = self.camera_manager.get_frame_bundle()

# Deep copy for thread safety
frames_dict = self._copy_camera_bundle(camera_bundle)

# Emit signal (safe for GUI thread)
self.frame_ready.emit(frames_dict)
```

### Custom Data Copying

If you need to pass custom data:

```python
import copy

# For simple objects
data_copy = copy.copy(data)

# For complex nested objects
data_copy = copy.deepcopy(data)

# For NumPy arrays
array_copy = array.copy()
```

## Error Handling

### Catching Errors in Worker

```python
try:
    # Processing code
    result = self.process_data()
except Exception as e:
    # Log error
    self.logger.error(f"Processing failed: {e}")
    
    # Emit error signal
    self.error_occurred.emit("Processing", str(e))
```

### Handling Errors in GUI

```python
def _on_worker_error(self, error_type: str, error_message: str):
    """Handle worker errors."""
    if error_type == "Fatal":
        # Show critical error dialog
        QMessageBox.critical(self, "Fatal Error", error_message)
        # Stop system
        self._on_stop_system()
    else:
        # Show in status bar
        self.statusBar().showMessage(f"Error: {error_message}", 5000)
```

## Performance Monitoring

### Accessing Performance Metrics

```python
def _on_performance_ready(self, metrics: dict):
    """Handle performance metrics."""
    fps = metrics['fps']
    total_latency = metrics['total_latency_ms']
    module_latencies = metrics['module_latencies']
    
    # Update displays
    print(f"FPS: {fps:.2f}")
    print(f"Latency: {total_latency:.2f}ms")
    
    for module, latency in module_latencies.items():
        print(f"  {module}: {latency:.2f}ms")
```

### Performance Metrics Structure

```python
{
    'fps': 30.5,                    # Current frames per second
    'frame_count': 1000,            # Total frames processed
    'loop_time_ms': 32.5,           # Current loop time
    'module_latencies': {           # Average latencies per module
        'camera': 5.2,
        'bev': 14.8,
        'segmentation': 15.1,
        'detection': 19.5,
        'dms': 24.3,
        'intelligence': 9.8,
        'alerts': 2.1
    },
    'total_latency_ms': 90.8        # Sum of all module latencies
}
```

## Lifecycle Management

### Worker Lifecycle

```
Created → Initialized → Running → Stopping → Stopped
   ↓          ↓           ↓          ↓          ↓
  new()    start()    run()      stop()    cleanup()
```

### Graceful Shutdown

```python
# Request stop
worker.stop()

# Wait for completion (5 second timeout)
if not worker.wait(5000):
    # Force termination if needed
    worker.terminate()
    worker.wait()
```

### Cleanup

The worker automatically cleans up:
- Stops camera capture
- Stops recording if active
- Clears GPU cache
- Closes all resources

## Common Patterns

### Pattern 1: Update Display on Frame Ready

```python
def _on_frames_ready(self, frames: dict):
    """Update video displays."""
    for camera_id, frame in frames.items():
        if frame is not None:
            self.video_display.update_frame(camera_id, frame)
```

### Pattern 2: Update Multiple Widgets

```python
def _on_driver_state_ready(self, driver_state):
    """Update driver-related widgets."""
    # Update driver state panel
    self.driver_panel.update_driver_state(driver_state)
    
    # Update attention zones on BEV
    self.bev_canvas.update_attention_zones({
        'current_zone': driver_state.gaze['attention_zone']
    })
    
    # Update status indicator
    self.status_indicator.update_readiness(
        driver_state.readiness_score
    )
```

### Pattern 3: Conditional Processing

```python
def _on_risks_ready(self, risk_assessment):
    """Handle risk assessment."""
    # Calculate overall risk
    overall_risk = max(
        r.contextual_score 
        for r in risk_assessment.top_risks
    ) if risk_assessment.top_risks else 0.0
    
    # Update display
    self.risk_panel.update_risk_score(overall_risk)
    
    # Trigger alert if high risk
    if overall_risk > 0.7:
        self.show_high_risk_warning()
```

## Debugging

### Enable Debug Logging

```python
import logging

# Set worker logger to DEBUG
logging.getLogger('sentinel.worker').setLevel(logging.DEBUG)
```

### Monitor Signal Emissions

```python
from PyQt6.QtTest import QSignalSpy

# Create signal spy
spy = QSignalSpy(worker.frame_ready)

# Check emissions
print(f"Signals emitted: {len(spy)}")
print(f"Last signal data: {spy[-1]}")
```

### Check Thread State

```python
# Check if worker is running
if worker.isRunning():
    print("Worker is running")

# Check if worker finished
if worker.isFinished():
    print("Worker has finished")
```

## Best Practices

### DO ✓

- Always deep copy data before emitting signals
- Use try-except blocks in processing code
- Emit error signals for recoverable errors
- Log all significant events
- Clean up resources in cleanup method
- Use QMutex for shared state
- Wait for worker to finish before destroying

### DON'T ✗

- Don't access GUI widgets from worker thread
- Don't share mutable objects between threads
- Don't block the worker thread with long operations
- Don't ignore error signals
- Don't terminate worker without trying graceful stop
- Don't create GUI objects in worker thread
- Don't use global variables for thread communication

## Troubleshooting

### Worker Won't Start

```python
# Check if already running
if worker.isRunning():
    print("Worker already running")
    return

# Check initialization
if not worker._initialize_modules():
    print("Module initialization failed")
    return
```

### GUI Freezing

```python
# Ensure data is copied, not referenced
# BAD:
self.frame_ready.emit(camera_bundle)  # Shared reference

# GOOD:
frames_copy = self._copy_camera_bundle(camera_bundle)
self.frame_ready.emit(frames_copy)  # Independent copy
```

### Memory Leaks

```python
# Limit history size
if len(self.module_latencies['camera']) > 100:
    self.module_latencies['camera'] = \
        self.module_latencies['camera'][-100:]
```

### Worker Won't Stop

```python
# Check stop flag in loop
while self._running and not self._stop_requested:
    # Processing...
    pass

# Use timeout when waiting
if not worker.wait(5000):
    worker.terminate()
```

## Advanced Usage

### Custom Worker Subclass

```python
class CustomSentinelWorker(SentinelWorker):
    """Custom worker with additional functionality."""
    
    # Add custom signal
    custom_data_ready = pyqtSignal(dict)
    
    def _processing_loop(self):
        """Override processing loop."""
        # Call parent implementation
        super()._processing_loop()
        
        # Add custom processing
        custom_data = self._process_custom_data()
        self.custom_data_ready.emit(custom_data)
```

### Multiple Workers

```python
# Create multiple workers for different tasks
perception_worker = SentinelWorker(config)
recording_worker = RecordingWorker(config)

# Start both
perception_worker.start()
recording_worker.start()

# Stop both
perception_worker.stop()
recording_worker.stop()
```

## References

- Qt Threading Documentation: https://doc.qt.io/qt-6/threads.html
- PyQt6 Signals and Slots: https://www.riverbankcomputing.com/static/Docs/PyQt6/
- SENTINEL Architecture: `GUI_ARCHITECTURE.md`
- Task 24 Summary: `.kiro/specs/sentinel-safety-system/TASK_24_SUMMARY.md`
