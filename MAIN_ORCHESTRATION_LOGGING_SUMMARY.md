# Main System Orchestration Logging Summary

## Overview
The `src/main.py` module serves as the main entry point and orchestrator for the SENTINEL system. It has comprehensive logging implemented across all critical operations including initialization, processing loops, performance monitoring, state management, and shutdown procedures.

## Logger Configuration

### Module Logger
```python
logger = logging.getLogger(__name__)  # src.main
```

### System Loggers
```python
self.logger = logging.getLogger('sentinel.system')
self.perf_logger = logging.getLogger('sentinel.performance')
```

### Configuration in `configs/logging.yaml`
```yaml
src.main:
  level: INFO
  handlers: [file_all]
  propagate: false

sentinel.system:
  level: INFO
  handlers: [file_all]
  propagate: false

sentinel.performance:
  level: INFO
  handlers: [file_performance, file_all]
  propagate: false
```

## Logging Coverage

### 1. System Initialization (`__init__` and `_initialize_modules`)

**Current Logging:**
- ✅ Module initialization start: `"Initializing SENTINEL system modules..."`
- ✅ Individual module initialization: `"Initializing Camera Manager..."`, `"Initializing BEV Generator..."`, etc.
- ✅ Successful initialization: `"All modules initialized successfully"`
- ✅ Initialization failures: `"Failed to initialize modules: {e}"`

**Log Levels:**
- INFO: Normal initialization progress
- ERROR: Initialization failures with full traceback

### 2. System Startup (`start`)

**Current Logging:**
- ✅ Start request: `"Starting SENTINEL system..."`
- ✅ Already running warning: `"System is already running"`
- ✅ Camera capture start: `"Starting camera capture..."`
- ✅ Visualization server start: `"Starting visualization server..."`
- ✅ Successful start: `"SENTINEL system started successfully"`
- ✅ Start failures: `"Failed to start system: {e}"`
- ✅ Performance monitoring start: `"Starting performance monitoring..."`

**Log Levels:**
- INFO: Normal startup operations
- WARNING: Already running condition
- ERROR: Startup failures

### 3. Performance Monitoring (`_performance_monitoring_loop`, `_log_performance_metrics`)

**Current Logging:**
- ✅ Monitoring start: `"Starting performance monitoring..."`
- ✅ Monitoring stop: `"Performance monitoring stopped"`
- ✅ Monitoring errors: `"Performance monitoring error: {e}"`
- ✅ Periodic metrics (every 10 seconds):
  - FPS: `"FPS: {current_fps:.2f}"`
  - Frame count: `"Frames processed: {self.frame_count}"`
  - CPU usage: `"CPU usage: {avg_cpu:.1f}%"`
  - Memory usage: `"Memory usage: {avg_memory:.1f} MB"`
  - GPU memory: `"GPU memory: {avg_gpu_memory:.1f} MB"`
  - Module latencies: `"{module}: {latency:.2f}ms"`
  - Total pipeline latency: `"Total pipeline latency: {total_latency:.2f}ms"`

**Log Levels:**
- INFO: Performance metrics and statistics
- ERROR: Monitoring errors

### 4. Main Processing Loop (`_processing_loop`)

**Current Logging:**
- ✅ Loop start: `"Starting main processing loop..."`
- ✅ Loop stop: `"Processing loop stopped"`
- ✅ Periodic frame info (every 30 frames): `"Frame {frame_count}: {loop_time*1000:.1f}ms ({fps:.1f} FPS)"`
- ✅ DMS processing errors: `"DMS processing error: {e}"`
- ✅ Perception processing errors: `"Perception processing error: {e}"`
- ✅ Loop errors: `"Error in processing loop: {e}"` with full traceback
- ✅ Visualization streaming errors (DEBUG): `"Visualization streaming error: {e}"`

**Log Levels:**
- INFO: Normal processing progress and periodic statistics
- ERROR: Processing errors with tracebacks
- DEBUG: Non-critical errors (visualization streaming)

### 5. State Management (`_restore_system_state`, `_save_system_state`, `_periodic_state_save`)

**Current Logging:**
- ✅ No previous state: `"No previous state found, starting fresh"`
- ✅ Old state: `"Previous state is too old ({state_age/60:.1f} minutes), starting fresh"`
- ✅ State recovery info:
  - `"Previous session: {prev_frame_count} frames, {prev_runtime:.1f}s runtime"`
  - `"Recovery time: {recovery_time*1000:.1f}ms"`
  - `"✓ Recovery time requirement met (<2s)"` or `"✗ Recovery time exceeded 2s: {recovery_time:.2f}s"`
- ✅ State restore failure: `"Failed to restore system state: {e}"`
- ✅ State save success: `"System state saved to {state_file}"`
- ✅ State save failure: `"Failed to save system state: {e}"`
- ✅ Periodic save failure (DEBUG): `"Periodic state save failed: {e}"`

**Log Levels:**
- INFO: State operations and recovery metrics
- WARNING: State restore failures, old state detection
- DEBUG: Periodic save failures

### 6. System Shutdown (`stop`, `_close_resources`)

**Current Logging:**
- ✅ Shutdown initiation: `"Stopping SENTINEL system..."`
- ✅ Processing thread wait: `"Waiting for processing thread to finish..."`
- ✅ Recording stop: `"Stopping active recording..."`
- ✅ Scenario export: `"Final scenario exported to: {scenario_path}"`
- ✅ Camera stop: `"Stopping camera capture..."`
- ✅ Visualization stop: `"Stopping visualization server..."`
- ✅ Successful shutdown: `"SENTINEL system stopped successfully"`
- ✅ Shutdown errors: `"Error during shutdown: {e}"` with full traceback
- ✅ Resource cleanup: `"GPU cache cleared"`, `"All resources closed"`
- ✅ Resource cleanup warnings: `"Error closing resources: {e}"`

**Log Levels:**
- INFO: Normal shutdown operations
- WARNING: Resource cleanup issues
- ERROR: Shutdown errors with tracebacks

### 7. Final Statistics (`_log_final_statistics`)

**Current Logging:**
- ✅ Comprehensive final report including:
  - Total frames processed
  - Total runtime
  - Average FPS
  - Per-module average and P95 latencies
  - CPU usage (average and max)
  - Memory usage (average and max)
  - GPU memory usage (average and max)
  - Requirement validation:
    - `"FPS ≥ 30: {'✓ PASS' if fps_ok else '✗ FAIL'} ({avg_fps:.2f})"`
    - `"Latency < 100ms (p95): {'✓ PASS' if latency_ok else '✗ FAIL'} ({p95_total:.2f}ms)"`
    - `"CPU ≤ 60%: {'✓ PASS' if cpu_ok else '✗ FAIL'} ({avg_cpu:.1f}%)"`
    - `"GPU memory ≤ 8GB: {'✓ PASS' if gpu_ok else '✗ FAIL'} ({max_gpu_gb:.2f}GB)"`

**Log Levels:**
- INFO: All final statistics and requirement validation

### 8. Main Function (`main`)

**Current Logging:**
- ✅ System banner: `"SENTINEL System v{version}"`
- ✅ Configuration loaded: `"Configuration loaded from: {args.config}"`
- ✅ Log level: `"Log level: {log_level}"`
- ✅ Configuration errors: `"ERROR: Invalid configuration file"`, `"ERROR: {e}"`, `"ERROR: Failed to load configuration: {e}"`
- ✅ Initialization failure: `"Failed to initialize SENTINEL system: {e}"`
- ✅ Signal handling: `"Received signal {signum}, initiating shutdown..."`
- ✅ System ready: `"System ready. Press Ctrl+C to stop."`
- ✅ Keyboard interrupt: `"Keyboard interrupt received"`
- ✅ System errors: `"System error: {e}"` with full traceback

**Log Levels:**
- INFO: Normal operations and status messages
- ERROR: Configuration and system errors

## Performance Considerations

### Logging Overhead Minimization
1. **Periodic Logging**: Frame-by-frame logging only every 30 frames (~1 second)
2. **Performance Metrics**: Logged every 10 seconds to avoid overhead
3. **DEBUG Level**: Used for non-critical errors (visualization streaming)
4. **Conditional Logging**: Statistics only logged when data is available

### Timing Tracking
- All module latencies tracked in milliseconds
- Separate performance logger for detailed timing analysis
- P95 latency calculations for requirement validation

### Resource Monitoring
- CPU, memory, and GPU usage tracked continuously
- History limited to last 60 samples (1 minute) to prevent memory growth
- Periodic cleanup of old data

## Integration with Other Modules

The main orchestrator logs interactions with:
- **Camera Manager**: Start/stop, frame bundle retrieval
- **BEV Generator**: Frame transformation timing
- **Semantic Segmentor**: Segmentation timing
- **Object Detector**: Detection timing
- **Driver Monitor**: DMS analysis timing
- **Contextual Intelligence**: Risk assessment timing
- **Alert System**: Alert generation timing
- **Scenario Recorder**: Recording triggers and exports
- **Visualization Server**: Streaming operations

## Requirement Compliance

### Real-Time Performance (30+ FPS, <100ms latency)
- ✅ Periodic logging (not per-frame) minimizes overhead
- ✅ Performance metrics logged to separate file
- ✅ Module latencies tracked for bottleneck identification
- ✅ P95 latency validation against 100ms requirement

### System Reliability
- ✅ Comprehensive error logging with tracebacks
- ✅ State persistence for crash recovery
- ✅ Graceful degradation logging
- ✅ Resource cleanup tracking

### Operational Monitoring
- ✅ FPS and throughput tracking
- ✅ CPU/memory/GPU usage monitoring
- ✅ Per-module performance breakdown
- ✅ Requirement validation reporting

## Example Log Output

### Startup
```
INFO - sentinel.system - Initializing SENTINEL system modules...
INFO - sentinel.system - Initializing Camera Manager...
INFO - sentinel.system - Initializing BEV Generator...
INFO - sentinel.system - Initializing Semantic Segmentor...
INFO - sentinel.system - Initializing Object Detector...
INFO - sentinel.system - Initializing Driver Monitoring System...
INFO - sentinel.system - Initializing Contextual Intelligence Engine...
INFO - sentinel.system - Initializing Alert System...
INFO - sentinel.system - Initializing Scenario Recorder...
INFO - sentinel.system - Initializing Visualization Server...
INFO - sentinel.system - All modules initialized successfully
INFO - sentinel.system - Starting SENTINEL system...
INFO - sentinel.system - Starting camera capture...
INFO - sentinel.system - Starting visualization server...
INFO - sentinel.system - SENTINEL system started successfully
INFO - sentinel.performance - Starting performance monitoring...
INFO - sentinel.system - Starting main processing loop...
```

### Runtime Performance Metrics
```
INFO - sentinel.performance - ============================================================
INFO - sentinel.performance - PERFORMANCE METRICS
INFO - sentinel.performance - ============================================================
INFO - sentinel.performance - FPS: 31.25
INFO - sentinel.performance - Frames processed: 312
INFO - sentinel.performance - CPU usage: 45.3%
INFO - sentinel.performance - Memory usage: 2847.2 MB
INFO - sentinel.performance - GPU memory: 3421.8 MB
INFO - sentinel.performance - Module latencies (avg):
INFO - sentinel.performance -   alerts: 3.12ms
INFO - sentinel.performance -   bev: 12.45ms
INFO - sentinel.performance -   camera: 4.23ms
INFO - sentinel.performance -   detection: 18.67ms
INFO - sentinel.performance -   dms: 22.34ms
INFO - sentinel.performance -   intelligence: 8.91ms
INFO - sentinel.performance -   recording: 2.45ms
INFO - sentinel.performance -   segmentation: 13.78ms
INFO - sentinel.performance -   visualization: 1.23ms
INFO - sentinel.performance - Total pipeline latency: 87.18ms
INFO - sentinel.performance - ============================================================
```

### Shutdown and Final Statistics
```
INFO - sentinel.system - Stopping SENTINEL system...
INFO - sentinel.system - Waiting for processing thread to finish...
INFO - sentinel.system - Stopping camera capture...
INFO - sentinel.system - Stopping visualization server...
INFO - sentinel.system - System state saved to state/system_state.pkl
INFO - sentinel.system - GPU cache cleared
INFO - sentinel.system - All resources closed
INFO - sentinel.system - ============================================================
INFO - sentinel.system - FINAL STATISTICS
INFO - sentinel.system - ============================================================
INFO - sentinel.system - Total frames processed: 1247
INFO - sentinel.system - Total runtime: 41.23 seconds
INFO - sentinel.system - Average FPS: 30.25
INFO - sentinel.system - camera: avg=4.15ms, p95=5.23ms
INFO - sentinel.system - bev: avg=12.67ms, p95=14.89ms
INFO - sentinel.system - segmentation: avg=13.45ms, p95=15.12ms
INFO - sentinel.system - detection: avg=18.23ms, p95=21.45ms
INFO - sentinel.system - dms: avg=22.78ms, p95=25.34ms
INFO - sentinel.system - intelligence: avg=9.12ms, p95=11.23ms
INFO - sentinel.system - alerts: avg=3.45ms, p95=4.67ms
INFO - sentinel.system - recording: avg=2.34ms, p95=3.12ms
INFO - sentinel.system - visualization: avg=1.45ms, p95=2.01ms
INFO - sentinel.system - CPU usage: avg=47.2%, max=58.9%
INFO - sentinel.system - Memory: avg=2834.5MB, max=3012.7MB
INFO - sentinel.system - GPU memory: avg=3398.2MB, max=3654.1MB
INFO - sentinel.system - ============================================================
INFO - sentinel.system - REQUIREMENT VALIDATION
INFO - sentinel.system - ============================================================
INFO - sentinel.system - FPS ≥ 30: ✓ PASS (30.25)
INFO - sentinel.system - Latency < 100ms (p95): ✓ PASS (92.34ms)
INFO - sentinel.system - CPU ≤ 60%: ✓ PASS (47.2%)
INFO - sentinel.system - GPU memory ≤ 8GB: ✓ PASS (3.57GB)
INFO - sentinel.system - SENTINEL system stopped successfully
```

## Conclusion

The `src/main.py` module has **comprehensive logging already implemented** that covers:
- ✅ All initialization and startup operations
- ✅ Real-time performance monitoring with minimal overhead
- ✅ Detailed error tracking with full tracebacks
- ✅ State management and crash recovery
- ✅ Graceful shutdown procedures
- ✅ Final statistics and requirement validation
- ✅ Integration with all system modules

The logging is designed to support real-time performance requirements (30+ FPS, <100ms latency) while providing sufficient detail for debugging, monitoring, and requirement validation.

**No additional logging changes are needed for this module.**
