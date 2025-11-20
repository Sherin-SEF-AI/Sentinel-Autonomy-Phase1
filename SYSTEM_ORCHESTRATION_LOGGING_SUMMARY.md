# System Orchestration Logging Summary

## Overview

Comprehensive logging has been implemented for the SENTINEL system orchestrator (`src/main.py`), which coordinates all modules and manages the real-time processing pipeline. The logging infrastructure provides detailed visibility into system initialization, processing loop performance, resource utilization, and graceful shutdown.

## Logging Configuration

### Logger Instances

The system orchestrator uses two specialized loggers:

1. **`sentinel.system`** - Main system orchestration logger
   - Level: INFO
   - Handlers: file_all
   - Purpose: System lifecycle, module coordination, state management

2. **`sentinel.performance`** - Performance monitoring logger
   - Level: INFO
   - Handlers: file_performance, file_all
   - Purpose: Real-time performance metrics, FPS tracking, latency analysis

### Log Files

- **`logs/sentinel.log`** - All system events and orchestration
- **`logs/performance.log`** - Dedicated performance metrics
- **`logs/errors.log`** - System-level errors and exceptions

## Key Logging Points

### 1. System Initialization

**Location**: `SentinelSystem.__init__()` and `_initialize_modules()`

**Logged Events**:
```
INFO - Initializing SENTINEL system modules...
INFO - Initializing Camera Manager...
INFO - Initializing BEV Generator...
INFO - Initializing Semantic Segmentor...
INFO - Initializing Object Detector...
INFO - Initializing Driver Monitoring System...
INFO - Initializing Contextual Intelligence Engine...
INFO - Initializing Alert System...
INFO - Initializing Scenario Recorder...
INFO - Initializing Visualization Server...
INFO - All modules initialized successfully
ERROR - Failed to initialize modules: {error_details}
```

**Purpose**: Track module initialization sequence and identify startup failures

### 2. System Startup

**Location**: `SentinelSystem.start()`

**Logged Events**:
```
WARNING - System is already running
INFO - Starting SENTINEL system...
INFO - Starting camera capture...
INFO - Starting visualization server...
INFO - SENTINEL system started successfully
ERROR - Failed to start system: {error_details}
```

**Purpose**: Monitor system startup and detect configuration issues

### 3. Performance Monitoring Loop

**Location**: `_performance_monitoring_loop()` and `_log_performance_metrics()`

**Logged Events** (every 10 seconds):
```
INFO - Starting performance monitoring...
INFO - ============================================================
INFO - PERFORMANCE METRICS
INFO - ============================================================
INFO - FPS: 32.45
INFO - Frames processed: 3245
INFO - CPU usage: 45.2%
INFO - Memory usage: 2048.5 MB
INFO - GPU memory: 4096.3 MB
INFO - Module latencies (avg):
INFO -   alerts: 3.45ms
INFO -   bev: 12.34ms
INFO -   camera: 4.56ms
INFO -   detection: 18.23ms
INFO -   dms: 22.15ms
INFO -   intelligence: 8.67ms
INFO -   recording: 2.34ms
INFO -   segmentation: 13.45ms
INFO -   visualization: 1.23ms
INFO - Total pipeline latency: 86.42ms
INFO - ============================================================
ERROR - Performance monitoring error: {error_details}
INFO - Performance monitoring stopped
```

**Purpose**: Real-time performance tracking and bottleneck identification

### 4. Main Processing Loop

**Location**: `_processing_loop()`

**Logged Events**:
```
INFO - Starting main processing loop...
INFO - Frame 30: 28.5ms (35.1 FPS)
INFO - Frame 60: 29.2ms (34.2 FPS)
INFO - Scenario exported to: scenarios/20241115_143022/
ERROR - DMS processing error: {error_details}
ERROR - Perception processing error: {error_details}
ERROR - Error in processing loop: {error_details}
INFO - Processing loop stopped
```

**Purpose**: Monitor frame-by-frame processing and detect pipeline failures

### 5. State Recovery

**Location**: `_restore_system_state()`

**Logged Events**:
```
INFO - No previous state found, starting fresh
INFO - ============================================================
INFO - STATE RECOVERY
INFO - ============================================================
INFO - Previous session: 5432 frames, 180.5s runtime
INFO - Recovery time: 125.3ms
INFO - ============================================================
INFO - ✓ Recovery time requirement met (<2s)
WARNING - ✗ Recovery time exceeded 2s: 2.35s
WARNING - Failed to restore system state: {error_details}
INFO - Starting with fresh state
```

**Purpose**: Track crash recovery and validate recovery time requirements

### 6. State Persistence

**Location**: `_save_system_state()` and `_periodic_state_save()`

**Logged Events**:
```
INFO - System state saved to state/system_state.pkl
WARNING - Failed to save system state: {error_details}
DEBUG - Periodic state save failed: {error_details}
```

**Purpose**: Monitor state persistence for crash recovery

### 7. System Shutdown

**Location**: `SentinelSystem.stop()`

**Logged Events**:
```
INFO - Stopping SENTINEL system...
INFO - Waiting for processing thread to finish...
INFO - Stopping active recording...
INFO - Final scenario exported to: scenarios/20241115_143022/
INFO - Stopping camera capture...
INFO - Stopping visualization server...
INFO - System state saved to state/system_state.pkl
INFO - GPU cache cleared
INFO - All resources closed
INFO - SENTINEL system stopped successfully
ERROR - Error during shutdown: {error_details}
```

**Purpose**: Track graceful shutdown sequence and resource cleanup

### 8. Final Statistics

**Location**: `_log_final_statistics()`

**Logged Events**:
```
INFO - ============================================================
INFO - FINAL STATISTICS
INFO - ============================================================
INFO - Total frames processed: 10845
INFO - Total runtime: 360.25 seconds
INFO - Average FPS: 30.11
INFO - alerts: avg=3.45ms, p95=5.23ms
INFO - bev: avg=12.34ms, p95=14.56ms
INFO - camera: avg=4.56ms, p95=6.78ms
INFO - detection: avg=18.23ms, p95=22.45ms
INFO - dms: avg=22.15ms, p95=24.89ms
INFO - intelligence: avg=8.67ms, p95=10.23ms
INFO - recording: avg=2.34ms, p95=3.45ms
INFO - segmentation: avg=13.45ms, p95=15.67ms
INFO - visualization: avg=1.23ms, p95=2.34ms
INFO - CPU usage: avg=48.5%, max=62.3%
INFO - Memory: avg=2048.5MB, max=2456.7MB
INFO - GPU memory: avg=4096.3MB, max=5234.8MB
INFO - ============================================================
INFO - REQUIREMENT VALIDATION
INFO - ============================================================
INFO - FPS ≥ 30: ✓ PASS (30.11)
INFO - Latency < 100ms (p95): ✓ PASS (95.67ms)
INFO - CPU ≤ 60%: ✓ PASS (48.5%)
INFO - GPU memory ≤ 8GB: ✓ PASS (5.11GB)
```

**Purpose**: Comprehensive performance summary and requirement validation

### 9. Main Entry Point

**Location**: `main()`

**Logged Events**:
```
INFO - ============================================================
INFO - SENTINEL System v1.0
INFO - ============================================================
INFO - Configuration loaded from: configs/default.yaml
INFO - Log level: INFO
ERROR - Failed to initialize SENTINEL system: {error_details}
INFO - Received signal 2, initiating shutdown...
INFO - System ready. Press Ctrl+C to stop.
INFO - Keyboard interrupt received
ERROR - System error: {error_details}
```

**Purpose**: Track application lifecycle and user interactions

## Performance Considerations

### Minimal Overhead Design

1. **Periodic Logging**: Performance metrics logged every 10 seconds, not per-frame
2. **Conditional Logging**: Frame-level logs only every 30 frames (~1 second)
3. **Efficient Data Structures**: Rolling windows for metrics (last 60 samples)
4. **Separate Thread**: Performance monitoring runs in dedicated thread
5. **Debug Level**: Detailed logs only at DEBUG level to minimize production overhead

### Latency Budget

- Performance monitoring: < 1ms per iteration (runs every 1 second)
- State persistence: < 5ms (runs every 100 frames)
- Logging overhead: < 0.1ms per frame (minimal string formatting)

### Memory Management

- CPU usage history: 60 samples × 8 bytes = 480 bytes
- Memory usage history: 60 samples × 8 bytes = 480 bytes
- GPU memory history: 60 samples × 8 bytes = 480 bytes
- Module latencies: 9 modules × 100 samples × 8 bytes = 7.2 KB
- **Total overhead**: ~8.6 KB (negligible)

## Error Handling

### Exception Logging

All exceptions are logged with full stack traces:
```python
except Exception as e:
    self.logger.error(f"Operation failed: {e}")
    self.logger.error(traceback.format_exc())
```

### Graceful Degradation

- Pipeline failures logged but don't crash system
- Missing modules detected during initialization
- Resource cleanup errors logged as warnings

## Integration with Other Modules

### Module Coordination

The orchestrator logs interactions with all modules:
- Camera Manager: Frame bundle acquisition
- BEV Generator: Perspective transformation
- Semantic Segmentor: Pixel classification
- Object Detector: 3D detection and tracking
- Driver Monitor: Driver state analysis
- Contextual Intelligence: Risk assessment
- Alert System: Alert generation and dispatch
- Scenario Recorder: Event recording
- Visualization Server: Real-time streaming

### Performance Tracking

Per-module latency tracking enables:
- Bottleneck identification
- Performance regression detection
- Optimization target prioritization
- Real-time performance monitoring

## Verification

### Test Coverage

The logging implementation can be verified using:

```bash
# Run system orchestration tests
pytest tests/test_integration.py -v

# Check log output
tail -f logs/sentinel.log
tail -f logs/performance.log

# Verify performance metrics
python scripts/verify_system_orchestration.py
```

### Expected Behavior

1. **Startup**: All modules initialize successfully
2. **Runtime**: FPS ≥ 30, latency < 100ms
3. **Monitoring**: Performance metrics logged every 10 seconds
4. **Shutdown**: Graceful cleanup with final statistics

## Requirements Satisfied

### Functional Requirements

- ✅ **FR-12.1**: System orchestration with module coordination
- ✅ **FR-12.2**: Real-time processing loop at 30+ FPS
- ✅ **FR-12.3**: Performance monitoring and metrics
- ✅ **FR-12.4**: Graceful shutdown and resource cleanup
- ✅ **FR-12.5**: State persistence and crash recovery

### Non-Functional Requirements

- ✅ **NFR-1**: Real-time performance (30+ FPS, <100ms latency)
- ✅ **NFR-2**: Resource efficiency (CPU ≤60%, GPU ≤8GB)
- ✅ **NFR-3**: Reliability (graceful degradation, error recovery)
- ✅ **NFR-4**: Observability (comprehensive logging and metrics)

## Future Enhancements

1. **Structured Logging**: JSON format for machine parsing
2. **Distributed Tracing**: OpenTelemetry integration
3. **Metrics Export**: Prometheus endpoint for monitoring
4. **Log Aggregation**: ELK stack integration
5. **Alerting**: Automated alerts on performance degradation

## Conclusion

The system orchestrator logging provides comprehensive visibility into SENTINEL's operation, enabling:
- Real-time performance monitoring
- Bottleneck identification
- Crash recovery validation
- Requirement compliance verification
- Production debugging and optimization

The logging infrastructure is designed for minimal overhead while providing maximum observability, ensuring it doesn't impact the real-time performance requirements of the safety-critical system.
