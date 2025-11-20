# Performance Monitoring Dock Widget

## Overview

The Performance Monitoring Dock provides comprehensive real-time visualization and logging of SENTINEL system performance metrics. It helps operators and engineers monitor system health, identify bottlenecks, and ensure performance requirements are met.

## Features

### 1. FPS Graph Tab
Monitors frame rate to ensure the system maintains 30+ FPS throughput.

**Displays:**
- Real-time FPS over last 60 seconds
- 30 FPS target line (dashed orange)
- Current FPS with color coding (green ≥30, red <30)

**Use Case:** Verify the system is processing frames at the required rate.

### 2. Latency Graph Tab
Tracks end-to-end processing latency from camera capture to alert generation.

**Displays:**
- Real-time latency over last 60 seconds
- 100ms threshold line (dashed red)
- Current latency
- P95 latency (95th percentile)
- Violation count (samples >100ms)

**Use Case:** Ensure latency stays under 100ms at P95 for timely alerts.

### 3. Module Breakdown Tab
Shows time distribution across pipeline stages to identify bottlenecks.

**Displays:**
- Stacked bar chart of module timings
- Color-coded modules (Camera, BEV, Segmentation, Detection, DMS, Intelligence, Alerts)
- Total pipeline time with color coding
- Tooltips with exact timing values

**Use Case:** Identify which pipeline stage is consuming the most time.

### 4. Resource Usage Tab
Monitors GPU and CPU resource consumption.

**Displays:**
- GPU memory gauge (max 8GB)
  - Green: 0-70%
  - Yellow: 70-85%
  - Red: 85-100%
- CPU usage gauge (max 60%)
  - Green: 0-48%
  - Yellow: 48-60%
  - Red: 60-100%
- Current and peak values for both

**Use Case:** Ensure resource usage stays within operational limits.

### 5. Performance Logging Tab
Records performance metrics to file for offline analysis.

**Features:**
- Start/Stop logging button
- Automatic log file creation in `logs/performance/`
- CSV format: timestamp, fps, latency_ms, gpu_memory_mb, cpu_percent
- Export performance report (text summary)
- Export raw data (CSV)
- Performance summary with statistics

**Use Case:** Collect data for performance analysis, debugging, and optimization.

## Usage

### Basic Integration

```python
from PyQt6.QtWidgets import QDockWidget
from PyQt6.QtCore import Qt
from gui.widgets import PerformanceDockWidget

# Create dock widget
perf_dock = QDockWidget("Performance Monitor", main_window)
perf_widget = PerformanceDockWidget()
perf_dock.setWidget(perf_widget)
main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, perf_dock)

# Start monitoring
perf_widget.start_monitoring()
```

### Updating Metrics

```python
# Update individual metrics
perf_widget.update_fps(32.5)
perf_widget.update_latency(85.3)
perf_widget.update_module_timings({
    'Camera': 5.2,
    'BEV': 15.1,
    'Segmentation': 14.8,
    'Detection': 20.3,
    'DMS': 25.1,
    'Intelligence': 10.2,
    'Alerts': 4.5
})
perf_widget.update_resources(4096.0, 45.2)

# Or update all at once
perf_widget.update_all_metrics(
    fps=32.5,
    latency_ms=85.3,
    module_timings=timings_dict,
    gpu_memory_mb=4096.0,
    cpu_percent=45.2
)
```

### Controlling Monitoring

```python
# Start monitoring (enables 1 Hz update timer)
perf_widget.start_monitoring()

# Stop monitoring
perf_widget.stop_monitoring()

# Clear all data
perf_widget.clear_all()
```

## Performance Targets

The widget uses these targets based on SENTINEL requirements:

| Metric | Target | Threshold |
|--------|--------|-----------|
| FPS | ≥30 FPS | 30 FPS |
| Latency (P95) | ≤100ms | 100ms |
| GPU Memory | ≤8GB | 8192 MB |
| CPU Usage | ≤60% | 60% |

Color coding indicates status:
- **Green:** Within target
- **Yellow:** Approaching limit
- **Red:** Exceeding limit

## Log File Format

Performance logs are saved in CSV format:

```csv
timestamp,fps,latency_ms,gpu_memory_mb,cpu_percent
2024-11-16T10:30:45.123456,32.5,85.3,4096.0,45.2
2024-11-16T10:30:46.123456,31.8,87.1,4102.5,46.1
...
```

Log files are automatically named with timestamp: `perf_YYYYMMDD_HHMMSS.log`

## Performance Summary

The summary tab calculates and displays:

**FPS Statistics:**
- Mean, Median, Min, Max, Standard Deviation

**Latency Statistics:**
- Mean, Median, P95, P99, Min, Max

**Performance Assessment:**
- PASS/FAIL against FPS target (≥30 FPS)
- PASS/FAIL against latency target (≤100ms at P95)

## Tips

1. **Monitor During Development:** Keep the performance dock open while developing to catch performance regressions early.

2. **Use Logging for Benchmarks:** Enable logging when running performance tests to collect data for analysis.

3. **Check Module Breakdown:** If latency is high, check the module breakdown to identify the bottleneck.

4. **Watch Resource Usage:** If GPU or CPU usage is consistently high, consider optimization or hardware upgrades.

5. **Export Reports:** Use the export feature to share performance data with the team or include in documentation.

## Troubleshooting

**Issue:** Graphs not updating
- **Solution:** Ensure `start_monitoring()` has been called

**Issue:** High latency spikes
- **Solution:** Check module breakdown to identify which stage is slow

**Issue:** Logging not working
- **Solution:** Check that `logs/performance/` directory is writable

**Issue:** Memory usage growing
- **Solution:** The widget automatically limits history to last 60 points (FPS/latency) or 1000 points (logging), so memory usage should be stable

## API Reference

### PerformanceDockWidget

#### Methods

```python
def start_monitoring() -> None:
    """Start performance monitoring with 1 Hz update timer"""

def stop_monitoring() -> None:
    """Stop performance monitoring"""

def clear_all() -> None:
    """Clear all performance data from all tabs"""

def update_fps(fps: float) -> None:
    """Update FPS display"""

def update_latency(latency_ms: float) -> None:
    """Update latency display"""

def update_module_timings(timings: Dict[str, float]) -> None:
    """Update module breakdown. Keys: module names, Values: time in ms"""

def update_resources(gpu_memory_mb: float, cpu_percent: float) -> None:
    """Update resource usage displays"""

def update_all_metrics(fps: float, latency_ms: float, 
                      module_timings: Dict[str, float],
                      gpu_memory_mb: float, cpu_percent: float) -> None:
    """Update all metrics at once (more efficient)"""
```

## Dependencies

- PyQt6 >= 6.5.0
- PyQtGraph >= 0.13.0
- NumPy >= 1.24.0

## Related Components

- `CircularGaugeWidget` - Used for resource usage gauges
- `SENTINELMainWindow` - Main window that hosts the dock
- Performance monitoring in `src/main.py` - Collects actual metrics

## Future Enhancements

Planned improvements:
- Configurable time windows
- Alert thresholds with notifications
- Historical data playback
- Session comparison
- Network performance metrics
- Custom metric plugins
