# Task 21: Performance Monitoring Dock - Implementation Summary

## Overview
Implemented a comprehensive performance monitoring dock widget for the SENTINEL GUI that provides real-time visualization and logging of system performance metrics.

## Completed Subtasks

### 21.1 FPS Graph ✓
**Implementation:** `FPSGraphWidget` in `src/gui/widgets/performance_dock.py`

**Features:**
- Real-time FPS plotting using PyQtGraph over last 60 seconds
- 30 FPS target line (dashed orange line)
- Color coding: green when ≥30 FPS, red when <30 FPS
- Auto-scaling Y-axis
- Current FPS display with color-coded label
- Scrolling time axis (seconds ago)

**Requirements Met:** 17.1

### 21.2 Latency Graph ✓
**Implementation:** `LatencyGraphWidget` in `src/gui/widgets/performance_dock.py`

**Features:**
- Real-time latency plotting over last 60 seconds
- 100ms threshold line (dashed red line)
- P95 latency calculation and display
- Violation tracking (count of samples >100ms)
- Color coding: blue for normal, red for violations
- Statistics display: Current, P95, and violation count

**Requirements Met:** 17.2

### 21.3 Module Breakdown ✓
**Implementation:** `ModuleBreakdownWidget` in `src/gui/widgets/performance_dock.py`

**Features:**
- Stacked bar chart showing time spent in each pipeline stage
- Color-coded modules:
  - Camera: #ff6b6b (red)
  - BEV: #4ecdc4 (teal)
  - Segmentation: #45b7d1 (blue)
  - Detection: #96ceb4 (green)
  - DMS: #ffeaa7 (yellow)
  - Intelligence: #dfe6e9 (gray)
  - Alerts: #fd79a8 (pink)
- Total time display with color coding (green <80ms, orange 80-100ms, red >100ms)
- Tooltips with exact timing values
- Updates at 1 Hz

**Requirements Met:** 17.3

### 21.4 Resource Usage Displays ✓
**Implementation:** `ResourceUsageWidget` in `src/gui/widgets/performance_dock.py`

**Features:**
- GPU Memory Gauge:
  - Max threshold: 8GB (8192 MB)
  - Color zones: green (0-70%), yellow (70-85%), red (85-100%)
  - Current and peak value tracking
  - Percentage display
- CPU Usage Gauge:
  - Max threshold: 60%
  - Color zones: green (0-48%), yellow (48-60%), red (60-100%)
  - Current and peak value tracking
- Uses existing CircularGaugeWidget for consistent UI
- Grouped display with QGroupBox

**Requirements Met:** 17.4, 17.5

### 21.5 Performance Logging ✓
**Implementation:** `PerformanceLoggingWidget` in `src/gui/widgets/performance_dock.py`

**Features:**
- Start/Stop logging button with visual state indication
- Automatic log file creation in `logs/performance/` directory
- CSV format logging: timestamp, fps, latency_ms, gpu_memory_mb, cpu_percent
- Entry counter showing number of logged samples
- Export capabilities:
  - Export Performance Report (text file with statistics)
  - Export Raw Data (CSV file)
- Performance Summary:
  - FPS statistics (mean, median, min, max, std)
  - Latency statistics (mean, median, P95, P99, min, max)
  - Performance assessment (PASS/FAIL against targets)
  - Total sample count
- Refresh summary button
- Automatic history management (keeps last 1000 samples in memory)

**Requirements Met:** 17.6

## Main Widget Integration

### PerformanceDockWidget
**Location:** `src/gui/widgets/performance_dock.py`

**Structure:**
- Tabbed interface with 5 tabs:
  1. FPS - Frame rate monitoring
  2. Latency - End-to-end latency tracking
  3. Modules - Pipeline stage breakdown
  4. Resources - GPU/CPU usage
  5. Logging - Performance logging controls

**API Methods:**
```python
# Individual updates
update_fps(fps: float)
update_latency(latency_ms: float)
update_module_timings(timings: Dict[str, float])
update_resources(gpu_memory_mb: float, cpu_percent: float)

# Batch update
update_all_metrics(fps, latency_ms, module_timings, gpu_memory_mb, cpu_percent)

# Control
start_monitoring()  # Start 1 Hz update timer
stop_monitoring()   # Stop updates
clear_all()         # Clear all data
```

**Mock Data Generation:**
- Included for testing purposes
- Simulates realistic performance metrics:
  - FPS: 25-35 (around 30 target)
  - Latency: 60-95ms with 10% chance of spikes (100-130ms)
  - Module timings: Realistic ranges for each stage
  - GPU: 3-7.5 GB usage
  - CPU: 30-70% usage

## Files Created/Modified

### New Files:
1. `src/gui/widgets/performance_dock.py` - Main implementation (500+ lines)
2. `test_performance_dock.py` - Visual test script with instructions
3. `tests/unit/test_performance_dock.py` - Unit tests
4. `.kiro/specs/sentinel-safety-system/TASK_21_SUMMARY.md` - This summary

### Modified Files:
1. `src/gui/widgets/__init__.py` - Added exports for new widgets

## Testing

### Visual Test
Run `python test_performance_dock.py` to launch interactive test window:
- All 5 tabs functional
- Mock data updates every second
- Test logging, export, and all visualizations
- Dark theme applied for better visibility

### Unit Tests
Located in `tests/unit/test_performance_dock.py`:
- Widget initialization tests
- Data update tests
- Clear functionality tests
- Logging functionality tests (with mocks)

## Integration with Main Window

The performance dock can be integrated into the main SENTINEL window as a dockable widget:

```python
from gui.widgets import PerformanceDockWidget

# In SENTINELMainWindow.__init__()
self.perf_dock = QDockWidget("Performance Monitor", self)
self.perf_widget = PerformanceDockWidget()
self.perf_dock.setWidget(self.perf_widget)
self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.perf_dock)

# Add to View menu for show/hide
self.docks_menu.addAction(self.perf_dock.toggleViewAction())

# Start monitoring when system starts
self.perf_widget.start_monitoring()

# Update with real data from SENTINEL system
self.perf_widget.update_all_metrics(
    fps=system_fps,
    latency_ms=system_latency,
    module_timings=system_timings,
    gpu_memory_mb=gpu_usage,
    cpu_percent=cpu_usage
)
```

## Performance Characteristics

- **Update Rate:** 1 Hz (configurable via timer interval)
- **Memory Usage:** Minimal - uses deque with max 60 points for graphs
- **CPU Overhead:** Very low - PyQtGraph is highly optimized
- **Logging Overhead:** Negligible - simple CSV append operations

## Dependencies

- PyQt6 - GUI framework
- PyQtGraph - High-performance plotting
- NumPy - Statistical calculations
- Python standard library (time, collections, logging, os, datetime)

## Future Enhancements

Potential improvements for future tasks:
1. Configurable time windows (30s, 60s, 5min)
2. Alert thresholds with notifications
3. Historical data playback
4. Comparison with previous sessions
5. Network performance metrics
6. Disk I/O monitoring
7. Custom metric plugins
8. Real-time performance profiling integration

## Verification

All subtasks completed and verified:
- ✓ 21.1 FPS graph with target line and color coding
- ✓ 21.2 Latency graph with P95 and violation tracking
- ✓ 21.3 Module breakdown with stacked bar chart
- ✓ 21.4 Resource usage gauges for GPU and CPU
- ✓ 21.5 Performance logging with export capabilities

The performance monitoring dock is fully functional and ready for integration with the SENTINEL system.
