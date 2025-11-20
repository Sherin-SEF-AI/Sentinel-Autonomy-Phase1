# Analytics Module Logging Summary

## Overview

The analytics module logging has been configured to track trip analytics, behavior reporting, risk heatmap generation, dashboard operations, and report exports.

## Logging Configuration

### Logger Hierarchy

```
src.analytics (INFO)
├── src.analytics.trip_analytics (INFO)
├── src.analytics.behavior_report (INFO)
├── src.analytics.risk_heatmap (INFO)
├── src.analytics.analytics_dashboard (INFO)
└── src.analytics.report_exporter (INFO)
```

### Log Files

- **logs/sentinel.log**: All analytics logs with detailed formatting
- **logs/errors.log**: Error-level logs only
- **Console**: INFO level and above

## Module Components

### 1. TripAnalytics (`src/analytics/trip_analytics.py`)

**Purpose**: Analyzes trip data and generates statistics

**Key Logging Points**:
- Trip start/end events
- Segment analysis completion
- Statistics calculation
- Performance metrics

**Example Logs**:
```
INFO - src.analytics.trip_analytics - Trip started: trip_id=abc123
INFO - src.analytics.trip_analytics - Segment analyzed: duration=120.5s, distance=2.3km
INFO - src.analytics.trip_analytics - Trip completed: total_distance=15.2km, duration=25.5min
DEBUG - src.analytics.trip_analytics - Statistics calculated: avg_speed=35.7km/h, max_risk=0.65
```

### 2. BehaviorReportGenerator (`src/analytics/behavior_report.py`)

**Purpose**: Generates driver behavior reports

**Key Logging Points**:
- Report generation start/completion
- Behavior pattern detection
- Safety score calculation
- Recommendation generation

**Example Logs**:
```
INFO - src.analytics.behavior_report - Report generation started: driver_id=driver_001
INFO - src.analytics.behavior_report - Behavior patterns analyzed: aggressive_events=3, cautious_score=0.75
INFO - src.analytics.behavior_report - Safety score calculated: score=82/100
INFO - src.analytics.behavior_report - Report generated: recommendations=5, duration=0.15s
```

### 3. RiskHeatmap (`src/analytics/risk_heatmap.py`)

**Purpose**: Creates spatial risk heatmaps

**Key Logging Points**:
- Heatmap generation start/completion
- Grid cell processing
- Risk aggregation
- Visualization rendering

**Example Logs**:
```
INFO - src.analytics.risk_heatmap - Heatmap generation started: grid_size=50x50
DEBUG - src.analytics.risk_heatmap - Grid cells processed: count=2500, high_risk_cells=45
INFO - src.analytics.risk_heatmap - Risk aggregated: max_risk=0.92, avg_risk=0.23
INFO - src.analytics.risk_heatmap - Heatmap generated: duration=0.08s
```

### 4. AnalyticsDashboard (`src/analytics/analytics_dashboard.py`)

**Purpose**: PyQt6 dashboard for analytics visualization

**Key Logging Points**:
- Dashboard initialization
- Data updates
- Chart rendering
- User interactions

**Example Logs**:
```
INFO - src.analytics.analytics_dashboard - Dashboard initialized: widgets=8
DEBUG - src.analytics.analytics_dashboard - Data updated: trips=15, total_distance=250.5km
INFO - src.analytics.analytics_dashboard - Charts rendered: duration=0.05s
DEBUG - src.analytics.analytics_dashboard - User interaction: action=export_report, format=pdf
```

### 5. ReportExporter (`src/analytics/report_exporter.py`)

**Purpose**: Exports analytics reports in various formats

**Key Logging Points**:
- Export start/completion
- Format conversion
- File writing
- Error handling

**Example Logs**:
```
INFO - src.analytics.report_exporter - Export started: format=pdf, report_type=behavior
DEBUG - src.analytics.report_exporter - Data serialized: size=1.2MB
INFO - src.analytics.report_exporter - Report exported: path=reports/behavior_20241118.pdf, duration=0.25s
ERROR - src.analytics.report_exporter - Export failed: format=csv, error=permission_denied
```

## Configuration in logging.yaml

```yaml
# Analytics Module
src.analytics:
  level: INFO
  handlers: [file_all]
  propagate: false

src.analytics.trip_analytics:
  level: INFO
  handlers: [file_all]
  propagate: false

src.analytics.behavior_report:
  level: INFO
  handlers: [file_all]
  propagate: false

src.analytics.risk_heatmap:
  level: INFO
  handlers: [file_all]
  propagate: false

src.analytics.analytics_dashboard:
  level: INFO
  handlers: [file_all]
  propagate: false

src.analytics.report_exporter:
  level: INFO
  handlers: [file_all]
  propagate: false
```

## Usage Examples

### Basic Logging

```python
import logging
from analytics import TripAnalytics

logger = logging.getLogger(__name__)

# Initialize trip analytics
trip_analytics = TripAnalytics()
logger.info("TripAnalytics initialized")

# Start trip
trip_analytics.start_trip()
logger.info(f"Trip started: trip_id={trip_analytics.current_trip_id}")

# End trip
summary = trip_analytics.end_trip()
logger.info(f"Trip completed: distance={summary.total_distance:.2f}km, "
           f"duration={summary.duration:.1f}s")
```

### Performance Logging

```python
import time
import logging

logger = logging.getLogger('src.analytics.risk_heatmap')

start_time = time.time()
heatmap = risk_heatmap.generate()
duration = time.time() - start_time

logger.debug(f"Heatmap generated: duration={duration*1000:.2f}ms, "
            f"cells={heatmap.grid_size[0]*heatmap.grid_size[1]}")
```

### Error Logging

```python
import logging

logger = logging.getLogger('src.analytics.report_exporter')

try:
    exporter.export_report(report, format='pdf')
    logger.info(f"Report exported successfully: path={output_path}")
except PermissionError as e:
    logger.error(f"Export failed - permission denied: path={output_path}, error={e}")
except Exception as e:
    logger.error(f"Export failed: error={e}", exc_info=True)
```

## Verification

Run the verification script to test logging setup:

```bash
python scripts/verify_analytics_logging.py
```

Expected output:
```
============================================================
ANALYTICS MODULE LOGGING VERIFICATION
============================================================

1. Testing analytics module import...
   ✓ Analytics module imported successfully

2. Testing individual component imports...
   ✓ TripAnalytics imported successfully
   ✓ BehaviorReportGenerator imported successfully
   ✓ RiskHeatmap imported successfully
   ✓ AnalyticsDashboard imported successfully
   ✓ ReportExporter imported successfully

3. Testing logger instances...
   ✓ Logger 'src.analytics' exists
   ✓ Logger 'src.analytics.trip_analytics' exists
   ✓ Logger 'src.analytics.behavior_report' exists
   ✓ Logger 'src.analytics.risk_heatmap' exists
   ✓ Logger 'src.analytics.analytics_dashboard' exists
   ✓ Logger 'src.analytics.report_exporter' exists

4. Testing logging output...
   ✓ Logging output test completed

5. Testing component-specific logging...
   ✓ TripAnalytics logging works
   ✓ BehaviorReportGenerator logging works
   ✓ RiskHeatmap logging works
   ✓ ReportExporter logging works

============================================================
VERIFICATION COMPLETE
============================================================
```

## Performance Considerations

The analytics module is not in the real-time critical path (30+ FPS, <100ms latency), so:

- **Log Level**: INFO is appropriate for production
- **Frequency**: Analytics operations are periodic (trip summaries, reports)
- **Impact**: Minimal impact on system performance
- **Debug Mode**: Can use DEBUG level for detailed analysis without affecting real-time processing

## Best Practices

1. **Trip Events**: Log trip start/end with trip_id for traceability
2. **Performance**: Log duration for report generation and export operations
3. **Errors**: Always log export failures with full context
4. **Statistics**: Log key metrics (distance, duration, safety scores)
5. **User Actions**: Log dashboard interactions for UX analysis

## Integration with Main System

The analytics module integrates with:
- **Recording Module**: Analyzes recorded trip data
- **Intelligence Module**: Uses risk assessments for heatmaps
- **Profiling Module**: Generates behavior reports from driver profiles
- **GUI**: Provides dashboard visualization

All analytics operations are logged to the same log files as the main system for unified monitoring.

## Troubleshooting

### No Logs Appearing

1. Check log level in `configs/logging.yaml`
2. Verify logger name matches module path
3. Ensure log directory exists and is writable

### Too Many Logs

1. Increase log level from DEBUG to INFO
2. Reduce logging frequency in tight loops
3. Use log rotation settings in logging.yaml

### Missing Component Logs

1. Verify component has `logger = logging.getLogger(__name__)`
2. Check logger configuration in logging.yaml
3. Ensure component is properly imported

## Summary

✅ **Logging Setup Complete**
- All analytics components have logging configured
- Logger hierarchy established
- Configuration added to logging.yaml
- Verification script created
- Documentation complete

The analytics module logging is now fully integrated with the SENTINEL system logging infrastructure.
