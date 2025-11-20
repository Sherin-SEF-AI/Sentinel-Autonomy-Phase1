# Analytics Module - Quick Reference

## Overview
The analytics module provides trip tracking, risk heatmaps, behavior reports, and data export for SENTINEL.

## Quick Start

### 1. Trip Analytics

```python
from src.analytics import TripAnalytics

analytics = TripAnalytics(config)

# Start trip
trip_id = analytics.start_trip(driver_id="driver_001")

# Update during trip
analytics.update(
    timestamp=time.time(),
    position=(x, y),
    speed=speed_mps,
    risk_score=risk,
    alerts=alert_list
)

# End trip
summary = analytics.end_trip()
print(f"Safety Score: {summary.safety_score:.1f}")
```

### 2. Risk Heatmap

```python
from src.analytics import RiskHeatmap

heatmap = RiskHeatmap(config)

# Add risk points
heatmap.add_risk_point((x, y), risk_score=0.8, radius=5.0)

# Get visualization
colored = heatmap.get_heatmap_colored(colormap='hot')

# Export
heatmap.export_heatmap_image('heatmap.png')
```

### 3. Export Data

```python
from src.analytics import ReportExporter

exporter = ReportExporter(config)

# Export trips
exporter.export_trips_csv(trip_summaries)

# Export driver profile
exporter.export_driver_profile_csv(profile)

# Export complete bundle
exporter.export_analytics_bundle(profile, trips, heatmap)
```

### 4. Generate Reports

```python
from src.analytics import BehaviorReportGenerator

generator = BehaviorReportGenerator(config)

# PDF report
generator.generate_pdf_report(profile, trips, 'report.pdf')

# Excel report
generator.generate_excel_report(profile, trips, 'report.xlsx')
```

### 5. Analytics Dashboard (GUI)

```python
from src.analytics import AnalyticsDashboard

dashboard = AnalyticsDashboard(config)

# Update data
dashboard.update_trip_data(trip_summaries)
dashboard.update_driver_profile(driver_id, profile)
dashboard.update_fleet_stats(fleet_stats)

# Connect export
dashboard.export_requested.connect(handle_export)
```

## Configuration

```yaml
analytics:
  segment_duration: 60.0
  high_risk_threshold: 0.7
  heatmap:
    grid_size: 2.0
    max_range: 100.0
    decay_factor: 0.95
  output_dir: "reports/"
```

## Key Classes

### TripAnalytics
- `start_trip(driver_id)` → trip_id
- `update(timestamp, position, speed, risk_score, alerts)`
- `end_trip()` → TripSummary
- `get_statistics()` → Dict

### RiskHeatmap
- `add_risk_point(position, risk_score, radius)`
- `get_heatmap(normalize)` → ndarray
- `get_heatmap_colored(colormap)` → RGB image
- `export_heatmap_image(filepath, colormap)`

### ReportExporter
- `export_trips_csv(trips, filepath)`
- `export_driver_profile_csv(profile, filepath)`
- `export_visualization_png(image, filepath)`
- `export_analytics_bundle(profile, trips, heatmap)`

### BehaviorReportGenerator
- `generate_pdf_report(profile, trips, output_path)`
- `generate_excel_report(profile, trips, output_path)`

### AnalyticsDashboard (PyQt6)
- `update_trip_data(trip_summaries)`
- `update_driver_profile(driver_id, profile)`
- `update_fleet_stats(fleet_stats)`
- Signal: `export_requested(str)`

## Data Structures

### TripSummary
```python
@dataclass
class TripSummary:
    trip_id: str
    start_time: datetime
    end_time: datetime
    duration: float  # seconds
    distance: float  # meters
    avg_speed: float  # m/s
    max_speed: float  # m/s
    safety_score: float  # 0-100
    alert_counts: Dict[str, int]
    high_risk_segments: List[TripSegment]
    driver_id: Optional[str]
```

### TripSegment
```python
@dataclass
class TripSegment:
    start_time: float
    end_time: float
    start_position: Tuple[float, float]
    end_position: Tuple[float, float]
    distance: float
    avg_speed: float
    max_risk: float
    alert_count: int
    critical_alert_count: int
```

## Dependencies

### Required
- numpy
- PyQt6 (for dashboard)

### Optional
- reportlab (PDF reports)
- matplotlib (charts)
- pandas + openpyxl (Excel)
- opencv-python (images)

## Common Patterns

### Complete Trip Analysis
```python
# Initialize
analytics = TripAnalytics(config)
heatmap = RiskHeatmap(config)
exporter = ReportExporter(config)

# Track trip
trip_id = analytics.start_trip("driver_001")
for frame in trip_frames:
    analytics.update(frame.timestamp, frame.position, 
                    frame.speed, frame.risk, frame.alerts)
    heatmap.add_risk_point(frame.position, frame.risk)

# Generate results
summary = analytics.end_trip()
exporter.export_trips_csv([summary])
heatmap.export_heatmap_image('heatmap.png')
```

### Generate Driver Report
```python
# Get data
trips = analytics.get_driver_trips("driver_001", limit=50)
profile = profile_manager.get_profile("driver_001")

# Generate report
generator = BehaviorReportGenerator(config)
generator.generate_pdf_report(profile, trips, 'report.pdf')
```

### GUI Integration
```python
# Create dashboard
analytics_dock = QDockWidget("Analytics", self)
dashboard = AnalyticsDashboard(config)
analytics_dock.setWidget(dashboard)
self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, analytics_dock)

# Connect signals
worker.trip_completed.connect(
    lambda s: dashboard.update_trip_data([s])
)
```

## Export Formats

- **CSV**: Trip data, driver profiles, heatmap data
- **PDF**: Professional reports with charts
- **Excel**: Multi-sheet workbooks
- **PNG**: Heatmaps and visualizations

Check available formats:
```python
formats = exporter.get_export_formats()
```

## Performance Tips

1. **Trip Analytics**: Updates are O(1), very fast
2. **Risk Heatmap**: Use appropriate grid size for your needs
3. **Report Generation**: Run PDF/Excel generation async
4. **Dashboard**: 5-second refresh is optimal
5. **Heatmap Decay**: Apply periodically to manage memory

## Troubleshooting

### PDF Generation Fails
```bash
pip install reportlab matplotlib
```

### Excel Export Fails
```bash
pip install pandas openpyxl
```

### Image Export Fails
```bash
pip install opencv-python
```

### Import Errors
Make sure you're importing from the correct path:
```python
from src.analytics import TripAnalytics  # Correct
```

## Examples

See `examples/analytics_example.py` for a complete demonstration.

Run example:
```bash
python examples/analytics_example.py
```

## Documentation

Full documentation: `src/analytics/README.md`
