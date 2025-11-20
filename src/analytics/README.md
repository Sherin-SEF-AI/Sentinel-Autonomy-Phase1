# Analytics and Reporting Module

The analytics module provides comprehensive trip analytics, risk heatmaps, driver behavior reports, and data export functionality for the SENTINEL system.

## Components

### 1. TripAnalytics

Tracks and analyzes trip metrics in real-time.

**Features:**
- Track trip duration, distance, and average speed
- Calculate trip safety scores
- Count alerts by type (info, warning, critical)
- Identify high-risk segments
- Maintain trip history

**Usage:**
```python
from src.analytics import TripAnalytics

analytics = TripAnalytics(config)

# Start a trip
trip_id = analytics.start_trip(driver_id="driver_001")

# Update during trip
analytics.update(
    timestamp=time.time(),
    position=(x, y),
    speed=speed_mps,
    risk_score=risk,
    alerts=alert_list
)

# End trip and get summary
summary = analytics.end_trip()
print(f"Safety Score: {summary.safety_score}")
print(f"High-risk segments: {len(summary.high_risk_segments)}")
```

### 2. RiskHeatmap

Generates spatial risk heatmaps by aggregating risk scores by location.

**Features:**
- Aggregate risk by location with configurable grid size
- Generate colored heatmap visualizations
- Identify high-risk locations
- Export heatmap images
- Overlay on map images
- Temporal decay for historical data

**Usage:**
```python
from src.analytics import RiskHeatmap

heatmap = RiskHeatmap(config)

# Add risk points
heatmap.add_risk_point(position=(x, y), risk_score=0.8, radius=5.0)

# Add trajectory risk
heatmap.add_trajectory_risk(trajectory, risk_scores)

# Get heatmap
heatmap_array = heatmap.get_heatmap(normalize=True)
colored_heatmap = heatmap.get_heatmap_colored(colormap='hot')

# Export
heatmap.export_heatmap_image('heatmap.png', colormap='hot')

# Get high-risk locations
high_risk = heatmap.get_high_risk_locations(threshold=0.7)
```

### 3. BehaviorReportGenerator

Generates comprehensive driver behavior reports in PDF and Excel formats.

**Features:**
- Generate PDF reports with charts
- Include safety scores and trends
- Add personalized recommendations
- Export to Excel format
- Automatic chart generation (requires matplotlib)

**Usage:**
```python
from src.analytics import BehaviorReportGenerator

generator = BehaviorReportGenerator(config)

# Generate PDF report
success = generator.generate_pdf_report(
    driver_profile=profile,
    trip_summaries=trips,
    output_path='report.pdf'
)

# Generate Excel report
success = generator.generate_excel_report(
    driver_profile=profile,
    trip_summaries=trips,
    output_path='report.xlsx'
)
```

### 4. ReportExporter

Exports analytics data to various formats.

**Features:**
- Export trips to CSV
- Export driver profiles to CSV
- Export risk heatmaps to CSV
- Export visualizations to PNG
- Export summary reports to PDF/Excel
- Bundle export with all formats

**Usage:**
```python
from src.analytics import ReportExporter

exporter = ReportExporter(config)

# Export trips
exporter.export_trips_csv(trip_summaries, 'trips.csv')

# Export driver profile
exporter.export_driver_profile_csv(profile, 'profile.csv')

# Export visualization
exporter.export_visualization_png(image, 'viz.png')

# Export complete bundle
bundle_dir = exporter.export_analytics_bundle(
    driver_profile=profile,
    trip_summaries=trips,
    heatmap=heatmap_array
)
```

### 5. AnalyticsDashboard

PyQt6 widget for displaying analytics in the GUI.

**Features:**
- Display trip statistics
- Show driver performance metrics
- Plot trends over time
- Compare against fleet averages
- Export functionality integration

**Usage:**
```python
from src.analytics import AnalyticsDashboard

dashboard = AnalyticsDashboard(config)

# Update with trip data
dashboard.update_trip_data(trip_summaries)

# Update driver profile
dashboard.update_driver_profile(driver_id, profile)

# Update fleet statistics
dashboard.update_fleet_stats(fleet_stats)

# Connect export signal
dashboard.export_requested.connect(handle_export)
```

## Configuration

Add analytics configuration to `configs/default.yaml`:

```yaml
analytics:
  # Trip analytics
  segment_duration: 60.0  # seconds
  high_risk_threshold: 0.7
  
  # Risk heatmap
  heatmap:
    grid_size: 2.0  # meters per cell
    max_range: 100.0  # meters
    decay_factor: 0.95  # temporal decay
  
  # Output
  output_dir: "reports/"
```

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
    alert_counts: Dict[str, int]  # by urgency
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
- reportlab (for PDF reports)
- matplotlib (for charts)
- pandas (for Excel export)
- openpyxl (for Excel export)
- opencv-python (for image export)

Install optional dependencies:
```bash
pip install reportlab matplotlib pandas openpyxl opencv-python
```

## Examples

### Complete Trip Analysis

```python
from src.analytics import TripAnalytics, RiskHeatmap, ReportExporter

# Initialize
analytics = TripAnalytics(config)
heatmap = RiskHeatmap(config)
exporter = ReportExporter(config)

# Start trip
trip_id = analytics.start_trip(driver_id="driver_001")

# During trip
for frame in trip_frames:
    # Update analytics
    analytics.update(
        timestamp=frame.timestamp,
        position=frame.position,
        speed=frame.speed,
        risk_score=frame.risk_score,
        alerts=frame.alerts
    )
    
    # Update heatmap
    heatmap.add_risk_point(frame.position, frame.risk_score)

# End trip
summary = analytics.end_trip()

# Export results
exporter.export_trips_csv([summary])
heatmap.export_heatmap_image('trip_heatmap.png')
```

### Generate Driver Report

```python
from src.analytics import TripAnalytics, BehaviorReportGenerator

analytics = TripAnalytics(config)
generator = BehaviorReportGenerator(config)

# Get trip history
trips = analytics.get_driver_trips(driver_id="driver_001", limit=50)

# Get driver profile (from profiling module)
from src.profiling import ProfileManager
profile_mgr = ProfileManager(config)
profile = profile_mgr.get_profile(driver_id="driver_001")

# Generate report
generator.generate_pdf_report(
    driver_profile=profile,
    trip_summaries=trips,
    output_path='driver_001_report.pdf'
)
```

## Integration with GUI

The analytics dashboard can be added as a dock widget:

```python
from src.analytics import AnalyticsDashboard

# In main window
self.analytics_dock = QDockWidget("Analytics", self)
self.analytics_dashboard = AnalyticsDashboard(self.config)
self.analytics_dock.setWidget(self.analytics_dashboard)
self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.analytics_dock)

# Connect to data sources
self.sentinel_worker.trip_completed.connect(
    lambda summary: self.analytics_dashboard.update_trip_data([summary])
)
```

## Performance Considerations

- **Trip Analytics**: Minimal overhead, updates in O(1) time
- **Risk Heatmap**: Grid-based aggregation, O(n) for n risk points
- **Report Generation**: PDF/Excel generation can be slow for large datasets
- **Dashboard Updates**: Updates every 5 seconds by default

## Testing

```bash
# Run analytics tests
pytest tests/test_analytics.py -v

# Test specific component
pytest tests/test_analytics.py::test_trip_analytics -v
```

## Logging

The analytics module uses Python's logging framework:

```python
import logging

# Enable debug logging
logging.getLogger('src.analytics').setLevel(logging.DEBUG)
```

## Future Enhancements

- Real-time chart updates in dashboard
- Advanced statistical analysis
- Machine learning-based predictions
- Integration with cloud analytics
- Custom report templates
- Interactive heatmap visualization
