# Analytics Module Logging - Quick Reference

## Quick Start

```python
import logging
from analytics import TripAnalytics, BehaviorReportGenerator, RiskHeatmap

# Get logger
logger = logging.getLogger(__name__)

# Log analytics operations
logger.info("Analytics operation started")
```

## Logger Names

| Component | Logger Name |
|-----------|-------------|
| Module | `src.analytics` |
| Trip Analytics | `src.analytics.trip_analytics` |
| Behavior Report | `src.analytics.behavior_report` |
| Risk Heatmap | `src.analytics.risk_heatmap` |
| Dashboard | `src.analytics.analytics_dashboard` |
| Report Exporter | `src.analytics.report_exporter` |

## Common Logging Patterns

### Trip Analytics

```python
# Trip start
logger.info(f"Trip started: trip_id={trip_id}, timestamp={timestamp}")

# Segment analysis
logger.debug(f"Segment analyzed: duration={duration:.1f}s, distance={distance:.2f}km")

# Trip completion
logger.info(f"Trip completed: total_distance={dist:.2f}km, avg_speed={speed:.1f}km/h")
```

### Behavior Report

```python
# Report generation
logger.info(f"Report generation started: driver_id={driver_id}")

# Safety score
logger.info(f"Safety score calculated: score={score}/100, grade={grade}")

# Report completion
logger.info(f"Report generated: recommendations={count}, duration={duration:.2f}s")
```

### Risk Heatmap

```python
# Heatmap generation
logger.info(f"Heatmap generation started: grid_size={width}x{height}")

# Risk aggregation
logger.debug(f"Risk aggregated: max_risk={max_risk:.2f}, avg_risk={avg_risk:.2f}")

# Completion
logger.info(f"Heatmap generated: high_risk_cells={count}, duration={duration:.3f}s")
```

### Report Export

```python
# Export start
logger.info(f"Export started: format={format}, report_type={type}")

# Success
logger.info(f"Report exported: path={path}, size={size_mb:.2f}MB")

# Error
logger.error(f"Export failed: format={format}, error={error}")
```

## Log Levels

- **DEBUG**: Detailed analytics calculations, grid processing
- **INFO**: Trip events, report generation, exports (default)
- **WARNING**: Unusual patterns, missing data
- **ERROR**: Export failures, calculation errors

## Verification

```bash
# Test logging setup
python3 scripts/verify_analytics_logging.py

# Check logs
tail -f logs/sentinel.log | grep analytics
```

## Performance Notes

- Analytics is **not** in the real-time critical path
- INFO level logging has minimal performance impact
- Use DEBUG for detailed analysis without affecting system FPS
- Report generation typically takes 50-250ms

## Key Metrics to Log

1. **Trip Statistics**: Distance, duration, average speed
2. **Safety Scores**: Overall score, component scores
3. **Risk Metrics**: Max risk, average risk, high-risk events
4. **Export Operations**: Format, file size, duration
5. **Dashboard Updates**: Data refresh rate, chart rendering time

## Integration Points

- **Recording Module**: Trip data source
- **Intelligence Module**: Risk assessment data
- **Profiling Module**: Driver behavior data
- **GUI**: Dashboard visualization

All analytics logs appear in `logs/sentinel.log` alongside main system logs.
