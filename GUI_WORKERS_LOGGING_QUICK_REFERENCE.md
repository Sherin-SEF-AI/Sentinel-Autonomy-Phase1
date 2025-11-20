# GUI Workers Logging - Quick Reference

## Logger Names

```python
# Module-level logger
logger = logging.getLogger('src.gui.workers')

# Worker thread logger (used in SentinelWorker)
logger = logging.getLogger('sentinel.worker')
```

## Configuration Location

**File:** `configs/logging.yaml`

```yaml
src.gui.workers:
  level: INFO
  handlers: [file_all]
  propagate: false

sentinel.worker:
  level: INFO
  handlers: [file_all]
  propagate: false
```

## Log Levels

- **INFO** (default): Normal operations, state changes
- **DEBUG**: Detailed diagnostics (enable for troubleshooting)
- **ERROR**: Errors with full tracebacks

## Key Log Messages

### Initialization
```
INFO - sentinel.worker - SentinelWorker initialized
INFO - sentinel.worker - SentinelWorker thread starting...
INFO - sentinel.worker - Initializing system modules...
INFO - sentinel.worker - All modules initialized successfully
```

### Processing
```
INFO - sentinel.worker - SentinelWorker processing loop started
ERROR - sentinel.worker - DMS processing error: <details>
ERROR - sentinel.worker - Perception processing error: <details>
```

### Shutdown
```
INFO - sentinel.worker - Stop requested for SentinelWorker
INFO - sentinel.worker - Processing loop stopped
INFO - sentinel.worker - Cleaning up SentinelWorker resources...
INFO - sentinel.worker - Cleanup complete
INFO - sentinel.worker - SentinelWorker thread stopped
```

## Viewing Logs

```bash
# Real-time monitoring
tail -f logs/sentinel.log | grep "sentinel.worker"

# Check for errors
grep "ERROR.*sentinel.worker" logs/sentinel.log

# View initialization sequence
grep "Initializing" logs/sentinel.log
```

## Enable Debug Logging

Edit `configs/logging.yaml`:

```yaml
sentinel.worker:
  level: DEBUG  # Change from INFO
  handlers: [file_all]
  propagate: false
```

## Performance Impact

- **INFO level:** <0.1ms overhead per log
- **No per-frame logging:** Only periodic and event-based
- **Thread-safe:** Python logging is thread-safe by default
- **Async I/O:** Buffered file handlers

## Integration Example

```python
from src.gui.workers import SentinelWorker

# Create worker
worker = SentinelWorker(config)

# Connect signals
worker.error_occurred.connect(handle_error)
worker.status_changed.connect(update_status)

# Start (logs automatically)
worker.start()

# Stop (logs automatically)
worker.stop()
worker.wait()
```

## Troubleshooting

**No logs appearing?**
- Check `logs/` directory exists
- Verify `configs/logging.yaml` is loaded
- Check file permissions

**Too many logs?**
- Use INFO level (not DEBUG)
- Check for duplicate handlers

**Performance issues?**
- Ensure INFO level (not DEBUG)
- Check disk I/O performance
- Verify log rotation is working

## Log Files

- **Main log:** `logs/sentinel.log` (rotates at 10MB)
- **Error log:** `logs/errors.log` (ERROR level only)
- **Backups:** `logs/sentinel.log.1`, `.2`, etc. (5 backups)

## Summary

✅ Logging configured for GUI workers module  
✅ Thread-safe operation  
✅ Minimal performance impact  
✅ Comprehensive error tracking  
✅ Integration with GUI error handling  

All worker thread operations are logged automatically with no additional code required.
