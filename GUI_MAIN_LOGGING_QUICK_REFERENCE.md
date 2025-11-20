# GUI Main Entry Point Logging - Quick Reference

## Module: `src/gui_main.py`

PyQt6 GUI application entry point with comprehensive logging.

## Logger Configuration

```yaml
# configs/logging.yaml
src.gui_main:
  level: INFO
  handlers: [file_all]
  propagate: false
```

## Key Logging Points

### 1. Application Startup
```python
logger.info("SENTINEL GUI Application - Starting")
```

### 2. Configuration Loading
```python
logger.debug(f"Configuration loaded: duration={duration*1000:.2f}ms")
logger.info("Configuration validated successfully")
```

### 3. Qt Application Initialization
```python
logger.debug(f"QApplication created: duration={duration*1000:.2f}ms")
logger.info("Qt application initialized: name=SENTINEL, domain=sentinel-safety.com")
```

### 4. Theme Manager
```python
logger.debug(f"Theme manager initialized: duration={duration*1000:.2f}ms")
logger.info("Theme manager created successfully")
```

### 5. Main Window
```python
logger.debug(f"Main window created: duration={duration*1000:.2f}ms")
logger.info("Main window displayed successfully")
```

### 6. Startup Complete
```python
logger.info(f"SENTINEL GUI Application started successfully: total_duration={duration*1000:.2f}ms")
```

### 7. Event Loop
```python
logger.debug("Entering Qt event loop...")
logger.info(f"Qt event loop exited: exit_code={exit_code}")
```

## Error Logging

All errors include exception tracebacks:

```python
logger.error(f"Configuration loading failed: {e}", exc_info=True)
logger.error(f"QApplication creation failed: {e}", exc_info=True)
logger.error(f"Theme manager initialization failed: {e}", exc_info=True)
logger.error(f"Main window creation failed: {e}", exc_info=True)
logger.error(f"Qt event loop error: {e}", exc_info=True)
```

## Performance Timing

Tracked initialization steps:
- Configuration loading
- QApplication creation
- Theme manager initialization
- Main window creation
- Total startup time

All logged in milliseconds with 2 decimal precision.

## Verification

```bash
# Verify logging setup
python scripts/verify_gui_main_logging.py
```

Expected output:
```
✓ PASS: Logging Configuration
✓ PASS: Module Imports
✓ PASS: Logging Statements
✓ ALL CHECKS PASSED
```

## Log Levels

- **DEBUG**: Initialization steps, timing details, event loop entry
- **INFO**: Major milestones, successful operations, startup summary
- **WARNING**: Non-critical failures (e.g., DPI scaling)
- **ERROR**: Critical failures with tracebacks

## Typical Startup Log Sequence

```
INFO - ============================================================
INFO - SENTINEL GUI Application - Starting
INFO - ============================================================
DEBUG - Loading configuration from configs/default.yaml...
DEBUG - Configuration loaded: duration=15.23ms
INFO - Configuration validated successfully
DEBUG - Configuring high DPI scaling...
DEBUG - High DPI scaling configured: policy=PassThrough
DEBUG - Creating QApplication instance...
DEBUG - QApplication created: duration=45.67ms
INFO - Qt application initialized: name=SENTINEL, domain=sentinel-safety.com
DEBUG - Initializing theme manager...
DEBUG - Theme manager initialized: duration=12.34ms
INFO - Theme manager created successfully
DEBUG - Creating main window...
DEBUG - Main window created: duration=234.56ms
DEBUG - Showing main window...
INFO - Main window displayed successfully
INFO - ============================================================
INFO - SENTINEL GUI Application started successfully: total_duration=312.45ms
INFO - ============================================================
DEBUG - Entering Qt event loop...
```

## Common Issues

### PyQt6 Not Installed
```
ERROR - QApplication creation failed: No module named 'PyQt6'
```
**Solution**: Install PyQt6: `pip install PyQt6`

### Configuration File Missing
```
ERROR - Configuration loading failed: file not found - configs/default.yaml
```
**Solution**: Ensure `configs/default.yaml` exists

### Theme Loading Error
```
ERROR - Theme manager initialization failed: ...
```
**Solution**: Check theme files in `src/gui/themes/`

## Integration Points

- **Logging System**: Uses `LoggerSetup.setup()` from `src/core/logging.py`
- **Configuration**: Loads from `configs/default.yaml` via `ConfigManager`
- **Theme Manager**: Initializes from `src/gui/themes/theme_manager.py`
- **Main Window**: Creates `SENTINELMainWindow` from `src/gui/main_window.py`

## Performance Targets

- Configuration loading: < 50ms
- QApplication creation: < 100ms
- Theme manager init: < 50ms
- Main window creation: < 500ms
- **Total startup: < 1000ms**

Actual timing is logged for monitoring and optimization.
