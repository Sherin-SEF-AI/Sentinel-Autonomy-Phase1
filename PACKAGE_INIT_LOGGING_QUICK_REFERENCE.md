# Package Initialization Logging - Quick Reference

## Log Messages

### Success Case
```
INFO - src - SENTINEL package initialized successfully: modules=['SentinelSystem', 'gui_main', 'main']
```

### Failure Case
```
WARNING - src - SENTINEL package initialization incomplete: missing_dependencies=<error>
```

## Configuration

**Logger**: `src`  
**Level**: INFO  
**Handlers**: file_all  
**Location**: `configs/logging.yaml`

## Verification

```bash
python scripts/verify_package_init_logging.py
```

## Common Issues

| Issue | Log Message | Solution |
|-------|-------------|----------|
| Missing PyQt6 | `missing_dependencies=No module named 'PyQt6'` | `pip install PyQt6` |
| Missing torch | `missing_dependencies=No module named 'torch'` | `pip install torch` |
| Missing opencv | `missing_dependencies=No module named 'cv2'` | `pip install opencv-python` |

## Integration

- Logs to: `logs/sentinel_*.log`
- Format: `timestamp - logger - level - message`
- Rotation: 10MB max, 5 backups

## Performance

- **Overhead**: < 1ms (one-time at import)
- **Impact**: None on runtime performance
- **Frequency**: Once per Python session
