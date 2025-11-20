# Visualization Server Logging Implementation Summary

## Overview
Comprehensive logging has been implemented for `src/visualization/backend/server.py` to monitor the FastAPI backend server that provides real-time data streaming and REST API endpoints for the SENTINEL visualization dashboard.

## Logger Configuration

### Logger Name
- **Primary Logger**: `VisualizationBackend`
- **Module Loggers**: 
  - `src.visualization.backend.server`
  - `src.visualization.backend.streaming`
  - `src.visualization.backend.data_serializer`

### Log Level
- **Default**: INFO (production)
- **Debug Mode**: DEBUG (development/troubleshooting)

### Log Handlers
- Console output (INFO level)
- File output to `logs/sentinel.log` (DEBUG level)
- Error-only output to `logs/errors.log` (ERROR level)

## Logging Implementation Details

### 1. ConnectionManager Class

#### WebSocket Connection Events
```python
# Connection established
logger.info(f"WebSocket client connected: client={websocket.client}, total_connections={len(self.active_connections)}")

# Connection closed
logger.info(f"WebSocket client disconnected: client={websocket.client}, remaining_connections={len(self.active_connections)}")
```

#### Broadcast Operations
```python
# Successful broadcast
logger.debug(f"Broadcast completed successfully: clients={success_count}")

# Broadcast with failures
logger.warning(f"Broadcast completed with failures: success={success_count}, failed={len(disconnected)}, remaining_connections={len(self.active_connections)}")

# Individual client failure
logger.error(f"Broadcast failed to client: client={connection.client}, error={e}")
```

### 2. VisualizationServer Class

#### Initialization
```python
logger.debug("Initializing VisualizationServer")
logger.debug("Configuring CORS middleware")
logger.debug("Registering API routes")
logger.info(f"Frontend static files mounted: path={frontend_path}")
logger.info(f"VisualizationServer initialized: scenarios_path={self.scenarios_path}")
```

#### REST API Endpoints

**Configuration Management**
```python
# GET /api/config
logger.debug("Configuration requested via API")

# POST /api/config
logger.debug(f"Configuration update requested: keys={list(config_update.keys())}")
logger.info(f"Configuration updated successfully: updates={config_update}")
logger.error(f"Configuration update failed: error={e}", exc_info=True)
```

**Scenario Management**
```python
# GET /api/scenarios
logger.debug(f"Listing scenarios from: path={self.scenarios_path}")
logger.debug(f"Scenarios listed: count={len(scenarios)}")

# GET /api/scenarios/{scenario_id}
logger.debug(f"Scenario details requested: scenario_id={scenario_id}")
logger.debug(f"Scenario metadata loaded: scenario_id={scenario_id}")
logger.debug(f"Scenario annotations loaded: scenario_id={scenario_id}, frame_count={len(annotations.get('frames', []))}")

# DELETE /api/scenarios/{scenario_id}
logger.info(f"Scenario deletion requested: scenario_id={scenario_id}")
logger.info(f"Scenario deleted successfully: scenario_id={scenario_id}, path={scenario_dir}")
logger.warning(f"Scenario not found for deletion: scenario_id={scenario_id}, path={scenario_dir}")
```

**WebSocket Streaming**
```python
# Initial data sent
logger.debug(f"Sending initial data to new client: client={websocket.client}")

# Client messages
logger.debug(f"Ping received from client: client={websocket.client}")
logger.debug(f"Client message received: client={websocket.client}, type={message.get('type', 'unknown')}")

# Disconnection
logger.debug(f"WebSocket disconnected normally: client={websocket.client}")
logger.error(f"WebSocket error occurred: client={websocket.client}, error={e}", exc_info=True)
```

#### Data Streaming with Performance Monitoring
```python
# Normal streaming
logger.debug(f"Data streamed: duration={duration_ms:.2f}ms, clients={len(self.connection_manager.active_connections)}")

# Slow streaming (> 5ms)
logger.warning(f"Data streaming slow: duration={duration_ms:.2f}ms, clients={len(self.connection_manager.active_connections)}")
```

#### Server Lifecycle
```python
# Server start
logger.info(f"Starting visualization server: host={host}, port={port}")

# Server stop
logger.info("Visualization server stopped normally")

# Server error
logger.error(f"Visualization server error: error={e}", exc_info=True)
```

## Log Message Patterns

### Standard Format
All log messages follow the pattern: `"Action completed/failed: key=value, key2=value2"`

### Key Principles
1. **Past tense** for completed actions
2. **Structured data** with key=value pairs
3. **Relevant context** (client info, timing, counts)
4. **Concise but informative**

## Performance Considerations

### Real-Time Constraints
- **Target**: Streaming latency < 5ms
- **Monitoring**: Automatic warning if streaming exceeds 5ms
- **Debug logging**: Minimal overhead, only when enabled

### Log Levels by Operation
- **DEBUG**: Routine operations, detailed flow, performance metrics
- **INFO**: State changes, connections, configuration updates
- **WARNING**: Performance degradation, missing resources, partial failures
- **ERROR**: Operation failures, exceptions with stack traces

## Integration with SENTINEL System

### Visualization Pipeline
```
System Data → stream_data() → ConnectionManager.broadcast() → WebSocket Clients
     ↓              ↓                      ↓                          ↓
  [DEBUG]       [DEBUG/WARN]           [DEBUG/ERROR]              [DEBUG]
```

### API Request Flow
```
HTTP Request → FastAPI Route → Business Logic → Response
     ↓              ↓                ↓              ↓
  [DEBUG]       [DEBUG/INFO]    [INFO/ERROR]   [DEBUG]
```

## Monitoring and Troubleshooting

### Key Metrics to Monitor
1. **Connection count**: Track active WebSocket connections
2. **Broadcast duration**: Monitor streaming performance
3. **Failed broadcasts**: Identify problematic clients
4. **API errors**: Track configuration and scenario management issues

### Common Issues and Log Patterns

**Slow Streaming**
```
WARNING - Data streaming slow: duration=12.34ms, clients=5
```
Action: Check network conditions, reduce data payload, or optimize serialization

**Client Connection Issues**
```
ERROR - Broadcast failed to client: client=('127.0.0.1', 54321), error=...
WARNING - Broadcast completed with failures: success=4, failed=1, remaining_connections=4
```
Action: Client may have disconnected abruptly, connection cleaned up automatically

**Scenario Not Found**
```
WARNING - Scenario not found: scenario_id=20241115_103045, path=scenarios/20241115_103045
```
Action: Verify scenario exists or was not deleted

## Configuration in logging.yaml

```yaml
loggers:
  VisualizationBackend:
    level: INFO
    handlers: [file_all]
    propagate: false
  
  src.visualization.backend.server:
    level: INFO
    handlers: [file_all]
    propagate: false
  
  src.visualization.backend.streaming:
    level: INFO
    handlers: [file_all]
    propagate: false
  
  src.visualization.backend.data_serializer:
    level: INFO
    handlers: [file_all]
    propagate: false
```

## Testing Logging

### Verify Logging Setup
```bash
# Run visualization backend tests
pytest tests/test_visualization_backend.py -v

# Check log output
tail -f logs/sentinel.log | grep VisualizationBackend
```

### Example Log Output
```
2024-11-15 13:45:23 - VisualizationBackend - INFO - VisualizationServer initialized: scenarios_path=scenarios/
2024-11-15 13:45:30 - VisualizationBackend - INFO - WebSocket client connected: client=('127.0.0.1', 54321), total_connections=1
2024-11-15 13:45:30 - VisualizationBackend - DEBUG - Sending initial data to new client: client=('127.0.0.1', 54321)
2024-11-15 13:45:31 - VisualizationBackend - DEBUG - Data streamed: duration=2.34ms, clients=1
2024-11-15 13:45:32 - VisualizationBackend - DEBUG - Ping received from client: client=('127.0.0.1', 54321)
2024-11-15 13:45:45 - VisualizationBackend - DEBUG - Configuration requested via API
2024-11-15 13:46:00 - VisualizationBackend - INFO - Scenario deletion requested: scenario_id=20241115_103045
2024-11-15 13:46:00 - VisualizationBackend - INFO - Scenario deleted successfully: scenario_id=20241115_103045, path=scenarios/20241115_103045
```

## Summary

The visualization server logging implementation provides:
- ✅ Comprehensive coverage of all major operations
- ✅ Performance monitoring with automatic warnings
- ✅ Structured log messages with relevant context
- ✅ Appropriate log levels for different scenarios
- ✅ Integration with SENTINEL's centralized logging system
- ✅ Minimal performance overhead in production (INFO level)
- ✅ Detailed debugging capability when needed (DEBUG level)

This logging setup enables effective monitoring, troubleshooting, and performance analysis of the visualization backend server while maintaining SENTINEL's real-time performance requirements.
