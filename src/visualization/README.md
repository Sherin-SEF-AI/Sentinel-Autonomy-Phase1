# Visualization Module

Complete visualization system for SENTINEL with real-time dashboard and scenario playback.

## Overview

The visualization module provides:
- **Real-time Dashboard**: Live view of BEV, detections, driver state, and risks at 30 Hz
- **Scenario Playback**: Frame-by-frame review of recorded scenarios with annotations
- **REST API**: Configuration management and scenario access
- **WebSocket Streaming**: Real-time data streaming to connected clients

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Visualization System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Backend (Python/FastAPI)                                    │
│  ├── Server                                                  │
│  │   ├── REST API endpoints                                 │
│  │   ├── WebSocket streaming                                │
│  │   └── Static file serving                                │
│  ├── Data Serializer                                         │
│  │   ├── Image encoding (JPEG base64)                       │
│  │   ├── Data structure conversion                          │
│  │   └── JSON serialization                                 │
│  └── Streaming Manager                                       │
│      ├── Performance monitoring                              │
│      ├── 30 Hz data streaming                               │
│      └── Client connection management                        │
│                                                               │
│  Frontend (HTML/JavaScript)                                  │
│  ├── Live Dashboard (index.html)                            │
│  │   ├── BEV visualization                                  │
│  │   ├── Driver state panel                                 │
│  │   ├── Risk assessment panel                              │
│  │   └── Performance metrics                                │
│  └── Playback Interface (playback.html)                     │
│      ├── Scenario browser                                   │
│      ├── Frame-by-frame controls                            │
│      ├── Timeline scrubbing                                 │
│      └── Annotation overlay                                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Backend

#### VisualizationServer (`backend/server.py`)
FastAPI server with:
- REST endpoints for configuration and scenarios
- WebSocket endpoint for real-time streaming
- Static file serving for frontend
- Connection management for multiple clients

#### Data Serializer (`backend/data_serializer.py`)
Converts SENTINEL data structures to JSON:
- Base64 image encoding for efficient transmission
- Numpy array to list conversion
- Dataclass to dictionary serialization
- Colored segmentation overlays

#### Streaming Manager (`backend/streaming.py`)
Manages real-time data flow:
- 30 Hz streaming to connected clients
- Performance monitoring (FPS, latency)
- Data caching and updates
- Integration with SENTINEL system

### Frontend

#### Live Dashboard (`frontend/index.html`, `frontend/app.js`)
Real-time visualization:
- Bird's Eye View with detections
- Driver state metrics (readiness, attention, drowsiness)
- Top-3 risk assessment
- Performance graphs
- Multi-modal alerts

#### Playback Interface (`frontend/playback.html`, `frontend/playback.js`)
Scenario review:
- Scenario browser with metadata
- Frame-by-frame navigation
- Timeline scrubbing
- Playback speed control (0.25x - 2x)
- Annotation overlay toggle

## Usage

### Starting the Server

```python
from src.visualization.backend import create_server

# Load configuration
config = {...}

# Create and run server
server = create_server(config)
server.run(host="0.0.0.0", port=8080)
```

### Integrating with SENTINEL

```python
from src.visualization.backend import create_streaming_manager, StreamingIntegration

# Create server and streaming manager
server = create_server(config)
streaming_manager = create_streaming_manager(server, target_fps=30)
integration = StreamingIntegration(streaming_manager)

# In main processing loop
async def process_frame(frame_bundle):
    # Process frame...
    bev = bev_generator.generate(...)
    detections = detector.detect(...)
    driver_state = dms.analyze(...)
    risk_assessment = intelligence.assess(...)
    alerts = alert_system.process(...)
    
    # Push to visualization
    integration.push_frame_data(
        timestamp=frame_bundle.timestamp,
        bev=bev,
        detections=detections,
        driver_state=driver_state,
        risk_assessment=risk_assessment,
        alerts=alerts,
        latencies={...}
    )
    
    # Stream to clients
    await integration.stream_current_frame(frame_bundle.timestamp)
```

### Accessing the Dashboard

1. **Live View**: `http://localhost:8080/`
2. **Playback**: `http://localhost:8080/playback.html`
3. **API Docs**: `http://localhost:8080/docs`

## API Endpoints

### REST API

```
GET  /                          - Live dashboard (HTML)
GET  /playback.html             - Playback interface (HTML)
GET  /api/config                - Get system configuration
POST /api/config                - Update configuration
GET  /api/scenarios             - List recorded scenarios
GET  /api/scenarios/{id}        - Get scenario details
DEL  /api/scenarios/{id}        - Delete scenario
GET  /api/status                - System status
```

### WebSocket

```
WS   /ws/stream                 - Real-time data stream (30 Hz)
```

## Data Format

### WebSocket Stream Message

```json
{
  "type": "frame_data",
  "timestamp": 1.0,
  "server_time": "2024-11-15T10:30:45.123Z",
  "bev": {
    "timestamp": 1.0,
    "image": "base64_encoded_jpeg...",
    "mask": [[true, true, ...], ...]
  },
  "segmentation": {
    "timestamp": 1.0,
    "class_map": [[1, 1, 2, ...], ...],
    "confidence": [[0.95, 0.92, ...], ...],
    "overlay": "base64_encoded_jpeg..."
  },
  "detections": [
    {
      "bbox_3d": {"x": 10.0, "y": 2.0, "z": 0.0, ...},
      "class_name": "vehicle",
      "confidence": 0.95,
      "velocity": {"vx": 5.0, "vy": 0.0, "vz": 0.0},
      "track_id": 1
    }
  ],
  "driver_state": {
    "face_detected": true,
    "readiness_score": 85.0,
    "head_pose": {...},
    "gaze": {...},
    "drowsiness": {...},
    "distraction": {...}
  },
  "risk_assessment": {
    "top_risks": [
      {
        "hazard": {...},
        "contextual_score": 0.65,
        "driver_aware": true,
        "urgency": "medium"
      }
    ]
  },
  "alerts": [],
  "performance": {
    "fps": 30.5,
    "latency": {...},
    "cpu_percent": 45.2,
    "gpu_memory_mb": 3456
  }
}
```

## Features

### Live Dashboard

**Main View**
- Switch between BEV, camera feeds, and 3D scene
- Real-time object detection overlays
- Color-coded object classes
- Track ID persistence

**Driver State Panel**
- Readiness score with color-coded bar (0-100)
- Current attention zone (8 zones around vehicle)
- Drowsiness level (PERCLOS-based)
- Distraction type and duration

**Risk Assessment Panel**
- Top 3 contextual risks
- Urgency levels (critical, high, medium, low)
- Time-to-collision (TTC)
- Driver awareness indicator

**Performance Panel**
- Real-time FPS
- Per-module latency breakdown
- CPU and GPU utilization
- Total end-to-end latency

**Alert System**
- Floating alerts with urgency colors
- Auto-dismiss after 5 seconds
- Slide-in animation
- Multi-modal indicators

### Playback Interface

**Scenario Browser**
- List of recorded scenarios
- Timestamp and duration
- Trigger type
- Click to load

**Playback Controls**
- Play/Pause
- Previous/Next frame
- Timeline scrubbing
- Speed control (0.25x, 0.5x, 1x, 2x)
- Annotation toggle

**Info Panel**
- Trigger type
- Duration
- Total detections
- Maximum risk score

**Annotations**
- 3D bounding boxes
- Object labels and track IDs
- Risk indicators
- Driver state metrics

## Performance

### Backend
- **Streaming Rate**: 30 Hz (33ms per frame)
- **Image Encoding**: JPEG quality 85
- **WebSocket Overhead**: ~5-10ms per broadcast
- **Multi-Client**: Supports 10+ simultaneous connections

### Frontend
- **Rendering**: 60 FPS UI updates
- **Canvas**: Hardware-accelerated 2D rendering
- **Image Decoding**: Browser-native JPEG decoding
- **Memory**: ~50MB for dashboard, ~100MB for playback

## Configuration

### Server Configuration

```yaml
visualization:
  enabled: true
  port: 8080
  update_rate: 30  # Hz
```

### Frontend Configuration

Edit `frontend/app.js` to customize:
- WebSocket URL
- Reconnection interval
- Color schemes
- Layout preferences

## Examples

### Example 1: Basic Server

```python
from src.visualization.backend import create_server
from src.core.config import ConfigManager

config_manager = ConfigManager("configs/default.yaml")
server = create_server(config_manager.config)
server.run(host="0.0.0.0", port=8080)
```

See: `examples/visualization_backend_example.py`

### Example 2: Streaming Integration

```python
from src.visualization.backend import create_server, create_streaming_manager, StreamingIntegration

server = create_server(config)
streaming_manager = create_streaming_manager(server, target_fps=30)
integration = StreamingIntegration(streaming_manager)

# Push data
integration.push_frame_data(
    timestamp=1.0,
    bev=bev_output,
    detections=detections,
    driver_state=driver_state,
    risk_assessment=risk_assessment,
    alerts=alerts
)

# Stream
await integration.stream_current_frame(1.0)
```

See: `examples/visualization_streaming_example.py`

## Testing

Run visualization tests:

```bash
pytest tests/test_visualization_backend.py -v
```

## Dependencies

- `fastapi>=0.103.0`: Web framework
- `uvicorn[standard]>=0.23.0`: ASGI server
- `websockets>=11.0`: WebSocket support
- `opencv-python>=4.8.0`: Image encoding
- `numpy>=1.24.0`: Array operations

## Browser Requirements

- Modern browser with WebSocket support
- JavaScript enabled
- Canvas API support
- Recommended: Chrome, Firefox, Edge, Safari

## Troubleshooting

### WebSocket Connection Failed
- Check backend is running on correct port
- Verify firewall allows WebSocket connections
- Check browser console for errors

### Images Not Displaying
- Verify base64 encoding is correct
- Check JPEG quality settings
- Ensure CORS headers are set

### Performance Issues
- Reduce streaming frame rate
- Lower JPEG quality
- Use binary WebSocket protocol
- Enable compression

### Playback Not Working
- Verify scenarios exist in storage path
- Check scenario JSON format
- Ensure annotations are complete

## Security

For production deployment:

1. **HTTPS**: Use TLS/SSL encryption
2. **Authentication**: Add JWT or API key auth
3. **CORS**: Restrict origins to specific domains
4. **Rate Limiting**: Prevent WebSocket abuse
5. **Input Validation**: Sanitize all inputs

## Future Enhancements

- [ ] Three.js 3D scene visualization
- [ ] Attention heatmap overlay
- [ ] Historical data graphs
- [ ] Video playback (MP4 streaming)
- [ ] Multi-vehicle dashboard
- [ ] Mobile-responsive layout
- [ ] Binary WebSocket protocol
- [ ] Compression for reduced bandwidth
- [ ] Recording export (screenshots, videos)
- [ ] Configuration editor UI
