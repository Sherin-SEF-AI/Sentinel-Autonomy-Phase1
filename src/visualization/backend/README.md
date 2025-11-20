# Visualization Backend

FastAPI-based backend server for SENTINEL visualization dashboard.

## Features

- **WebSocket Streaming**: Real-time data streaming at 30 Hz to connected clients
- **REST API**: Configuration management and scenario playback endpoints
- **Data Serialization**: Efficient conversion of SENTINEL data structures to JSON
- **Multi-Client Support**: Broadcast data to multiple connected WebSocket clients

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  REST Endpoints:                                             │
│  - GET  /                      Health check                  │
│  - GET  /api/config            Get configuration             │
│  - POST /api/config            Update configuration          │
│  - GET  /api/scenarios         List scenarios                │
│  - GET  /api/scenarios/{id}    Get scenario details          │
│  - DEL  /api/scenarios/{id}    Delete scenario               │
│  - GET  /api/status            System status                 │
│                                                               │
│  WebSocket:                                                   │
│  - WS   /ws/stream             Real-time data stream         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### VisualizationServer

Main server class that manages:
- FastAPI application and routes
- WebSocket connection management
- Configuration state
- Scenario storage access

### ConnectionManager

Manages WebSocket connections:
- Accept new connections
- Disconnect clients
- Broadcast messages to all connected clients

### Data Serializer

Converts SENTINEL data structures to JSON:
- `serialize_bev_output()`: BEV images with base64 encoding
- `serialize_segmentation_output()`: Segmentation maps with colored overlays
- `serialize_detection_3d()`: 3D bounding boxes
- `serialize_driver_state()`: Driver monitoring metrics
- `serialize_risk_assessment()`: Risk scores and hazards
- `serialize_alert()`: Alert messages
- `serialize_frame_data()`: Complete frame data bundle

## Usage

### Starting the Server

```python
from src.visualization.backend import create_server
from src.core.config import ConfigManager

# Load configuration
config_manager = ConfigManager("configs/default.yaml")
config = config_manager.config

# Create and run server
server = create_server(config)
server.run(host="0.0.0.0", port=8080)
```

### Streaming Data

```python
import asyncio
from src.visualization.backend import serialize_frame_data

# Serialize frame data
data = serialize_frame_data(
    timestamp=1.0,
    bev=bev_output,
    segmentation=seg_output,
    detections=detections_3d,
    driver_state=driver_state,
    risk_assessment=risk_assessment,
    alerts=alerts,
    performance=performance_metrics
)

# Stream to all connected clients
await server.stream_data(data)
```

### WebSocket Client (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/stream');

ws.onopen = () => {
    console.log('Connected to SENTINEL stream');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    // Access frame data
    console.log('Timestamp:', data.timestamp);
    console.log('FPS:', data.performance.fps);
    
    // Update visualization
    updateBEV(data.bev.image);  // Base64 encoded image
    updateDetections(data.detections);
    updateDriverState(data.driver_state);
    updateRisks(data.risk_assessment.top_risks);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};

ws.onclose = () => {
    console.log('Disconnected from SENTINEL stream');
};
```

### REST API Examples

#### Get Configuration

```bash
curl http://localhost:8080/api/config
```

#### Update Configuration

```bash
curl -X POST http://localhost:8080/api/config \
  -H "Content-Type: application/json" \
  -d '{"alerts": {"suppression": {"max_simultaneous": 3}}}'
```

#### List Scenarios

```bash
curl http://localhost:8080/api/scenarios
```

#### Get Scenario Details

```bash
curl http://localhost:8080/api/scenarios/20241115_103045
```

#### Delete Scenario

```bash
curl -X DELETE http://localhost:8080/api/scenarios/20241115_103045
```

## Data Format

### Frame Data Structure

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
      "bbox_3d": {
        "x": 10.0, "y": 2.0, "z": 0.0,
        "w": 4.5, "h": 1.8, "l": 2.0,
        "theta": 0.1
      },
      "class_name": "vehicle",
      "confidence": 0.95,
      "velocity": {"vx": 5.0, "vy": 0.0, "vz": 0.0},
      "track_id": 1
    }
  ],
  "driver_state": {
    "face_detected": true,
    "readiness_score": 85.0,
    "head_pose": {"roll": 0.1, "pitch": -0.05, "yaw": 0.02},
    "gaze": {"pitch": -0.1, "yaw": 0.05, "attention_zone": "front"},
    "drowsiness": {"score": 0.2, "yawn_detected": false},
    "distraction": {"type": "none", "confidence": 0.95}
  },
  "risk_assessment": {
    "top_risks": [
      {
        "hazard": {
          "object_id": 1,
          "type": "vehicle",
          "position": {"x": 10.0, "y": 2.0, "z": 0.0},
          "ttc": 2.5,
          "zone": "front"
        },
        "contextual_score": 0.65,
        "driver_aware": true,
        "urgency": "medium"
      }
    ]
  },
  "alerts": [],
  "performance": {
    "fps": 30.5,
    "latency": {
      "camera": 5.2,
      "bev": 14.8,
      "segmentation": 13.5,
      "detection": 18.3,
      "dms": 22.1,
      "intelligence": 8.7,
      "total": 82.6
    },
    "gpu_memory_mb": 3456,
    "cpu_percent": 45.2
  }
}
```

## Performance

- **Streaming Rate**: 30 Hz (33ms per frame)
- **Image Encoding**: JPEG with quality 85 for efficient transmission
- **WebSocket Overhead**: ~5-10ms per broadcast
- **Multi-Client**: Supports 10+ simultaneous connections

## Dependencies

- `fastapi>=0.103.0`: Web framework
- `uvicorn[standard]>=0.23.0`: ASGI server
- `websockets>=11.0`: WebSocket support
- `pydantic>=2.0.0`: Data validation
- `opencv-python>=4.8.0`: Image encoding

## Testing

Run backend tests:

```bash
pytest tests/test_visualization_backend.py -v
```

## Integration

The backend integrates with the main SENTINEL system:

```python
from src.visualization.backend import create_server, serialize_frame_data

class SentinelSystem:
    def __init__(self, config):
        # Create visualization server
        self.viz_server = create_server(config)
        
        # Start server in background thread
        import threading
        self.viz_thread = threading.Thread(
            target=self.viz_server.run,
            kwargs={'host': '0.0.0.0', 'port': 8080}
        )
        self.viz_thread.daemon = True
        self.viz_thread.start()
    
    async def process_frame(self, frame_bundle):
        # Process frame...
        bev = self.bev_generator.generate(...)
        detections = self.detector.detect(...)
        driver_state = self.dms.analyze(...)
        risk_assessment = self.intelligence.assess(...)
        alerts = self.alert_system.process(...)
        
        # Serialize and stream
        data = serialize_frame_data(
            timestamp=frame_bundle.timestamp,
            bev=bev,
            detections=detections,
            driver_state=driver_state,
            risk_assessment=risk_assessment,
            alerts=alerts,
            performance=self.get_performance_metrics()
        )
        
        await self.viz_server.stream_data(data)
```

## Security Considerations

For production deployment:

1. **CORS Configuration**: Restrict `allow_origins` to specific domains
2. **Authentication**: Add JWT or API key authentication
3. **Rate Limiting**: Implement rate limiting for REST endpoints
4. **HTTPS**: Use TLS/SSL for encrypted communication
5. **Input Validation**: Validate all configuration updates

## Future Enhancements

- [ ] Authentication and authorization
- [ ] Rate limiting and throttling
- [ ] Compression for WebSocket messages
- [ ] Binary protocol for reduced bandwidth
- [ ] Recording playback streaming
- [ ] Multi-room support for multiple vehicles
