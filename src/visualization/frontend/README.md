# Visualization Frontend

Web-based dashboard for SENTINEL real-time visualization.

## Features

- **Live Bird's Eye View**: Real-time BEV with semantic segmentation overlay
- **Multi-View Layout**: Switch between BEV, camera feeds, and 3D scene
- **Driver Monitoring**: Real-time driver state metrics and attention visualization
- **Risk Assessment**: Top-3 hazards with contextual risk scores
- **Performance Monitoring**: FPS, latency, and resource usage metrics
- **Alert System**: Real-time safety alerts with urgency levels

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Dashboard                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────┐  ┌──────────────────────────────┐ │
│  │   Main View Area    │  │      Side Panel              │ │
│  │                     │  │                              │ │
│  │  - BEV View         │  │  - Driver State              │ │
│  │  - Camera Feeds     │  │    • Readiness Score         │ │
│  │  - 3D Scene         │  │    • Attention Zone          │ │
│  │                     │  │    • Drowsiness              │ │
│  │  [Detections        │  │    • Distraction             │ │
│  │   overlaid]         │  │                              │ │
│  │                     │  │  - Risk Assessment           │ │
│  │                     │  │    • Top 3 Hazards           │ │
│  │                     │  │    • TTC & Awareness         │ │
│  │                     │  │                              │ │
│  │                     │  │  - Performance Metrics       │ │
│  │                     │  │    • FPS & Latency           │ │
│  │                     │  │    • CPU & GPU Usage         │ │
│  └─────────────────────┘  └──────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Main View Area

**BEV View**
- Displays bird's eye view with semantic segmentation overlay
- Shows 3D bounding boxes projected onto BEV
- Color-coded object classes (vehicles, pedestrians, cyclists)
- Track IDs for persistent object tracking

**Camera Feeds** (placeholder)
- Multi-camera view layout
- Interior camera for DMS
- External cameras (front-left, front-right)

**3D Scene** (placeholder)
- Three.js-based 3D visualization
- Interactive camera controls
- 3D bounding boxes in vehicle coordinate frame

### Side Panel

**Driver State**
- Readiness score (0-100) with color-coded bar
- Current attention zone (front, left, right, etc.)
- Drowsiness level (PERCLOS-based)
- Distraction type (phone, passenger, controls, etc.)

**Risk Assessment**
- Top 3 contextual risks
- Urgency levels (critical, high, medium, low)
- Time-to-collision (TTC)
- Driver awareness indicator

**Performance Metrics**
- Real-time FPS
- Per-module latency breakdown
- CPU and GPU utilization
- Total end-to-end latency

### Alert System

- Floating alerts in top-right corner
- Color-coded by urgency (critical=red, warning=orange, info=blue)
- Auto-dismiss after 5 seconds
- Slide-in animation

## Usage

### Serving the Frontend

The frontend is served by the FastAPI backend:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="src/visualization/frontend", html=True), name="frontend")
```

Access the dashboard at: `http://localhost:8080/`

### WebSocket Connection

The dashboard automatically connects to the backend WebSocket:

```javascript
const wsUrl = `ws://localhost:8080/ws/stream`;
const ws = new WebSocket(wsUrl);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Update visualization
};
```

### Data Flow

1. Backend streams frame data at 30 Hz via WebSocket
2. Frontend receives JSON messages with:
   - Base64-encoded BEV and segmentation images
   - 3D detection bounding boxes
   - Driver state metrics
   - Risk assessment scores
   - Performance metrics
3. Dashboard updates visualizations in real-time

## Layout

### Grid Layout

```
┌─────────────────────────────────────────────────────────┐
│                      Header                              │
│  SENTINEL | Status: Connected | FPS: 30.5               │
├──────────────────────────────┬──────────────────────────┤
│                              │                          │
│                              │   Driver State           │
│        Main View             │   ┌──────────────────┐   │
│                              │   │ Readiness: 85    │   │
│   [BEV | Cameras | 3D]       │   │ Attention: front │   │
│                              │   └──────────────────┘   │
│   ┌────────────────────┐     │                          │
│   │                    │     │   Risk Assessment        │
│   │   BEV with         │     │   ┌──────────────────┐   │
│   │   Detections       │     │   │ Vehicle: 65%     │   │
│   │                    │     │   │ TTC: 2.5s        │   │
│   └────────────────────┘     │   └──────────────────┘   │
│                              │                          │
│                              │   Performance            │
│                              │   ┌──────────────────┐   │
│                              │   │ FPS: 30.5        │   │
│                              │   │ Latency: 82ms    │   │
│                              │   └──────────────────┘   │
├──────────────────────────────┴──────────────────────────┤
│                      Footer                              │
│  Timestamp: 10.5s | SENTINEL v1.0                       │
└─────────────────────────────────────────────────────────┘
```

## Styling

### Color Scheme

- Background: `#1a1a1a` (dark)
- Panels: `#2a2a2a` (medium dark)
- Accents: `#3a3a3a` (light dark)
- Primary: `#00ff88` (green)
- Critical: `#ff4444` (red)
- Warning: `#ffaa00` (orange)
- Info: `#00aaff` (blue)

### Typography

- Font: Segoe UI, Tahoma, Geneva, Verdana, sans-serif
- Headers: 18-24px, bold
- Body: 12-14px, regular
- Metrics: 18-20px, bold

## Browser Compatibility

- Chrome/Edge: ✅ Full support
- Firefox: ✅ Full support
- Safari: ✅ Full support
- Mobile: ⚠️ Limited (desktop-optimized layout)

## Performance

- **Rendering**: 60 FPS UI updates
- **WebSocket**: 30 Hz data stream
- **Image Decoding**: Hardware-accelerated JPEG
- **Canvas**: 2D rendering for BEV and overlays

## Future Enhancements

### Phase 2
- [ ] Three.js 3D scene visualization
- [ ] Interactive camera controls
- [ ] Attention heatmap overlay
- [ ] Trajectory prediction visualization
- [ ] Historical data graphs

### Phase 3
- [ ] Scenario playback controls
- [ ] Frame-by-frame scrubbing
- [ ] Annotation overlay toggle
- [ ] Export screenshots/videos
- [ ] Configuration editor

### Phase 4
- [ ] Multi-vehicle dashboard
- [ ] Fleet monitoring view
- [ ] Comparative analytics
- [ ] Custom alert rules
- [ ] Mobile-responsive layout

## Development

### File Structure

```
src/visualization/frontend/
├── index.html          # Main HTML structure
├── app.js              # Dashboard application logic
├── README.md           # This file
└── assets/             # (future) Images, icons, etc.
```

### Adding New Views

1. Add tab button in HTML:
```html
<button class="tab" data-view="myview">My View</button>
```

2. Add view panel:
```html
<div id="myview-view" class="view-panel">
    <canvas id="myview-canvas"></canvas>
</div>
```

3. Implement view logic in `app.js`:
```javascript
updateMyView(data) {
    // Render view
}
```

### Customizing Styles

Edit the `<style>` section in `index.html` to customize:
- Colors and themes
- Layout and spacing
- Fonts and typography
- Animations and transitions

## Integration Example

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.visualization.backend import create_server

app = FastAPI()

# Serve frontend
app.mount("/", StaticFiles(directory="src/visualization/frontend", html=True), name="frontend")

# Add backend routes
# ... (WebSocket and REST endpoints)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

Access dashboard at: `http://localhost:8080/`

## Troubleshooting

### WebSocket Connection Failed

- Check backend is running on correct port
- Verify firewall allows WebSocket connections
- Check browser console for error messages

### Images Not Displaying

- Verify base64 encoding is correct
- Check image format (JPEG/PNG)
- Ensure CORS headers are set correctly

### Performance Issues

- Reduce streaming frame rate (e.g., 15 Hz instead of 30 Hz)
- Lower JPEG quality for smaller payloads
- Use binary WebSocket protocol instead of JSON
- Enable compression on WebSocket connection

## Security

For production deployment:

1. **HTTPS**: Use TLS/SSL for encrypted communication
2. **Authentication**: Add login page and JWT tokens
3. **CORS**: Restrict origins to specific domains
4. **CSP**: Implement Content Security Policy headers
5. **Rate Limiting**: Prevent abuse of WebSocket connections
