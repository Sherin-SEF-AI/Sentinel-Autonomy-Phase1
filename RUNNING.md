# How to Run SENTINEL

This guide explains how to install dependencies, configure, and run the SENTINEL Contextual Safety Intelligence Platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)
7. [Testing Without Hardware](#testing-without-hardware)

---

## Quick Start

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Run the GUI application (recommended)
python3 src/gui_main.py

# OR run the console/backend version
python3 run_sentinel.py
```

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows 10+
- **Python**: 3.9 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for real-time performance)
  - 8GB+ VRAM for optimal performance
  - CPU-only mode available (reduced performance)
- **CPU**: 8-core processor (Intel i7/Ryzen 7 or better)
- **RAM**: 16GB minimum, 32GB recommended
- **Cameras**: Up to 3 USB cameras (or simulated feeds)

### Hardware Setup (Optional)

For full functionality, you'll need:
- **3 USB Cameras**:
  - 1x Interior (driver monitoring) - 640x480
  - 2x Front (left/right) - 1280x720
- **GPS Device** (optional):
  - NMEA-compatible GPS receiver
  - Connected via USB (e.g., `/dev/ttyUSB0`)

---

## Installation

### 1. Clone the Repository

```bash
cd /home/user
git clone <repository-url> Sentinel-Autonomy-Phase1
cd Sentinel-Autonomy-Phase1
```

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip3 install -r requirements.txt
```

### 3. Install CUDA (GPU Acceleration)

**For NVIDIA GPUs** (recommended for best performance):

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvidia-smi
```

If CUDA is not available, the system will fall back to CPU mode automatically.

### 4. Download Model Weights

Place model files in the `models/` directory:

```
models/
├── bev_segmentation.pth        # BEV semantic segmentation
├── yolov8m_automotive.pt       # YOLOv8 object detection
├── l2cs_gaze.pth              # Gaze estimation
├── drowsiness_model.pth        # Drowsiness detection
└── distraction_clf.pth         # Distraction classification
```

**Note**: Model weights are not included in the repository due to size. Contact the project maintainers or train your own models.

---

## Configuration

### Edit Configuration File

The main configuration file is `configs/default.yaml`. Key sections:

#### 1. Camera Configuration

```yaml
cameras:
  sync_tolerance_ms: 100.0  # Frame sync tolerance

  interior:
    device: 0  # Camera index (0, 1, 2, etc.)
    resolution: [640, 480]
    fps: 30

  front_left:
    device: 1
    resolution: [1280, 720]
    fps: 30

  front_right:
    device: 2
    resolution: [1280, 720]
    fps: 30
```

**Find your camera devices**:
```bash
# Linux
ls -l /dev/video*

# List available cameras with their capabilities
v4l2-ctl --list-devices
```

#### 2. GPU/CPU Configuration

```yaml
models:
  segmentation:
    device: "cuda"  # Change to "cpu" if no GPU
    precision: "fp16"  # Use "fp32" for CPU

  detection:
    # Uses PyTorch device auto-detection
```

#### 3. Feature Toggles

Enable/disable features as needed:

```yaml
features:
  lane_detection:
    enabled: true

  blind_spot:
    enabled: true

  collision_warning:
    enabled: true

  traffic_signs:
    enabled: true

  interaction_prediction:
    enabled: true  # NEW: Multi-object interaction prediction

  gps:
    enabled: false  # Enable when GPS hardware is connected
    simulation: true  # Use simulation mode for testing
```

#### 4. Performance Tuning

Adjust based on your hardware:

```yaml
tracking:
  max_age: 30  # Reduce for faster processing
  min_hits: 3  # Increase for more stable tracks

risk_assessment:
  trajectory_prediction:
    horizon: 3.0  # Reduce for faster computation
    num_hypotheses: 3  # Reduce to 1 for CPU mode
```

---

## Running the Application

### Option 1: GUI Application (Recommended)

The GUI provides full visualization and control:

```bash
python3 src/gui_main.py
```

**GUI Features**:
- Live camera feeds with overlays
- Bird's Eye View (BEV) display
- Driver state monitoring
- Risk assessment panel
- Alert notifications
- Advanced features:
  - Lane detection visualization
  - Blind spot indicators
  - Collision warnings
  - Traffic sign recognition
  - **NEW**: Interaction predictions
  - **NEW**: GPS location and speed limits
  - **NEW**: Analytics dashboard
  - **NEW**: Incident review system
- Performance metrics

**Using the GUI**:

1. **Start System**: Click `System → Start System` or press `F5`
2. **View Features**: Click tabs on the left/right docks
3. **Analytics**: Go to `Analytics → Analytics Dashboard`
4. **Incident Review**: Go to `Analytics → Incident Review`
5. **Stop System**: Press `F6` or `System → Stop System`

### Option 2: Console/Backend Mode

For headless operation or debugging:

```bash
python3 run_sentinel.py
```

**Options**:
```bash
# Custom config file
python3 run_sentinel.py --config my_config.yaml

# Change log level
python3 run_sentinel.py --log-level DEBUG
```

The console version:
- Runs all processing pipelines
- Logs performance metrics
- Records scenarios automatically
- No visualization UI
- Suitable for deployment on embedded systems

### Option 3: Direct Module Execution

```bash
# Run GUI directly
python3 -m src.gui_main

# Run backend directly
python3 -m src.main --config configs/default.yaml
```

---

## Testing Without Hardware

### Simulated Camera Feeds

If you don't have physical cameras, modify the configuration:

```yaml
cameras:
  interior:
    device: 0  # Will open first available camera or fail gracefully

  # Or create video file sources (requires code modification)
```

**Create test video files**:
```bash
# Generate sample video with OpenCV (Python)
import cv2
import numpy as np

writer = cv2.VideoWriter('test_camera.mp4',
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, (1280, 720))

for i in range(300):  # 10 seconds at 30 FPS
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    writer.write(frame)

writer.release()
```

### GPS Simulation Mode

GPS is pre-configured for simulation:

```yaml
features:
  gps:
    enabled: false  # Set to true to test GPS
    simulation: true  # Uses simulated GPS data
```

The simulator:
- Generates GPS coordinates (starts at San Francisco: 37.7749, -122.4194)
- Simulates movement (north-east at 50 km/h)
- Provides satellite count and fix quality
- Works without hardware

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

```
ModuleNotFoundError: No module named 'PyQt6'
```

**Solution**:
```bash
pip3 install -r requirements.txt
```

#### 2. Camera Not Found

```
ERROR: Failed to open camera device 0
```

**Solutions**:
- Check camera connections: `ls -l /dev/video*`
- Test camera: `ffplay /dev/video0`
- Try different device numbers (0, 1, 2, etc.)
- On Linux, add user to video group:
  ```bash
  sudo usermod -a -G video $USER
  # Log out and back in
  ```

#### 3. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce batch size in config
- Use CPU mode: set `device: "cpu"` in config
- Close other GPU applications
- Reduce number of trajectory hypotheses

#### 4. Low FPS (<30 FPS)

**Solutions**:
- Enable GPU acceleration (CUDA)
- Reduce image resolution in config
- Disable non-essential features
- Use `fp16` precision instead of `fp32`
- Check `configs/default.yaml` performance settings

#### 5. GUI Won't Start

```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

**Solution (Linux)**:
```bash
sudo apt install libxcb-xinerama0 libxcb-cursor0
export QT_DEBUG_PLUGINS=1  # For debugging
```

#### 6. Model Files Missing

```
FileNotFoundError: models/yolov8m_automotive.pt
```

**Temporary Solution**:
- Download pre-trained YOLOv8:
  ```bash
  pip install ultralytics
  python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
  mv yolov8m.pt models/yolov8m_automotive.pt
  ```
- Other models require training or project-specific weights

---

## Running Specific Components

### Test Individual Widgets

```bash
# Test analytics dashboard
python3 -c "
from PyQt6.QtWidgets import QApplication
import sys
sys.path.insert(0, 'src')
from src.gui.widgets.analytics_dashboard import AnalyticsDashboard

app = QApplication(sys.argv)
widget = AnalyticsDashboard()
widget.show()
sys.exit(app.exec())
"

# Test GPS widget
python3 test_gps_widget.py  # If test file exists

# Test incident review
python3 -c "
from PyQt6.QtWidgets import QApplication
import sys
sys.path.insert(0, 'src')
from src.gui.widgets.incident_review_widget import IncidentReviewWidget

app = QApplication(sys.argv)
widget = IncidentReviewWidget()
widget.show()
sys.exit(app.exec())
"
```

### Run Backend with Custom Config

```bash
# Copy default config
cp configs/default.yaml configs/my_config.yaml

# Edit your config
nano configs/my_config.yaml

# Run with custom config
python3 run_sentinel.py --config configs/my_config.yaml
```

---

## Performance Monitoring

### Check System Performance

The console output shows:

```
PERFORMANCE METRICS
============================================================
FPS: 32.45
Frames processed: 1000
CPU usage: 45.2%
Memory usage: 4567.3 MB
GPU memory: 3210.5 MB
Module latencies (avg):
  camera: 12.34ms
  bev: 8.92ms
  segmentation: 23.45ms
  detection: 31.23ms
  ...
Total pipeline latency: 85.67ms
============================================================
```

### Optimization Tips

1. **Enable GPU**: Ensure CUDA is working
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Profile Slow Modules**: Enable profiling
   ```yaml
   risk_assessment:
     trajectory_prediction:
       enable_profiling: true
   ```

3. **Monitor Logs**: Check `logs/` directory
   ```bash
   tail -f logs/sentinel.log
   tail -f logs/performance.log
   ```

---

## Next Steps

### Accessing New Features

#### 1. Analytics Dashboard
1. Start GUI application
2. Go to `Analytics → Analytics Dashboard`
3. View historical trip data, safety trends, and performance metrics

#### 2. Incident Review
1. Start GUI, trigger safety events (hard braking, etc.)
2. Scenarios are auto-recorded to `scenarios/`
3. Go to `Analytics → Incident Review`
4. Browse and replay recorded incidents

#### 3. GPS Tracking
1. Enable in config: `features.gps.enabled: true`
2. Connect GPS or use simulation mode
3. View in `Advanced Features → GPS tab`
4. See real-time position, speed limits, and violations

#### 4. Interaction Prediction
1. Enabled by default
2. Predicts pedestrian crossings, lane changes, merges
3. View warnings in main display
4. Check logs for interaction events

---

## Additional Resources

- **Documentation**: See `README.md` for architecture details
- **Configuration Reference**: See `configs/default.yaml` with inline comments
- **Data Formats**: See `data/README.md` and `scenarios/README.md`
- **Changelog**: See `CHANGELOG.md` for recent updates
- **API Documentation**: (Coming soon)

---

## Getting Help

If you encounter issues:

1. Check logs in `logs/` directory
2. Enable debug logging: `--log-level DEBUG`
3. Review `TROUBLESHOOTING.md` (if available)
4. Open an issue on the project repository
5. Contact project maintainers

---

## License

See LICENSE file for details.
