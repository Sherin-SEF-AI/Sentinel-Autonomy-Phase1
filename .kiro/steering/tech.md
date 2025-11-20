# Technology Stack

## Language & Runtime

- **Python 3.10+**: Primary development language
- **CUDA 11.8+**: GPU acceleration for deep learning inference

## Core Dependencies

### Computer Vision & Deep Learning
- **OpenCV**: Camera capture, image processing, calibration
- **PyTorch**: Deep learning framework for model inference
- **TensorRT**: Model optimization for NVIDIA GPUs
- **ONNX**: Model portability and deployment
- **MediaPipe**: Face detection and landmark extraction

### Models
- **BEVFormer-Tiny**: Semantic segmentation of bird's eye view
- **YOLOv8**: Multi-view object detection
- **L2CS-Net**: Gaze estimation
- **DeepSORT**: Multi-object tracking
- **MobileNetV3**: Distraction classification
- **LSTM**: Trajectory prediction model
- **FaceNet**: Face recognition for driver identification

### GUI Framework
- **PyQt6**: Desktop GUI application framework
- **PyQtGraph**: High-performance plotting and graphics
- **QOpenGLWidget**: GPU-accelerated rendering

### Backend & Visualization
- **FastAPI**: Web API and WebSocket server
- **React + Three.js**: Real-time 3D visualization dashboard (web-based)
- **NumPy**: Vectorized numerical operations

### Advanced Features
- **python-can**: CAN bus interface for vehicle telemetry
- **requests**: HTTP client for cloud API
- **lanelet2** or **OpenDRIVE**: HD map formats
- **scipy**: Scientific computing for trajectory prediction

### Configuration & Data
- **YAML**: Configuration file format
- **JSON**: Scenario recording and annotations
- **DBC**: CAN bus message definitions

## Hardware Requirements

- **GPU**: NVIDIA GPU with compute capability ≥ 7.0, 8GB+ VRAM
- **CPU**: 8-core processor
- **Cameras**: 3x USB 3.0 cameras (1 interior, 2 external)
- **Storage**: SSD for fast model loading

## Development Setup

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py

# Calibrate cameras
python scripts/calibrate.py --cameras 3
```

### Running the System
```bash
# Start SENTINEL with default config
python main.py --config configs/default.yaml

# Access dashboard
# Open browser to http://localhost:8080
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/test_integration.py

# Run performance benchmarks
pytest tests/test_performance.py
```

## Deployment

### Docker
```bash
# Build image
docker build -t sentinel:latest .

# Run with GPU support
docker run --gpus all -v /dev/video0:/dev/video0 sentinel:latest
```

## Performance Optimization

- **FP16 mixed precision** for inference
- **TensorRT optimization** for production deployment
- **Multi-threaded camera capture** for parallel processing
- **Vectorized NumPy operations** for CPU efficiency
- **GPU memory pre-allocation** to avoid dynamic allocation
