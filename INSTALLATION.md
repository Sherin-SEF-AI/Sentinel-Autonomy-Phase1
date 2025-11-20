# SENTINEL System Installation Guide

This guide provides detailed instructions for installing and deploying the SENTINEL contextual safety intelligence platform.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Installation Methods](#installation-methods)
  - [Method 1: Docker Installation (Recommended)](#method-1-docker-installation-recommended)
  - [Method 2: Native Installation](#method-2-native-installation)
- [Camera Setup and Calibration](#camera-setup-and-calibration)
- [Model Download](#model-download)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Troubleshooting](#troubleshooting)

---

## Hardware Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with compute capability ≥ 7.0 (e.g., RTX 2060 or better)
  - 8GB+ VRAM
  - CUDA 11.8+ support
- **CPU**: 8-core processor (Intel i7/AMD Ryzen 7 or better)
- **RAM**: 16GB system memory
- **Storage**: 
  - 50GB free space for system and models
  - SSD recommended for fast model loading
- **Cameras**: 3x USB 3.0 cameras
  - 1x interior camera (640x480 @ 30fps minimum)
  - 2x external cameras (1280x720 @ 30fps minimum)

### Recommended Requirements

- **GPU**: NVIDIA RTX 3070 or better (12GB+ VRAM)
- **CPU**: 12-core processor
- **RAM**: 32GB system memory
- **Storage**: NVMe SSD with 100GB+ free space
- **Cameras**: High-quality USB 3.0 cameras with global shutter

### Supported Platforms

- Ubuntu 20.04 LTS or 22.04 LTS
- NVIDIA Driver version 520.61.05 or newer
- Docker 20.10+ with nvidia-docker2 (for Docker installation)

---

## Software Dependencies

### Core Dependencies

- Python 3.10+
- CUDA 11.8+
- cuDNN 8.6+
- OpenCV 4.5+
- PyTorch 2.0+
- TensorRT 8.5+ (optional, for optimization)

### Python Packages

All Python dependencies are listed in:
- `requirements.txt` - Core dependencies
- `requirements-gpu.txt` - GPU-specific dependencies
- `requirements-dev.txt` - Development dependencies

---

## Installation Methods

### Method 1: Docker Installation (Recommended)

Docker installation provides the easiest setup with all dependencies pre-configured.

#### Prerequisites

1. **Install Docker**:
```bash
# Update package index
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

2. **Install NVIDIA Docker Runtime**:
```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker daemon
sudo systemctl restart docker
```

3. **Verify GPU Access**:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

#### Build and Run

1. **Clone Repository**:
```bash
git clone https://github.com/your-org/sentinel.git
cd sentinel
```

2. **Build Docker Image**:
```bash
docker build -t sentinel:latest .
```

3. **Run with Docker Compose** (Recommended):
```bash
# Edit docker-compose.yml to configure camera devices
# Then start the system
docker-compose up -d

# View logs
docker-compose logs -f sentinel
```

4. **Or Run with Docker Command**:
```bash
docker run -d \
  --name sentinel \
  --gpus all \
  --device=/dev/video0:/dev/video0 \
  --device=/dev/video1:/dev/video1 \
  --device=/dev/video2:/dev/video2 \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/scenarios:/app/scenarios \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/configs:/app/configs \
  sentinel:latest
```

5. **Access Dashboard**:
```bash
# Open browser to http://localhost:8080
```

---

### Method 2: Native Installation

Native installation provides more control but requires manual dependency management.

#### Step 1: Install System Dependencies

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install build tools
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl

# Install Python 3.10
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip

# Install OpenCV dependencies
sudo apt-get install -y \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install USB camera support
sudo apt-get install -y \
    libusb-1.0-0 \
    v4l-utils
```

#### Step 2: Install CUDA and cuDNN

```bash
# Download and install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

#### Step 3: Clone Repository

```bash
git clone https://github.com/your-org/sentinel.git
cd sentinel
```

#### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

#### Step 5: Install Python Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install GPU dependencies
pip install -r requirements-gpu.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### Step 6: Verify Installation

```bash
# Test PyTorch GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Test OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

---

## Camera Setup and Calibration

### Camera Mounting

1. **Interior Camera**: Mount facing driver, capturing full face and upper body
2. **Front-Left Camera**: Mount on left side of windshield, angled 45° left
3. **Front-Right Camera**: Mount on right side of windshield, angled 45° right

### Camera Configuration

1. **Identify Camera Devices**:
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera capture
ffplay /dev/video0
```

2. **Update Configuration**:
Edit `configs/default.yaml` and set correct device indices:
```yaml
cameras:
  interior:
    device: 0  # Update with correct device index
  front_left:
    device: 1
  front_right:
    device: 2
```

### Camera Calibration

Calibration is required for accurate BEV generation and 3D object detection.

1. **Print Calibration Pattern**:
```bash
# Download checkerboard pattern (9x6 squares, 25mm square size)
wget https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png
# Print on A4 paper
```

2. **Run Calibration Script**:
```bash
python scripts/calibrate_camera.py \
    --camera-id 1 \
    --camera-name front_left \
    --pattern-size 9 6 \
    --square-size 0.025
```

3. **Follow On-Screen Instructions**:
- Capture 20+ images of checkerboard at various angles
- Press 's' to save image, 'q' to finish
- Script will compute intrinsics, extrinsics, and homography

4. **Repeat for All Cameras**:
```bash
# Interior camera
python scripts/calibrate_camera.py --camera-id 0 --camera-name interior

# Front-right camera
python scripts/calibrate_camera.py --camera-id 2 --camera-name front_right
```

5. **Validate Calibration**:
```bash
python scripts/validate_calibration.py --camera front_left
```

Calibration files are saved to `configs/calibration/{camera_name}.yaml`.

---

## Model Download

### Automatic Download

```bash
# Download all pretrained models
python scripts/download_models.py

# Verify downloads
python scripts/download_models.py --verify-only
```

### Manual Download

If automatic download fails, manually download models:

1. **BEV Segmentation Model** (~45 MB)
   - URL: [Model URL]
   - Save to: `models/bev_segmentation.pth`

2. **YOLOv8 Model** (~52 MB)
   - URL: [Model URL]
   - Save to: `models/yolov8m_automotive.pt`

3. **L2CS-Net Gaze Model** (~28 MB)
   - URL: [Model URL]
   - Save to: `models/l2cs_gaze.pth`

4. **Drowsiness Model** (~15 MB)
   - URL: [Model URL]
   - Save to: `models/drowsiness_model.pth`

5. **Distraction Model** (~12 MB)
   - URL: [Model URL]
   - Save to: `models/distraction_clf.pth`

### Verify Model Checksums

```bash
# Verify all models
python scripts/download_models.py --verify-only
```

---

## Configuration

### Basic Configuration

Edit `configs/default.yaml` to customize system behavior:

```yaml
# Camera settings
cameras:
  interior:
    device: 0
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

# Model settings
models:
  segmentation:
    device: "cuda"  # or "cpu"
    precision: "fp16"  # or "fp32"

# Risk assessment thresholds
risk_assessment:
  thresholds:
    intervention: 0.7
    critical: 0.9

# Alert settings
alerts:
  modalities:
    visual:
      enabled: true
    audio:
      enabled: true
      volume: 0.8
    haptic:
      enabled: false
```

### Advanced Configuration

See `configs/default.yaml` for all available configuration options.

---

## Running the System

### Native Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Run system
python src/main.py --config configs/default.yaml

# Access dashboard at http://localhost:8080
```

### Docker Installation

```bash
# Start system
docker-compose up -d

# View logs
docker-compose logs -f

# Stop system
docker-compose down
```

### Verification

1. **Check System Status**:
```bash
# View logs
tail -f logs/sentinel.log

# Check GPU usage
nvidia-smi
```

2. **Access Dashboard**:
- Open browser to `http://localhost:8080`
- Verify camera feeds are visible
- Check BEV generation
- Monitor system performance metrics

3. **Run Example Scripts**:
```bash
# Test camera capture
python examples/camera_example.py

# Test BEV generation
python examples/bev_example.py

# Test full pipeline
python examples/visualization_complete_example.py
```

---

## Troubleshooting

### Camera Issues

**Problem**: Cameras not detected
```bash
# Check camera devices
ls -l /dev/video*

# Check camera permissions
sudo usermod -aG video $USER
newgrp video

# Test camera access
v4l2-ctl --list-devices
ffplay /dev/video0
```

**Problem**: Camera permission denied in Docker
```bash
# Add user to video group
sudo usermod -aG video $USER

# Or run container with privileged mode
docker run --privileged ...
```

### GPU Issues

**Problem**: CUDA not available
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: Out of GPU memory
```bash
# Reduce batch size or use FP16 precision
# Edit configs/default.yaml:
models:
  segmentation:
    precision: "fp16"
```

### Performance Issues

**Problem**: Low FPS or high latency
```bash
# Check GPU utilization
nvidia-smi -l 1

# Check CPU usage
htop

# Enable performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit (adjust for your GPU)

# Optimize models with TensorRT (advanced)
python scripts/optimize_models.py
```

**Problem**: System crashes or freezes
```bash
# Check system logs
journalctl -xe

# Check SENTINEL logs
tail -f logs/sentinel.log
tail -f logs/errors.log

# Monitor system resources
htop
nvidia-smi -l 1
```

### Model Issues

**Problem**: Model download fails
```bash
# Check internet connection
ping google.com

# Try manual download
# See "Manual Download" section above

# Verify existing models
python scripts/download_models.py --verify-only
```

**Problem**: Model inference errors
```bash
# Check model files exist
ls -lh models/

# Verify model checksums
python scripts/download_models.py --verify-only

# Check GPU memory
nvidia-smi

# Try CPU inference (slower)
# Edit configs/default.yaml:
models:
  segmentation:
    device: "cpu"
```

### Calibration Issues

**Problem**: Poor BEV quality
```bash
# Re-run calibration with more images
python scripts/calibrate_camera.py --camera-id 1 --camera-name front_left

# Validate calibration
python scripts/validate_calibration.py --camera front_left

# Check calibration files
cat configs/calibration/front_left.yaml
```

**Problem**: 3D detection inaccurate
```bash
# Verify camera extrinsics are correct
# Measure actual camera positions and update calibration files
# Re-run calibration if needed
```

### Docker Issues

**Problem**: Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t sentinel:latest .
```

**Problem**: Container exits immediately
```bash
# Check container logs
docker logs sentinel

# Run interactively for debugging
docker run -it --rm --gpus all sentinel:latest /bin/bash
```

### Network Issues

**Problem**: Cannot access dashboard
```bash
# Check if port is in use
sudo netstat -tulpn | grep 8080

# Try different port
docker run -p 8081:8080 ...

# Check firewall
sudo ufw status
sudo ufw allow 8080
```

### Getting Help

If you encounter issues not covered here:

1. **Check Logs**:
   - System log: `logs/sentinel.log`
   - Error log: `logs/errors.log`
   - Module-specific logs: `logs/{module}.log`

2. **Run Verification Scripts**:
   ```bash
   python scripts/verify_system_structure.py
   python scripts/verify_system_orchestration.py
   ```

3. **Enable Debug Logging**:
   ```yaml
   # Edit configs/default.yaml
   system:
     log_level: "DEBUG"
   ```

4. **Report Issues**:
   - GitHub Issues: [Repository URL]
   - Include: System info, logs, error messages, steps to reproduce

---

## Next Steps

After successful installation:

1. **Run System Tests**:
   ```bash
   pytest tests/
   ```

2. **Explore Examples**:
   ```bash
   ls examples/
   python examples/visualization_complete_example.py
   ```

3. **Review Documentation**:
   - `README.md` - Project overview
   - `configs/default.yaml` - Configuration reference
   - Module READMEs in `src/*/README.md`

4. **Customize Configuration**:
   - Adjust risk thresholds
   - Configure alert modalities
   - Tune performance parameters

5. **Start Development**:
   - See `requirements-dev.txt` for development tools
   - Run tests: `pytest tests/`
   - Check code style: `black src/ tests/`

---

## Support

For additional support:
- Documentation: [Docs URL]
- Community Forum: [Forum URL]
- Email: support@sentinel-system.com
