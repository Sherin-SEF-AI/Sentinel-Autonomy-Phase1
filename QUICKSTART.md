# SENTINEL Quick Start Guide

Get SENTINEL up and running in minutes.

## Prerequisites

- Ubuntu 20.04/22.04 LTS
- NVIDIA GPU (RTX 2060 or better)
- 3x USB cameras
- Docker installed (recommended)

## Option 1: Docker (Fastest)

```bash
# 1. Clone repository
git clone https://github.com/your-org/sentinel.git
cd sentinel

# 2. Build image
docker build -t sentinel:latest .

# 3. Run system
docker-compose up -d

# 4. Access dashboard
# Open browser to http://localhost:8080
```

## Option 2: Native Installation

```bash
# 1. Clone repository
git clone https://github.com/your-org/sentinel.git
cd sentinel

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-gpu.txt

# 3. Download models
python3 scripts/download_models.py

# 4. Calibrate cameras
python3 scripts/calibrate_camera.py --camera-id 0 --camera-name interior
python3 scripts/calibrate_camera.py --camera-id 1 --camera-name front_left
python3 scripts/calibrate_camera.py --camera-id 2 --camera-name front_right

# 5. Run system
python3 src/main.py --config configs/default.yaml

# 6. Access dashboard
# Open browser to http://localhost:8080
```

## Verify Installation

```bash
# Check system status
docker-compose logs -f  # For Docker
# or
tail -f logs/sentinel.log  # For native

# Run tests
pytest tests/

# Run examples
python3 examples/camera_example.py
python3 examples/visualization_complete_example.py
```

## Common Issues

### Camera not detected
```bash
# List cameras
ls -l /dev/video*

# Test camera
ffplay /dev/video0
```

### GPU not available
```bash
# Check GPU
nvidia-smi

# Verify CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Port already in use
```bash
# Use different port
docker run -p 8081:8080 ...
```

## Next Steps

- Read [INSTALLATION.md](INSTALLATION.md) for detailed setup
- Read [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Explore examples in `examples/` directory
- Customize configuration in `configs/default.yaml`

## Support

- Documentation: See README.md
- Issues: GitHub Issues
- Email: support@sentinel-system.com
