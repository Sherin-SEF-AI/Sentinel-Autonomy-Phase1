# SENTINEL - Contextual Safety Intelligence Platform
<img width="1912" height="1042" alt="image" src="https://github.com/user-attachments/assets/eab827a1-a2b7-4ae6-b735-16029a14e530" />

SENTINEL is a real-time contextual safety intelligence platform for vehicles that prevents accidents by understanding both the environment and driver state in real-time.

## Features

- **360° Bird's Eye View Perception**: Multi-camera sensor fusion creates a unified top-down view
- **Driver Monitoring System**: Tracks driver attention, drowsiness, and distraction
- **Contextual Risk Assessment**: Correlates environmental threats with driver awareness
- **Intelligent Safety Interventions**: Context-aware alerts adapted to driver cognitive load

## Quick Start

### Docker Installation (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access dashboard at http://localhost:8080
```

### Native Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-gpu.txt

# Download pretrained models
python scripts/download_models.py

# Calibrate cameras
python scripts/calibrate_camera.py --camera-id 0 --camera-name interior
python scripts/calibrate_camera.py --camera-id 1 --camera-name front_left
python scripts/calibrate_camera.py --camera-id 2 --camera-name front_right

# Run system
python src/main.py --config configs/default.yaml
```

For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md).

For production deployment, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Usage

```bash
# Start SENTINEL with default configuration
python src/main.py --config configs/default.yaml

# Start with custom log level
python src/main.py --config configs/default.yaml --log-level DEBUG
```

## Project Structure

```
sentinel/
├── src/                    # Source code
│   ├── camera/            # Camera management
│   ├── perception/        # Perception pipeline (BEV, segmentation, detection)
│   ├── dms/              # Driver monitoring system
│   ├── intelligence/     # Contextual intelligence engine
│   ├── alerts/           # Alert system
│   ├── recording/        # Scenario recording
│   ├── visualization/    # Dashboard
│   ├── core/             # Core infrastructure
│   └── main.py           # System orchestration
├── configs/              # Configuration files
├── models/               # Pretrained models
├── scenarios/            # Recorded scenarios
├── tests/                # Test suite
└── scripts/              # Utility scripts
```

## Configuration

All system parameters are configured via YAML files in the `configs/` directory:

- `default.yaml`: Main system configuration
- `calibration/*.yaml`: Camera calibration parameters

## Documentation

- [INSTALLATION.md](INSTALLATION.md) - Complete installation guide with troubleshooting
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment and operations guide
- [configs/default.yaml](configs/default.yaml) - Configuration reference
- Module READMEs in `src/*/README.md` - Component-specific documentation

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_camera.py

# Run with coverage
pytest --cov=src tests/
```

### Running Examples

```bash
# Test camera capture
python examples/camera_example.py

# Test BEV generation
python examples/bev_example.py

# Test full visualization
python examples/visualization_complete_example.py
```

## Performance

- Real-time processing at 30+ FPS
- End-to-end latency under 100ms
- GPU memory usage ≤ 8GB
- CPU usage ≤ 60% on 8-core processor

## License

Copyright © 2024 SENTINEL Project

## Hardware Requirements

### Minimum
- GPU: NVIDIA GPU with compute capability ≥ 7.0, 8GB+ VRAM
- CPU: 8-core processor
- RAM: 16GB
- Storage: 50GB free space (SSD recommended)
- Cameras: 3x USB 3.0 cameras

### Recommended
- GPU: NVIDIA RTX 3070 or better (12GB+ VRAM)
- CPU: 12-core processor
- RAM: 32GB
- Storage: NVMe SSD with 100GB+ free space

See [INSTALLATION.md](INSTALLATION.md) for complete hardware and software requirements.

## Status

✅ **Core System Complete** - All modules implemented and tested. Ready for deployment and integration testing.
