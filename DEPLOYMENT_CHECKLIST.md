# SENTINEL Deployment Checklist

Use this checklist to ensure a successful deployment of the SENTINEL system.

## Pre-Deployment

### Hardware Verification
- [ ] GPU meets requirements (NVIDIA, compute capability ≥ 7.0, 8GB+ VRAM)
- [ ] CPU has 8+ cores
- [ ] System has 16GB+ RAM (32GB recommended)
- [ ] Storage has 50GB+ free space
- [ ] SSD is used for models and system files
- [ ] All 3 cameras are connected via USB 3.0
- [ ] Camera mounting is secure and stable
- [ ] Cameras have clear, unobstructed views

### Software Verification
- [ ] Ubuntu 20.04 or 22.04 LTS installed
- [ ] NVIDIA drivers installed (version 520.61.05+)
- [ ] `nvidia-smi` command works
- [ ] CUDA 11.8+ installed
- [ ] Docker installed (for Docker deployment)
- [ ] nvidia-docker2 installed (for Docker deployment)
- [ ] Python 3.10+ installed (for native deployment)

### Model Preparation
- [ ] All models downloaded: `python3 scripts/download_models.py`
- [ ] Model checksums verified: `python3 scripts/download_models.py --verify-only`
- [ ] Models directory has ~152 MB of files
- [ ] Model manifest created: `models/manifest.json`

### Camera Calibration
- [ ] Interior camera calibrated
- [ ] Front-left camera calibrated
- [ ] Front-right camera calibrated
- [ ] Calibration files exist in `configs/calibration/`
- [ ] Calibration validated: `python3 scripts/validate_calibration.py`

### Configuration
- [ ] Configuration file customized: `configs/default.yaml` or `configs/production.yaml`
- [ ] Camera device indices set correctly
- [ ] Model paths configured
- [ ] Risk thresholds tuned (if needed)
- [ ] Alert modalities configured
- [ ] Recording settings configured
- [ ] Visualization settings configured

### Testing
- [ ] Unit tests pass: `pytest tests/`
- [ ] Integration tests pass: `pytest tests/test_integration.py`
- [ ] Performance tests pass: `pytest tests/test_performance.py`
- [ ] Camera capture works: `python3 examples/camera_example.py`
- [ ] BEV generation works: `python3 examples/bev_example.py`
- [ ] Detection works: `python3 examples/detection_example.py`
- [ ] DMS works: `python3 examples/dms_example.py`
- [ ] Full pipeline works: `python3 examples/visualization_complete_example.py`

## Deployment

### Docker Deployment
- [ ] Dockerfile builds successfully: `docker build -t sentinel:latest .`
- [ ] Docker Compose configuration valid: `docker-compose config`
- [ ] Camera devices configured in `docker-compose.yml`
- [ ] Volume mounts configured
- [ ] Resource limits set appropriately
- [ ] Container starts: `docker-compose up -d`
- [ ] Container is healthy: `docker-compose ps`
- [ ] Logs show no errors: `docker-compose logs -f`

### Native Deployment
- [ ] Virtual environment created
- [ ] Dependencies installed: `pip install -r requirements.txt requirements-gpu.txt`
- [ ] System service created (if using systemd)
- [ ] Service enabled: `sudo systemctl enable sentinel`
- [ ] Service starts: `sudo systemctl start sentinel`
- [ ] Service is active: `sudo systemctl status sentinel`
- [ ] Logs show no errors: `tail -f logs/sentinel.log`

### Verification
- [ ] System is running
- [ ] GPU is being utilized: `nvidia-smi`
- [ ] All cameras are capturing frames
- [ ] BEV is being generated
- [ ] Objects are being detected
- [ ] Driver state is being monitored
- [ ] Risks are being assessed
- [ ] Alerts are being generated (when appropriate)
- [ ] Dashboard is accessible: `http://localhost:8080`
- [ ] Performance metrics are acceptable (FPS ≥ 25, latency < 150ms)

## Post-Deployment

### Monitoring Setup
- [ ] Log monitoring configured
- [ ] Performance monitoring configured
- [ ] GPU monitoring configured
- [ ] Disk space monitoring configured
- [ ] Health checks configured
- [ ] Alerting configured (for system issues)

### Backup Configuration
- [ ] Configuration backup scheduled
- [ ] Model backup completed (one-time)
- [ ] Scenario backup scheduled
- [ ] System state backup scheduled
- [ ] Backup destination configured
- [ ] Backup restoration tested

### Documentation
- [ ] Deployment documented (date, version, configuration)
- [ ] Team trained on system operation
- [ ] Troubleshooting guide accessible
- [ ] Contact information updated
- [ ] Maintenance schedule established

### Security
- [ ] Dashboard authentication enabled (if needed)
- [ ] API tokens configured (if needed)
- [ ] File permissions set correctly
- [ ] Firewall configured
- [ ] TLS/SSL enabled (if needed)
- [ ] Data retention policy configured
- [ ] Privacy settings configured

### Performance Optimization
- [ ] GPU persistence mode enabled: `sudo nvidia-smi -pm 1`
- [ ] GPU power limit set (if needed)
- [ ] CPU governor set to performance (if needed)
- [ ] FP16 precision enabled (if needed)
- [ ] Unnecessary services disabled

## 24-Hour Stability Check

After deployment, monitor for 24 hours:

- [ ] System runs continuously for 24 hours
- [ ] No crashes or restarts
- [ ] No memory leaks (memory usage stable)
- [ ] No GPU memory leaks
- [ ] No disk space issues
- [ ] Performance metrics remain stable
- [ ] All cameras remain functional
- [ ] Alerts are appropriate (not too many, not too few)
- [ ] Logs show no recurring errors

## Sign-Off

### Deployment Information
- **Date**: _______________
- **Version**: _______________
- **Deployed By**: _______________
- **Environment**: [ ] Development [ ] Testing [ ] Production
- **Deployment Method**: [ ] Docker [ ] Native [ ] Kubernetes

### Verification
- [ ] All checklist items completed
- [ ] System tested and verified
- [ ] Documentation updated
- [ ] Team notified

### Approvals
- **Technical Lead**: _______________ Date: _______________
- **Operations**: _______________ Date: _______________
- **Project Manager**: _______________ Date: _______________

## Rollback Plan

If deployment fails:

1. **Docker Deployment**:
   ```bash
   docker-compose down
   docker-compose -f docker-compose.backup.yml up -d
   ```

2. **Native Deployment**:
   ```bash
   sudo systemctl stop sentinel
   # Restore configuration from backup
   tar -xzf backup-config-YYYYMMDD.tar.gz
   sudo systemctl start sentinel
   ```

3. **Verify Rollback**:
   - [ ] System is running
   - [ ] Previous version is active
   - [ ] All functionality works
   - [ ] Document rollback reason

## Notes

Use this space for deployment-specific notes:

```
[Add notes here]
```

## References

- [INSTALLATION.md](INSTALLATION.md) - Installation guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [README.md](README.md) - Project overview
