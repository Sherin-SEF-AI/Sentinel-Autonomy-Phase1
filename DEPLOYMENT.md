# SENTINEL System Deployment Guide

This guide covers deployment strategies, production considerations, and operational best practices for the SENTINEL system.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Deployment Strategies](#deployment-strategies)
- [Production Configuration](#production-configuration)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Backup and Recovery](#backup-and-recovery)
- [Security Considerations](#security-considerations)
- [Performance Optimization](#performance-optimization)

---

## Deployment Overview

SENTINEL can be deployed in several configurations:

1. **Standalone Vehicle Deployment**: Single vehicle with local processing
2. **Fleet Deployment**: Multiple vehicles with centralized management
3. **Development/Testing**: Lab environment for testing and validation

---

## Pre-Deployment Checklist

### Hardware Verification

- [ ] GPU meets minimum requirements (compute capability ≥ 7.0, 8GB+ VRAM)
- [ ] CPU has sufficient cores (8+ recommended)
- [ ] System has adequate RAM (16GB+ minimum, 32GB recommended)
- [ ] Storage has sufficient space (50GB+ free)
- [ ] All 3 cameras are connected and functional
- [ ] Camera mounting is secure and stable
- [ ] USB 3.0 connections are used for cameras

### Software Verification

- [ ] Operating system is Ubuntu 20.04/22.04 LTS
- [ ] NVIDIA drivers are installed (version 520.61.05+)
- [ ] CUDA 11.8+ is installed and functional
- [ ] Docker and nvidia-docker2 are installed (for Docker deployment)
- [ ] All pretrained models are downloaded and verified
- [ ] Camera calibration is completed for all cameras
- [ ] Configuration files are customized for deployment environment

### Testing Verification

- [ ] All unit tests pass: `pytest tests/`
- [ ] Integration tests pass: `pytest tests/test_integration.py`
- [ ] Performance benchmarks meet requirements: `pytest tests/test_performance.py`
- [ ] Camera capture works correctly
- [ ] BEV generation produces valid output
- [ ] Object detection identifies objects correctly
- [ ] DMS detects driver state accurately
- [ ] Alerts are generated appropriately
- [ ] Visualization dashboard is accessible

---

## Deployment Strategies

### Strategy 1: Docker Deployment (Recommended)

**Advantages**:
- Consistent environment across deployments
- Easy updates and rollbacks
- Isolated dependencies
- Simplified deployment process

**Steps**:

1. **Prepare Environment**:
```bash
# Install Docker and nvidia-docker2
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. **Build Image**:
```bash
cd sentinel
docker build -t sentinel:v1.0.0 .
docker tag sentinel:v1.0.0 sentinel:latest
```

3. **Configure Deployment**:
```bash
# Edit docker-compose.yml for production
# Update camera devices, volume mounts, resource limits
nano docker-compose.yml
```

4. **Deploy**:
```bash
docker-compose up -d
```

5. **Verify**:
```bash
docker-compose logs -f
curl http://localhost:8080/health
```

### Strategy 2: Native Deployment

**Advantages**:
- Direct hardware access
- Lower overhead
- Easier debugging

**Steps**:

1. **Install Dependencies**:
```bash
# Follow INSTALLATION.md for complete setup
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

2. **Configure System Service**:
```bash
# Create systemd service
sudo nano /etc/systemd/system/sentinel.service
```

```ini
[Unit]
Description=SENTINEL Safety Intelligence System
After=network.target

[Service]
Type=simple
User=sentinel
WorkingDirectory=/opt/sentinel
Environment="PYTHONPATH=/opt/sentinel"
ExecStart=/opt/sentinel/venv/bin/python src/main.py --config configs/production.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

3. **Enable and Start Service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable sentinel
sudo systemctl start sentinel
sudo systemctl status sentinel
```

### Strategy 3: Kubernetes Deployment (Fleet)

For fleet deployments with multiple vehicles:

```yaml
# sentinel-deployment.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: sentinel
  namespace: fleet
spec:
  selector:
    matchLabels:
      app: sentinel
  template:
    metadata:
      labels:
        app: sentinel
    spec:
      nodeSelector:
        vehicle: "true"
      containers:
      - name: sentinel
        image: sentinel:v1.0.0
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
          requests:
            nvidia.com/gpu: 1
            memory: 8Gi
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: scenarios
          mountPath: /app/scenarios
        - name: dev-video
          mountPath: /dev
      volumes:
      - name: models
        hostPath:
          path: /opt/sentinel/models
      - name: scenarios
        hostPath:
          path: /opt/sentinel/scenarios
      - name: dev-video
        hostPath:
          path: /dev
```

---

## Production Configuration

### Configuration File Structure

Create a production-specific configuration:

```bash
cp configs/default.yaml configs/production.yaml
nano configs/production.yaml
```

### Key Production Settings

```yaml
system:
  name: "SENTINEL-PROD"
  log_level: "INFO"  # Use INFO in production, DEBUG for troubleshooting
  
# Optimize for production performance
models:
  segmentation:
    precision: "fp16"  # Use FP16 for better performance
    device: "cuda"
  detection:
    confidence_threshold: 0.6  # Adjust based on validation
  dms:
    face_detection: "MediaPipe"

# Production risk thresholds (tune based on field testing)
risk_assessment:
  thresholds:
    hazard_detection: 0.3
    intervention: 0.7
    critical: 0.9

# Alert configuration
alerts:
  suppression:
    duplicate_window: 5.0
    max_simultaneous: 2
  modalities:
    visual:
      enabled: true
    audio:
      enabled: true
      volume: 0.8
    haptic:
      enabled: true  # Enable if hardware supports

# Recording configuration
recording:
  enabled: true
  triggers:
    risk_threshold: 0.7
    ttc_threshold: 1.5
  storage_path: "/data/scenarios/"
  max_duration: 30.0
  retention_days: 30  # Auto-delete old scenarios

# Visualization (disable in production if not needed)
visualization:
  enabled: false  # Disable to save resources
```

### Environment Variables

Set environment variables for production:

```bash
# Create .env file
cat > .env << EOF
SENTINEL_ENV=production
SENTINEL_CONFIG=/app/configs/production.yaml
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=INFO
ENABLE_TELEMETRY=true
EOF
```

---

## Monitoring and Maintenance

### System Monitoring

1. **Log Monitoring**:
```bash
# Real-time log monitoring
tail -f logs/sentinel.log

# Error monitoring
tail -f logs/errors.log

# Performance monitoring
tail -f logs/performance.log
```

2. **Resource Monitoring**:
```bash
# GPU monitoring
nvidia-smi -l 1

# System resources
htop

# Disk usage
df -h
du -sh scenarios/
```

3. **Health Checks**:
```bash
# HTTP health endpoint
curl http://localhost:8080/health

# System status
systemctl status sentinel  # For systemd deployment
docker-compose ps  # For Docker deployment
```

### Automated Monitoring

Set up monitoring with Prometheus and Grafana:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sentinel'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
```

### Alerting

Configure alerts for critical conditions:

```yaml
# alertmanager.yml
route:
  receiver: 'sentinel-alerts'
  
receivers:
  - name: 'sentinel-alerts'
    email_configs:
      - to: 'ops@example.com'
        from: 'sentinel@example.com'
        
groups:
  - name: sentinel
    rules:
      - alert: HighGPUMemory
        expr: gpu_memory_used > 7000
        for: 5m
      - alert: LowFPS
        expr: system_fps < 25
        for: 1m
      - alert: CameraFailure
        expr: camera_healthy == 0
        for: 30s
```

### Maintenance Tasks

**Daily**:
- Check system logs for errors
- Verify camera functionality
- Monitor disk space usage

**Weekly**:
- Review recorded scenarios
- Check performance metrics
- Verify model accuracy on sample data

**Monthly**:
- Update system dependencies
- Review and tune risk thresholds
- Clean up old scenario recordings
- Backup configuration and calibration files

**Quarterly**:
- Re-calibrate cameras if needed
- Update pretrained models
- Performance benchmarking
- Security audit

---

## Backup and Recovery

### Backup Strategy

1. **Configuration Backup**:
```bash
# Backup configuration files
tar -czf backup-config-$(date +%Y%m%d).tar.gz configs/

# Backup to remote storage
rsync -avz configs/ backup-server:/backups/sentinel/configs/
```

2. **Model Backup**:
```bash
# Backup models (one-time or after updates)
tar -czf backup-models-$(date +%Y%m%d).tar.gz models/
```

3. **Scenario Backup**:
```bash
# Backup critical scenarios
rsync -avz scenarios/ backup-server:/backups/sentinel/scenarios/

# Or use automated backup script
python scripts/backup_scenarios.py --days 7 --destination /backup/
```

4. **System State Backup**:
```bash
# Backup system state
cp -r .kiro/state/ backup-state-$(date +%Y%m%d)/
```

### Recovery Procedures

**Scenario 1: System Crash**
```bash
# System automatically restores state on restart
# Verify recovery
systemctl status sentinel
tail -f logs/sentinel.log
```

**Scenario 2: Configuration Corruption**
```bash
# Restore from backup
tar -xzf backup-config-YYYYMMDD.tar.gz
systemctl restart sentinel
```

**Scenario 3: Model Corruption**
```bash
# Re-download models
python scripts/download_models.py --force

# Or restore from backup
tar -xzf backup-models-YYYYMMDD.tar.gz
```

**Scenario 4: Complete System Failure**
```bash
# Reinstall system
# Restore configuration
tar -xzf backup-config-YYYYMMDD.tar.gz
# Restore models
tar -xzf backup-models-YYYYMMDD.tar.gz
# Restart system
systemctl start sentinel
```

---

## Security Considerations

### Access Control

1. **Dashboard Authentication**:
```yaml
# Add authentication to visualization dashboard
visualization:
  enabled: true
  auth:
    enabled: true
    username: "admin"
    password_hash: "bcrypt_hash_here"
```

2. **API Security**:
```python
# Use API tokens for programmatic access
# Configure in configs/production.yaml
api:
  enabled: true
  require_auth: true
  tokens:
    - "secure_token_1"
    - "secure_token_2"
```

3. **File Permissions**:
```bash
# Restrict access to sensitive files
chmod 600 configs/production.yaml
chmod 700 scenarios/
chown -R sentinel:sentinel /opt/sentinel
```

### Data Privacy

1. **Encryption at Rest**:
```bash
# Encrypt scenario recordings
# Use LUKS for disk encryption
sudo cryptsetup luksFormat /dev/sdX
sudo cryptsetup open /dev/sdX sentinel-data
```

2. **Data Retention**:
```yaml
# Configure automatic deletion
recording:
  retention_days: 30
  auto_cleanup: true
```

3. **Anonymization**:
```python
# Blur faces in recorded scenarios (optional)
recording:
  anonymize_faces: true
```

### Network Security

1. **Firewall Configuration**:
```bash
# Allow only necessary ports
sudo ufw enable
sudo ufw allow 8080/tcp  # Dashboard
sudo ufw allow 22/tcp    # SSH
```

2. **TLS/SSL**:
```yaml
# Enable HTTPS for dashboard
visualization:
  ssl:
    enabled: true
    cert_file: "/etc/ssl/certs/sentinel.crt"
    key_file: "/etc/ssl/private/sentinel.key"
```

---

## Performance Optimization

### GPU Optimization

1. **Enable Persistence Mode**:
```bash
sudo nvidia-smi -pm 1
```

2. **Set Power Limit**:
```bash
# Adjust based on your GPU
sudo nvidia-smi -pl 300
```

3. **Use TensorRT** (Advanced):
```bash
# Convert models to TensorRT
python scripts/optimize_models.py --backend tensorrt
```

### CPU Optimization

1. **Set CPU Governor**:
```bash
sudo cpupower frequency-set -g performance
```

2. **Pin Processes to Cores**:
```bash
# Use taskset to pin to specific cores
taskset -c 0-7 python src/main.py
```

### Memory Optimization

1. **Reduce Model Precision**:
```yaml
models:
  segmentation:
    precision: "fp16"  # Use FP16 instead of FP32
```

2. **Limit Scenario Recording**:
```yaml
recording:
  max_duration: 20.0  # Reduce from 30s
  compression: "h264"  # Use compression
```

### Disk I/O Optimization

1. **Use SSD for Models**:
```bash
# Mount SSD for models directory
sudo mount /dev/nvme0n1 /opt/sentinel/models
```

2. **Optimize Logging**:
```yaml
logging:
  level: "INFO"  # Reduce from DEBUG
  rotation:
    max_bytes: 10485760  # 10MB
    backup_count: 3
```

---

## Troubleshooting Production Issues

### High Latency

1. Check GPU utilization: `nvidia-smi`
2. Check CPU usage: `htop`
3. Review performance logs: `tail -f logs/performance.log`
4. Reduce model precision to FP16
5. Disable visualization if not needed

### System Crashes

1. Check system logs: `journalctl -xe`
2. Check SENTINEL logs: `tail -f logs/errors.log`
3. Verify GPU memory: `nvidia-smi`
4. Check disk space: `df -h`
5. Review crash dumps in `logs/crashes/`

### Camera Issues

1. Check camera connections: `ls -l /dev/video*`
2. Test camera capture: `ffplay /dev/video0`
3. Verify calibration: `python scripts/validate_calibration.py`
4. Check USB bandwidth: `lsusb -t`

### Model Inference Errors

1. Verify model files: `ls -lh models/`
2. Check GPU memory: `nvidia-smi`
3. Validate model checksums: `python scripts/download_models.py --verify-only`
4. Try CPU inference temporarily
5. Re-download models if corrupted

---

## Support and Resources

- **Documentation**: See `INSTALLATION.md` for setup details
- **Issue Tracking**: GitHub Issues
- **Community Forum**: [Forum URL]
- **Email Support**: support@sentinel-system.com
- **Emergency Contact**: [Emergency contact info]

---

## Appendix

### Deployment Checklist

```
Pre-Deployment:
[ ] Hardware verified
[ ] Software installed
[ ] Models downloaded
[ ] Cameras calibrated
[ ] Configuration customized
[ ] Tests passed

Deployment:
[ ] System deployed
[ ] Health checks passing
[ ] Monitoring configured
[ ] Backups configured
[ ] Documentation updated

Post-Deployment:
[ ] System running stable for 24h
[ ] Performance metrics acceptable
[ ] Alerts functioning correctly
[ ] Team trained on operation
[ ] Maintenance schedule established
```

### Performance Targets

| Metric | Target | Acceptable | Action if Below |
|--------|--------|------------|-----------------|
| FPS | ≥30 | ≥25 | Optimize models, reduce precision |
| Latency (p95) | <100ms | <150ms | Check GPU utilization, reduce load |
| GPU Memory | <7GB | <8GB | Reduce batch size, use FP16 |
| CPU Usage | <60% | <80% | Optimize code, add cores |
| Uptime | >99.9% | >99% | Investigate crashes, improve stability |

### Contact Information

For deployment support:
- Technical Lead: [Name, Email]
- Operations: [Name, Email]
- Emergency: [Phone Number]
