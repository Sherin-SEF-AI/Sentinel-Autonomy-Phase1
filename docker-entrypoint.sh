#!/bin/bash
# Docker entrypoint script for SENTINEL system

set -e

echo "=========================================="
echo "SENTINEL System - Docker Container"
echo "=========================================="

# Check if models directory is empty
if [ ! "$(ls -A /app/models 2>/dev/null)" ]; then
    echo ""
    echo "⚠️  Models directory is empty"
    echo "Downloading pretrained models..."
    python3 scripts/download_models.py || {
        echo "❌ Model download failed"
        echo "You can manually download models or mount a volume with models"
        echo "Example: docker run -v /path/to/models:/app/models ..."
        exit 1
    }
fi

# Verify GPU access
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo ""
    echo "⚠️  No GPU detected - system will run in CPU mode (not recommended)"
fi

# Check camera devices
echo ""
echo "Camera devices:"
if ls /dev/video* 1> /dev/null 2>&1; then
    ls -l /dev/video* | awk '{print "  " $NF}'
else
    echo "  ⚠️  No camera devices found"
    echo "  Make sure to pass camera devices with --device flag"
    echo "  Example: docker run --device=/dev/video0 ..."
fi

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set permissions
chmod -R 755 /app/scripts
chmod -R 777 /app/logs
chmod -R 777 /app/scenarios

echo ""
echo "Starting SENTINEL system..."
echo "=========================================="
echo ""

# Execute the main command
exec "$@"
