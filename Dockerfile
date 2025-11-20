# SENTINEL System Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0 \
    v4l-utils \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy requirements files
COPY requirements.txt requirements-gpu.txt ./

# Upgrade pip and install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -r requirements-gpu.txt

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Create necessary directories
RUN mkdir -p models scenarios logs .kiro/specs

# Copy model download script and make executable
RUN chmod +x scripts/download_models.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports for visualization dashboard
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Entry point script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python3", "src/main.py", "--config", "configs/default.yaml"]
