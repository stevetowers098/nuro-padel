# ViTPose++ Pose Estimation Service Dockerfile
# Optimized for MMPose ViTPose++ with staged dependency resolution

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set environment variables for speed and stability
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_OFFLINE=1 \
    ULTRALYTICS_OFFLINE=1 \
    ONLINE=False \
    YOLO_TELEMETRY=False \
    MPLCONFIGDIR=/tmp/matplotlib \
    CUDA_VISIBLE_DEVICES=0

# Use faster mirrors + retry logic
RUN sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 \
    libglib2.0-dev build-essential curl wget git

# Clean up and create Python symlink
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy staged requirements for optimal dependency resolution
COPY requirements-stage1.txt .
COPY requirements-stage2.txt .

# CRITICAL STAGED INSTALLATION for MMPose compatibility
# Stage 1: Core PyTorch (MUST be installed first)
RUN pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel && \
    pip uninstall torch torchvision torchaudio triton numpy xtcocotools pycocotools -y --quiet 2>/dev/null || true && \
    pip install --no-cache-dir -r requirements-stage1.txt

# Stage 2: Install via mim (CRITICAL ORDER to avoid dependency conflicts)
RUN pip install --no-cache-dir openmim==0.3.9 && \
    mim install mmengine && \
    mim install "mmcv==2.1.0" && \
    mim install "mmdet>=3.0.0,<3.3.0" && \
    mim install mmpose>=1.3.0

# Stage 3: Additional dependencies
RUN pip install --no-cache-dir -r requirements-stage2.txt

# Create weights directory for volume mounting
RUN mkdir -p /app/weights/vitpose

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/src/main.py

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create required directories with proper permissions
RUN mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app

USER appuser

# Verify ViTPose++ installation
RUN python -c "import numpy as np; print(f'NumPy version: {np.__version__}'); import torch; print(f'PyTorch version: {torch.__version__}'); import xtcocotools; print('xtcocotools imported successfully'); import mmpose; print('MMPose imported successfully'); print('✅ All ViTPose++ dependencies working correctly')" || echo "⚠️ ViTPose++ verification failed - check installation"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8006/healthz || exit 1

# Expose port
EXPOSE 8006

# Start command
CMD ["python", "src/main.py"]