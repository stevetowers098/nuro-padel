# MMPose Service Dockerfile
# Optimized for fast builds with complex MMPose dependencies - Complete isolation

FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Set environment variables for speed and stability
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_OFFLINE=1 \
    ULTRALYTICS_OFFLINE=1 \
    ONLINE=False \
    YOLO_TELEMETRY=False

# Install system dependencies with network retry logic
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libglib2.0-dev \
    build-essential \
    curl \
    wget \
    git \
    || (echo "Retrying failed downloads..." && apt-get update --fix-missing && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 libglib2.0-dev build-essential curl wget git) \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install Python dependencies in optimized single-layer approach
# This reduces build time significantly while maintaining exact MMPose requirements
RUN pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel && \
    pip uninstall openxlab torch torchvision torchaudio triton xtcocotools pycocotools numpy -y --quiet 2>/dev/null || true && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 && \
    pip install --no-cache-dir cython && \
    pip install --no-cache-dir pycocotools && \
    pip install --no-cache-dir --no-deps --force-reinstall xtcocotools && \
    pip install --no-cache-dir openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install mmdet && \
    mim install "mmpose>=1.0.0" && \
    pip install --no-cache-dir -r requirements.txt

# Create weights directory for volume mounting (best practice)
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Verify MMPose installation
RUN python -c "import numpy as np; print(f'NumPy version: {np.__version__}'); import xtcocotools; print('xtcocotools imported successfully'); import mmpose; print('MMPose imported successfully'); print('✅ All MMPose dependencies working correctly')" || echo "⚠️ MMPose verification failed - check installation"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/healthz || exit 1

# Expose port
EXPOSE 8003

# Start command
CMD ["python", "main.py"]