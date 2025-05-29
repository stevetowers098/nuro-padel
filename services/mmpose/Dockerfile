# MMPose Service Dockerfile
# Complex installation requirements - must install MMPose stack in specific order

FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# YOLO offline mode for production safety
ENV YOLO_OFFLINE=1
ENV ULTRALYTICS_OFFLINE=1
ENV ONLINE=False
ENV YOLO_TELEMETRY=False

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
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
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install basic Python dependencies first
RUN pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel

# Clean up any conflicting packages that might cause issues
RUN pip uninstall openxlab torch torchvision torchaudio triton xtcocotools pycocotools numpy -y --quiet 2>/dev/null || true

# Install compatible NumPy version first
RUN pip install --no-cache-dir numpy==1.24.3

# Install PyTorch with compatible versions
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install Cython before cocotools
RUN pip install --no-cache-dir cython

# Install cocotools in correct order
RUN pip install --no-cache-dir pycocotools
RUN pip install --no-cache-dir --no-deps --force-reinstall xtcocotools

# Install openmim for MMPose installation
RUN pip install --no-cache-dir openmim

# Install MMPose stack using mim (correct order is critical)
RUN mim install mmengine
RUN mim install "mmcv>=2.0.1"
RUN mim install mmdet
RUN mim install "mmpose>=1.0.0"

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create weights directory
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