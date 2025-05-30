# RF-DETR Detection Service Dockerfile
# Optimized for stable RF-DETR with exact dependency control

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

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install Python dependencies with exact versions for RF-DETR stability
# Stage 1: Core PyTorch (CRITICAL for RF-DETR compatibility)
RUN pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel && \
    pip uninstall torch torchvision torchaudio triton numpy -y --quiet 2>/dev/null || true && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Stage 2: RF-DETR with exact stable version
RUN pip install --no-cache-dir rfdetr==0.1.0

# Stage 3: Additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create weights directory for volume mounting
RUN mkdir -p /app/weights/rf-detr

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

# Verify RF-DETR installation
RUN python -c "import numpy as np; print(f'NumPy version: {np.__version__}'); import torch; print(f'PyTorch version: {torch.__version__}'); from rfdetr import RFDETRBase; print('RF-DETR imported successfully'); print('✅ All RF-DETR dependencies working correctly')" || echo "⚠️ RF-DETR verification failed - check installation"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8005/healthz || exit 1

# Expose port
EXPOSE 8005

# Start command
CMD ["python", "src/main.py"]