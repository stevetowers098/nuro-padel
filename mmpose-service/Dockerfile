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

# Install system dependencies in single optimized layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
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
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install exact openxlab dependency versions first, then MMPose
# Clean cache during install to save space
RUN pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel && \
    pip install --no-cache-dir pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0 "numpy>=1.21.0,<2.0" && \
    pip install --no-cache-dir mmpose && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

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

# Verify MMPose installation - fail fast if broken
RUN python -c "import numpy as np; print(f'NumPy version: {np.__version__}'); import xtcocotools; print('xtcocotools imported successfully'); import mmpose; print('MMPose imported successfully'); print('✅ All MMPose dependencies working correctly')"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/healthz || exit 1

# Expose port
EXPOSE 8003

# Start command
CMD ["python", "main.py"]