# YOLO-NAS Service Dockerfile
# Optimized for fast builds with super-gradients compatibility - Complete isolation
# ⚠️ RISKY: Deci AI dissolved - super-gradients may have compatibility issues

FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

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
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
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

# Install Python dependencies in optimized single-layer approach with error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir 'torch>=2.0.0,<2.6.0' 'torchvision>=0.15.0,<0.21.0' 'torchaudio>=2.0.0,<2.6.0' && \
    pip install --no-cache-dir numpy scipy && \
    (pip install --no-cache-dir super-gradients || echo "⚠️ super-gradients installation failed - service will run in fallback mode") && \
    pip install --no-cache-dir -r requirements.txt

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

# Verify super-gradients installation (with fallback)
RUN python -c "try: from super_gradients.training import models; print('✅ super-gradients imported successfully')\nexcept ImportError as e: print('⚠️ super-gradients not available:', e)" || echo "⚠️ super-gradients verification failed"

# Health check (longer start period due to potential model download)
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8004/healthz || exit 1

# Expose port
EXPOSE 8004

# Start command
CMD ["python", "main.py"]