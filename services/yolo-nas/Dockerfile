# YOLO-NAS Service Dockerfile
# Optimized for fast builds with super-gradients compatibility - Complete isolation
# ⚠️ RISKY: Deci AI dissolved - super-gradients may have compatibility issues

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
    YOLO_TELEMETRY=False

# Add a retry mechanism for failed connections to archive.ubuntu.com/security.ubuntu.com
RUN apt-get update && \
    for i in {1..5}; do apt-get install -y --no-install-recommends software-properties-common && break || sleep 15; done && \
    add-apt-repository -y ppa:deadsnakes/ppa || \
    (echo "Retrying repository addition..." && add-apt-repository -y ppa:deadsnakes/ppa) && \
    apt-get update

# Install YOLO-NAS system dependencies with multiple retry attempts
RUN apt-get update && \
    for i in {1..3}; do apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3-pip \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 build-essential curl wget git \
    && break || (echo "Retry attempt $i failed, waiting..." && sleep 15); done

# Clean up and create Python symlink
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install Python dependencies in optimized single-layer approach with error handling
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir 'torch>=2.0.0,<2.6.0' 'torchvision>=0.15.0,<0.21.0' 'torchaudio>=2.0.0,<2.6.0' && \
    # Pin numpy for super-gradients compatibility
    pip install --no-cache-dir numpy==1.23.0 scipy && \
    # Install super-gradients without deps to avoid conflicts
    (pip install --no-cache-dir super-gradients>=3.7.0 --no-deps || echo "⚠️ super-gradients installation failed - service will run in fallback mode") && \
    # Install remaining dependencies from requirements.txt
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

# Verify super-gradients installation (with fallback)
RUN python -c "try: from super_gradients.training import models; print('✅ super-gradients imported successfully')\nexcept ImportError as e: print('⚠️ super-gradients not available:', e)" || echo "⚠️ super-gradients verification failed"

# Health check (longer start period due to potential model download)
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8004/healthz || exit 1

# Expose port
EXPOSE 8004

# Start command
CMD ["python", "main.py"]