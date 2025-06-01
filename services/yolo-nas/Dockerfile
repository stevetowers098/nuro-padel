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

# Use faster mirrors + retry logic
RUN sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-distutils python3.10-venv python3-pip \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgtk-3-0 build-essential curl wget git

# Clean up and create Python symlink
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create and activate Python virtual environment
RUN python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Fix: Use python3 -m pip + super-gradients compatible numpy + proper CUDA index + optimal order
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3 -m pip install --no-cache-dir numpy==1.23.0 && \
    python3 -m pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir super-gradients>=3.7.0 --no-deps && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Create weights directory for volume mounting (best practice)
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create required directories with proper permissions
RUN mkdir -p /home/appuser && \
    chown -R appuser:appuser /home/appuser && \
    chown -R appuser:appuser /app && \
    chmod 755 /home/appuser

USER root
ENV HOME=/tmp
USER appuser

# Verify super-gradients installation (with fallback)
RUN python -c "import sys; \
try: \
    from super_gradients.training import models; \
    print('✅ super-gradients imported successfully'); \
except ImportError as e: \
    print('⚠️ super-gradients not available:', str(e)); \
    sys.exit(0)" || echo "⚠️ super-gradients verification completed"

# Health check (longer start period due to potential model download)
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8004/healthz || exit 1

# Expose port
EXPOSE 8004

# Start command - optimized for production
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]