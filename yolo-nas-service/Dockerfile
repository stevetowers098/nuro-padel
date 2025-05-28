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

# Fix symbolic link issue - Device or resource busy error resolution
RUN rm -f /etc/resolv.conf && ln -s /run/systemd/resolve/resolv.conf /etc/resolv.conf

# Install system dependencies with enhanced Python 3.10 support and venv dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && (add-apt-repository ppa:deadsnakes/ppa || true) \
    && apt-get update || apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-venv \
    python3-pip \
    python3-dev \
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
    apt-transport-https \
    ca-certificates \
    gnupg \
    dbus \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
    && apt-get update && apt-get install -y google-cloud-cli \
    && mkdir -p /var/run/dbus \
    && service dbus start \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment with enhanced error handling to avoid Python version conflicts
RUN echo "🔧 Creating Python virtual environment with Python 3.10..." && \
    python3.10 -m venv /opt/venv && \
    echo "✅ Virtual environment created successfully" && \
    /opt/venv/bin/python -m ensurepip --upgrade && \
    echo "✅ Virtual environment pip ensured and upgraded" && \
    /opt/venv/bin/python --version && \
    echo "✅ Python version verified in virtual environment"
ENV PATH="/opt/venv/bin:$PATH"

# Verify Python version meets networkx requirements (>=3.9)
RUN echo "🔍 PYTHON VERSION CHECK: Verifying Python >=3.9 for networkx compatibility..." && \
    python --version && \
    python -c "import sys; assert sys.version_info >= (3, 9), f'Python {sys.version_info} < 3.9 required for networkx'; print('✅ Python version meets networkx requirements')"

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install dependencies with conflict resolution - exclude problematic packages
RUN echo "🔍 RESOLVING DEPENDENCIES: Installing with conflict resolution..." && \
    echo "Step 1: Upgrading pip tools..." && \
    pip install --no-cache-dir --upgrade pip setuptools==65.7.0 wheel --verbose && \
    echo "Step 2: Installing super-gradients compatible versions first..." && \
    pip install --no-cache-dir "numpy==1.23.0" "requests==2.31.0" --verbose && \
    echo "Step 3: Installing super-gradients with conflicting packages excluded..." && \
    pip install --no-cache-dir super-gradients==3.7.1 --no-deps --verbose && \
    echo "Step 4: Installing super-gradients core dependencies manually..." && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --verbose && \
    echo "Step 5: Installing remaining requirements..." && \
    pip install --no-cache-dir -r requirements.txt --verbose && \
    echo "Step 6: Listing installed packages for verification..." && \
    pip list

# Create weights directory
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Verify super-gradients installation - fail fast if broken
RUN echo "🔍 VERIFICATION: Testing super-gradients import..." && \
    python -c "from super_gradients.training import models; print('✅ super-gradients imported successfully')"

# Additional verification for debugging (non-critical imports)
RUN echo "🔍 ADDITIONAL CHECKS: Testing other key imports for debugging..." && \
    python -c "import cv2; print('✅ opencv imported successfully')" && \
    python -c "import fastapi; print('✅ fastapi imported successfully')" && \
    python -c "import numpy; print('✅ numpy imported successfully')" && \
    python -c "import requests; print('✅ requests imported successfully')" && \
    echo "📊 All imports verified successfully"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check (longer start period due to potential model download)
HEALTHCHECK --interval=30s --timeout=30s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8004/healthz || exit 1

# Expose port
EXPOSE 8004

# Start command
CMD ["python", "main.py"]