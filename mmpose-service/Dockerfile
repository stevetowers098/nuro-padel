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

# Fix potential resolv.conf symbolic link issue in Docker
RUN ln -sfT /run/systemd/resolve/resolv.conf /etc/resolv.conf || echo "Skipping resolv.conf symlink"

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
    dbus \
    && mkdir -p /var/run/dbus \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install PyTorch 1.12.x (MIM-compatible version) and MMPose ecosystem with verbose output
# Clean cache during install to save space
RUN echo "🔍 VERBOSE INSTALL: Installing MMPose dependencies with detailed output..." && \
    echo "Step 1: Upgrading pip and installing base dependencies..." && \
    pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel --verbose && \
    echo "Step 2: Installing numpy..." && \
    pip install --no-cache-dir numpy --verbose && \
    echo "Step 3: Installing PyTorch ecosystem..." && \
    pip install --no-cache-dir torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117 --verbose && \
    echo "Step 4: Installing OpenMIM..." && \
    pip install --no-cache-dir openmim --verbose && \
    echo "Step 5: Installing compatible package versions..." && \
    pip install --no-cache-dir pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0 --verbose && \
    echo "Step 6: Installing MMPose ecosystem via MIM..." && \
    mim install mmcv-full mmpose xtcocotools && \
    echo "Step 7: Installing additional requirements..." && \
    pip install --no-cache-dir -r requirements.txt --verbose && \
    echo "Step 8: Cleaning up..." && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create weights directory
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Verify MMPose installation with comprehensive testing
RUN echo "🔍 VERIFICATION: Testing all MMPose dependencies..." && \
    python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}'); import xtcocotools; print('✅ xtcocotools imported successfully'); import mmpose; print('✅ MMPose imported successfully'); print('📊 All MMPose dependencies working correctly')"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8003/healthz || exit 1

# Expose port
EXPOSE 8003

# Start command
CMD ["python", "main.py"]