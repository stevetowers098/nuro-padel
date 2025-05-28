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

# Fix symbolic link issue - Device or resource busy error resolution
RUN ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf

# Install system dependencies in single optimized layer with enhanced Python venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
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
    libglib2.0-dev \
    build-essential \
    curl \
    wget \
    git \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create virtual environment with enhanced error handling to avoid pip root warnings
RUN echo "🔧 Creating Python virtual environment..." && \
    python3 -m venv /opt/venv && \
    echo "✅ Virtual environment created successfully" && \
    /opt/venv/bin/python -m ensurepip --upgrade && \
    echo "✅ Virtual environment pip ensured and upgraded"
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install PyTorch 1.12.x (MIM-compatible version) and MMPose ecosystem with enhanced error handling
# Clean cache during install to save space
RUN echo "🔍 ENHANCED INSTALL: Installing MMPose dependencies with error handling..." && \
    echo "Step 1: Upgrading pip and installing base dependencies..." && \
    pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel --verbose && \
    echo "Step 2: Installing numpy explicitly..." && \
    pip install --no-cache-dir numpy --verbose && \
    echo "Step 3: Installing PyTorch ecosystem..." && \
    pip install --no-cache-dir torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu117 --verbose && \
    echo "Step 4: Installing OpenMIM..." && \
    pip install --no-cache-dir openmim --verbose && \
    echo "Step 5: Installing compatible package versions for openxlab..." && \
    pip install --no-cache-dir pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0 --verbose && \
    echo "Step 6: Installing MMPose ecosystem via MIM..." && \
    mim install mmcv-full mmpose xtcocotools && \
    echo "Step 7: Explicitly ensuring core dependencies..." && \
    pip install --no-cache-dir numpy xtcocotools mmpose --verbose && \
    echo "Step 8: Installing additional requirements..." && \
    pip install --no-cache-dir -r requirements.txt --verbose && \
    echo "Step 9: Cleaning up..." && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create weights directory
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Verify MMPose installation with enhanced error handling
RUN echo "🔍 VERIFICATION: Testing all MMPose dependencies individually..." && \
    echo "Testing NumPy..." && \
    python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')" && \
    echo "Testing xtcocotools..." && \
    python -c "import xtcocotools; print('✅ xtcocotools imported successfully')" && \
    echo "Testing MMPose..." && \
    python -c "import mmpose; print('✅ MMPose imported successfully')" && \
    echo "Testing OpenCV..." && \
    python -c "import cv2; print('✅ OpenCV imported successfully')" && \
    echo "Testing FastAPI..." && \
    python -c "import fastapi; print('✅ FastAPI imported successfully')" && \
    echo "📊 All MMPose dependencies verified successfully"

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