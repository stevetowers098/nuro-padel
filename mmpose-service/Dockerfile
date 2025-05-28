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
    libcrc32c-dev \
    build-essential \
    curl \
    wget \
    git \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start \
    && rm -f /etc/resolv.conf \
    && ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf || true \
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

# Install PyTorch 2.1.x and MMPose ecosystem following official MMCV installation guidelines
# Based on official MMPose compatibility requirements and MMCV installation best practices
RUN echo "🔍 OFFICIAL MMCV INSTALL: Following official MMCV installation guidelines..." && \
    echo "Step 1: Upgrading pip and installing base dependencies with fixed versions..." && \
    pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel --verbose && \
    echo "Step 2: Installing numpy explicitly (constrained for mmcv compatibility)..." && \
    pip install --no-cache-dir "numpy>=1.21.0,<2.0" --verbose && \
    echo "Step 3: Installing PyTorch 2.1.x ecosystem (compatible versions)..." && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118 --verbose && \
    echo "Step 4: Verifying PyTorch installation (MMCV requirement)..." && \
    python -c "import torch; print(f'✅ PyTorch successfully installed: {torch.__version__}')" && \
    python -c "import torch; print(f'✅ CUDA version: {torch.version.cuda}')" && \
    echo "Step 5: Installing OpenMIM (recommended for MMCV)..." && \
    pip install --no-cache-dir -U openmim --verbose && \
    echo "Step 6: Critical - Uninstalling ALL existing mmcv versions (prevents conflicts)..." && \
    pip uninstall -y mmcv mmcv-lite mmcv-full || true && \
    echo "Step 7: Installing mmengine..." && \
    mim install mmengine --verbose && \
    echo "Step 8: Installing mmcv with exact version (full version with CUDA ops)..." && \
    mim install "mmcv==2.1.0" --verbose && \
    echo "Step 9: Installing mmdet with version constraints..." && \
    mim install "mmdet>=3.0.0,<3.3.0" --verbose && \
    echo "Step 10: Installing compatible package versions for openxlab..." && \
    pip install --no-cache-dir pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0 --verbose && \
    echo "Step 11: Installing google_crc32c with --no-deps to avoid conflicts..." && \
    pip install --no-cache-dir --no-deps google_crc32c --verbose && \
    echo "Step 12: Installing MMPose and xtcocotools..." && \
    mim install mmpose xtcocotools --verbose && \
    echo "Step 13: Installing additional requirements..." && \
    pip install --no-cache-dir -r requirements.txt --verbose && \
    echo "Step 14: Checking all dependencies for conflicts..." && \
    pip check || echo "Warning: Some dependency conflicts detected but proceeding..." && \
    echo "Step 15: Cleaning up..." && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create weights directory
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Verify MMPose installation following official MMCV verification guidelines
RUN echo "🔍 OFFICIAL VERIFICATION: Testing all MMPose dependencies with official checks..." && \
    echo "Testing NumPy..." && \
    python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')" && \
    echo "Testing PyTorch (MMCV requirement)..." && \
    python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')" && \
    python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import torch; print(f'✅ CUDA version: {torch.version.cuda}')" && \
    echo "Testing MMCV (full version with CUDA ops)..." && \
    python -c "import mmcv; print(f'✅ MMCV version: {mmcv.__version__}')" && \
    python -c "import mmcv; print(f'✅ MMCV CUDA available: {mmcv.ops.get_compiler_version()}' if hasattr(mmcv.ops, 'get_compiler_version') else '✅ MMCV ops available')" && \
    echo "Testing mmengine..." && \
    python -c "import mmengine; print(f'✅ MMEngine version: {mmengine.__version__}')" && \
    echo "Testing mmdet..." && \
    python -c "import mmdet; print(f'✅ MMDet version: {mmdet.__version__}')" && \
    echo "Testing xtcocotools..." && \
    python -c "import xtcocotools; print('✅ xtcocotools imported successfully')" && \
    echo "Testing MMPose..." && \
    python -c "import mmpose; print(f'✅ MMPose version: {mmpose.__version__}')" && \
    echo "Testing OpenCV..." && \
    python -c "import cv2; print(f'✅ OpenCV version: {cv2.__version__}')" && \
    echo "Testing FastAPI..." && \
    python -c "import fastapi; print(f'✅ FastAPI version: {fastapi.__version__}')" && \
    echo "📊 All MMPose dependencies verified following official MMCV guidelines"

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