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

# Install system dependencies with enhanced Python 3.10 support and venv dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    software-properties-common \
    && apt-add-repository ppa:deadsnakes/ppa \
    && apt-get update --fix-missing \
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
    && apt-get update --fix-missing && apt-get install -y google-cloud-cli \
    && mkdir -p /var/run/dbus \
    && (rm -f /etc/resolv.conf && ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf 2>/dev/null || echo "resolv.conf already managed, skipping...") \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

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
    echo "Step 3a: DIAGNOSTIC - Checking super-gradients dependencies before install..." && \
    pip show super-gradients || echo "super-gradients not yet installed" && \
    echo "Step 3b: Installing super-gradients without --no-deps to include required dependencies..." && \
    pip install --no-cache-dir super-gradients==3.7.1 --verbose && \
    echo "Step 3c: DIAGNOSTIC - Verifying super-gradients dependencies were installed..." && \
    pip show albumentations boto3 psutil || echo "Some dependencies missing - this confirms the issue" && \
    echo "Step 4: Installing super-gradients core dependencies manually..." && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --verbose && \
    echo "Step 5: Installing remaining requirements..." && \
    pip install --no-cache-dir -r requirements.txt --verbose && \
    echo "Step 6: Installing super-gradients missing dependencies explicitly..." && \
    pip install --no-cache-dir \
        "albumentations~=1.3" \
        "boto3>=1.17.15" \
        psutil --verbose && \
    echo "Step 7: DIAGNOSTIC - Final verification of all dependencies..." && \
    pip show albumentations boto3 psutil super-gradients && \
    echo "Step 8: Listing installed packages for verification..." && \
    pip list

# Create weights directory
RUN mkdir -p /app/weights

# Copy application code
COPY . .

# Set proper permissions
RUN chmod +x /app/main.py

# Verify super-gradients installation and all required dependencies
RUN echo "🔍 VERIFICATION: Testing super-gradients and its dependencies..." && \
    echo "Checking albumentations..." && \
    python -c "import albumentations; print(f'✅ albumentations version: {albumentations.__version__}')" && \
    echo "Checking boto3..." && \
    python -c "import boto3; print(f'✅ boto3 version: {boto3.__version__}')" && \
    echo "Checking psutil..." && \
    python -c "import psutil; print(f'✅ psutil version: {psutil.__version__}')" && \
    echo "Testing super-gradients import..." && \
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