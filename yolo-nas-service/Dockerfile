# YOLO-NAS Service Dockerfile
# ⚠️ RISKY: Deci AI dissolved - super-gradients may have compatibility issues

FROM nvidia/cuda:11.8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# YOLO offline mode for production safety
ENV YOLO_OFFLINE=1
ENV ULTRALYTICS_OFFLINE=1
ENV ONLINE=False
ENV YOLO_TELEMETRY=False

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies with conservative approach
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch first with conservative versions for YOLO-NAS compatibility
RUN pip install --no-cache-dir 'torch>=2.0.0,<2.6.0' 'torchvision>=0.15.0,<0.21.0' 'torchaudio>=2.0.0,<2.6.0' --index-url https://download.pytorch.org/whl/cu118

# Install other core dependencies
RUN pip install --no-cache-dir numpy scipy

# Try to install super-gradients with error handling
RUN pip install --no-cache-dir super-gradients || echo "⚠️ super-gradients installation failed - service will run in fallback mode"

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

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