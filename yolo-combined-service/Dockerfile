# YOLO Combined Service Dockerfile (YOLO11 + YOLOv8)
# Optimized for fast builds while maintaining complete service isolation

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

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

# Install system dependencies in single layer for better caching
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    curl \
    wget \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for optimal Docker layer caching
COPY requirements.txt .

# Install Python dependencies in optimized order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
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

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

# Expose port
EXPOSE 8001

# Start command
CMD ["python", "main.py"]