# YOLO Combined Service Dockerfile (YOLO11 + YOLOv8)
# Multi-stage build for optimized container size

FROM ubuntu:22.04 AS base

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
    libglib2.0-0 \
    libgtk-3-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
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

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

# Expose port
EXPOSE 8001

# Start command
CMD ["python", "main.py"]