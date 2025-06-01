FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
# Shared directory removed for isolation; no COPY of shared

# Removed shared directory copy for isolation
# No shared directory for YOLO11; remove COPY of shared

# Create weights directory
RUN mkdir -p /app/weights

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8007/healthz || exit 1

# Expose port
EXPOSE 8007

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007", "--workers", "1"]