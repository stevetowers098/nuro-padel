# NuroPadel Deployment Guide

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB+ available space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for acceleration)
- **OS**: Ubuntu 20.04+ or compatible Docker environment

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- Google Cloud SDK (for GCS uploads)

## Exact Version Requirements

### Critical Dependency Versions

#### YOLO-NAS Service
```txt
# Super-gradients compatibility requirements
numpy==1.23.0          # Must be ≤1.23 for super-gradients
sphinx==4.0.2          # Must be ~4.0.2 for super-gradients
super-gradients==3.7.1
requests==2.31.0
docutils==0.18.0
```

#### MMPose Service
```txt
# OpenXLab compatibility requirements
torch==1.12.1
torchvision==0.13.1
mmcv-full==1.7.2
mmpose==1.1.0
pytz==2023.3           # Must be 2023.3 for openxlab
requests==2.28.2
rich==13.4.2
tqdm==4.65.0
xtcocotools
```

#### YOLO Combined Service
```txt
torch==2.3.1
torchvision==0.18.1
ultralytics==8.2.97
opencv-python-headless==4.10.0.84
fastapi==0.111.0
google-cloud-storage==2.10.0
```

## Known Issues & Solutions

### 1. Symbolic Link Creation Failures

**Error**: `ln: failed to create symbolic link '/etc/resolv.conf': Device or resource busy`

**Root Cause**: `/etc/resolv.conf` is already in use or locked by another process

**Solution Applied in Dockerfiles**:
```dockerfile
# Enhanced conditional check for MMPose
RUN if [ ! -L /etc/resolv.conf ]; then \
    ln -sfT /run/systemd/resolve/resolv.conf /etc/resolv.conf; \
    else echo "Symbolic link already exists"; \
fi

# YOLO-NAS force removal approach
RUN rm -f /etc/resolv.conf && ln -sf /run/systemd/resolve/resolv.conf /etc/resolv.conf || true
```

### 2. Super-gradients Dependency Conflicts

**Error**: `ModuleNotFoundError: No module named 'super_gradients'`

**Root Cause**: Multiple packages require different versions:
- `albumentations 1.4.18` requires `numpy>=1.24.4`
- `sphinx-rtd-theme 3.0.2` requires `sphinx>=6,<9`
- `pyhanko 0.27.1` requires `requests>=2.31.0`

**Solution Applied - Use Globally Compatible Versions**:
```dockerfile
# Install globally compatible versions that satisfy ALL packages
RUN pip install "numpy==1.24.4" "requests==2.31.0" "sphinx>=6,<9" "docutils>=0.18,<0.22"
RUN pip install super-gradients==3.7.1
RUN pip install -r requirements.txt
```

**Key Insight**: Instead of downgrading to super-gradients' preferred versions, upgrade to versions that satisfy all packages.

### 3. MMPose OpenXLab Conflicts

**Error**: `openxlab 0.1.2 has requirement pytz~=2023.3, but you'll have pytz 2025.2`

**Solution Applied**:
```dockerfile
# Install specific compatible versions
RUN pip install pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0
RUN mim install mmcv-full mmpose xtcocotools
```

### 4. Missing numpy Dependency

**Error**: `ModuleNotFoundError: No module named 'numpy'`

**Solution Applied**:
```dockerfile
# Install numpy explicitly before other dependencies
RUN pip install numpy
# Ensure numpy is available during verification
RUN pip install --no-cache-dir numpy xtcocotools mmpose --verbose
```

### 5. Missing Python.h Development Headers

**Error**: `fatal error: Python.h: No such file or directory`

**Root Cause**: Python development headers missing for C extensions

**Solution Applied**:
```dockerfile
# Install Python development headers
RUN apt-get update && apt-get install -y python3.10-dev python3-dev
```

### 6. Pip Running as Root Warning

**Warning**: `WARNING: Running pip as the 'root' user can result in broken permissions`

**Solution Applied**:
```dockerfile
# Create virtual environment to avoid root pip issues
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -r requirements.txt
```

### 7. Enhanced Dependency Verification

**Error**: `ERROR: failed to solve: process "/bin/sh -c echo "🔍 VERIFICATION..."`

**Solution Applied**:
```dockerfile
# Test each dependency individually for better error isolation
RUN echo "Testing NumPy..." && \
    python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')" && \
    echo "Testing xtcocotools..." && \
    python -c "import xtcocotools; print('✅ xtcocotools imported successfully')" && \
    echo "Testing MMPose..." && \
    python -c "import mmpose; print('✅ MMPose imported successfully')"
```

## Deployment Methods

### Method 1: Standard Docker Compose

```bash
# Start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f [service-name]
```

### Method 2: Sequential Deployment (Space Optimized)

For environments with limited disk space (GitHub runners, small VMs):

```bash
# Deploy services one at a time
./deploy.sh --deploy-sequential

# Deploy specific service
./deploy.sh --deploy-seq yolo-combined
./deploy.sh --deploy-seq mmpose
./deploy.sh --deploy-seq yolo-nas

# Clean up disk space aggressively
./deploy.sh --cleanup-disk
```

### Method 3: GitHub Actions Sequential

```yaml
# Deploy single service
gh workflow run sequential-deploy.yml -f service=yolo-combined

# Deploy all services sequentially  
gh workflow run sequential-deploy.yml -f service=all
```

## Disk Space Optimization

### Problem
GitHub runners have ~14GB disk space. PyTorch dependencies are ~2GB each, causing deployment failures.

### Solutions Implemented

#### 1. Aggressive Cleanup
```bash
# Remove unnecessary system packages
sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost
sudo rm -rf "$AGENT_TOOLSDIRECTORY" /usr/local/lib/android
sudo docker system prune -af --volumes
```

#### 2. No-Cache Installation
```dockerfile
ENV PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir -r requirements.txt
```

#### 3. CPU-Only PyTorch (Space Saving)
```txt
torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu
torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Split Installation Strategy
```dockerfile
# Install in stages to manage memory
RUN pip install --no-cache-dir numpy opencv-python fastapi
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu  
RUN pip install --no-cache-dir -r requirements.txt
```

## Installation Order Importance

### YOLO-NAS Service
```dockerfile
# CRITICAL: Install compatible versions BEFORE super-gradients
RUN pip install numpy==1.23.0 sphinx==4.0.2
RUN pip install super-gradients==3.7.1
RUN pip install -r requirements.txt  # Other dependencies
```

### MMPose Service  
```dockerfile
# CRITICAL: Install base dependencies first
RUN pip install numpy
RUN pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install openmim
RUN pip install pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0
RUN mim install mmcv-full mmpose xtcocotools
RUN pip install -r requirements.txt
```

## Model Setup

### YOLO Combined Service
Models auto-downloaded by Ultralytics on first run:
- `yolo11n.pt`, `yolo11n-pose.pt`
- `yolov8n.pt`, `yolov8n-pose.pt`

### TrackNet Weights
```bash
# Download TrackNet weights (if available)
wget -O yolo-combined-service/models/tracknet_v2.pth [MODEL_URL]

# Or service runs with random initialization for testing
```

### MMPose Models
Auto-downloaded via OpenMMLab Model Zoo on first inference.

### YOLO-NAS Models
Auto-downloaded by super-gradients:
- `yolo_nas_pose_n` with `coco_pose` weights
- `yolo_nas_s` with `coco` weights

## Environment Configuration

### Required Environment Variables
```bash
# Docker containers
DEBIAN_FRONTEND=noninteractive
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
YOLO_OFFLINE=1
ULTRALYTICS_OFFLINE=1

# GCS configuration
GCS_BUCKET_NAME=padel-ai
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### Port Configuration
```yaml
# docker-compose.yml
services:
  yolo-combined: 
    ports: ["8001:8001"]
  mmpose:
    ports: ["8003:8003"] 
  yolo-nas:
    ports: ["8004:8004"]
  nginx:
    ports: ["80:80"]
```

## Verification & Testing

### Health Checks
```bash
# Individual services
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose  
curl http://localhost:8004/healthz  # YOLO-NAS

# Load balancer
curl http://localhost/api/healthz
```

### Expected Health Check Response
```json
{
  "status": "healthy",
  "models": {
    "yolo11_pose": true,
    "yolov8_object": true,
    "tracknet": true,
    "mmpose_loaded": true,
    "super_gradients_available": true
  }
}
```

### Testing Imports
```bash
# In containers
docker-compose exec mmpose-service python -c "import mmpose; print('✅ MMPose OK')"
docker-compose exec yolo-nas-service python -c "from super_gradients.training import models; print('✅ Super-gradients OK')"
docker-compose exec yolo-combined-service python -c "import ultralytics; print('✅ YOLO OK')"
```

## Troubleshooting

### Common Errors

#### 1. "Device or resource busy" during build
```bash
# Solution: Use -T flag for ln command
RUN ln -sfT /run/systemd/resolve/resolv.conf /etc/resolv.conf || echo "Skipping"
```

#### 2. "ModuleNotFoundError: No module named 'super_gradients'"
```bash
# Check dependency order in Dockerfile
# Ensure numpy==1.23.0 and sphinx==4.0.2 installed first
```

#### 3. "protobuf version conflicts"
```bash
# Force compatible protobuf version
pip install protobuf>=5.26.1,<6.0.0
```

#### 4. Out of disk space during build
```bash
# Use sequential deployment
./deploy.sh --deploy-sequential

# Or use CPU-only PyTorch
# Edit requirements.txt to use +cpu versions
```

### Container Debugging
```bash
# Shell into container
docker-compose exec mmpose-service bash

# Check installed packages
pip list | grep -E "(torch|mmpose|super)"

# Check GPU availability
nvidia-smi

# Check disk space
df -h
```

### Service-Specific Debugging

#### YOLO-NAS Service
```bash
# Check super-gradients installation
python -c "from super_gradients.training import models; print('OK')"

# Check model loading
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

#### MMPose Service
```bash
# Check MMPose installation
python -c "import mmpose; print(mmpose.__version__)"

# Check MMCV compatibility
python -c "import mmcv; print(mmcv.__version__)"
```

## Production Considerations

### Resource Allocation
```yaml
# docker-compose.yml resource limits
services:
  mmpose-service:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Monitoring
```bash
# Resource usage
docker stats

# Service health monitoring
curl -f http://localhost/api/healthz || exit 1
```

### Backup & Recovery
```bash
# Backup model weights
tar -czf models_backup.tar.gz */models/

# Backup configuration
tar -czf config_backup.tar.gz docker-compose.yml nginx.conf
```

## Performance Tuning

### GPU Optimization
```python
# Half precision for faster inference
model.half()
torch.backends.cudnn.benchmark = True
```

### Memory Management
```dockerfile
# Reduce Docker layer size
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
```

### Network Optimization
```nginx
# nginx.conf optimizations
client_max_body_size 100M;
proxy_read_timeout 300s;
proxy_send_timeout 300s;
```

## Security

### Container Security
```dockerfile
# Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

### Network Security
```yaml
# Internal network isolation
networks:
  internal:
    driver: bridge
    internal: true
```

## Maintenance

### Regular Updates
```bash
# Update base images
docker-compose pull
docker-compose up -d

# Update Python packages (test in staging first)
pip install --upgrade ultralytics mmpose
```

### Log Management
```bash
# Rotate logs
docker-compose logs --tail=1000 > service_logs.txt

# Clean old containers
docker system prune -f
```

This deployment guide provides comprehensive instructions for reliable deployment of the NuroPadel platform with all known issues addressed.