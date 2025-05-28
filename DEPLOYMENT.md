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

**Solution Applied - MMpose Enhanced Approach**:
```dockerfile
# MMpose uses the enhanced symlink creation approach
RUN rm -f /etc/resolv.conf && ln -s /run/systemd/resolve/resolv.conf /etc/resolv.conf
```

**Why This Works**: Forcibly removes any existing file/link before creating the new symlink using a cleaner approach without temporary flags, avoiding "resource busy" errors completely.

### 2. Super-gradients Dependency Conflicts

**Error**: `ModuleNotFoundError: No module named 'super_gradients'`

**Root Cause**: Impossible dependency conflict within super-gradients itself:
- `super-gradients 3.7.1` requires `numpy<=1.23`
- `albumentations/albucore` (dependencies of super-gradients) require `numpy>=1.24.4`

**Solution Applied - Install with Dependency Exclusion**:
```dockerfile
# Install super-gradients compatible versions first
RUN pip install "numpy==1.23.0" "requests==2.31.0"
# Install super-gradients without automatic dependency resolution
RUN pip install super-gradients==3.7.1 --no-deps
# Manually install essential dependencies (PyTorch, etc.)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
```

**Key Insight**: Use `--no-deps` to avoid conflicting sub-dependencies, then manually install only the essential ones.

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

### 6. Python Virtual Environment Creation Failures

**Error**: `python3 -m venv /opt/venv` fails or `python3 -m ensurepip` fails to upgrade pip

**Root Cause**: Missing Python virtual environment dependencies or dbus system bus socket

**Solution Applied - MMpose Enhanced Virtual Environment Setup**:
```dockerfile
# Install comprehensive Python venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-venv \
    python3-venv \
    python3-pip \
    python3-dev \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start

# Create virtual environment with enhanced error handling
RUN echo "🔧 Creating Python virtual environment..." && \
    python3 -m venv /opt/venv && \
    echo "✅ Virtual environment created successfully" && \
    /opt/venv/bin/python -m ensurepip --upgrade && \
    echo "✅ Virtual environment pip ensured and upgraded"
ENV PATH="/opt/venv/bin:$PATH"
```

**Why This Works**:
- Installs both `python3.10-venv` and `python3-venv` for comprehensive support
- Ensures dbus service is running to avoid missing system bus socket errors
- Uses explicit ensurepip upgrade within the virtual environment
- Provides detailed logging for debugging virtual environment creation

### 7. Pip Running as Root Warning

**Warning**: `WARNING: Running pip as the 'root' user can result in broken permissions`

**Solution Applied**:
```dockerfile
# Create virtual environment to avoid root pip issues (see solution #6 above)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install -r requirements.txt
```

### 8. Missing System Bus Socket Error

**Error**: `Failed to open connection to "system" message bus due to missing /var/run/dbus/system_bus_socket`

**Root Cause**: dbus service not properly configured or running during Docker build

**Solution Applied - MMpose dbus Configuration**:
```dockerfile
# Install and configure dbus properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start
```

**Why This Works**: Ensures dbus is installed, the socket directory exists, and the service is started during the build process.

### 9. Python Version Compatibility Issues (YOLO-NAS)

**Error**: `networkx` requires Python >=3.9 but system Python is 3.8.10

**Root Cause**: Default system Python version doesn't meet dependency requirements for modern packages like networkx

**Solution Applied - YOLO-NAS Enhanced Python Setup**:
```dockerfile
# Ensure Python 3.10 is properly linked as default Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Create virtual environment explicitly with Python 3.10
RUN echo "🔧 Creating Python virtual environment with Python 3.10..." && \
    python3.10 -m venv /opt/venv && \
    /opt/venv/bin/python -m ensurepip --upgrade && \
    /opt/venv/bin/python --version
ENV PATH="/opt/venv/bin:$PATH"

# Verify Python version meets networkx requirements
RUN echo "🔍 PYTHON VERSION CHECK: Verifying Python >=3.9 for networkx compatibility..." && \
    python --version && \
    python -c "import sys; assert sys.version_info >= (3, 9), f'Python {sys.version_info} < 3.9 required for networkx'; print('✅ Python version meets networkx requirements')"
```

**Why This Works**:
- Explicitly creates virtual environment with Python 3.10 instead of system default
- Links Python 3.10 as the default python and python3 commands
- Verifies version compatibility before dependency installation
- Ensures networkx and other modern packages have required Python version

### 10. Enhanced Dependency Verification

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
# CRITICAL: Enhanced system dependencies for Python 3.10 and virtual environment support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-venv \
    python3-pip \
    python3-dev \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# CRITICAL: Enhanced virtual environment creation with Python 3.10
RUN echo "🔧 Creating Python virtual environment with Python 3.10..." && \
    python3.10 -m venv /opt/venv && \
    /opt/venv/bin/python -m ensurepip --upgrade && \
    /opt/venv/bin/python --version
ENV PATH="/opt/venv/bin:$PATH"

# CRITICAL: Verify Python version meets networkx requirements (>=3.9)
RUN python --version && \
    python -c "import sys; assert sys.version_info >= (3, 9), f'Python {sys.version_info} < 3.9 required for networkx'; print('✅ Python version meets networkx requirements')"

# CRITICAL: Install compatible versions BEFORE super-gradients
RUN pip install numpy==1.23.0 sphinx==4.0.2
RUN pip install super-gradients==3.7.1
RUN pip install -r requirements.txt  # Other dependencies
```

### MMPose Service
```dockerfile
# CRITICAL: Enhanced system dependencies for virtual environment support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-venv \
    python3-venv \
    python3-pip \
    python3-dev \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start

# CRITICAL: Enhanced virtual environment creation with error handling
RUN echo "🔧 Creating Python virtual environment..." && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m ensurepip --upgrade
ENV PATH="/opt/venv/bin:$PATH"

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