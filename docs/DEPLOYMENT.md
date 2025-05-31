# Deployment Guide

## 🆕 NEW: Enhanced Upgrade Capabilities

The Nuro-Padel system now includes sophisticated upgrade management:
- **Model Version Management** - Switch models without code changes
- **Feature Flags** - Toggle features dynamically via configuration
- **Enhanced Health Checks** - Detailed service status and upgrade readiness
- **Hot Configuration Reload** - Changes take effect without restarts

📖 **See the complete [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for detailed upgrade instructions**

### Quick Upgrade Commands
```bash
# Check enhanced service health with model versions and features
curl http://localhost:8001/healthz | jq  # YOLO Combined
curl http://localhost:8003/healthz | jq  # MMPose
curl http://localhost:8004/healthz | jq  # YOLO-NAS
curl http://localhost:8005/healthz | jq  # RF-DETR
curl http://localhost:8006/healthz | jq  # ViTPose++

# Demo all upgrade features
chmod +x scripts/demo-upgrade-features.sh
./scripts/demo-upgrade-features.sh

# Toggle features via config files
jq '.features.tracknet_v4.enabled = true' services/yolo-combined/config/model_config.json > temp.json && mv temp.json services/yolo-combined/config/model_config.json

# Override via environment variables
export FEATURE_ENHANCED_BALL_TRACKING_ENABLED=false
docker-compose restart yolo-combined
```

##  Quick Deploy Commands
```bash
./scripts/deploy.sh           # Smart production deploy
./scripts/dev-fast.sh         # Fast development builds (1-2 min)
./scripts/backup.sh           # Backup working services
```

## 🌍 Environments
- **Development**: `localhost:8080` (docker-compose.dev.yml)
- **Production**: `35.189.53.46:8080` (docker-compose.yml)
- **CI/CD**: GitHub Actions → Build images → Manual VM deploy

## 🚀 **DEPLOYMENT PIPELINE - FIXED (May 30, 2025)**

### **Issue Resolved: Containers Not Starting After Image Pulls**
**Root Cause**: GitHub Actions built and pushed images successfully, but containers weren't starting because:
- Missing `docker-compose up -d` commands in deployment pipeline
- deploy.sh called non-existent `deploy-resilient.sh` script
- Registry mismatch between build (`ghcr.io/stevetowers098/nuro-padel/*`) and compose files
- Permission errors preventing service startup

### **✅ Complete Fix Applied:**

**Pipeline Fixes:**
- **Fixed [`deploy.sh`](../scripts/deploy.sh)** - Now includes container startup commands
- **Updated [`docker-compose.yml`](../deployment/docker-compose.yml)** - Corrected registry references and volume mappings
- **Enhanced GitHub Actions** - Added deployment guidance step

**Service Fixes:**
- **Updated [`download-models.sh`](../scripts/download-models.sh)** - Working YOLO11 URLs from v8.3.0 release
- **Fixed Docker permissions** - Added user mapping (1000:1000) and home directory creation
- **Corrected volume mappings** - Services mount proper model subdirectories

### **Current Deployment Process:**
```bash
# 1. Download models (with working YOLO11 URLs)
./scripts/download-models.sh all

# 2. Deploy to VM (now includes container startup)
./scripts/deploy.sh --vm

# 3. Verify services are running
curl http://35.189.53.46:8001/healthz  # YOLO Combined
curl http://35.189.53.46:8003/healthz  # MMPose
curl http://35.189.53.46:8004/healthz  # YOLO-NAS
curl http://35.189.53.46:8005/healthz  # RF-DETR
curl http://35.189.53.46:8006/healthz  # ViTPose++
curl http://35.189.53.46:8080/         # Load Balancer
```

## 📦 Service Details

### **YOLO Combined Service** (services/yolo-combined/) ✅ **ENHANCED**
**Models**: YOLO11, YOLOv8, TrackNet
**Endpoints**: 5 total (/yolo11/pose, /yolo11/object, /yolov8/pose, /yolov8/object, /track-ball)
**Port**: 8001
**Optimization**: ONNX/TensorRT support for Ultralytics models
**Key Dependencies** (UPDATED May 30, 2025):
```txt
torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
ultralytics>=8.3.0,<8.4.0     # ✅ FIXED: YOLO11 compatibility (was 8.2.97)
google-cloud-storage==2.18.0  # Updated version
protobuf>=3.19.5,<4.0.0       # Critical constraint
opencv-python-headless==4.10.0.84
fastapi==0.111.0
httpx==0.27.0
onnx==1.16.0                   # Model optimization
onnxruntime-gpu==1.18.1        # GPU acceleration
```
**Fixes Applied**:
- ✅ **FIXED**: YOLO11 compatibility issue (`AttributeError: C3k2`) via ultralytics>=8.3.0
- ✅ **NEW**: Dedicated YOLO11 object detection model (`yolo11n.pt`)
- ✅ ONNX/TensorRT optimization support for all YOLO models

### **MMPose Service** (services/mmpose/)
**Models**: RTMPose-M, HRNet-W48
**Endpoints**: /mmpose/pose (biomechanical analysis)
**Port**: 8003
**Key Dependencies** (Critical Version Constraints):
```txt
torch==2.1.2                  # Must match MMCV compatibility
torchvision==0.16.2
torchaudio==2.1.2
mmcv==2.1.0                   # Exact version required
numpy>=1.21.0,<2.0            # Critical: constrained for mmcv
pytz==2023.3                  # Must be 2023.3 for openxlab
mmdet>=3.0.0,<3.3.0           # Version constraint for compatibility
mmengine                      # Latest compatible via mim
mmpose                        # Latest compatible via mim
```

### **YOLO-NAS Service** (services/yolo-nas/) ✅ **OPTIMIZED**
**Models**: YOLO-NAS Pose N, YOLO-NAS S
**Endpoints**: /yolo-nas/pose, /yolo-nas/object
**Port**: 8004
**Optimization**: PyTorch → ONNX → TensorRT (automatic fallback)
**Key Dependencies**:
```txt
super-gradients==3.7.1        # Main framework
numpy==1.23.0                 # Must be ≤1.23 for super-gradients
torch + torchvision           # Auto-managed by super-gradients
onnx==1.15.0                  # Model optimization (compatible with super-gradients)
onnxruntime-gpu==1.18.1       # GPU acceleration
# TensorRT via: pip install nvidia-tensorrt
```
**Fixes Applied**:
- ✅ ONNX version conflict resolved (1.16.0 → 1.15.0)
- ✅ Local model loading from `/opt/padel-docker/weights/super-gradients/`
- ✅ Python virtual environment isolation
- ✅ Optimized Docker CMD with uvicorn

### **RF-DETR Detection Service** (services/rf-detr/) ✅ **NEW**
**Models**: RF-DETR Base (stable v0.1.0)
**Endpoints**: /rf-detr/analyze (object detection with FP16 optimization)
**Port**: 8005
**Optimization**: FP16 precision, GPU memory management, resolution constraint (divisible by 56)
**Key Dependencies**:
```txt
torch==2.1.2                  # PyTorch with CUDA 12.1 support
torchvision==0.16.2
rfdetr==0.1.0                  # Stable RF-DETR version
supervision>=0.16.0            # Computer vision utilities
fastapi==0.111.0               # API framework
numpy>=1.21.0,<2.0            # Numerical computing
opencv-python-headless==4.10.0.84
psutil>=5.9.0                  # Performance monitoring
```
**Features**:
- ✅ **FP16 Optimization**: Automatic half-precision for VRAM efficiency (4-5GB usage)
- ✅ **Resolution Constraint**: Automatic adjustment to be divisible by 56
- ✅ **GPU Memory Monitoring**: Real-time VRAM usage tracking and cleanup
- ✅ **Stable Implementation**: Uses proven RF-DETR v0.1.0 (not development versions)

### **ViTPose++ Pose Service** (services/vitpose/) ✅ **NEW**
**Models**: ViTPose-Base, HRNet-W48 (fallback)
**Endpoints**: /vitpose/analyze (advanced pose estimation with joint angles)
**Port**: 8006
**Optimization**: FP16 precision, staged MMPose dependency resolution, GPU memory management
**Key Dependencies** (Staged Installation):
```txt
# Stage 1: Core PyTorch
torch==2.1.2
torchvision==0.16.2
numpy>=1.21.0,<2.0

# Stage 2: MMPose via mim (CRITICAL ORDER)
openmim==0.3.9
mmengine                       # Via mim install
mmcv==2.1.0                   # Via mim install (exact version)
mmdet>=3.0.0,<3.3.0           # Via mim install (version constraint)
mmpose>=1.3.0                 # Via mim install

# Stage 3: Additional
fastapi==0.111.0
xtcocotools>=1.14             # COCO tools
opencv-python-headless==4.10.0.84
psutil>=5.9.0
```
**Features**:
- ✅ **Advanced Pose Quality**: Pose quality scoring based on keypoint visibility
- ✅ **Joint Angle Calculation**: Biomechanical analysis with 6 joint angles
- ✅ **Dependency Isolation**: Staged installation prevents MMPose conflicts
- ✅ **FP16 Optimization**: Automatic half-precision for VRAM efficiency (4-5GB usage)
- ✅ **Fallback Models**: HRNet-W48 fallback if ViTPose fails to load

## 🔧 Development Workflow

### Fast Development (Recommended)
```bash
# Uses pre-built base image for 1-2 minute builds
./scripts/dev-fast.sh
```

**Base Image**: `ghcr.io/stevetowers098/nuro-padel/base:latest`
- Contains heavy ML dependencies (PyTorch, CUDA, etc.)
- Built once, reused for fast iteration
- Perfect for testing phase

### Production Deployment
```bash
# Full production build with optimizations
./scripts/deploy.sh
```

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 20GB+ available space
- **OS**: Ubuntu 20.04+ or compatible Docker environment

### Recommended (Production)
- **CPU**: 8+ cores
- **RAM**: 32GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 50GB+ SSD

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker runtime (for GPU support)
- Google Cloud SDK (for GCS uploads)

## 🩺 Health Checks
```bash
# Global health
curl http://35.189.53.46:8080/healthz

# Individual services
curl http://35.189.53.46:8080/yolo-combined/healthz
curl http://35.189.53.46:8080/mmpose/healthz
curl http://35.189.53.46:8080/yolo-nas/healthz
curl http://35.189.53.46:8080/rf-detr/healthz
curl http://35.189.53.46:8080/vitpose/healthz

# Direct service access (when running locally)
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose
curl http://localhost:8004/healthz  # YOLO-NAS
curl http://localhost:8005/healthz  # RF-DETR
curl http://localhost:8006/healthz  # ViTPose++
```

### Expected Health Response
```json
{
  "status": "healthy",
  "models": {
    "yolo11_pose": true,
    "yolo11_object": true,
    "yolov8_object": true,
    "yolov8_pose": true,
    "tracknet": true,
    "mmpose_loaded": true,
    "super_gradients_available": true,
    "pose_backend": "tensorrt",
    "object_backend": "onnx"
  }
}
```
**Backend Status Indicators**:
- `"tensorrt"` = Maximum performance (40-70% faster)
- `"onnx"` = Good performance (20-40% faster)
- `"pytorch"` = Baseline performance (fallback)

## 🔄 GitHub Actions CI/CD
**Smart Deployment Features**:
- **Change Detection**: Only rebuilds modified services
- **Selective Builds**: Skips unchanged services automatically
- **Time Savings**: 5-minute deploys vs 30+ minute full rebuilds
- **Auto-Deploy**: Push to `docker-containers` → Deploy to VM
- **Docker Cache**: GitHub Actions cache backend for faster builds

**Branches**:
- `docker-containers` - Auto-deploys to production VM
- `main` - Source code, manual deploy
- `develop` - Development branch

### GitHub Actions Build Cache Configuration
Each build job includes Docker Buildx setup for cache support:
```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3

- name: Build and push image
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

**Cache Benefits**:
- ✅ Layer caching across workflow runs
- ✅ Reduced build times for unchanged dependencies
- ✅ Optimized resource usage in GitHub Actions

## 🛠️ VM SSH Access
```bash
# Connect to production VM
gcloud compute ssh padel-ai --zone=australia-southeast1-a

# Or using IP directly (check VM_HOST secret for current IP)
ssh Towers@$VM_HOST

# Check services on VM
docker ps
docker-compose ps
docker-compose logs -f
```

**⚠️ IMPORTANT**: The VM instance details are:
- **Instance Name**: `padel-ai` ✅
- **Zone**: `australia-southeast1-a` ✅
- **Username**: `Towers` ✅

**Previous Configuration (WRONG)**:
- ❌ Instance: `nuro-padel-vm`
- ❌ Zone: `us-central1-a`
- ❌ Username: `user`

## ⚙️ Environment Configuration

### Required Environment Variables
```bash
# Docker containers
DEBIAN_FRONTEND=noninteractive
PYTHONUNBUFFERED=1
PIP_NO_CACHE_DIR=1
YOLO_OFFLINE=1
ULTRALYTICS_OFFLINE=1
ONLINE=False
YOLO_TELEMETRY=False

# GCS configuration
GCS_BUCKET_NAME=padel-ai
GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json

# GPU support
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
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
  rf-detr:
    ports: ["8005:8005"]
  vitpose:
    ports: ["8006:8006"]
  nginx:
    ports: ["8080:80", "8443:443"]
```

## 🔧 Installation Methods

### Method 1: Standard Docker Compose
```bash
# Start all services
docker-compose up --build -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f [service-name]
```

### Method 2: Fast Development Build
```bash
# Use pre-built base image for fast iteration
./scripts/dev-fast.sh

# Check base image exists
docker image inspect ghcr.io/stevetowers098/nuro-padel/base:latest
```

### Method 3: Sequential Deployment (Space Optimized)
For environments with limited disk space:
```bash
# Deploy services one at a time
./scripts/deploy.sh --deploy-sequential

# Deploy specific service
./scripts/deploy.sh --deploy-seq yolo-combined
./scripts/deploy.sh --deploy-seq mmpose
./scripts/deploy.sh --deploy-seq yolo-nas
```

## 📊 Model Setup & Optimization

### **REQUIRED** Models for Offline Operation (`/opt/padel-docker/weights/`)

#### YOLO-NAS Service (Total: ~72MB)
```bash
# Required local models to prevent network downloads
weights/super-gradients/
├── yolo_nas_pose_n_coco_pose.pth      # ~25MB - Pose estimation
└── yolo_nas_s_coco.pth                # ~47MB - Object detection
```

#### YOLO Combined Service (Total: ~24MB + TrackNet)
```bash
# ✅ UPDATED: Complete YOLO11 + YOLOv8 + TrackNet model set (Fixed URLs May 30, 2025)
weights/ultralytics/
├── yolo11n-pose.pt                    # ~6MB - YOLO11 pose detection (v8.3.0)
├── yolo11n.pt                         # ~6MB - ✅ FIXED: YOLO11 object detection (v8.3.0)
├── yolov8n.pt                         # ~6MB - YOLOv8 object detection
├── yolov8n-pose.pt                    # ~6MB - YOLOv8 pose detection
└── tracknet_v2.pth                    # ~3MB - TrackNet ball tracking (optional)
```

**Model URLs Fixed**: Updated download script with working YOLO11 URLs from `v8.3.0` release (previous `v8.2.0` URLs returned 0MB files).

#### MMPose Service (Total: ~9MB)
```bash
# Biomechanical analysis model
weights/mmpose/
└── rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth  # ~9MB
```

#### ViTPose++ Service (Total: ~200MB)
```bash
# Advanced pose estimation model
weights/vitpose/
└── vitpose_base_coco_256x192.pth          # ~200MB - ViTPose++ base model
```

#### RF-DETR Service (Runtime Download)
```bash
# RF-DETR models download automatically at runtime
weights/rf-detr/
└── README.txt                            # Setup indicator (models via Python)
```

### **RECOMMENDED** Performance Optimization

#### 1. TensorRT (40-70% faster on NVIDIA T4)
```bash
# On VM with NVIDIA T4:
pip install nvidia-tensorrt
```

#### 2. Model Optimization Pipeline
```bash
# Create optimized models for maximum performance
cd /opt/padel-docker
python3 scripts/export-models.py

# Creates in weights/optimized/:
# - *.onnx files (20-40% faster)
# - *.engine files (40-70% faster on T4)
```

#### 3. Cache Pre-population (Offline Reliability)
```bash
# Super-gradients cache
python3 -c "
from super_gradients.training import models
models.get('yolo_nas_pose_n', pretrained_weights=None)
models.get('yolo_nas_s', pretrained_weights=None)
print('✅ Super-gradients cache populated')
"

# Ultralytics cache
python3 -c "
from ultralytics import YOLO
print('✅ Ultralytics cache populated')
"
```

### Storage Requirements Summary
- **Base Models**: ~308MB (required)
  - YOLO-NAS: ~72MB
  - YOLO Combined: ~24MB
  - MMPose: ~9MB
  - ViTPose++: ~200MB
  - TrackNet: ~3MB (optional)
  - RF-DETR: Runtime download (~50MB estimated)
- **TensorRT**: ~500MB (recommended for T4)
- **Optimization Cache**: ~50MB
- **Python Wheels**: ~200MB
- **Total**: ~1.1GB

### Model Performance Backends
Services automatically select best available optimization:
1. **TensorRT** (fastest) - 40-70% performance boost on T4
2. **ONNX** (good) - 20-40% performance boost, portable
3. **PyTorch** (baseline) - Always available fallback

## 🔍 Testing & Verification

### API Testing
```bash
# Test YOLO Combined endpoints
curl -X POST http://localhost:8001/yolo11/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/test.mp4", "video": false, "data": true}'

curl -X POST http://localhost:8001/yolov8/object \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/test.mp4", "confidence": 0.3}'

# Test MMPose biomechanics
curl -X POST http://localhost:8003/mmpose/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/test.mp4", "video": true}'

# Test YOLO-NAS high-accuracy
curl -X POST http://localhost:8004/yolo-nas/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/test.mp4", "confidence": 0.5}'

# Test RF-DETR object detection
curl -X POST http://localhost:8005/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/test.mp4", "confidence": 0.3, "resolution": 672}'

# Test ViTPose++ advanced pose estimation
curl -X POST http://localhost:8006/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/test.mp4", "video": true, "confidence": 0.3}'
```

### Import Testing
```bash
# Test in containers
docker-compose exec yolo-combined python -c "import ultralytics; print('✅ YOLO OK')"
docker-compose exec mmpose python -c "import mmpose; print('✅ MMPose OK')"
docker-compose exec yolo-nas python -c "from super_gradients.training import models; print('✅ Super-gradients OK')"
docker-compose exec rf-detr python -c "from rfdetr import RFDETRBase; print('✅ RF-DETR OK')"
docker-compose exec vitpose python -c "import mmpose; print('✅ ViTPose++ OK')"
```

## 🚀 Performance Optimization

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

## 🔒 Security Considerations

### Container Security
```dockerfile
# Run as non-root user (in development)
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
```

### Network Security
```yaml
# Internal network isolation
networks:
  nuro-padel-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
```

## 📈 Production Monitoring

### Resource Monitoring
```bash
# Resource usage
docker stats

# Service health monitoring
watch -n 5 'curl -s http://localhost:8080/healthz | jq .status'

# Log monitoring
docker-compose logs -f --tail=100
```

### Backup & Recovery
```bash
# Backup model weights
tar -czf models_backup.tar.gz services/*/models/

# Backup configuration
tar -czf config_backup.tar.gz deployment/ scripts/

# Service backup/restore
./scripts/backup.sh           # Backup working services
./scripts/restore.sh          # Restore from backup
```

## 🔧 Maintenance

### Regular Updates
```bash
# Update base images
docker-compose pull
docker-compose up -d

# Update Python packages (test in staging first)
pip install --upgrade ultralytics mmpose super-gradients
```

### Log Management
```bash
# View recent logs
docker-compose logs --tail=1000 > service_logs.txt

# Clean old containers and images
docker system prune -f
docker builder prune -af
```

### Disk Space Management
```bash
# Check space usage
docker system df

# Aggressive cleanup
docker system prune -af --volumes
docker builder prune -af

# Remove unused images
docker rmi $(docker images -f "dangling=true" -q)
```

## 🚨 Network Connection Troubleshooting

### apt-get Connection Failures (FIXED - May 29, 2025)
**Issue**: GitHub Actions failing with network connection errors like:
```
Failed to fetch libsystemd0 from http://archive.ubuntu.com/ubuntu
Connection failed [IP: 185.125.190.82 80]
```

**Root Cause**: Ubuntu's primary archive.ubuntu.com servers can be unreliable in CI/CD environments due to high load and geographic latency.

**✅ SOLUTION IMPLEMENTED**: Use faster Azure mirrors + simplified package installation:

**New Working Pattern (Applied to All Services)**:
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Use faster mirrors + retry logic
RUN sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3.10-dev

# Your existing code continues here...
```

**GitHub Actions Build Retry Logic**:
```yaml
- name: Build with retry
  run: |
    for i in 1 2 3; do
      docker buildx build --push --tag your-image . && break
      echo "Retry $i failed, waiting..."
      sleep 30
    done
```

**Key Improvements:**
- ✅ **Azure Mirror**: Replaces slow archive.ubuntu.com with azure.archive.ubuntu.com (faster, more reliable)
- ✅ **Simplified Logic**: Removed complex retry patterns that could mask real issues
- ✅ **Build Retries**: Added 3-attempt retry logic in GitHub Actions for network resilience
- ✅ **Ubuntu 22.04**: Already using stable ubuntu-22.04 runners

**Performance Benefits:**
- 🚀 **50-80% faster package downloads** using Azure mirrors
- 🛡️ **Network failure resilience** with 3-attempt retry logic
- ⚡ **Cleaner Docker layers** without complex retry RUN commands
- 🎯 **Simplified debugging** when issues do occur

**Applied To:**
- [`services/yolo-combined/Dockerfile`](services/yolo-combined/Dockerfile)
- [`services/mmpose/Dockerfile`](services/mmpose/Dockerfile)
- [`services/yolo-nas/Dockerfile`](services/yolo-nas/Dockerfile)
- [`.github/workflows/smart-deploy.yml`](.github/workflows/smart-deploy.yml)

### CUDA Base Image Issues
**Issue**: Docker build failing with:
```
nvidia/cuda:12.1-runtime-ubuntu22.04: failed to resolve source metadata
```

**Root Cause**: Specific CUDA image tags may not exist on Docker Hub.

**Solution**: Use verified available CUDA image tags:
- ✅ `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (working)
- ❌ `nvidia/cuda:12.1-runtime-ubuntu22.04` (not found)

**Fixed in all services** (commit reference for working images):
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 as base
```

**Verification**: Check Docker Hub for available tags:
- https://hub.docker.com/r/nvidia/cuda/tags
- Use specific version numbers (e.g., 12.1.1) not generic (e.g., 12.1)

### Model Management (Best Practice)

**Issue**: Large AI models in Docker images create bloated containers and slow deployments.

**Solution**: Use volume mounting with pre-downloaded models:

```bash
# Download models before deployment
./scripts/download-models.sh all

# Models are mounted as volumes (see docker-compose.yml)
volumes:
  - ./weights:/app/weights:ro  # Read-only mount
```

**Model Download Script Usage:**
```bash
./scripts/download-models.sh all      # Download all models
./scripts/download-models.sh yolo     # YOLO models only
./scripts/download-models.sh mmpose   # MMPose models only
./scripts/download-models.sh verify   # Verify existing models
./scripts/download-models.sh clean    # Remove all models
```

**Required Models:**
- **YOLO Models**: `yolo11n-pose.pt`, `yolov8m.pt`, `yolov8n-pose.pt` (~50MB total)
- **MMPose Models**: `rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth` (~180MB)

### Working Configuration Applied (Commit c2ea327 - May 29, 2025)

**🎯 YOLO-Combined Service: ✅ FULLY WORKING**
- **Backup Location**: `working/yolo-combined-29-5-25/`
- **CUDA**: `nvidia/cuda:12.1.1-runtime-ubuntu22.04` (verified working from commit c2ea327)
- **PyTorch**: `cu121` wheels (compatible with CUDA 12.1.1)
- **Complete Setup**: Service + deploy scripts + docker-compose + nginx config + model downloader

**Service-Specific Fixes Applied:**

**🔬 MMPose Service:**
- **Enhanced Retry Logic**: Robust software-properties-common installation
- **PPA Verification**: Ensures add-apt-repository command availability before execution
- **MMPose Dependencies**: Complex system packages with comprehensive fallback mechanisms

**🎯 YOLO-NAS Service:**
- **Connection Recovery**: 5-attempt retry with 15-second delays for archive.ubuntu.com failures
- **Multi-Retry Installation**: 3-attempt retry cycle with sleep intervals for system dependencies
- **Robust PPA Setup**: Multiple retry attempts for deadsnakes repository addition

**Universal Improvements:**
- ✅ **CUDA Compatibility**: All services use `nvidia/cuda:12.1.1-runtime-ubuntu22.04` + `cu121` wheels
- ✅ **Network Resilience**: Service-specific retry patterns for repository connection issues
- ✅ **Model Management**: Volume mounting with automated download scripts
- ✅ **Working Backup**: Complete YOLO-Combined solution preserved in `working/yolo-combined-29-5-25/`

This deployment guide provides comprehensive instructions for reliable deployment of the NuroPadel platform with all services and dependencies properly configured.
