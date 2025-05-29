# Deployment Guide

## 🚀 Quick Deploy Commands
```bash
./scripts/deploy.sh           # Smart production deploy
./scripts/dev-fast.sh         # Fast development builds (1-2 min)
./scripts/backup.sh           # Backup working services
```

## 🌍 Environments
- **Development**: `localhost:8080` (docker-compose.dev.yml)
- **Production**: `35.189.53.46:8080` (docker-compose.yml)
- **CI/CD**: GitHub Actions → Auto-deploy on push to `docker-containers`

## 📦 Service Details

### **YOLO Combined Service** (services/yolo-combined/)
**Models**: YOLO11, YOLOv8, TrackNet
**Endpoints**: 5 total (/yolo11/pose, /yolo11/object, /yolov8/pose, /yolov8/object, /track-ball)
**Port**: 8001
**Key Dependencies**:
```txt
torch==2.3.1
torchvision==0.18.1
ultralytics==8.2.97
google-cloud-storage==2.10.0  # Specific version for protobuf compatibility
protobuf>=3.19.5,<4.0.0       # Critical constraint
opencv-python-headless==4.10.0.84
fastapi==0.111.0
httpx==0.27.0
```

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

### **YOLO-NAS Service** (services/yolo-nas/)
**Models**: YOLO-NAS Pose N, YOLO-NAS S
**Endpoints**: /yolo-nas/pose, /yolo-nas/object
**Port**: 8004
**Key Dependencies**:
```txt
super-gradients==3.7.1        # Main framework
numpy==1.23.0                 # Must be ≤1.23 for super-gradients
torch + torchvision           # Auto-managed by super-gradients
requests==2.31.0
```

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

# Direct service access (when running locally)
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose
curl http://localhost:8004/healthz  # YOLO-NAS
```

### Expected Health Response
```json
{
  "status": "healthy",
  "models": {
    "yolo11_pose": true,
    "yolov8_object": true,
    "yolov8_pose": true,
    "tracknet": true,
    "mmpose_loaded": true,
    "super_gradients_available": true
  }
}
```

## 🔄 GitHub Actions CI/CD
**Smart Deployment Features**:
- **Change Detection**: Only rebuilds modified services
- **Selective Builds**: Skips unchanged services automatically  
- **Time Savings**: 5-minute deploys vs 30+ minute full rebuilds
- **Auto-Deploy**: Push to `docker-containers` → Deploy to VM

**Branches**:
- `docker-containers` - Auto-deploys to production VM
- `main` - Source code, manual deploy
- `develop` - Development branch

## 🛠️ VM SSH Access
```bash
# Connect to production VM
gcloud compute ssh nuro-vm --zone=us-central1-a

# Or using IP directly
ssh user@35.189.53.46

# Check services on VM
docker ps
docker-compose ps
docker-compose logs -f
```

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

## 📊 Model Setup

### YOLO Combined Service
Models auto-downloaded by Ultralytics on first run:
- `yolo11n-pose.pt` - YOLO11 pose detection
- `yolov8m.pt` - YOLOv8 object detection
- `yolov8n-pose.pt` - YOLOv8 pose detection

### TrackNet Weights
```bash
# Download TrackNet weights (if available)
wget -O services/yolo-combined/models/tracknet_v2.pth [MODEL_URL]

# Or service runs with TrackNet disabled for testing
```

### MMPose Models
Auto-downloaded via OpenMMLab Model Zoo on first inference:
- RTMPose-M (recommended)
- HRNet-W48 (fallback)

### YOLO-NAS Models
Auto-downloaded by super-gradients:
- `yolo_nas_pose_n` with `coco_pose` weights
- `yolo_nas_s` with `coco` weights

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
```

### Import Testing
```bash
# Test in containers
docker-compose exec yolo-combined python -c "import ultralytics; print('✅ YOLO OK')"
docker-compose exec mmpose python -c "import mmpose; print('✅ MMPose OK')"
docker-compose exec yolo-nas python -c "from super_gradients.training import models; print('✅ Super-gradients OK')"
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

### apt-get Connection Failures
**Issue**: GitHub Actions failing with network connection errors like:
```
Failed to fetch libsystemd0 from http://archive.ubuntu.com/ubuntu
Connection failed [IP: 185.125.190.82 80]
```

**Root Cause**: Complex `apt-get` patterns with `--fix-missing`, PPA additions, and Ubuntu 20.04 causing network instability.

**Solution**: Use proven simple patterns from working commit `311f2f7` (1:44 AM Sydney time):
- Ubuntu 22.04 (not 20.04)
- CUDA 12.1 (not 12.2.0)
- Simple `apt-get update && apt-get install -y` (no `--fix-missing`)
- No PPA additions that can cause network timeouts

**Working Pattern**:
```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    # ... other packages
    && rm -rf /var/lib/apt/lists/*
```

**Avoid These Patterns** (cause network failures):
```dockerfile
# DON'T USE - causes connection failures
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update --fix-missing
```

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

This deployment guide provides comprehensive instructions for reliable deployment of the NuroPadel platform with all services and dependencies properly configured.