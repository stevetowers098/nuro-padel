# Nuro Padel AI - Computer Vision Services

## Summary

Production-ready Docker-based microservices for AI-powered padel video analysis with object detection, pose estimation, and automated GCS video uploads. Features YOLO-NAS, YOLO11/v8, and MMPose models with dependency conflict resolution and optimized builds.

## Architecture

### Services Overview

| Service | Models | Object Detection | Pose Detection | Video Upload | Port |
|---------|---------|------------------|----------------|--------------|------|
| **yolo-nas** | YOLO-NAS | ✅ | ✅ | ✅ GCS (gsutil CLI) | 8004 |
| **yolo-combined** | YOLO11/v8 | ✅ | ✅ | ❌ | 8001 |
| **mmpose** | RTMPose/HRNet | ❌ | ✅ | ✅ GCS (Python lib) | 8003 |

### Core Technology Stack

- **Runtime**: Python 3.10 on NVIDIA CUDA 12.2
- **Web Framework**: FastAPI with async processing
- **Computer Vision**: OpenCV, PIL, PyTorch
- **Containerization**: Docker with multi-stage builds
- **Load Balancing**: Nginx reverse proxy

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA Docker runtime (for GPU support)
- Google Cloud credentials (for video uploads)

### Run All Services
```bash
# Build and start all services
docker-compose up --build

# Verify services are running
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8004/healthz  # YOLO-NAS  
curl http://localhost:8003/healthz  # MMPose
```

### Test Analysis
```bash
# Object + Pose detection with YOLO-NAS (includes video upload)
curl -X POST http://localhost:8004/yolo-nas/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "video": true}'

# Fast analysis with YOLO11 (data only)
curl -X POST http://localhost:8001/yolo11/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "data": true}'
```

## Dependency Management & Fixes

### Critical Issues Resolved ✅

#### 1. YOLO-NAS Dependency Installation Order Issue
**Problem**: super-gradients installs incompatible old versions of core dependencies
- super-gradients installs numpy 1.23.0, but albumentations requires >=1.24.4
- super-gradients installs requests 2.22.0, but pyhanko requires >=2.31.0
- super-gradients installs docutils 0.17.1, but sphinx-rtd-theme requires >0.18,<0.22
- super-gradients installs sphinx 4.0.3, but sphinx-rtd-theme requires >=6,<9

**Solution Applied**:
```dockerfile
# Install compatible dependencies FIRST to prevent super-gradients from installing old versions
RUN pip install --no-cache-dir "numpy>=1.24.4" "requests>=2.31.0" "docutils>=0.18,<0.22" "sphinx>=6,<9" && \
    pip install --no-cache-dir super-gradients==3.7.1 && \
    pip install --no-cache-dir -r requirements.txt
```

#### 2. MMPose Dependency Version Conflicts
**Problem**: MMPose installs incompatible dependency versions causing build failures
- numpy 2.0+ breaks binary compatibility with xtcocotools
- pytz version conflicts with openxlab compatibility requirements

**Solution Applied**:
```dockerfile
# Install compatible versions first to prevent MMPose from installing incompatible dependencies
RUN pip install --no-cache-dir "pytz==2023.3" "numpy>=1.21.0,<2.0" && \
    mim install mmpose
```

#### 3. Removed Deprecated Pip Options
**Problem**: `--use-feature=2020-resolver` option not recognized by recent pip versions

**Solution Applied**:
- Removed deprecated pip flags from all Dockerfiles
- Modern pip uses advanced dependency resolution by default

#### 4. Fixed policy-rc.d Execution Errors
**Problem**: System-level package installation denied during Docker builds

**Solution Applied**:
```dockerfile
# Fix policy-rc.d denied execution error
RUN echo '#!/bin/sh\nexit 0' > /usr/sbin/policy-rc.d
```

### Production-Grade Dependency Strategy

#### YOLO-NAS Service
- **Core ML Dependencies**: Managed by `super-gradients` (PyTorch, numpy, protobuf)
- **Installation Order**: `super-gradients` → application dependencies
- **GCS Integration**: gsutil CLI (zero protobuf conflicts)
- **Setuptools**: Pinned to `65.7.0` (prevents InvalidVersion errors)

#### MMPose Service  
- **PyTorch Version**: `1.13.1` (MMPose requirement, not 2.x compatible)
- **Installation Order**: PyTorch → MIM packages → application dependencies
- **GCS Integration**: Python library (compatible with PyTorch 1.13.1)

#### YOLO Combined Service
- **Standard Stack**: FastAPI + YOLO11/v8
- **No GCS**: Local processing only (can be extended with gsutil)

## API Endpoints

### Core Analysis Endpoints
```bash
# Object Detection
POST /{service}/detect
POST /{service}/objects

# Pose Estimation  
POST /{service}/pose

# Combined Analysis
POST /{service}/analyze
```

### Service-Specific Features

#### YOLO-NAS (`/yolo-nas/*`)
- High-accuracy object detection
- Advanced pose estimation
- **Video uploads to GCS** via gsutil CLI
- Handles large video files efficiently

#### YOLO11/v8 (`/yolo11/*`)
- Ultra-fast processing
- Real-time analysis capability
- Data-only responses (no video uploads)

#### MMPose (`/mmpose/*`)
- Professional pose estimation
- RTMPose and HRNet models
- **Video uploads to GCS** via Python library

### Request/Response Format
```json
{
  "video_url": "https://example.com/video.mp4",
  "data": true,    // Return analysis data
  "video": true    // Upload annotated video (YOLO-NAS, MMPose only)
}
```

## Development

### File Structure
```
├── yolo-nas-service/          # YOLO-NAS with GCS uploads
│   ├── Dockerfile            # CUDA + super-gradients + gsutil
│   ├── requirements.txt      # Fixed dependency versions
│   └── main.py              # FastAPI with gsutil integration
├── yolo-combined-service/     # YOLO11/v8 fast processing
├── mmpose-service/           # RTMPose with Python GCS
├── nginx.conf               # Load balancer configuration
└── docker-compose.yml       # Full stack deployment
```

### Local Development
```bash
# Build specific service
docker build -t yolo-nas ./yolo-nas-service

# Run with volume mounting for development
docker run -v $(pwd)/yolo-nas-service:/app -p 8004:8004 yolo-nas

# Test dependency installation
docker run --rm yolo-nas python -c "
from super_gradients.training import models
import requests, numpy, docutils, sphinx
print('✅ All dependencies imported successfully')
"
```

### Environment Variables
```bash
# Google Cloud Authentication
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCS_BUCKET_NAME=your-bucket-name

# Service Configuration  
PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=0
```

## Troubleshooting

### Common Build Issues

#### Dependency Conflicts
```bash
# Clear Docker cache and rebuild
docker builder prune -f
docker-compose build --no-cache
```

#### CUDA/GPU Issues
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-runtime-ubuntu20.04 nvidia-smi
```

#### Service Health Checks
```bash
# Check all service status
docker-compose ps

# View service logs
docker-compose logs yolo-nas-service
docker-compose logs mmpose-service
```

### Performance Optimization

#### Memory Management
- YOLO-NAS: ~4GB GPU memory per request
- MMPose: ~2GB GPU memory per request  
- Use `CUDA_VISIBLE_DEVICES` to control GPU allocation

#### Processing Speed
- **Fastest**: YOLO Combined (data only) - ~1-2s per video
- **Balanced**: YOLO-NAS (data + video) - ~5-10s per video
- **Accurate**: MMPose (pose estimation) - ~10-15s per video

## Production Deployment

### Docker Compose Production
```bash
# Production deployment with resource limits
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Google Cloud Integration
```bash
# Setup service account authentication
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# Test GCS upload functionality
gsutil cp test-video.mp4 gs://your-bucket/test/
```

### Monitoring & Health Checks
- **Health Endpoints**: `GET /{service}/healthz`
- **Resource Monitoring**: Docker stats, NVIDIA-SMI
- **Log Aggregation**: Docker logs with structured JSON output

## License & Support

This project implements production-grade AI video analysis with robust dependency management and scalable microservices architecture.

For technical support or deployment assistance, refer to the troubleshooting guides in [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) and [`DOCKER_DEPENDENCY_FIXES.md`](DOCKER_DEPENDENCY_FIXES.md).
