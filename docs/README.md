# NuroPadel - AI-Powered Padel Analysis Platform

## Overview

NuroPadel is a comprehensive padel analysis platform that combines multiple AI models for player pose estimation, object detection, and advanced ball tracking. The platform leverages YOLO models for real-time detection and TrackNet for enhanced ball trajectory analysis.

## Architecture

### Services
- **YOLO Combined Service** (Port 8001) - Main service with TrackNet integration
- **MMPose Service** (Port 8003) - Advanced pose estimation 
- **YOLO-NAS Service** (Port 8004) - High-accuracy object detection

### Key Features
- **Enhanced Ball Tracking**: YOLO + TrackNet fusion for superior accuracy
- **Pose Estimation**: 17-keypoint human pose detection
- **Object Detection**: Person, sports ball, tennis racket detection
- **Video Processing**: Full video annotation with GCS upload
- **Real-time Analysis**: Sub-second processing times

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (optional, for acceleration)
- 8GB+ RAM
- Google Cloud SDK (for GCS uploads)

### Smart Development Workflow
```bash
# Code in VS Code, push to GitHub - deployment is automatic!
git add .
git commit -m "Update pose detection algorithm"
git push origin docker-containers

# GitHub Actions automatically:
# 1. Detects which services changed
# 2. Builds only changed services
# 3. Deploys to VM at 35.189.53.46
# 4. Health checks all services
# 5. Completes in ~5 minutes vs 30+ minutes
```

### Local Development
```bash
# Start all services locally
docker-compose up --build

# Health check
curl http://localhost:8080/healthz
```

## 🌐 External API Endpoints

**Base URL**: `http://35.189.53.46:8080` (Production VM)
**Alternative**: `http://padel-ai.com` (if DNS configured)

### JSON Request Format
All endpoints accept the same JSON format:

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3
}
```

### 🎯 Pose Detection Endpoints

#### YOLO Combined Service
- **YOLO11 Pose**: `POST http://35.189.53.46:8080/yolo11/pose`
- **YOLOv8 Pose**: `POST http://35.189.53.46:8080/yolov8/pose`

#### MMPose Biomechanics Service
- **MMPose Analysis**: `POST http://35.189.53.46:8080/mmpose/pose`

#### YOLO-NAS High-Accuracy Service
- **YOLO-NAS Pose**: `POST http://35.189.53.46:8080/yolo-nas/pose`

### 🎯 Object Detection Endpoints

#### YOLO Combined Service
- **YOLO11 Object**: `POST http://35.189.53.46:8080/yolo11/object`
- **YOLOv8 Object**: `POST http://35.189.53.46:8080/yolov8/object`

#### YOLO-NAS High-Accuracy Service
- **YOLO-NAS Object**: `POST http://35.189.53.46:8080/yolo-nas/object`

### 🎾 Enhanced Ball Tracking
- **TrackNet Enhanced**: `POST http://35.189.53.46:8080/track-ball`

### 🩺 Health Check Endpoints
- **Global Health**: `GET http://35.189.53.46:8080/healthz`
- **Service Discovery**: `GET http://35.189.53.46:8080/`
- **Individual Services**:
  - `GET http://35.189.53.46:8080/yolo-combined/healthz`
  - `GET http://35.189.53.46:8080/mmpose/healthz`
  - `GET http://35.189.53.46:8080/yolo-nas/healthz`

## 📋 Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `video_url` | string (URL) | ✅ Yes | - | **Public URL to your video file** |
| `video` | boolean | ❌ No | `false` | **Return annotated video?** `true` = get processed video back |
| `data` | boolean | ❌ No | `true` | **Return JSON data?** `true` = get detection/pose data |
| `confidence` | float | ❌ No | `0.3` | **Detection threshold** (0.0-1.0) - only for YOLO services |

## 🚀 API Usage Examples

### Pose Detection Example
```bash
curl -X POST http://35.189.53.46:8080/yolo11/pose \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/my-bucket/padel-video.mp4",
    "video": true,
    "data": true,
    "confidence": 0.5
  }'
```

### MMPose Biomechanical Analysis
```bash
curl -X POST http://35.189.53.46:8080/mmpose/pose \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/my-bucket/padel-video.mp4",
    "video": true,
    "data": true
  }'
```

### Object Detection Example
```bash
curl -X POST http://35.189.53.46:8080/yolo11/object \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/my-bucket/padel-video.mp4",
    "video": false,
    "data": true,
    "confidence": 0.3
  }'
```

### Enhanced Ball Tracking
```bash
curl -X POST http://35.189.53.46:8080/track-ball \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/my-bucket/padel-video.mp4",
    "video": true,
    "data": true,
    "confidence": 0.4
  }'
```

## Project Structure

```
nuro-padel/
├── yolo-combined-service/      # Main service with TrackNet
│   ├── tracknet/              # TrackNet integration
│   ├── models/                # Model weights
│   └── main.py
├── mmpose-service/            # Advanced pose estimation
├── yolo-nas-service/         # High-accuracy detection
├── docker-compose.yml        # Service orchestration
├── nginx.conf               # Load balancer
├── README.md               # This file
└── DEPLOYMENT.md          # Technical deployment guide
```

## Technology Stack

- **ML Frameworks**: PyTorch, Ultralytics, MMPose, Super-gradients
- **API Framework**: FastAPI
- **Containerization**: Docker
- **Load Balancing**: Nginx
- **Cloud Storage**: Google Cloud Storage
- **Video Processing**: OpenCV, FFMPEG

## Core Dependencies

**Important**: Versions vary by service to ensure compatibility

### MMPose Service
```txt
torch==2.1.2
torchvision==0.16.2
mmcv==2.1.0  # Full version with CUDA ops
mmengine (latest compatible)
mmdet>=3.0.0,<3.3.0
mmpose (latest compatible)
numpy>=1.21.0,<2.0  # Constrained for mmcv compatibility
```

### YOLO-NAS Service
```txt
super-gradients==3.7.1
numpy==1.23.0  # Required by super-gradients
torch (managed by super-gradients for cu118)
```

### YOLO Combined Service
```txt
torch==2.3.1
torchvision==0.18.1
ultralytics==8.2.97
```

### Common Dependencies
```txt
fastapi==0.111.0
google-cloud-storage==2.18.0
opencv-python-headless==4.10.0.84
```

**Note**: Different PyTorch versions across services ensure compatibility with respective ML frameworks. See [`DEPLOYMENT.md`](DEPLOYMENT.md) for detailed version management.

## Performance

- **Ball Tracking**: 95%+ precision with TrackNet enhancement
- **Processing Speed**: <50ms per 3-frame sequence
- **Pose Detection**: 17-keypoint accuracy on padel scenarios
- **Video Output**: Real-time annotation with GCS upload

## Use Cases

1. **Player Analysis**: Pose estimation and movement tracking
2. **Ball Trajectory**: Enhanced tracking with occlusion handling
3. **Game Analytics**: Object detection and spatial analysis
4. **Training Tools**: Automated video annotation for coaching

## Getting Started

For detailed deployment instructions, troubleshooting, and technical specifications, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Support

- Check service health at `/healthz` endpoints
- Review container logs: `docker-compose logs [service-name]`
- Verify model weights are present in `*/models/` directories
- Ensure adequate GPU memory for TrackNet processing

## License

NuroPadel is proprietary software for padel analysis applications.
