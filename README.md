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

### Start Services
```bash
# Start all services
docker-compose up --build

# Health check
curl http://localhost/api/healthz

# Test ball tracking
curl -X POST "http://localhost/api/track-ball" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/padel-video.mp4",
    "video": true,
    "data": true,
    "confidence": 0.3
  }'
```

## API Endpoints

### YOLO Combined Service
- `POST /yolo11/pose` - YOLO11 pose estimation
- `POST /yolo11/object` - YOLO11 object detection
- `POST /yolov8/pose` - YOLOv8 pose estimation  
- `POST /yolov8/object` - YOLOv8 object detection
- `POST /track-ball` - Enhanced ball tracking (YOLO + TrackNet)

### MMPose Service
- `POST /mmpose/pose` - High-precision pose estimation

### YOLO-NAS Service  
- `POST /yolo-nas/pose` - YOLO-NAS pose detection
- `POST /yolo-nas/object` - YOLO-NAS object detection

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

```txt
# ML Stack
torch==2.3.1
ultralytics==8.2.97
mmcv-full==1.7.2
super-gradients==3.7.1

# API & Cloud
fastapi==0.111.0
google-cloud-storage==2.18.0
```

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
