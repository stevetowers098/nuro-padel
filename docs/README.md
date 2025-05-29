# NuroPadel - AI-Powered Padel Analysis Platform

## Overview

NuroPadel is a comprehensive padel analysis platform that combines multiple AI models for player pose estimation, object detection, and advanced ball tracking. The platform leverages YOLO models for real-time detection and TrackNet for enhanced ball trajectory analysis.

## Architecture

### Services
- **YOLO Combined Service** (Port 8001) - Main service with TrackNet integration
- **MMPose Service** (Port 8003) - Advanced pose estimation 
- **YOLO-NAS Service** (Port 8004) - High-accuracy object detection

### Key Features
- **Enhanced Ball Tracking**: YOLO + TrackNet V2 ✅ fusion for superior accuracy (V4 pending release)
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

## 🤖 Model Setup & Downloads

### Automatic Model Download
The project includes a comprehensive model download script that handles all required AI models:

```bash
# Download all models (recommended)
./scripts/download-models.sh all

# Download specific model sets
./scripts/download-models.sh yolo       # YOLO models only
./scripts/download-models.sh mmpose     # MMPose models only
./scripts/download-models.sh tracknet   # TrackNet models only
./scripts/download-models.sh yolo-nas   # YOLO-NAS models only

# Verify existing models
./scripts/download-models.sh verify

# Show diagnostic information
./scripts/download-models.sh diagnose
```

### Model Organization
Models are automatically organized into subdirectories:
```
./weights/
├── ultralytics/          # YOLO models (~24MB)
│   ├── yolov8n.pt
│   ├── yolov8n-pose.pt
│   └── yolo11n-pose.pt
├── mmpose/               # MMPose models (~50MB)
│   └── rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
├── tracknet/             # TrackNet V2 ✅ models (~3MB, V4 pending release)
│   └── tracknet_v2.pth
└── super-gradients/      # YOLO-NAS models (~72MB)
    ├── yolo_nas_pose_n_coco_pose.pth
    └── yolo_nas_s_coco.pth
```

### Model Download Requirements
- **Total Size**: ~166MB for all models
- **Network**: Required for initial download
- **Storage**: Models are cached locally for offline operation

## 🔗 How Services Find Model Weights

### Volume Mounting Configuration
All services locate models through Docker volume mounting configured in [`deployment/docker-compose.yml`](deployment/docker-compose.yml):

```yaml
services:
  yolo-combined:
    volumes:
      - /opt/padel-docker/weights:/app/weights:ro  # Read-only mount
  mmpose:
    volumes:
      - /opt/padel-docker/weights:/app/weights:ro
  yolo-nas:
    volumes:
      - /opt/padel-docker/weights:/app/weights:ro
```

### Service Model Paths
Each service uses the `WEIGHTS_DIR = "/app/weights"` constant and looks for models in specific subdirectories:

#### YOLO-NAS Service ([`services/yolo-nas/main.py:104`](services/yolo-nas/main.py:104))
```python
WEIGHTS_DIR = "/app/weights"
local_pose_checkpoint = os.path.join(WEIGHTS_DIR, "super-gradients", "yolo_nas_pose_n_coco_pose.pth")
local_object_checkpoint = os.path.join(WEIGHTS_DIR, "super-gradients", "yolo_nas_s_coco.pth")
```

#### YOLO Combined Service ([`services/yolo-combined/main.py:50`](services/yolo-combined/main.py:50))
```python
WEIGHTS_DIR = "/app/weights"
model_path = os.path.join(WEIGHTS_DIR, "yolo11n-pose.pt")  # Looks for ultralytics/ models
```

#### MMPose Service
```python
WEIGHTS_DIR = "/app/weights"
# Looks for mmpose/ subdirectory models
```

### Path Mapping Summary
```
Host VM Path              → Container Path        → Service Access
/opt/padel-docker/weights → /app/weights         → WEIGHTS_DIR constant

Examples:
/opt/padel-docker/weights/super-gradients/yolo_nas_pose_n_coco_pose.pth
                         → /app/weights/super-gradients/yolo_nas_pose_n_coco_pose.pth
                         → YOLO-NAS service loads this automatically

/opt/padel-docker/weights/ultralytics/yolo11n-pose.pt
                         → /app/weights/yolo11n-pose.pt
                         → YOLO Combined service loads this automatically
```

### ✅ **No Configuration Required**
- Services automatically find models when volume is mounted correctly
- No environment variables needed for model paths
- Download script organization matches service expectations
- Docker Compose handles the path mapping automatically

## ️ VM Infrastructure & Deployment

### Production VM Details
- **Address**: `35.189.53.46` (Google Cloud VM)
- **User**: `towers`
- **Deployment Path**: `/opt/padel-docker`
- **OS**: Ubuntu 22.04 with NVIDIA T4 GPU
- **Docker**: Docker Compose v2 with NVIDIA runtime

### VM Directory Structure
```
/opt/padel-docker/
├── services/             # AI service containers
│   ├── yolo-combined/    # Port 8001
│   ├── mmpose/          # Port 8003
│   └── yolo-nas/        # Port 8004
├── deployment/          # Docker Compose configs
├── scripts/             # Deployment scripts
├── weights/             # Model weights (see above)
├── docs/               # Documentation
└── testing/            # Test suite
```

### Model Deployment Process

**Important**: Model weights must be downloaded to the VM before deployment.

#### Option 1: Manual Download (Recommended)
```bash
# Connect to VM
ssh towers@35.189.53.46

# Navigate to deployment directory
cd /opt/padel-docker

# Download all required models
./scripts/download-models.sh all

# Verify models are ready
./scripts/download-models.sh verify
```

#### Option 2: Local Download + Sync
```bash
# Download models locally
./scripts/download-models.sh all

# Deploy to VM (includes model sync)
./scripts/deploy.sh --vm
```

### VM SSH Access
The project includes SSH configuration for easy VM access:

```bash
# Using SSH config (recommended)
ssh padel-ai

# Or direct SSH
ssh Towers@35.189.53.46
```

**SSH Configuration**: Located in [`.ssh/config`](.ssh/config)
- **Host alias**: `padel-ai`
- **IP**: `35.189.53.46`
- **User**: `Towers`
- **Key**: `.ssh/padel-ai-key`

### Model Downloads in Deployment

**❓ Current Status**: Model downloads are **NOT** automatic in deployment scripts.

#### What the deploy script does:
```bash
./scripts/deploy.sh --vm
```
1. ✅ Syncs project files to VM
2. ✅ Runs deployment scripts
3. ❌ **Does NOT download models automatically**

#### Required Manual Step:
```bash
# Connect to VM first
ssh padel-ai

# Download models on VM
cd /opt/padel-docker
./scripts/download-models.sh all

# Then deploy services
./scripts/deploy.sh --all
```

### Complete VM Deployment Process
```bash
# Step 1: Deploy code to VM
./scripts/deploy.sh --vm

# Step 2: Connect to VM
ssh padel-ai

# Step 3: Download models (REQUIRED)
cd /opt/padel-docker
./scripts/download-models.sh all

# Step 4: Start services
./scripts/deploy.sh --all

# Step 5: Verify deployment
docker-compose ps
curl http://localhost:8080/healthz
```

### VM Service Status Commands
```bash
# On VM: Check service health
docker-compose ps
curl http://localhost:8080/healthz

# On VM: View logs
docker-compose logs -f [service-name]

# On VM: Restart services
docker-compose restart [service-name]
```

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
- **YOLO11 Object**: `POST http://35.189.53.46:8080/yolo11/object` ✅ **NEW: Dedicated YOLO11 Object Detection**
- **YOLOv8 Object**: `POST http://35.189.53.46:8080/yolov8/object`

#### YOLO-NAS High-Accuracy Service
- **YOLO-NAS Object**: `POST http://35.189.53.46:8080/yolo-nas/object`

### 🎾 Enhanced Ball Tracking
- **TrackNet V2 Enhanced**: `POST http://35.189.53.46:8080/track-ball` ✅ **(V4 pending release)**

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
├── services/                 # All AI services
│   ├── yolo-combined/        # Main service with TrackNet
│   │   ├── tracknet/         # TrackNet integration
│   │   ├── models/           # Model weights
│   │   └── main.py
│   ├── mmpose/               # Advanced pose estimation
│   └── yolo-nas/             # High-accuracy detection
├── deployment/               # Nginx configurations
├── docs/                     # Documentation
├── scripts/                  # Deployment scripts
├── testing/                  # Test suite
├── docker-compose.yml        # Service orchestration
└── DEPLOYMENT.md            # Technical deployment guide
```

## Technology Stack

- **ML Frameworks**: PyTorch, Ultralytics, MMPose, Super-gradients
- **Optimization**: ONNX Runtime, TensorRT (NVIDIA T4 optimized)
- **API Framework**: FastAPI
- **Containerization**: Docker + Virtual Environments
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

- **Enhanced Ball Tracking**:
  - **YOLO + Kalman Filtering**: Superior accuracy with gap filling for occlusions
  - **Trajectory Smoothing**: Polynomial interpolation for physics-based smooth trajectories
  - **TrackNet V2**: 95%+ precision (V4 upgrade ready when released)
  - **Real-time Processing**: Sub-frame latency with predictive tracking
- **Processing Speed**:
  - PyTorch: <50ms per 3-frame sequence (baseline)
  - ONNX: 20-40% faster inference on NVIDIA T4
  - TensorRT: 40-70% faster inference with FP16 optimization
- **Pose Detection**: 17-keypoint accuracy on padel scenarios
- **Video Output**: Real-time annotation with GCS upload
- **Model Backends**: Automatic fallback PyTorch → ONNX → TensorRT (best available)

## 🎾 Enhanced Ball Tracking Features

### Advanced YOLO Ball Detection
All `/object` endpoints now include enhanced ball tracking:

- **Kalman Filtering**: Predicts ball position during occlusions
- **Physics Priors**: Uses gravity and velocity models for realistic trajectories
- **Gap Filling**: Automatically interpolates missing detections
- **Trajectory Smoothing**: Polynomial interpolation removes jitter
- **Velocity Tracking**: Real-time speed and direction analysis

### TrackNet Integration Status
- **Current**: TrackNet V2 ✅ (Google Drive download available)
- **Upgrade Path**: TrackNet V4 🔄 (Plug-and-play when released)
- **Performance**: Enhanced YOLO tracking may outperform basic TrackNet V2

### Ball Tracking Response Format
```json
{
  "data": {
    "objects_per_frame": [
      {
        "class": "sports ball",
        "confidence": 0.85,
        "bbox": {"x1": 245.2, "y1": 156.8, "x2": 265.1, "y2": 176.3},
        "tracking": {
          "velocity_x": 12.4,
          "velocity_y": -8.1,
          "trajectory": [[240.1, 160.2], [245.2, 156.8]],
          "tracked": true,
          "interpolated": false,
          "smoothed": true
        }
      }
    ],
    "ball_tracking": {
      "enhanced": true,
      "kalman_filtered": true,
      "trajectory_smoothed": true,
      "fps": 30.0
    }
  }
}
```

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
