# NuroPadel - AI-Powered Padel Analysis Platform

> Smart GitHub Actions deployment with selective service rebuilding

## 🚀 Quick Start

**Simple Development Workflow:**
1. Code in VS Code
2. Push to GitHub
3. GitHub Actions automatically detects changes and rebuilds only what changed
4. Deploys to production in ~5 minutes

```bash
git add .
git commit -m "Update detection algorithm"
git push origin docker-containers  # Auto-deploys to VM
```

## 🌐 External API

**Production URL**: `http://35.189.53.46:8080`

### Pose Detection
- `POST /yolo11/pose` - Fast YOLO11 pose estimation
- `POST /mmpose/pose` - Advanced biomechanical analysis  
- `POST /yolo-nas/pose` - High-accuracy pose detection

### Object Detection
- `POST /yolo11/object` - Fast object detection
- `POST /yolo-nas/object` - High-accuracy object detection
- `POST /track-ball` - Enhanced ball tracking with TrackNet

### Request Format
```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": true,
  "data": true,
  "confidence": 0.3
}
```

## 🧠 Smart Deployment

- **Automatic change detection** using `git diff`
- **Selective rebuilding** - only changed services rebuild
- **Zero downtime** - unchanged services keep running
- **Health checks** with automatic rollback
- **5-minute deployments** vs 30+ minute full rebuilds

## 🏗️ Architecture

- **3 AI Services**: YOLO Combined, MMPose, YOLO-NAS
- **Nginx Load Balancer**: Routes traffic and health checks
- **Docker Compose**: Service orchestration
- **GitHub Actions**: Smart CI/CD pipeline
- **GCS Integration**: Video storage and processing

## 📁 Project Structure

```
nuro-padel/
├── .github/workflows/smart-deploy.yml    # Smart deployment workflow
├── yolo-combined-service/                # YOLO11 + YOLOv8 + TrackNet
├── mmpose-service/                       # Biomechanical analysis
├── yolo-nas-service/                     # High-accuracy detection
├── nginx.conf                            # Load balancer config
├── docker-compose.yml                    # Production deployment
├── docker-compose.dev.yml                # Development deployment
└── docs/README.md                        # Complete documentation
```

## 📚 Documentation

- **[Complete API Documentation](docs/README.md)** - Full endpoint details and examples
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Technical deployment instructions
- **[GitHub Setup](docs/GITHUB_RESILIENT_SETUP.md)** - CI/CD configuration

## 🎯 Key Features

- **Enhanced Ball Tracking**: YOLO + TrackNet fusion for 95%+ accuracy
- **17-Keypoint Pose Estimation**: Multiple model architectures
- **Real-time Processing**: Sub-second inference times
- **Automatic Video Annotation**: Generated and uploaded to GCS
- **Production Ready**: Load balanced, health checked, auto-scaling

## 🩺 Health Checks

```bash
# Global health
curl http://35.189.53.46:8080/healthz

# Service discovery
curl http://35.189.53.46:8080/

# Individual services
curl http://35.189.53.46:8080/yolo-combined/healthz
curl http://35.189.53.46:8080/mmpose/healthz
curl http://35.189.53.46:8080/yolo-nas/healthz
```

---

**NuroPadel** - Precision AI analysis for padel performance optimization