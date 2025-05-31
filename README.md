# NuroPadel - AI-Powered Padel Analysis Platform

## 🚀 Quick Start

**Production API Endpoint**: `http://35.189.53.46:8080`

```bash
# Download models and deploy
./scripts/download-models.sh all
./scripts/deploy.sh --vm
```

## 🏗️ Services Architecture

| Service | Port | Technology | Purpose |
|---------|------|------------|---------|
| [**YOLO Combined**](services/yolo-combined/) | 8001 | YOLO11/v8 + TrackNet | Enhanced ball tracking & pose detection |
| [**MMPose**](services/mmpose/) | 8003 | MMPose Framework | Advanced biomechanical analysis |
| [**YOLO-NAS**](services/yolo-nas/) | 8004 | Super-Gradients | High-accuracy object detection |
| [**RF-DETR**](services/rf-detr/) | 8005 | Transformer + FP16 | Transformer-based detection |
| [**ViTPose++**](services/vitpose/) | 8006 | Vision Transformer | Joint angle & pose analysis |
| **Load Balancer** | 8080 | Nginx | API Gateway & routing |

## 🌐 VM Infrastructure

### Production Environment
- **VM**: `35.189.53.46` (Google Cloud - Ubuntu 22.04 + NVIDIA T4)
- **User**: `towers`
- **Deployment Path**: `/opt/padel-docker`
- **SSH**: `ssh padel-ai` (see [`.ssh/config`](.ssh/config))

### Directory Structure
```
/opt/padel-docker/
├── services/         # Microservices (Docker containers)
├── weights/          # AI model weights (~396MB total)
│   ├── ultralytics/  # YOLO models
│   ├── mmpose/       # MMPose models  
│   ├── vitpose/      # ViTPose++ models
│   ├── rf-detr/      # RF-DETR models
│   ├── tracknet/     # TrackNet V2 models
│   └── super-gradients/ # YOLO-NAS models
├── deployment/       # Docker Compose configs
└── scripts/          # Deployment automation
```

## 🎯 API Endpoints

### Pose Detection
- **YOLO11 Pose**: `POST /yolo11/pose`
- **YOLOv8 Pose**: `POST /yolov8/pose`  
- **MMPose**: `POST /mmpose/pose`
- **YOLO-NAS**: `POST /yolo-nas/pose`
- **ViTPose++**: `POST /vitpose/analyze`

### Object Detection
- **YOLO11 Object**: `POST /yolo11/object`
- **YOLOv8 Object**: `POST /yolov8/object`
- **YOLO-NAS**: `POST /yolo-nas/object`
- **RF-DETR**: `POST /rf-detr/analyze`

### Enhanced Ball Tracking
- **TrackNet V2**: `POST /track-ball`

### Health Monitoring
- **Global Health**: `GET /healthz`
- **Load Balancer**: `GET /`

## 📝 Request Format

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,     // Return annotated video?
  "data": true,       // Return JSON analysis?
  "confidence": 0.3   // Detection threshold
}
```

## 🔧 Development Workflow

```bash
# Local development
docker-compose up --build

# Deploy to production VM
./scripts/deploy.sh --vm

# Health check
curl http://35.189.53.46:8080/healthz
```

## 📚 Documentation

- **[Complete Guide](docs/README.md)** - Comprehensive documentation
- **[Deployment](docs/DEPLOYMENT.md)** - Technical deployment guide  
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues & solutions
- **[GitHub Actions](docs/GITHUB_ACTIONS_SETUP.md)** - CI/CD setup

## 🔗 Service Documentation

- [**YOLO Combined Service**](services/yolo-combined/README.md) - Enhanced ball tracking
- [**MMPose Service**](services/mmpose/README.md) - Biomechanical analysis
- [**YOLO-NAS Service**](services/yolo-nas/README.md) - High-accuracy detection
- [**RF-DETR Service**](services/rf-detr/README.md) - Transformer detection
- [**ViTPose++ Service**](services/vitpose/README.md) - Advanced pose analysis

## ⚡ Performance

- **Processing**: Sub-second analysis per video
- **Ball Tracking**: 95%+ precision with gap filling
- **Pose Detection**: 17-keypoint COCO accuracy
- **GPU Optimization**: FP16 + TensorRT on NVIDIA T4
- **Scalability**: Microservices with load balancing

## 📊 Model Weights

Total: ~396MB across all services
- **YOLO Models**: ~24MB (ultralytics/)
- **MMPose Models**: ~50MB (mmpose/)  
- **ViTPose Models**: ~180MB (vitpose/)
- **RF-DETR Models**: ~50MB (rf-detr/)
- **TrackNet Models**: ~3MB (tracknet/)
- **YOLO-NAS Models**: ~72MB (super-gradients/)

## 🎾 Use Cases

1. **Player Analysis** - Pose estimation & movement tracking
2. **Ball Trajectory** - Enhanced tracking with occlusion handling  
3. **Game Analytics** - Object detection & spatial analysis
4. **Training Tools** - Automated video annotation for coaching

---

**For detailed API usage, deployment instructions, and technical specifications, see [docs/README.md](docs/README.md)**