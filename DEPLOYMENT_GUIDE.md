# 🚀 NuroPadel Docker Deployment Guide

Complete deployment strategy for 3 isolated AI services with smooth video processing and zero-downtime deployment.

## 📋 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Nginx Load Balancer                     │
│                    (Port 80/443)                          │
└─────────────────┬───────────────┬───────────────┬─────────┘
                  │               │               │
         ┌────────▼────────┐ ┌────▼────┐ ┌───────▼────────┐
         │ YOLO Combined   │ │ MMPose  │ │   YOLO-NAS     │
         │   Service       │ │ Service │ │   Service      │
         │ (Port 8001)     │ │(Port    │ │ (Port 8004)    │
         │                 │ │ 8003)   │ │                │
         │ • YOLO11 Pose   │ │ • Bio-  │ │ • High-Acc     │
         │ • YOLO11 Object │ │   mech  │ │   Pose         │
         │ • YOLOv8 Pose   │ │ • RTM   │ │ • High-Acc     │
         │ • YOLOv8 Object │ │   Pose  │ │   Object       │
         └─────────────────┘ └─────────┘ └────────────────┘
```

## 🔧 Prerequisites

### System Requirements
- **OS**: Ubuntu 22.04+ / Windows 11 with WSL2
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space
- **Docker**: 24.0+
- **Docker Compose**: 2.20+
- **NVIDIA Container Runtime**: Latest

### Installation Commands
```bash
# Install Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
```

## 📦 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/stevetowers098/nuro-padel.git
cd nuro-padel
git checkout docker-containers

# Make deployment script executable
chmod +x deploy.sh
```

### 2. Add Model Weights
```bash
# Create weights directory structure
mkdir -p weights/

# Add your model files:
# weights/yolo11n-pose.pt
# weights/yolov8m.pt
# weights/yolov8n-pose.pt
# weights/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

### 3. Build & Deploy
```bash
# Option 1: Full automated deployment
./deploy.sh --all

# Option 2: Step-by-step
./deploy.sh --build
./deploy.sh --test
./deploy.sh --deploy
```

### 4. Verify Deployment
```bash
# Check service health
curl http://localhost/healthz

# Test individual services
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose
curl http://localhost:8004/healthz  # YOLO-NAS

# Check containers
docker-compose ps
```

## 🌐 Service Endpoints

### YOLO Combined Service (Port 8001)
```bash
# YOLO11 Endpoints
POST /yolo11/pose      # 17-keypoint pose detection
POST /yolo11/object    # Object detection (person, ball, racket)

# YOLOv8 Endpoints  
POST /yolov8/pose      # 17-keypoint pose detection
POST /yolov8/object    # Object detection (person, ball, racket)
```

### MMPose Service (Port 8003)
```bash
# Biomechanical Analysis
POST /mmpose/pose      # High-precision pose + biomechanics
```

### YOLO-NAS Service (Port 8004)
```bash
# High-Accuracy Detection
POST /yolo-nas/pose    # State-of-the-art pose detection
POST /yolo-nas/object  # High-accuracy object detection
```

### Load Balancer (Port 80)
```bash
# Through Nginx (recommended for production)
GET  /                 # Service discovery
GET  /healthz          # Global health check

# Routed requests
POST /yolo11/pose      # → yolo-combined:8001
POST /mmpose/pose      # → mmpose:8003
POST /yolo-nas/pose    # → yolo-nas:8004
```

## 📊 API Usage Examples

### Request Format
```json
{
  "video_url": "https://example.com/video.mp4",
  "video": false,      // Return annotated video?
  "data": true,        // Return detection data?
  "confidence": 0.3    // Confidence threshold
}
```

### YOLO11 Pose Detection
```bash
curl -X POST "http://localhost/yolo11/pose" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/padel-video.mp4",
    "video": true,
    "data": true,
    "confidence": 0.5
  }'
```

### MMPose Biomechanical Analysis
```bash
curl -X POST "http://localhost/mmpose/pose" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/technique-video.mp4",
    "video": true,
    "data": true
  }'
```

## 🔄 VM Deployment

### Deploy to Google Cloud VM
```bash
# Method 1: Automated VM deployment
./deploy.sh --vm

# Method 2: Manual sync
VM_HOST="towers@35.189.53.46"
VM_PATH="/opt/padel-docker"

# Sync files
rsync -avz --delete . $VM_HOST:$VM_PATH/

# SSH and deploy
ssh $VM_HOST
cd $VM_PATH
./deploy.sh --all
```

### Production Environment Variables
```bash
# Create production config
echo "GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json" > .env.production
echo "GCS_BUCKET_NAME=padel-ai" >> .env.production
echo "LOG_LEVEL=INFO" >> .env.production
```

## 🚨 Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# If fails, reinstall NVIDIA runtime
sudo apt-get purge nvidia-container-runtime
sudo apt-get install -y nvidia-container-runtime
sudo systemctl restart docker
```

#### 2. Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Reduce batch sizes in services (edit main.py files)
# yolo-combined: batch_size = 4 (line 293, 391)
# mmpose: process fewer frames at once
# yolo-nas: batch_size = 4 (line 137, 202)
```

#### 3. Service Won't Start
```bash
# Check logs
docker-compose logs [service-name]

# Common fixes:
docker-compose down
docker system prune -f
./deploy.sh --build
```

#### 4. MMPose Installation Issues
```bash
# Rebuild MMPose with verbose logging
cd mmpose-service
docker build --no-cache --progress=plain .
```

### Performance Optimization

#### 1. Video Processing Smoothness
- ✅ **FIXED**: All services now extract ALL frames (`num_frames_to_extract=-1`)
- ✅ **OPTIMIZED**: Batch processing for GPU efficiency
- ✅ **FFMPEG**: High-quality video reconstruction with original FPS

#### 2. GPU Memory Management
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Optimize batch sizes per service:
# - YOLO Combined: 8 frames/batch
# - MMPose: 1 frame at a time (complex processing)
# - YOLO-NAS: 8 frames/batch with half precision
```

#### 3. Container Optimization
- ✅ **Multi-stage builds**: Reduced image sizes
- ✅ **Layer caching**: Requirements installed first
- ✅ **Non-root users**: Security best practices
- ✅ **Health checks**: Automatic service monitoring

## 📈 Monitoring & Maintenance

### Health Monitoring
```bash
# Global health check
curl http://localhost/healthz

# Individual service health
for port in 8001 8003 8004; do
  echo "Testing port $port:"
  curl -s http://localhost:$port/healthz | jq '.'
done
```

### Log Management
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f yolo-combined
docker-compose logs -f mmpose
docker-compose logs -f yolo-nas

# Log rotation (add to crontab)
docker system prune -f
```

### Updates & Rollbacks
```bash
# Update services
git pull origin docker-containers
./deploy.sh --build
docker-compose up -d --remove-orphans

# Rollback if needed
docker-compose down
docker-compose up -d --scale yolo-combined=1 --scale mmpose=1 --scale yolo-nas=1
```

## 🔐 Security Considerations

### Production Security
1. **Firewall**: Only expose necessary ports
2. **SSL/TLS**: Configure HTTPS in nginx.conf
3. **Secrets**: Use Docker secrets for API keys
4. **Updates**: Regular security updates for base images
5. **Monitoring**: Log analysis and alerting

### Network Security
```bash
# Create custom network
docker network create --driver bridge nuro-padel-secure

# Use in docker-compose.yml
networks:
  default:
    external:
      name: nuro-padel-secure
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Runtime](https://github.com/NVIDIA/nvidia-container-runtime)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [Super Gradients (YOLO-NAS)](https://github.com/Deci-AI/super-gradients)

## 🆘 Support

If you encounter issues:

1. **Check logs**: `docker-compose logs [service]`
2. **Verify GPU**: `nvidia-smi`
3. **Test individual services**: Use health endpoints
4. **Clean environment**: `docker system prune -af`
5. **Rebuild**: `./deploy.sh --build`

For advanced support, provide:
- Docker version: `docker --version`
- GPU info: `nvidia-smi`
- Service logs: `docker-compose logs`
- System specs: CPU, RAM, GPU model