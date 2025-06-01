# Service Isolation Deployment Guide

Complete guide for deploying and managing independent AI services.

## 🚀 Quick Deploy

```bash
# Download models for all services
./scripts/download-models-unified.sh all

# Deploy specific services
./services/yolo8/deploy.sh
./services/yolo11/deploy.sh
./services/mmpose/deploy.sh  # (existing)
```

## 🏗️ Service Architecture

### Independent Services
- **YOLO8** (Port 8002) - Stable YOLOv8 detection & pose
- **YOLO11** (Port 8007) - Latest generation with optimizations
- **MMPose** (Port 8003) - Advanced biomechanical analysis
- **YOLO-NAS** (Port 8004) - High-accuracy detection
- **RF-DETR** (Port 8005) - Transformer-based detection
- **ViTPose** (Port 8006) - Vision transformer pose analysis

### Key Benefits
✅ **Independent Failures** - One service down doesn't affect others  
✅ **Selective Deployment** - Update only changed services  
✅ **Resource Isolation** - Each service has dedicated GPU allocation  
✅ **Faster Iterations** - Rebuild only what changed  
✅ **Parallel Development** - Teams work on different services  

## 📦 Service-Specific Deployment

### YOLO8 Service
```bash
# Deploy
cd services/yolo8 && ./deploy.sh

# Verify
curl http://localhost:8002/healthz

# Test endpoints
curl -X POST http://localhost:8002/pose -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4"}'
```

### YOLO11 Service  
```bash
# Deploy
cd services/yolo11 && ./deploy.sh

# Verify
curl http://localhost:8007/healthz

# Test with optimizations
curl -X POST http://localhost:8007/object -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "confidence": 0.3}'
```

## 🔧 GitHub Actions Integration

### Smart Service Detection
- Detects changes per service directory
- Deploys only modified services
- Parallel deployment for faster CI/CD

### Workflow Triggers
```yaml
# Auto-deploy on push to specific service
on:
  push:
    paths:
      - 'services/yolo8/**'      # Triggers yolo8 deployment
      - 'services/yolo11/**'     # Triggers yolo11 deployment
```

## 📊 Model Requirements

### YOLO8 Service
```
weights/ultralytics/
├── yolov8n.pt        # ~6MB - Object detection
└── yolov8n-pose.pt   # ~6MB - Pose detection
```

### YOLO11 Service  
```
weights/ultralytics/
├── yolo11n.pt        # ~6MB - Object detection
└── yolo11n-pose.pt   # ~6MB - Pose detection
```

### Download All Models
```bash
# Unified download script
./scripts/download-models-unified.sh all

# Service-specific downloads
./scripts/download-models-unified.sh yolo8
./scripts/download-models-unified.sh yolo11
```

## 🩺 Health Monitoring

### Individual Service Health
```bash
# Check all services
for port in 8002 8007 8003 8004 8005 8006; do
  echo "=== Port $port ==="
  curl -s http://localhost:$port/healthz | jq .status
done

# Service-specific health
curl http://localhost:8002/healthz | jq .models.yolov8_object
curl http://localhost:8007/healthz | jq .deployment.torch_optimizations
```

### Load Balancer Integration
```bash
# Via nginx (when deployed)
curl http://localhost:8080/yolo8/healthz
curl http://localhost:8080/yolo11/healthz
```

## 🚨 Emergency Procedures

### Single Service Recovery
```bash
# Restart specific service
cd services/yolo8 && ./deploy.sh

# Check logs
docker-compose logs -f yolo8

# Emergency restart
docker-compose restart yolo8
```

### Rollback Service
```bash
# Backup current deployment
cp deployment/docker-compose.yml deployment/docker-compose.yml.backup

# Restore previous version
git checkout HEAD~1 -- services/yolo8/
cd services/yolo8 && ./deploy.sh
```

## 🔧 Configuration Management

### Service-Specific Config
Each service maintains independent configuration:
```
services/yolo8/
├── requirements.txt    # Independent dependencies
├── Dockerfile         # Service-specific build
├── deploy.sh          # Independent deployment
└── README.md          # Service documentation
```

### Environment Variables
```bash
# YOLO8 specific
YOLO_OFFLINE=1
CONFIDENCE_THRESHOLD=0.3

# YOLO11 specific  
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
torch.cuda.set_per_process_memory_fraction=0.8
```

## 📈 Scaling & Performance

### Independent Scaling
```bash
# Scale specific service
docker-compose up --scale yolo8=3

# Resource monitoring per service
docker stats nuro-padel-yolo8
docker stats nuro-padel-yolo11
```

### Performance Optimization
- YOLO8: Stable baseline performance
- YOLO11: Enhanced memory management + latest optimizations
- Independent GPU allocation prevents resource conflicts

## 🔗 Service Communication

Services are independent but can communicate via:
- **Direct HTTP calls** between services
- **Shared volumes** for temporary data
- **Message queues** for async processing (optional)

## 🎯 Best Practices

1. **Deploy incrementally** - Test one service at a time
2. **Monitor independently** - Each service has dedicated health checks  
3. **Version separately** - Services can use different dependency versions
4. **Document changes** - Update service-specific READMEs
5. **Test isolation** - Verify services work independently

---

**Next Steps**: See individual service READMEs for detailed configuration and troubleshooting.