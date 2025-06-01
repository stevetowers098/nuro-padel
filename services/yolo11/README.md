# YOLO11 Service

Latest generation YOLO11 object detection and pose estimation service.

## Quick Start
```bash
# Deploy independently
./deploy.sh

# Health check
curl http://localhost:8007/healthz

# Test detection
curl -X POST http://localhost:8007/object \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "confidence": 0.3}'
```

## Features
- YOLO11 object detection (person, sports ball, tennis racket)
- YOLO11 pose estimation (17 keypoints) 
- Enhanced ball tracking with Kalman filtering
- Memory optimization for stable performance

## Endpoints
- `GET /healthz` - Service health with model diagnostics
- `POST /pose` - 17-keypoint pose detection
- `POST /object` - Object detection with enhanced ball tracking

## Configuration
- **Port**: 8007
- **Models**: `yolo11n.pt`, `yolo11n-pose.pt`
- **Requirements**: Fixed ultralytics 8.3.11+ (C3k2 bug fix)
- **Memory**: Optimized with `torch.cuda.set_per_process_memory_fraction(0.8)`

## Performance Improvements
- PyTorch memory optimization enabled
- CUDA benchmark mode for fixed input sizes
- Protobuf compatibility fixes for GCS uploads
- Enhanced error handling and fallback modes

## Development
```bash
# Local development
cd deployment && docker-compose up --build yolo11

# View logs
docker-compose logs -f yolo11

# Shell access
docker exec -it nuro-padel-yolo11 bash
```

## Troubleshooting
- **C3k2 errors**: Fixed in ultralytics 8.3.11+
- **Memory issues**: Automatic fraction limiting enabled
- **Model loading**: Check weights directory `/app/weights/ultralytics/`
- **Health endpoint**: Shows model status and optimization backend