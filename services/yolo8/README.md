# YOLO8 Service

Independent YOLOv8 object detection and pose estimation service.

## Features
- YOLOv8 object detection (person, sports ball, tennis racket)
- YOLOv8 pose estimation (17 keypoints)
- Enhanced ball tracking with Kalman filtering
- Independent deployment and scaling

## Endpoints
- `GET /healthz` - Health check
- `POST /pose` - Pose detection
- `POST /object` - Object detection

## Port
- **8002** - Main service port

## Deployment
```bash
# Deploy this service independently
./deploy.sh

# Or from root directory
./services/yolo8/deploy.sh
```

## Development
```bash
# Build and run locally
cd deployment
docker-compose up --build yolo8

# View logs
docker-compose logs -f yolo8
```

## Models
- `yolov8n.pt` - Object detection
- `yolov8n-pose.pt` - Pose detection
- Custom models supported in `/app/weights/`

## Environment Variables
- `YOLO_OFFLINE=1` - Disable auto-downloads
- `CONFIDENCE_THRESHOLD=0.3` - Detection threshold
- `NVIDIA_VISIBLE_DEVICES=all` - GPU access

## Troubleshooting
- Check model files exist in weights directory
- Verify CUDA availability with health endpoint
- Monitor memory usage for batch processing