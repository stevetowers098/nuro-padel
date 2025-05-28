# NuroPadel - AI-Powered Padel Analysis Platform

## Overview

NuroPadel is a comprehensive padel analysis platform that combines multiple AI models for player pose estimation, object detection, and advanced ball tracking. The platform leverages YOLO models for real-time detection and TrackNet for enhanced ball trajectory analysis.

## Recent Updates: TrackNet Integration

### Implementation Strategy
**Extended yolo-combined-service** - Added TrackNet as ball trajectory refinement layer to existing YOLO detection pipeline.

### Architecture Overview
- **Input**: 3 consecutive frames (640×360px)
- **Model**: VGG16 backbone + DeconvNet decoder  
- **Output**: Gaussian heatmap with ball position
- **Integration**: YOLO detects → TrackNet refines trajectory

## Current Services

### 1. YOLO Combined Service (Port 8001)
Enhanced with TrackNet integration for superior ball tracking:

**Endpoints:**
- `/yolo11/pose` - YOLO11 pose estimation
- `/yolo11/object` - YOLO11 object detection
- `/yolov8/pose` - YOLOv8 pose estimation  
- `/yolov8/object` - YOLOv8 object detection (with TrackNet enhancement)
- `/track-ball` - Enhanced ball tracking (YOLO + TrackNet fusion)

**TrackNet Features:**
- **Occlusion handling**: Maintains tracking when ball hidden
- **Trajectory smoothing**: Reduces YOLO detection noise
- **Spin detection**: Foundation for advanced ball analysis
- **Real-time processing**: <50ms per 3-frame sequence

### 2. MMPose Service (Port 8002)
Advanced pose estimation using MMPose framework:
- `/analyze` - Comprehensive pose analysis

### 3. YOLO-NAS Service (Port 8003)
Legacy YOLO-NAS implementation:
- `/analyze` - Basic object detection

## Project Structure

```
nuro-padel/
├── yolo-combined-service/           # Main service with TrackNet
│   ├── tracknet/                    # TrackNet integration
│   │   ├── model.py                 # TrackNet architecture
│   │   ├── utils.py                 # Pre/post-processing
│   │   └── inference.py             # Ball tracking logic
│   ├── models/                      # Model weights directory
│   │   ├── README.md                # Model download instructions
│   │   └── tracknet_v2.pth          # Pre-trained TrackNet weights
│   ├── utils/
│   │   └── video_utils.py
│   ├── main.py                      # Enhanced with TrackNet endpoints
│   ├── requirements.txt             # Updated dependencies
│   └── Dockerfile
├── mmpose-service/                  # Advanced pose estimation
├── yolo-nas-service/               # Legacy detection service
├── docker-compose.yml              # Multi-service orchestration
└── nginx.conf                      # Load balancer configuration
```

## Dependencies

### Core ML Stack
```txt
torch==2.3.1
torchvision==0.18.1
ultralytics==8.2.97
```

### TrackNet Dependencies
```txt
matplotlib==3.8.2
scipy==1.11.4
tqdm==4.66.1
opencv-python-headless==4.10.0.84
```

### API & Cloud
```txt
fastapi==0.111.0
google-cloud-storage==2.10.0
```

## Quick Start

### 1. Start All Services
```bash
docker-compose up --build
```

### 2. Test Enhanced Ball Tracking
```bash
curl -X POST "http://localhost/api/track-ball" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/padel-video.mp4",
    "video": true,
    "data": true,
    "confidence": 0.3
  }'
```

### 3. Health Check
```bash
curl http://localhost/api/healthz
```
Expected response includes TrackNet status:
```json
{
  "status": "healthy",
  "models": {
    "yolo11_pose": true,
    "yolov8_object": true, 
    "yolov8_pose": true,
    "tracknet": true
  }
}
```

## TrackNet Integration Details

### API Response Format

Enhanced object detection now includes:

```json
{
  "data": {
    "objects_per_frame": [
      {
        "class": "sports ball",
        "confidence": 0.95,
        "bbox": {"x1": 320, "y1": 180, "x2": 340, "y2": 200},
        "center": {"x": 330, "y": 190},
        "tracking_method": "tracknet_refined"
      }
    ],
    "tracknet_enabled": true,
    "total_frames": 120
  },
  "video_url": "https://storage.googleapis.com/padel-ai/enhanced_ball_tracking/video.mp4"
}
```

### Tracking Methods
- `yolo_only` - Standard YOLO detection
- `tracknet_refined` - YOLO detection enhanced by TrackNet
- `tracknet_only` - TrackNet detection when YOLO misses ball

### Performance Targets
- **Accuracy**: 95%+ precision on padel videos
- **Speed**: <50ms per 3-frame sequence  
- **Memory**: +2GB GPU usage over YOLO baseline

## Model Setup

### TrackNet Weights
Place pre-trained weights in:
```
yolo-combined-service/models/tracknet_v2.pth
```

If weights not found, TrackNet runs with random initialization (testing only).

### YOLO Weights
Standard YOLO weights are auto-downloaded by Ultralytics.

## Development

### Local Development
```bash
cd yolo-combined-service
pip install -r requirements.txt
python main.py
```

### Testing TrackNet
```bash
# Test with TrackNet disabled
TRACKNET_AVAILABLE=false python main.py

# Test with TrackNet enabled (requires weights)
python main.py
```

## Deployment

### Production Environment
```bash
docker-compose -f docker-compose.yml up -d
```

### Scaling
- Horizontal scaling via Docker Swarm or Kubernetes
- Load balancing handled by nginx
- GPU acceleration for TrackNet processing

## API Documentation

### Enhanced Ball Tracking Endpoint

**POST** `/track-ball`

Enhanced ball tracking combining YOLO detection with TrackNet trajectory refinement.

**Request:**
```json
{
  "video_url": "string",
  "video": boolean,
  "data": boolean, 
  "confidence": float
}
```

**Response:**
- Annotated video with enhanced ball tracking visualization
- JSON data with refined ball positions and tracking metadata
- TrackNet enhancement status and performance metrics

## Key Benefits

1. **Superior Ball Tracking**: TrackNet reduces false negatives and tracking interruptions
2. **Occlusion Handling**: Maintains trajectory during ball occlusion by players/net
3. **Trajectory Smoothing**: Eliminates jitter in ball position data
4. **Foundation for Analytics**: Enables advanced metrics like spin analysis
5. **Minimal Infrastructure**: Reuses existing YOLO service architecture

## Support

For issues related to:
- **TrackNet Integration**: Check model weights and GPU availability
- **YOLO Models**: Verify Ultralytics installation and model downloads
- **Service Communication**: Check nginx configuration and port availability

## License

NuroPadel is proprietary software for padel analysis applications.
