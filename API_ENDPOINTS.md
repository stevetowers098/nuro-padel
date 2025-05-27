# Direct Model Endpoints Guide

## Overview
Each model service now has dedicated endpoints for specific capabilities, accessible directly on their respective ports.

## Available Endpoints by Port

### Port 8001 - YOLO11 Service
- **Base URL**: `http://localhost:8001`
- **Endpoints**:
  - `/yolo11/pose` - YOLO11 pose detection with 17 keypoints
  - `/yolo11/object` - YOLO11 object detection (person, sports ball, tennis racket)
  - `/yolo11` - Original unified endpoint
  - `/healthz` - Health check

### Port 8002 - YOLOv8 Service
- **Base URL**: `http://localhost:8002`
- **Endpoints**:
  - `/yolov8/object` - YOLOv8 object detection optimized for padel
  - `/yolov8/pose` - YOLOv8 pose detection with 17 keypoints
  - `/yolov8` - Original unified endpoint
  - `/healthz` - Health check

### Port 8003 - MMPose Service
- **Base URL**: `http://localhost:8003`
- **Endpoints**:
  - `/mmpose/pose` - High-precision biomechanical pose analysis
  - `/mmpose` - Original unified endpoint
  - `/healthz` - Health check

### Port 8004 - YOLO-NAS Service
- **Base URL**: `http://localhost:8004`
- **Endpoints**:
  - `/yolo-nas/pose` - High-accuracy pose detection with YOLO-NAS
  - `/yolo-nas/object` - High-accuracy object detection with YOLO-NAS
  - `/yolo-nas` - Original unified endpoint
  - `/healthz` - Health check

## Endpoint Details

### Object Detection Endpoints

#### `POST /yolov8/object` (Port 8002)
**Purpose**: Dedicated YOLOv8 object detection for padel videos
**Detects**: person (class 0), sports ball (class 32), tennis racket (class 38)
**Features**: 
- Optimized for padel object detection
- Bounding box annotations
- Confidence scoring

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true,
  "confidence": 0.3
}
```

#### `POST /yolo11/object` (Port 8001)
**Purpose**: YOLO11 object detection with latest architecture
**Detects**: person, sports ball, tennis racket
**Features**:
- Latest YOLO11 architecture
- Enhanced detection accuracy
- Batch processing optimization

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true,
  "confidence": 0.3
}
```

#### `POST /yolo-nas/object` (Port 8004)
**Purpose**: High-accuracy object detection with YOLO-NAS architecture
**Detects**: person, sports ball, tennis racket
**Features**:
- Maximum accuracy with YOLO-NAS
- Pretrained COCO weights
- Half precision GPU optimization
- Batch processing for performance

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true
}
```

### Pose Detection Endpoints

#### `POST /yolov8/pose` (Port 8002)
**Purpose**: Fast and reliable pose detection with YOLOv8
**Detects**: Person poses with 17 keypoints
**Features**:
- Proven YOLOv8 architecture
- Skeleton overlay visualization
- Good balance of speed and accuracy
- Compatible with YOLOv8 pose models

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true
}
```

#### `POST /yolo11/pose` (Port 8001)
**Purpose**: Fast and accurate pose detection with YOLO11
**Detects**: Person poses with 17 keypoints
**Features**:
- Skeleton overlay visualization
- Real-time performance
- Batch processing

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true
}
```

#### `POST /mmpose/pose` (Port 8003)
**Purpose**: High-precision biomechanical pose analysis
**Detects**: Detailed pose keypoints with biomechanical metrics
**Features**:
- RTMPose or HRNet models
- Joint angle calculations
- Biomechanical scoring:
  - Posture score
  - Balance score
  - Movement efficiency
  - Power potential

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true
}
```

#### `POST /yolo-nas/pose` (Port 8004)
**Purpose**: High-accuracy pose detection with YOLO-NAS
**Detects**: Precise pose keypoints with 17 joints
**Features**:
- YOLO-NAS architecture for maximum accuracy
- Half precision GPU optimization
- Confidence-based keypoint filtering

**Request Body**:
```json
{
  "video_url": "https://example.com/padel_video.mp4",
  "video": true,
  "data": true
}
```

## Response Format

All endpoints return consistent response format:

```json
{
  "data": {
    "objects_per_frame": [...],     // For object detection
    "poses_per_frame": [...],       // For pose detection
    "biomechanics_per_frame": [...] // For MMPose analysis
  },
  "video_url": "https://storage.googleapis.com/padel-ai/...",
  "timestamp": "2025-01-27T09:30:00.000Z"
}
```

## Usage Examples

### Object Detection with YOLOv8
```bash
curl -X POST "http://localhost:8002/yolov8/object" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/padel_match.mp4",
    "video": true,
    "data": true,
    "confidence": 0.3
  }'
```

### Pose Analysis with MMPose
```bash
curl -X POST "http://localhost:8003/mmpose/pose" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/padel_match.mp4",
    "video": true,
    "data": true
  }'
```

### High-Accuracy Pose with YOLO-NAS
```bash
curl -X POST "http://localhost:8004/yolo-nas/pose" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/padel_match.mp4",
    "video": true,
    "data": true
  }'
```

## Model Capabilities Summary

| Model | Port | Object Detection | Pose Detection | Special Features |
|-------|------|------------------|----------------|------------------|
| YOLOv8 | 8002 | ✅ Optimized | ✅ Fast | Proven architecture, dual capability |
| YOLO11 | 8001 | ✅ Latest arch | ✅ Fast | Latest YOLO architecture, dual capability |
| MMPose | 8003 | ❌ | ✅ Biomechanics | Joint angles, posture analysis |
| YOLO-NAS | 8004 | ✅ High accuracy | ✅ High accuracy | Maximum precision, dual capability |

## 🔥 Complete Endpoint Matrix

### Object Detection Capabilities
| Service | Endpoint | Classes Detected | Accuracy | Speed | GPU Optimized |
|---------|----------|------------------|----------|-------|---------------|
| YOLOv8 | `/yolov8/object` | person, ball, racket | High | Fast | ✅ Half precision |
| YOLO11 | `/yolo11/object` | person, ball, racket | Very High | Fast | ✅ Half precision |
| YOLO-NAS | `/yolo-nas/object` | person, ball, racket | Maximum | Medium | ✅ Half precision |

### Pose Detection Capabilities
| Service | Endpoint | Keypoints | Analysis Type | Accuracy | Speed |
|---------|----------|-----------|---------------|----------|-------|
| YOLOv8 | `/yolov8/pose` | 17 | Basic pose | High | Very Fast |
| YOLO11 | `/yolo11/pose` | 17 | Advanced pose | Very High | Fast |
| MMPose | `/mmpose/pose` | 17 | Biomechanical | Maximum | Medium |
| YOLO-NAS | `/yolo-nas/pose` | 17 | High precision | Maximum | Medium |

## Health Checks

Check service availability:
```bash
# Check all services
curl http://localhost:8001/healthz  # YOLO11
curl http://localhost:8002/healthz  # YOLOv8
curl http://localhost:8003/healthz  # MMPose
curl http://localhost:8004/healthz  # YOLO-NAS
```

## Production Deployment

When deploying, ensure:
1. All model weights are present in `/opt/padel/app/weights/`
2. GCS credentials are configured for video uploads
3. FFMPEG is installed for video processing
4. GPU drivers are properly configured (if using CUDA)

## Migration from Unified API

The unified API endpoints (`/pose`, `/object`) in the main API service are still available and route to these dedicated endpoints. You can:
- Use the unified API for dynamic model selection
- Use direct endpoints for specific model requirements
- Mix approaches based on your use case

Both approaches are supported and maintained.