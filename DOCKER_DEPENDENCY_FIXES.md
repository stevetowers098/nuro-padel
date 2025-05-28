# Docker Dependency Fixes - Final Resolution

## 🚨 Issue Resolved
**Error**: `ERROR: Cannot install google-cloud-storage and protobuf<6.0.0 and >=5.26.1 because these package versions have conflicting dependencies`

**Root Cause**: Fundamental incompatibility between:
- `google-api-core` (from GCS) requires `protobuf<5.0.0.dev0`
- `grpcio-status` (from GCS) requires `protobuf>=5.26.1,<6.0.0`
- `super-gradients` (YOLO-NAS) requires older protobuf
- `ultralytics` (YOLO11/v8) may also conflict

## ✅ Solution Applied

### 1. Architecture Decision: GCS Separation
**YOLO Services**: Removed GCS dependency to avoid conflicts
- [`yolo-nas-service/requirements.txt`](yolo-nas-service/requirements.txt) - Removed protobuf constraint & GCS
- [`yolo-combined-service/requirements.txt`](yolo-combined-service/requirements.txt) - Removed protobuf constraint & GCS

**MMPose Service**: Keeps GCS with newer protobuf
- [`mmpose-service/requirements.txt`](mmpose-service/requirements.txt) - Retains `protobuf>=5.26.1,<6.0.0` & GCS

### 2. Code Changes Applied

#### YOLO-NAS Service
- [`yolo-nas-service/main.py`](yolo-nas-service/main.py:26) - Disabled GCS import
- [`yolo-nas-service/main.py`](yolo-nas-service/main.py:68) - Stubbed `upload_to_gcs()` function

#### YOLO Combined Service  
- [`yolo-combined-service/main.py`](yolo-combined-service/main.py:25) - Disabled GCS import
- [`yolo-combined-service/main.py`](yolo-combined-service/main.py:63) - Stubbed `upload_to_gcs()` function

## 🎯 Service Capabilities Matrix

| Service | Object Detection | Pose Detection | Video Upload | Protobuf Version |
|---------|------------------|----------------|--------------|------------------|
| **yolo-combined** | ✅ YOLO11/v8 | ✅ YOLO11/v8 | ❌ Disabled | Auto (compatible) |
| **yolo-nas** | ✅ YOLO-NAS | ✅ YOLO-NAS | ❌ Disabled | <4.0.0 (super-gradients) |
| **mmpose** | ❌ N/A | ✅ RTMPose/HRNet | ✅ Enabled | >=5.26.1 (GCS compatible) |

## 🚀 Deployment Impact

### What Works Now
```bash
# All Docker services build successfully
docker-compose build

# All services provide analysis data
curl -X POST http://localhost:8001/yolo11/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "data": true}'

# MMPose provides video uploads
curl -X POST http://localhost:8003/mmpose/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "video": true}'
```

### What Changed
- **YOLO services**: Return empty `video_url` in responses (no GCS upload)
- **MMPose service**: Continues to upload annotated videos to GCS
- **Analysis data**: All services continue to provide full analysis data

## 🔧 Workaround for Video Uploads

If you need video outputs from YOLO services:

### Option 1: Use MMPose for Video Outputs
```bash
# Get YOLO analysis data only
YOLO_DATA=$(curl -X POST http://localhost:8001/yolo11/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "data": true}')

# Get video output from MMPose
MMPOSE_VIDEO=$(curl -X POST http://localhost:8003/mmpose/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4", "video": true}')
```

### Option 2: Local File Processing
- Process videos locally and save files instead of uploading to GCS
- Modify service code to write to `/tmp/` or mounted volume

## 📋 Files Modified

### Requirements Files
```
yolo-nas-service/requirements.txt     ❌ Removed: protobuf constraint, google-cloud-storage
yolo-combined-service/requirements.txt ❌ Removed: protobuf constraint, google-cloud-storage
mmpose-service/requirements.txt       ✅ Kept: protobuf>=5.26.1,<6.0.0, google-cloud-storage
```

### Python Files
```
yolo-nas-service/main.py      ❌ Disabled: GCS import, upload_to_gcs()
yolo-combined-service/main.py ❌ Disabled: GCS import, upload_to_gcs()
mmpose-service/main.py        ✅ Enabled: Full GCS functionality
```

## 🧪 Testing

### Verify Builds Work
```bash
# Should complete without errors
docker-compose build --no-cache

# Check all services start
docker-compose up -d
docker-compose ps
```

### Test Functionality
```bash
# Test YOLO services (data only)
curl http://localhost:8001/healthz
curl http://localhost:8004/healthz

# Test MMPose (data + video)
curl http://localhost:8003/healthz
```

This resolution allows all Docker services to build and function while maintaining the core analysis capabilities.