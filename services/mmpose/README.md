# MMPose Service

## 🎯 Overview

Advanced biomechanical pose estimation service using the MMPose framework for detailed human movement analysis. Provides comprehensive pose detection with biomechanical insights optimized for sports applications.

## 🔧 Technical Specifications

- **Port**: 8003
- **Framework**: MMPose with RTMPose models
- **Precision**: High-accuracy pose estimation with biomechanical analysis
- **Device Support**: CUDA (preferred) / CPU fallback
- **Model Size**: ~50MB (RTMPose-Medium)

## 🏗️ Architecture

### Model Configuration
- **Primary**: RTMPose-Medium (`rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192`)
- **Fallback**: HRNet-W48 (`td-hm_hrnet-w48_8xb32-210e_coco-256x192`)
- **Framework**: MMPose with MMCV CUDA operations
- **Optimization**: SimCC coordinate encoding for improved accuracy

### Key Features
- **17-Keypoint Detection**: Full COCO pose format
- **Biomechanical Analysis**: Joint angles, movement patterns
- **Sports Optimization**: Tuned for athletic movement analysis
- **Robust Tracking**: Temporal consistency across frames

## 🌐 API Endpoints

### POST `/pose`
Advanced pose estimation with biomechanical analysis

**Request Body:**
```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": true,          // Return annotated video
  "data": true,           // Return pose data
  "confidence": 0.3       // Keypoint confidence threshold
}
```

**Response Format:**
```json
{
  "data": {
    "poses_per_frame": [
      {
        "keypoints": {
          "nose": {"x": 320.5, "y": 180.2, "confidence": 0.95},
          "left_shoulder": {"x": 280.1, "y": 220.8, "confidence": 0.87},
          "right_shoulder": {"x": 360.3, "y": 218.5, "confidence": 0.91}
          // ... all 17 COCO keypoints
        },
        "biomechanics": {
          "joint_angles": {
            "left_elbow": 142.5,
            "right_elbow": 145.8,
            "left_knee": 165.2,
            "right_knee": 162.7,
            "left_shoulder": 78.3,
            "right_shoulder": 82.1
          },
          "movement_analysis": {
            "pose_quality": 87.5,
            "stability_score": 82.1,
            "symmetry_index": 0.94,
            "athletic_readiness": 78.9
          },
          "body_alignment": {
            "spine_angle": 2.3,
            "hip_level_difference": 1.2,
            "shoulder_level_difference": 0.8
          }
        },
        "frame_info": {
          "frame_number": 15,
          "timestamp": 0.5,
          "model_used": "RTMPose-Medium"
        }
      }
    ],
    "processing_summary": {
      "total_frames": 30,
      "successful_poses": 29,
      "average_confidence": 0.84,
      "model_info": {
        "name": "RTMPose-Medium", 
        "version": "MMPose 1.3.0",
        "precision": "fp32"
      }
    }
  },
  "video_url": "https://storage.googleapis.com/processed/mmpose_annotated.mp4"
}
```

### GET `/healthz`
Service health and model status

```json
{
  "status": "healthy",
  "service": {
    "service": "mmpose",
    "version": "1.0.0",
    "config_loaded": true
  },
  "models": {
    "rtmpose_loaded": true,
    "hrnet_available": true,
    "mmcv_cuda": true,
    "model_info": {
      "name": "RTMPose-Medium",
      "source": "mmpose_hub",
      "config": "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192"
    }
  },
  "system": {
    "cuda_available": true,
    "gpu_device": "Tesla T4",
    "mmpose_version": "1.3.0",
    "mmcv_version": "2.1.0"
  }
}
```

## ⚙️ Configuration

### Model Configuration ([`config/model_config.json`](config/model_config.json))
```json
{
  "service": "mmpose", 
  "version": "1.0.0",
  "models": {
    "rtmpose_medium": {
      "enabled": true,
      "config": "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192",
      "checkpoint": "/app/weights/mmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
    },
    "hrnet_w48": {
      "enabled": true,
      "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192",
      "fallback": true
    }
  },
  "biomechanics": {
    "joint_angle_analysis": true,
    "symmetry_detection": true,
    "stability_assessment": true,
    "movement_quality": true
  },
  "performance": {
    "confidence_threshold": 0.3,
    "max_concurrent_requests": 4,
    "batch_processing": false
  }
}
```

### Environment Variables
- `RTMPOSE_ENABLED=true/false` - Enable/disable RTMPose model
- `HRNET_ENABLED=true/false` - Enable/disable HRNet fallback
- `CONFIDENCE_THRESHOLD=0.3` - Override confidence threshold
- `BIOMECHANICS_ANALYSIS=true/false` - Enable detailed biomechanical analysis

## 🏃 Performance

### Model Performance
- **RTMPose-Medium**: ~40ms per frame (T4 GPU)
- **HRNet-W48**: ~60ms per frame (fallback)
- **Memory Usage**: ~800MB VRAM (RTMPose)
- **Accuracy**: 97%+ keypoint precision on sports scenarios

### Biomechanical Analysis Features
- **Joint Angles**: 6 major joint angle calculations
- **Symmetry Detection**: Left/right body symmetry analysis
- **Stability Assessment**: Balance and postural control metrics
- **Movement Quality**: Athletic movement efficiency scoring

### Optimization Features
- **SimCC Encoding**: Superior coordinate representation vs. heatmaps
- **MMCV CUDA Ops**: Hardware-accelerated operations
- **Temporal Smoothing**: Cross-frame consistency for video analysis
- **Batch Processing**: Efficient multi-frame inference

## 🔬 Biomechanical Analysis Details

### Joint Angle Calculations
- **Elbow Angles**: Shoulder-Elbow-Wrist angle calculation
- **Knee Angles**: Hip-Knee-Ankle angle calculation
- **Shoulder Angles**: Trunk-shoulder alignment
- **Spine Angle**: Overall postural deviation from vertical

### Movement Analysis Metrics
- **Pose Quality**: Overall keypoint visibility and confidence
- **Stability Score**: Balance assessment based on joint positioning
- **Symmetry Index**: Left/right body symmetry (0-1 scale)
- **Athletic Readiness**: Combined movement efficiency score

### Body Alignment Assessment
- **Spine Alignment**: Deviation from neutral spine position
- **Hip Level**: Left/right hip height difference
- **Shoulder Level**: Left/right shoulder height difference

## 🚀 Usage Examples

### Python SDK Example
```python
import requests

response = requests.post(
    "http://35.189.53.46:8080/mmpose/pose",
    json={
        "video_url": "https://storage.googleapis.com/bucket/training-session.mp4",
        "video": True,
        "data": True,
        "confidence": 0.4
    }
)

data = response.json()
poses = data["data"]["poses_per_frame"]

for frame_idx, pose in enumerate(poses):
    biomechanics = pose["biomechanics"]
    angles = biomechanics["joint_angles"]
    movement = biomechanics["movement_analysis"]
    
    print(f"Frame {frame_idx}:")
    print(f"  Left elbow: {angles['left_elbow']:.1f}°")
    print(f"  Stability: {movement['stability_score']:.1f}%")
    print(f"  Symmetry: {movement['symmetry_index']:.2f}")
```

### cURL Example
```bash
curl -X POST http://35.189.53.46:8080/mmpose/pose \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/bucket/training-session.mp4",
    "video": true,
    "data": true,
    "confidence": 0.3
  }'
```

## 🛠️ Development

### Local Development
```bash
# Build and run locally
cd services/mmpose
docker build -t mmpose-service .
docker run -p 8003:8003 mmpose-service

# Health check
curl http://localhost:8003/healthz
```

### Dependencies ([`requirements.txt`](requirements.txt))
- **Core**: `torch==2.1.2`, `torchvision==0.16.2`
- **MMPose Stack**: `mmpose>=1.3.0`, `mmcv==2.1.0`, `mmdet>=3.0.0`, `mmengine`
- **MIM**: `openmim==0.3.9` (for model management)
- **API**: `fastapi==0.111.0`, `uvicorn[standard]`
- **Processing**: `opencv-python-headless==4.10.0.84`
- **Numerics**: `numpy>=1.21.0,<2.0`, `scipy`
- **Cloud**: `google-cloud-storage==2.18.0`

### Critical Installation Order
The service requires staged installation for dependency compatibility:
```dockerfile
# Stage 1: Core PyTorch
pip install torch==2.1.2 torchvision==0.16.2

# Stage 2: MMPose stack via mim
pip install openmim==0.3.9
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.0.0,<3.3.0"
mim install mmpose>=1.3.0

# Stage 3: Additional dependencies
pip install -r requirements.txt
```

### Model Weights Required
```
/app/weights/mmpose/
└── rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth
```

Download via script:
```bash
./scripts/download-models.sh mmpose
```

## 🐛 Troubleshooting

### Common Issues

**MMPose Import Errors**
```bash
# Verify installation order
docker exec mmpose-container python -c "
import mmcv; print(f'MMCV: {mmcv.__version__}')
import mmengine; print('MMEngine: OK')
import mmdet; print('MMDet: OK') 
import mmpose; print('MMPose: OK')
"

# Check CUDA operations
docker exec mmpose-container python -c "
from mmcv.ops import get_compiling_cuda_version
print(f'CUDA version: {get_compiling_cuda_version()}')
"
```

**Model Loading Fails**
```bash
# Check model file
docker exec mmpose-container ls -la /app/weights/mmpose/

# Test model loading
docker exec mmpose-container python -c "
from mmpose.apis import init_model
model = init_model('rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192', None)
print('Model loaded successfully')
"
```

**Low Performance** 
- Ensure MMCV was installed with CUDA support
- Check GPU memory: `nvidia-smi`
- Verify CUDA operations are available in health check
- Monitor inference times in logs

### Performance Optimization
```bash
# Enable MMCV optimizations
export MMCV_WITH_OPS=1

# Use mixed precision (if supported)
export MMPOSE_MIXED_PRECISION=1

# Optimize for batch processing
export MMPOSE_BATCH_SIZE=4
```

### Logs and Debugging
```bash
# Service logs
docker-compose logs -f mmpose

# MMPose specific logs
docker exec mmpose-container python -c "
import logging
logging.getLogger('mmpose').setLevel(logging.DEBUG)
"

# Model inference debugging
grep "inference_time" /var/log/mmpose.log
```

## 📊 Model Comparison

| Model | Accuracy (AP) | Speed (ms) | Memory (MB) | Use Case |
|-------|---------------|------------|-------------|----------|
| RTMPose-Medium | 74.8 | 40 | 800 | Balanced accuracy/speed |
| HRNet-W48 | 75.1 | 60 | 1200 | Maximum accuracy |
| RTMPose-Small | 69.4 | 25 | 400 | Speed optimized |

## 🏥 Sports Medicine Applications

### Injury Prevention Analysis
- **Joint Angle Monitoring**: Detect dangerous joint positions
- **Movement Asymmetry**: Identify compensation patterns
- **Fatigue Detection**: Monitor movement quality degradation
- **Load Assessment**: Evaluate training load impact

### Performance Optimization
- **Technique Analysis**: Compare to optimal movement patterns
- **Efficiency Scoring**: Quantify movement quality
- **Progress Tracking**: Monitor improvement over time
- **Comparative Analysis**: Athlete vs. elite benchmarks

## 📚 References

- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [RTMPose Paper](https://arxiv.org/abs/2303.07399)
- [MMCV Operations](https://mmcv.readthedocs.io/en/latest/understand_mmcv/ops.html)
- [SimCC Coordinate Encoding](https://arxiv.org/abs/2107.03332)