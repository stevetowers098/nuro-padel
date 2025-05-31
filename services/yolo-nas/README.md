# YOLO-NAS Service

## 🎯 Overview

High-accuracy object detection and pose estimation service using YOLO-NAS (Neural Architecture Search) models from Super-Gradients. Optimized for superior accuracy with competitive inference speed.

## 🔧 Technical Specifications

- **Port**: 8004
- **Framework**: Super-Gradients YOLO-NAS
- **Models**: YOLO-NAS-Pose (N), YOLO-NAS-Object (S)
- **Device Support**: CUDA (preferred) / CPU fallback
- **Model Size**: ~72MB total (both models)

## 🏗️ Architecture

### Model Configuration
- **Pose Detection**: YOLO-NAS-Pose-N (`yolo_nas_pose_n_coco_pose.pth`)
- **Object Detection**: YOLO-NAS-S (`yolo_nas_s_coco.pth`)
- **Framework**: Super-Gradients with automatic optimization
- **Precision**: FP32 (default) with optional FP16 support

### Key Features
- **Neural Architecture Search**: Optimized model architecture
- **High Accuracy**: Superior performance vs. standard YOLO models
- **Dual Purpose**: Both object detection and pose estimation
- **Auto-Optimization**: Built-in TensorRT and ONNX conversion

## 🌐 API Endpoints

### POST `/pose`
High-accuracy pose estimation with YOLO-NAS-Pose

### POST `/object`
Precision object detection with YOLO-NAS-S

**Request Body:**
```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": true,          // Return annotated video
  "data": true,           // Return detection data
  "confidence": 0.3       // Detection confidence threshold
}
```

**Response Format (Pose Detection):**
```json
{
  "data": {
    "poses_per_frame": [
      {
        "person_id": 0,
        "bbox": {"x1": 245.2, "y1": 156.8, "x2": 365.1, "y2": 476.3},
        "bbox_confidence": 0.87,
        "keypoints": [
          {"name": "nose", "x": 320.5, "y": 180.2, "confidence": 0.95},
          {"name": "left_shoulder", "x": 280.1, "y": 220.8, "confidence": 0.89},
          {"name": "right_shoulder", "x": 360.3, "y": 218.5, "confidence": 0.92}
          // ... all 17 COCO keypoints
        ],
        "pose_score": 0.84,
        "frame_info": {
          "frame_number": 15,
          "timestamp": 0.5,
          "model_used": "YOLO-NAS-Pose-N"
        }
      }
    ],
    "processing_summary": {
      "total_frames": 30,
      "detected_poses": 29,
      "average_confidence": 0.82,
      "model_info": {
        "name": "YOLO-NAS-Pose-N",
        "accuracy": "high",
        "optimization": "auto"
      }
    }
  },
  "video_url": "https://storage.googleapis.com/processed/yolo_nas_annotated.mp4"
}
```

**Response Format (Object Detection):**
```json
{
  "data": {
    "objects_per_frame": [
      {
        "class": "person",
        "confidence": 0.89,
        "bbox": {"x1": 245.2, "y1": 156.8, "x2": 365.1, "y2": 476.3}
      },
      {
        "class": "sports ball", 
        "confidence": 0.76,
        "bbox": {"x1": 445.2, "y1": 256.8, "x2": 465.1, "y2": 276.3}
      },
      {
        "class": "tennis racket",
        "confidence": 0.82,
        "bbox": {"x1": 320.1, "y1": 280.5, "x2": 380.4, "y2": 350.2}
      }
    ],
    "frame_info": {
      "frame_number": 15,
      "timestamp": 0.5,
      "model_used": "YOLO-NAS-S"
    }
  }
}
```

### GET `/healthz`
Service health and model status

```json
{
  "status": "healthy",
  "service": {
    "service": "yolo-nas",
    "version": "1.0.0",
    "config_loaded": true
  },
  "models": {
    "pose_model_loaded": true,
    "object_model_loaded": true,
    "pose_model_info": {
      "name": "YOLO-NAS-Pose-N",
      "source": "local_checkpoint",
      "size": "nano",
      "accuracy": "high"
    },
    "object_model_info": {
      "name": "YOLO-NAS-S", 
      "source": "local_checkpoint",
      "size": "small",
      "classes": 80
    }
  },
  "system": {
    "cuda_available": true,
    "gpu_device": "Tesla T4",
    "super_gradients_version": "3.7.1"
  },
  "optimization": {
    "tensorrt_available": false,
    "onnx_available": true,
    "fp16_support": true
  }
}
```

## ⚙️ Configuration

### Model Configuration ([`config/model_config.json`](config/model_config.json))
```json
{
  "service": "yolo-nas",
  "version": "1.0.0", 
  "models": {
    "pose": {
      "enabled": true,
      "model_name": "yolo_nas_pose_n",
      "checkpoint": "/app/weights/super-gradients/yolo_nas_pose_n_coco_pose.pth",
      "confidence_threshold": 0.3,
      "nms_threshold": 0.5
    },
    "object": {
      "enabled": true,
      "model_name": "yolo_nas_s",
      "checkpoint": "/app/weights/super-gradients/yolo_nas_s_coco.pth", 
      "confidence_threshold": 0.3,
      "nms_threshold": 0.5,
      "classes": 80
    }
  },
  "optimization": {
    "auto_convert_onnx": true,
    "enable_tensorrt": false,
    "fp16_inference": false,
    "batch_size": 1
  },
  "performance": {
    "max_concurrent_requests": 3,
    "memory_optimization": true
  }
}
```

### Environment Variables
- `YOLO_NAS_POSE_ENABLED=true/false` - Enable/disable pose model
- `YOLO_NAS_OBJECT_ENABLED=true/false` - Enable/disable object model
- `CONFIDENCE_THRESHOLD=0.3` - Detection confidence threshold
- `NMS_THRESHOLD=0.5` - Non-maximum suppression threshold
- `ENABLE_TENSORRT=false` - Enable TensorRT optimization
- `FP16_INFERENCE=false` - Enable FP16 precision

## 🏃 Performance

### Model Performance
- **YOLO-NAS-Pose-N**: ~35ms per frame (T4 GPU)
- **YOLO-NAS-S**: ~28ms per frame (T4 GPU)
- **Memory Usage**: ~1.1GB VRAM (both models loaded)
- **Accuracy**: 
  - Pose: 68.4 AP (COCO pose validation)
  - Object: 47.5 mAP (COCO object validation)

### Accuracy Comparison
| Model | COCO AP | Speed (ms) | Memory (MB) |
|-------|---------|------------|-------------|
| YOLO-NAS-Pose-N | 68.4 | 35 | 600 |
| YOLOv8n-Pose | 50.4 | 25 | 400 |
| YOLO-NAS-S | 47.5 | 28 | 500 |
| YOLOv8s | 44.9 | 22 | 350 |

### Optimization Features
- **Neural Architecture Search**: Automatically optimized architecture
- **Auto ONNX Conversion**: Automatic ONNX export for faster inference
- **TensorRT Support**: Optional TensorRT optimization (when available)
- **FP16 Inference**: Half-precision support for memory efficiency
- **Quantization Ready**: Post-training quantization support

## 🚀 Usage Examples

### High-Accuracy Pose Detection
```bash
curl -X POST http://35.189.53.46:8080/yolo-nas/pose \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/bucket/precision-analysis.mp4",
    "video": true,
    "data": true,
    "confidence": 0.4
  }'
```

### Precision Object Detection
```bash
curl -X POST http://35.189.53.46:8080/yolo-nas/object \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://storage.googleapis.com/bucket/equipment-detection.mp4",
    "video": false,
    "data": true,
    "confidence": 0.5
  }'
```

### Python SDK Example
```python
import requests

# High-accuracy pose analysis
response = requests.post(
    "http://35.189.53.46:8080/yolo-nas/pose",
    json={
        "video_url": "https://storage.googleapis.com/bucket/technique-analysis.mp4",
        "video": True,
        "data": True,
        "confidence": 0.4
    }
)

data = response.json()
poses = data["data"]["poses_per_frame"]

for frame_idx, pose_data in enumerate(poses):
    person = pose_data[0]  # First detected person
    keypoints = person["keypoints"]
    pose_score = person["pose_score"]
    
    print(f"Frame {frame_idx}: Pose Score = {pose_score:.2f}")
    
    # Extract key joints
    nose = next(kp for kp in keypoints if kp["name"] == "nose")
    left_shoulder = next(kp for kp in keypoints if kp["name"] == "left_shoulder")
    
    print(f"  Nose: ({nose['x']:.1f}, {nose['y']:.1f}) conf={nose['confidence']:.2f}")
    print(f"  L-Shoulder: ({left_shoulder['x']:.1f}, {left_shoulder['y']:.1f})")
```

## 🛠️ Development

### Local Development
```bash
# Build and run locally
cd services/yolo-nas
docker build -t yolo-nas-service .
docker run -p 8004:8004 yolo-nas-service

# Health check
curl http://localhost:8004/healthz
```

### Dependencies ([`requirements.txt`](requirements.txt))
- **Core**: `super-gradients==3.7.1`
- **Constraints**: `numpy==1.23.0` (required by super-gradients)
- **PyTorch**: Managed automatically by super-gradients (cu118)
- **API**: `fastapi==0.111.0`, `uvicorn[standard]`
- **Processing**: `opencv-python-headless==4.10.0.84`
- **Cloud**: `google-cloud-storage==2.18.0`
- **Optimization**: `onnx`, `onnxruntime-gpu`

### Critical Dependencies Note
Super-Gradients has strict version requirements:
```txt
super-gradients==3.7.1    # Fixed version for stability
numpy==1.23.0             # Required exact version
torch                     # Automatically managed (cu118)
torchvision              # Automatically managed
```

### Model Weights Required
```
/app/weights/super-gradients/
├── yolo_nas_pose_n_coco_pose.pth    # Pose estimation model (~35MB)
└── yolo_nas_s_coco.pth              # Object detection model (~37MB)
```

Download via script:
```bash
./scripts/download-models.sh yolo-nas
```

## 🐛 Troubleshooting

### Common Issues

**Super-Gradients Version Conflicts**
```bash
# Check version compatibility
docker exec yolo-nas-container python -c "
import super_gradients; print(f'SG: {super_gradients.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')
import torch; print(f'PyTorch: {torch.__version__}')
"

# Expected output:
# SG: 3.7.1
# NumPy: 1.23.0
# PyTorch: 2.0.1+cu118
```

**Model Loading Fails**
```bash
# Check model files exist
docker exec yolo-nas-container ls -la /app/weights/super-gradients/

# Test model loading
docker exec yolo-nas-container python -c "
from super_gradients.training import models
pose_model = models.get('yolo_nas_pose_n', pretrained_weights='coco_pose')
print('Pose model loaded')
object_model = models.get('yolo_nas_s', pretrained_weights='coco')
print('Object model loaded')
"
```

**Poor Performance**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU memory: `nvidia-smi`
- Consider enabling ONNX optimization in config
- Monitor memory usage in health check

**Memory Issues**
```bash
# Check memory usage
curl -s http://localhost:8004/healthz | jq .system

# Reduce memory footprint
export YOLO_NAS_MEMORY_OPTIMIZATION=true
export YOLO_NAS_BATCH_SIZE=1
```

### Performance Optimization
```bash
# Enable ONNX auto-conversion
export AUTO_CONVERT_ONNX=true

# Enable FP16 (if supported)
export FP16_INFERENCE=true

# Enable TensorRT (if available)
export ENABLE_TENSORRT=true

# Optimize for single requests
export MAX_CONCURRENT_REQUESTS=1
```

### Logs and Debugging
```bash
# Service logs
docker-compose logs -f yolo-nas

# Super-Gradients specific logs
docker exec yolo-nas-container python -c "
import logging
logging.getLogger('super_gradients').setLevel(logging.DEBUG)
"

# Model performance monitoring
grep "inference_time" /var/log/yolo-nas.log
```

## 🔬 Model Architecture Details

### YOLO-NAS Innovations
- **Neural Architecture Search**: Automated model design optimization
- **Quantization-Aware Training**: Built-in quantization support
- **Knowledge Distillation**: Enhanced training with teacher models
- **Auto-Optimization**: Automatic ONNX/TensorRT conversion

### Architecture Variants
- **YOLO-NAS-S**: Small model, balanced speed/accuracy
- **YOLO-NAS-M**: Medium model, higher accuracy
- **YOLO-NAS-L**: Large model, maximum accuracy
- **YOLO-NAS-Pose-N**: Nano pose model, optimized for keypoints

### Accuracy Improvements vs. YOLOv8
- **Object Detection**: +2.6 mAP improvement
- **Pose Estimation**: +18 AP improvement
- **Inference Speed**: Comparable or faster
- **Memory Efficiency**: Similar or better

## 📊 Use Cases

### High-Accuracy Applications
- **Equipment Detection**: Precise racket/ball identification
- **Player Tracking**: Accurate person detection and tracking
- **Technique Analysis**: Detailed pose estimation for coaching
- **Competition Analysis**: Professional-grade accuracy for scoring

### Quality Control
- **False Positive Reduction**: Higher precision reduces noise
- **Confidence Scoring**: Better confidence calibration
- **Multi-Person Scenarios**: Improved performance with multiple players
- **Challenging Conditions**: Better performance in poor lighting/occlusion

## 📚 References

- [Super-Gradients Documentation](https://docs.deci.ai/super-gradients/)
- [YOLO-NAS Paper](https://arxiv.org/abs/2306.15852)
- [Neural Architecture Search](https://arxiv.org/abs/1611.01578)
- [COCO Dataset](https://cocodataset.org/)