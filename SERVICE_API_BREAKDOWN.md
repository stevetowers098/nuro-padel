# 🎯 NuroPadel AI Services - Complete API Breakdown

## 📋 Service Overview

| Service | Port | Technology Stack | ONNX Support | PyTorch Version |
|---------|------|------------------|--------------|-----------------|
| [YOLO8](#yolo8-service) | 8002 | Ultralytics Only | ❌ No | 2.1.2 |
| [YOLO11](#yolo11-service) | 8007 | Ultralytics Only | ❌ No | 2.1.2-2.4.0 |
| [MMPose](#mmpose-service) | 8003 | MMPose + MMCV | ❌ No | 2.1.2 (CUDA 12.1) |
| [YOLO-NAS](#yolo-nas-service) | 8004 | Super-Gradients + ONNX | ✅ Yes | ≥1.13.0 (auto-managed) |
| [RF-DETR](#rf-detr-service) | 8005 | Custom RF-DETR | ❌ No | Dockerfile-managed |
| [ViTPose++](#vitpose-service) | 8006 | MMPose Framework | ❌ No | 2.1.2 |

---

## 🎯 YOLO8 Service

### **Purpose**

Independent YOLOv8-based object and pose detection with enhanced ball tracking capabilities.

### **API Endpoints**

- **Health Check**: `GET /healthz`
- **Pose Detection**: `POST /pose`
- **Object Detection**: `POST /object`

### **Expected Input Format**

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3
}
```

### **Expected Output Examples**

#### Pose Detection Output

```json
{
  "data": {
    "poses_per_frame": [
      {
        "keypoints": {
          "nose": {"x": 320.5, "y": 140.2, "confidence": 0.95},
          "left_shoulder": {"x": 280.1, "y": 180.4, "confidence": 0.88},
          "right_shoulder": {"x": 360.7, "y": 175.9, "confidence": 0.92}
        },
        "confidence": 0.85,
        "bbox": {"x1": 250.0, "y1": 120.0, "x2": 390.0, "y2": 480.0}
      }
    ]
  },
  "video_url": "https://storage.googleapis.com/processed_yolov8_pose/video_20241201_143022_abc123.mp4"
}
```

#### Object Detection Output

```json
{
  "data": {
    "objects_per_frame": [
      {
        "class": "sports ball",
        "confidence": 0.92,
        "bbox": {"x1": 450.2, "y1": 200.1, "x2": 465.8, "y2": 215.7}
      },
      {
        "class": "person", 
        "confidence": 0.88,
        "bbox": {"x1": 200.0, "y1": 100.0, "x2": 400.0, "y2": 500.0}
      }
    ],
    "ball_tracking": {
      "enhanced": true,
      "kalman_filtered": true,
      "trajectory_smoothed": true,
      "fps": 30.0
    }
  }
}
```

### **What It Does**

1. **Downloads video** from provided URL
2. **Extracts all frames** for processing
3. **Loads YOLOv8 models** (pose and object detection)
4. **Processes in batches** (8 frames at a time) with GPU optimization
5. **Detects 17 keypoints** for pose estimation (COCO format)
6. **Identifies padel objects**: person, sports ball, tennis racket
7. **Applies enhanced ball tracking** with Kalman filtering and trajectory smoothing
8. **Creates annotated video** with skeleton overlays and bounding boxes
9. **Uploads results to Google Cloud Storage**

---

## 🚀 YOLO11 Service

### **Purpose**

Latest generation YOLO11 detection with memory optimization and enhanced performance.

### **API Endpoints**

- **Health Check**: `GET /healthz`
- **Pose Detection**: `POST /pose`
- **Object Detection**: `POST /object`

### **Expected Input Format**

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3
}
```

### **Expected Output Examples**

#### Enhanced Output with Performance Metrics

```json
{
  "data": {
    "objects_per_frame": [...],
    "ball_tracking": {
      "enhanced": true,
      "kalman_filtered": true,
      "trajectory_smoothed": true,
      "fps": 30,
      "yolo11_performance": "optimized"
    }
  },
  "video_url": "https://storage.googleapis.com/processed_yolo11_object/video_20241201_143022_def456.mp4"
}
```

### **What It Does**

1. **Memory optimization** with `torch.cuda.set_per_process_memory_fraction(0.8)`
2. **Protobuf compatibility** fixes for Google Cloud Storage
3. **Latest YOLO11 architecture** with improved accuracy
4. **FP16 precision** for faster inference on GPU
5. **Enhanced error handling** and memory management
6. **Same detection capabilities** as YOLO8 but with better performance
7. **Optimized batch processing** with CuDNN benchmarking

---

## 🧬 MMPose Service

### **Purpose**

Advanced biomechanical analysis using MMPose framework with RTMPose and HRNet models.

### **API Endpoints**

- **Health Check**: `GET /healthz`
- **Pose Analysis**: `POST /mmpose/pose`

### **Expected Input Format**

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true
}
```

### **Expected Output Examples**

#### **🤖 RAW MODEL OUTPUT (What MMPose Actually Returns):**
```json
{
  "keypoints": {
    "nose": {"x": 320.5, "y": 140.2, "confidence": 0.95},
    "left_elbow": {"x": 250.3, "y": 220.1, "confidence": 0.87}
  }
}
```

#### **⚙️ CURRENT OUTPUT (With Extensive Post-Processing):**
```json
{
  "data": {
    "biomechanics_per_frame": [
      {
        "keypoints": {...},
        "joint_angles": {
          "left_elbow": 125.6,
          "right_knee": 165.2,
          "left_shoulder": 95.4
        },
        "biomechanical_metrics": {
          "posture_score": 87.3,
          "balance_score": 82.1,
          "movement_efficiency": 91.5,
          "power_potential": 78.9
        }
      }
    ]
  }
}
```

### **What It Does**

1. **✅ Loads RTMPose-M or HRNet-W48** models with fallback strategy
2. **⚠️ POST-PROCESSING:** Calculates joint angles via trigonometry
3. **⚠️ POST-PROCESSING:** Assesses posture quality via custom scoring
4. **⚠️ POST-PROCESSING:** Evaluates balance using hip/ankle alignment math
5. **⚠️ POST-PROCESSING:** Measures movement efficiency against hardcoded ranges
6. **⚠️ POST-PROCESSING:** Estimates power potential via custom algorithms
7. **⚠️ POST-PROCESSING:** Generates comprehensive biomechanical reports

**🔥 ISSUE:** Extensive post-processing adds complexity and potential errors!

---

## ⚡ YOLO-NAS Service

### **Purpose**

High-accuracy detection using Super-Gradients YOLO-NAS with advanced optimization features.

### **API Endpoints**

- **Health Check**: `GET /healthz`
- **Pose Detection**: `POST /yolo-nas/pose`
- **Object Detection**: `POST /yolo-nas/object`
- **Enhanced Analysis**: `POST /yolo-nas/enhanced-analysis`
- **Model Optimization**: `POST /optimize-models`

### **Expected Input Format**

#### Standard Request

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3
}
```

#### Enhanced Analysis Request

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3,
  "enable_enhanced_analysis": true,
  "enable_joint_tracking": true,
  "enable_pose_quality": true,
  "enable_padel_analysis": true,
  "enable_batch_format": true
}
```

### **Expected Output Examples**

#### Enhanced Analysis Output

```json
{
  "format": "enhanced_batch",
  "features_enabled": {
    "enhanced_joint_confidence": true,
    "joint_tracking": true,
    "pose_quality": true,
    "padel_analysis": true,
    "batch_format": true
  },
  "data": {
    "poses_per_frame": [...],
    "enhanced_analysis": {
      "joint_analysis": [
        {
          "joint_confidence_avg": 0.89,
          "pose_quality_score": 0.91,
          "padel_stance_detected": false
        }
      ],
      "pose_quality": [0.91, 0.88, 0.93],
      "padel_specific": [false, true, false],
      "batch_predictions": {
        "num_detections": 45,
        "pred_boxes": [...],
        "pred_scores": [...],
        "pred_joints": [...],
        "pred_labels": [...]
      }
    }
  },
  "metadata": {
    "total_frames": 150,
    "processing_time": 5.2,
    "model_info": {"pose_model": "yolo_nas_pose_n (pytorch)", "object_model": "yolo_nas_s (pytorch)"},
    "nms_config": {
      "iou_threshold": 0.5,
      "score_threshold": 0.3,
      "max_detections": 10,
      "nms_per_class": true
    }
  }
}
```

### **What It Does**

1. **Loads YOLO-NAS models** with offline mode support and local caching
2. **Applies FP16 quantization** for memory efficiency
3. **Supports custom-trained models** for padel-specific detection
4. **Enhanced pose analysis** with joint confidence tracking
5. **Padel-specific stance detection** and movement analysis
6. **Advanced NMS configuration** for optimal detection
7. **Model optimization endpoints** for FP16/INT8 quantization
8. **Batch format output** for research and analysis
9. **Custom model loading** from `/app/weights/custom_padel_object.pth`

---

## 🔍 ViTPose++ Service

### **Purpose**

Vision Transformer-based pose estimation with advanced biomechanical insights and GPU memory optimization.

### **API Endpoints**

- **Health Check**: `GET /healthz`
- **Pose Analysis**: `POST /analyze`

### **Expected Input Format**

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3
}
```

### **Expected Output Examples**

#### Advanced Pose Analysis Output

```json
{
  "data": {
    "poses_per_frame": [
      {
        "keypoints": {...},
        "joint_angles": {
          "left_elbow": 125.6,
          "right_knee": 165.2,
          "hip_angle": 172.3,
          "spine_angle": 168.7
        },
        "pose_metrics": {
          "pose_quality_score": 89.4,
          "visible_keypoints": 15,
          "total_keypoints": 17,
          "confidence_threshold": 0.3,
          "model_used": "ViTPose-Base",
          "model_precision": "fp16",
          "biomechanical_insights": {
            "movement_efficiency": 88.7,
            "power_potential": 82.3,
            "balance_score": 85.9,
            "balance_status": "stable",
            "stability_metrics": {
              "overall_stability": 85.9,
              "postural_control": 87.3,
              "athletic_readiness": 85.5
            }
          }
        }
      }
    ],
    "processing_summary": {
      "total_frames": 150,
      "successful_analyses": 149,
      "total_keypoints": 2235,
      "confidence_threshold": 0.3
    }
  },
  "gpu_memory_usage": {
    "initial": {"total_mb": 8192, "allocated_mb": 1024, "free_mb": 7168},
    "final": {"total_mb": 8192, "allocated_mb": 1156, "free_mb": 7036}
  }
}
```

### **What It Does**

1. **Loads ViTPose-Base or HRNet-W48** with automatic fallback
2. **FP16 precision** for VRAM efficiency on GPU
3. **Advanced joint angle calculations** including spine and hip angles
4. **Balance assessment** with stability metrics
5. **Movement efficiency analysis** against optimal ranges
6. **Power potential estimation** for athletic performance
7. **GPU memory monitoring** and cleanup after each inference
8. **Athletic readiness scoring** for training optimization
9. **Comprehensive biomechanical insights** for detailed analysis

---

## 🎯 RF-DETR Service

### **Purpose**

Transformer-based object detection using RF-DETR with FP16 optimization and resolution constraints.

### **API Endpoints**

- **Health Check**: `GET /healthz`
- **Detection Analysis**: `POST /analyze`

### **Expected Input Format**

```json
{
  "video_url": "https://storage.googleapis.com/bucket/video.mp4",
  "video": false,
  "data": true,
  "confidence": 0.3,
  "resolution": 672
}
```

*Note: Resolution must be divisible by 56 for RF-DETR compatibility*

### **Expected Output Examples**

#### Object Detection Output

```json
{
  "data": {
    "detections_per_frame": [
      {
        "detections": [
          {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.89,
            "bbox": {
              "x1": 200.5,
              "y1": 150.2,
              "x2": 380.7,
              "y2": 520.1,
              "width": 180.2,
              "height": 369.9
            }
          }
        ],
        "detection_metrics": {
          "total_detections": 3,
          "confidence_threshold": 0.3,
          "resolution_used": 672,
          "model_used": "RF-DETR-Base",
          "model_precision": "fp16"
        }
      }
    ],
    "processing_summary": {
      "total_frames": 150,
      "successful_analyses": 150,
      "total_detections": 245,
      "confidence_threshold": 0.3,
      "resolution_used": 672
    }
  },
  "gpu_memory_usage": {
    "initial": {"total_mb": 8192, "allocated_mb": 1024, "free_mb": 7168},
    "final": {"total_mb": 8192, "allocated_mb": 1089, "free_mb": 7103}
  }
}
```

### **What It Does**

1. **Loads RF-DETR-Base** transformer model with FP16 precision
2. **Enforces resolution constraints** (must be divisible by 56)
3. **Automatic frame resizing** to required resolution with coordinate scaling
4. **FP16 autocast** for efficient GPU memory usage
5. **BGR to RGB conversion** for model compatibility
6. **Detailed bounding box information** with width/height calculations
7. **GPU memory monitoring** and cleanup
8. **Transformer-based detection** for improved accuracy over CNN approaches

---

## 🚀 Common Features Across All Services

### **GPU Optimization**

- **FP16 precision** where supported
- **Batch processing** (typically 8 frames)
- **Memory cleanup** after inference
- **CUDA device detection** and optimization

### **Video Processing**

- **FFmpeg integration** for video creation
- **Frame extraction** with OpenCV
- **GCS upload** with public URL generation
- **Temporary file cleanup**

### **Error Handling**

- **Graceful fallbacks** when models fail to load
- **Detailed error logging** with stack traces
- **Health check endpoints** with model status
- **Timeout handling** for video processing

### **Configuration**

- **Hot reloading** of configuration files
- **Environment variable overrides**
- **Feature flags** for dynamic behavior
- **Model path flexibility**

---

## ⚠️ **CRITICAL ISSUES IDENTIFIED**

### **1. Missing Utils Dependencies**
Several services are trying to import deleted utilities:
```python
# YOLO8 & YOLO11 Services - BROKEN IMPORTS:
from utils.video_utils import get_video_info, extract_frames
from utils.ball_tracker import smooth_ball_trajectory, draw_enhanced_ball_trajectory

# YOLO-NAS Service - BROKEN IMPORTS:
from utils.video_utils import get_video_info, extract_frames
from utils.model_optimizer import ModelOptimizer
```
**Status**: ❌ These utils directories were deleted during cleanup but services still reference them

### **2. Missing TrackNet Service**
**Expected**: TrackNet V2 service with `POST /track-ball` endpoint (mentioned in README)
**Status**: ❌ No TrackNet service exists - was removed with yolo-combined

---

## 🔧 **ACTUAL CURRENT ENDPOINTS** (Based on Code Analysis)

| Service | Port | Health Check | Primary Endpoints | Additional Endpoints |
|---------|------|--------------|-------------------|---------------------|
| **YOLO8** | 8002 | `GET /healthz` | `POST /pose`<br>`POST /object` | - |
| **YOLO11** | 8007 | `GET /healthz` | `POST /pose`<br>`POST /object` | - |
| **MMPose** | 8003 | `GET /healthz` | `POST /mmpose/pose` | - |
| **YOLO-NAS** | 8004 | `GET /healthz` | `POST /yolo-nas/pose`<br>`POST /yolo-nas/object` | `POST /yolo-nas/enhanced-analysis`<br>`POST /optimize-models` |
| **RF-DETR** | 8005 | `GET /healthz` | `POST /analyze` | - |
| **ViTPose++** | 8006 | `GET /healthz` | `POST /analyze` | - |
| **TrackNet** | ❌ Missing | ❌ | ❌ `POST /track-ball` | ❌ |

---

## 🎯 Use Case Recommendations

| Use Case | Recommended Service | Why |
|----------|-------------------|-----|
| **Real-time Detection** | RF-DETR | Transformer efficiency, FP16 optimization |
| **Detailed Biomechanics** | MMPose or ViTPose++ | Advanced joint analysis, stability metrics |
| **Ball Tracking** | ⚠️ **MISSING** | TrackNet service needs to be implemented |
| **High Accuracy** | YOLO-NAS | Super-Gradients architecture, custom models |
| **Latest Performance** | YOLO11 | Most recent YOLO architecture |
| **General Purpose** | YOLO8 | Reliable, well-tested performance |
| **Memory Constrained** | ViTPose++ | Efficient GPU memory management |

---

## 🚨 **IMMEDIATE ACTION REQUIRED**

### **Priority 1: Fix Broken Services**
1. **Create missing utils modules** or remove imports from:
   - `services/yolo8/main.py`
   - `services/yolo11/main.py`
   - `services/yolo-nas/main.py`

### **Priority 2: Implement TrackNet Service**
1. **Create TrackNet service** with `POST /track-ball` endpoint
2. **Port:** 8008 (next available)
3. **Model:** TrackNet V2 for specialized ball tracking

**🔥 Current services may fail to start due to missing utils imports!**
