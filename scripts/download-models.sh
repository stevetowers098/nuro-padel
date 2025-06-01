#!/bin/bash
# Pre-download YOLO-NAS models during build

echo "📥 Downloading YOLO-NAS models..."

# Install super-gradients temporarily
pip install super-gradients==3.7.1

# Download models to weights directory
python3 -c "
import os
from super_gradients.training import models

# Ensure weights directory exists
os.makedirs('/app/weights/super-gradients', exist_ok=True)

# Download YOLO-NAS models
print('Downloading YOLO-NAS object detection model...')
models.get('yolo_nas_s', pretrained_weights='coco')

print('Downloading YOLO-NAS pose estimation model...')  
models.get('yolo_nas_pose_n', pretrained_weights='coco_pose')

print('✅ Model download complete!')
"

echo "✅ YOLO-NAS models downloaded"

echo "📥 Downloading YOLOv8 models..."
mkdir -p ../weights/ultralytics
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt -P ../weights/ultralytics
wget -nc https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt -P ../weights/ultralytics
echo "✅ YOLOv8 models downloaded"

echo "📥 Downloading RF-DETR models..."
# RF-DETR models are typically downloaded by the library itself or require specific URLs.
# Placeholder for manual download if needed:
# wget -nc [RF-DETR_MODEL_URL] -P ../weights/rf-detr/
echo "⚠️ RF-DETR models may require manual download or are handled by the library."

echo "📥 Downloading ViTPose++ models..."
mkdir -p ../weights/vitpose
# ViTPose-Base checkpoint
wget -nc https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/vitpose/vitpose_base_coco_256x192-e210f354_20221220.pth -O ../weights/vitpose/vitpose_base_coco_256x192.pth
# HRNet-W48 fallback checkpoint
wget -nc https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220926.pth -O ../weights/vitpose/hrnet_w48_coco_256x192.pth
echo "✅ ViTPose++ models downloaded"