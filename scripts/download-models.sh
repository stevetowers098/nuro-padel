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