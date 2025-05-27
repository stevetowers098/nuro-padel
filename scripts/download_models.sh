#!/bin/bash
set -euo pipefail

# Debug info
whoami
pwd
ls -l "$PWD"
ls -ld "$PWD" || true

# Ensure script is running from the correct directory and weights dir exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
WEIGHTS_DIR="$SCRIPT_DIR/../weights"
if [ ! -d "$WEIGHTS_DIR" ]; then
  echo "ERROR: Weights directory $WEIGHTS_DIR does not exist!"
  exit 2
fi
cd "$WEIGHTS_DIR"
echo "Now in: $PWD"
ls -l "$PWD"

# Check write permissions
if [ ! -w "$PWD" ]; then
  echo "ERROR: No write permission to $PWD"
  exit 2
fi

# Check wget is installed
if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget is not installed!"
  exit 2
fi

# Download YOLOv8 model
if [ ! -f "yolov8m.pt" ]; then
  wget --timeout=60 --tries=2 -nv -O yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt || echo "WARNING: YOLOv8 model download failed, but continuing."
else
  echo "✅ yolov8m.pt already exists"
fi

# Download RTMPose weights
if [ ! -f "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth" ]; then
  wget --timeout=60 --tries=2 -nv -O rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth || echo "WARNING: RTMPose weights download failed, but continuing."
else
  echo "✅ RTMPose weights already exist"
fi

# Download HRNet weights
if [ ! -f "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth" ]; then
  wget --timeout=60 --tries=2 -nv -O hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth || echo "WARNING: HRNet weights download failed, but continuing."
else
  echo "✅ HRNet weights already exist"
fi

# Download YOLO11n pose model (if required)
if [ ! -f "yolo11n-pose.pt" ]; then
  wget --timeout=60 --tries=2 -nv -O yolo11n-pose.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt || echo "WARNING: YOLO11n pose model download failed, but continuing."
else
  echo "✅ yolo11n-pose.pt already exists"
fi

echo "Model download script complete."
