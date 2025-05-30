#!/bin/bash
# YOLO-NAS Model Download via Docker
# Uses the yolo-nas container itself to download models

set -euo pipefail

echo "🐳 YOLO-NAS Model Download via Docker Container"
echo "=============================================="

# Build the YOLO-NAS container if needed
echo "📦 Building YOLO-NAS container..."
cd deployment
docker-compose build yolo-nas

# Create weights/super-gradients directory
echo "📁 Creating weights directory structure..."
mkdir -p ../weights/super-gradients

# Download models using the container's Python environment
echo "⬇️ Downloading YOLO-NAS models using container..."
docker run --rm \
  -v "$(pwd)/../weights:/app/weights" \
  -v "$(pwd)/../scripts:/app/scripts:ro" \
  ghcr.io/stevetowers098/nuro-padel/yolo-nas:latest \
  python /app/scripts/download-yolo-nas.py

# Verify the downloads
echo "✅ Verifying downloaded models..."
if [ -f "../weights/super-gradients/yolo_nas_pose_n_coco_pose.pth" ] && \
   [ -f "../weights/super-gradients/yolo_nas_s_coco.pth" ]; then
    echo "✅ YOLO-NAS models successfully downloaded!"
    echo "📊 Model files:"
    ls -lh ../weights/super-gradients/*.pth
else
    echo "❌ Model download failed - files not found"
    echo "📁 Contents of weights/super-gradients/:"
    ls -la ../weights/super-gradients/ || echo "Directory does not exist"
    exit 1
fi

echo ""
echo "🚀 YOLO-NAS models ready! You can now restart the services:"
echo "   docker-compose down && docker-compose up -d"