#!/bin/bash

# NuroPadel Model Download Script
# Downloads all required AI model files for the padel analysis system

set -e  # Exit on any error

echo "🎾 NuroPadel Model Download Script"
echo "=================================="

# Create weights directory if it doesn't exist
WEIGHTS_DIR="./weights"
mkdir -p "$WEIGHTS_DIR"

echo "📁 Created weights directory: $WEIGHTS_DIR"

# Function to download with progress and verification
download_model() {
    local url="$1"
    local filename="$2"
    local description="$3"
    local expected_size="$4"  # Optional expected file size in MB
    
    local filepath="$WEIGHTS_DIR/$filename"
    
    echo ""
    echo "📥 Downloading $description..."
    echo "   URL: $url"
    echo "   File: $filename"
    
    if [[ -f "$filepath" ]]; then
        echo "   ✅ File already exists: $filepath"
        if [[ -n "$expected_size" ]]; then
            local actual_size=$(du -m "$filepath" | cut -f1)
            if [[ $actual_size -ge $expected_size ]]; then
                echo "   ✅ File size OK: ${actual_size}MB (expected: ~${expected_size}MB)"
                return 0
            else
                echo "   ⚠️  File size too small: ${actual_size}MB (expected: ~${expected_size}MB) - Re-downloading..."
            fi
        else
            return 0
        fi
    fi
    
    # Download with wget (show progress)
    if command -v wget &> /dev/null; then
        wget -O "$filepath" "$url" --progress=bar:force 2>&1
    elif command -v curl &> /dev/null; then
        curl -L -o "$filepath" "$url" --progress-bar
    else
        echo "❌ Error: Neither wget nor curl is available"
        exit 1
    fi
    
    if [[ -f "$filepath" ]]; then
        local file_size=$(du -h "$filepath" | cut -f1)
        echo "   ✅ Download complete: $filename ($file_size)"
    else
        echo "   ❌ Download failed: $filename"
        exit 1
    fi
}

echo "🚀 Starting model downloads..."

# MMPose Service Models
echo "📥 Downloading MMPose service models..."

# RTMPose Model (verified from main.py)
download_model \
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth" \
    "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth" \
    "RTMPose Model (Human Pose Estimation)" \
    100

# HRNet Model (verified fallback from main.py)
download_model \
    "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth" \
    "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth" \
    "HRNet Model (Fallback Pose Estimation)" \
    350

# TrackNet Model (for YOLO Combined Service ball tracking)
echo "📥 TrackNet model for ball tracking..."
echo "⚠️  TrackNet model URL needs verification - check TrackNet repository"
echo "   Expected file: tracknet_v2.pth"
echo "   Used by: YOLO Combined Service for enhanced ball tracking"
echo "   Location: /app/weights/tracknet_v2.pth"
echo "   Note: This model may need to be manually obtained from TrackNet repository"

# YOLO Combined Service Models (fact-checked from main.py)
echo "📥 Downloading YOLO Combined Service models..."

# YOLO11 Pose Model (yolo11n-pose.pt)
download_model \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt" \
    "yolo11n-pose.pt" \
    "YOLO11 Nano Pose (17 keypoints)" \
    6

# YOLOv8 Object Detection Model (yolov8m.pt)
download_model \
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt" \
    "yolov8m.pt" \
    "YOLOv8 Medium Object Detection" \
    52

# YOLOv8 Pose Model (yolov8n-pose.pt)
download_model \
    "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt" \
    "yolov8n-pose.pt" \
    "YOLOv8 Nano Pose (17 keypoints)" \
    6

# YOLO-NAS Models (for YOLO-NAS service)
echo "📥 Note: YOLO-NAS models are downloaded automatically by super-gradients library"
echo "   Models used in main.py:"
echo "   - yolo_nas_pose_n (with coco_pose weights)"
echo "   - yolo_nas_s (with coco weights)"
echo "   Location: Auto-downloaded to cache during first run"
echo "   No manual download required for YOLO-NAS service"

# Summary
echo ""
echo "🎉 Model Download Complete!"
echo "=============================="
echo "📊 Download Summary:"
ls -lh "$WEIGHTS_DIR" | grep -E '\.(pt|pth)$' || echo "No model files found"

echo ""
echo "📁 All models saved to: $WEIGHTS_DIR"
echo "🐳 Ready for Docker build!"
echo ""
echo "💡 Next steps:"
echo "   1. Run: docker-compose build"
echo "   2. Run: docker-compose up"
echo ""
echo "🔍 For development mode:"
echo "   docker-compose -f docker-compose.dev.yml up"