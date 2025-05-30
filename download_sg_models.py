#!/usr/bin/env python3
"""
Download SuperGradients YOLO-NAS models directly
"""
import os
import sys

# Add the directory to Python path if needed
sys.path.append('.')

# Set up directories
weights_dir = "weights/super-gradients"
os.makedirs(weights_dir, exist_ok=True)

try:
    from super_gradients.training import models
    print("✅ SuperGradients available")
    
    # Download YOLO-NAS pose model
    print("📦 Downloading YOLO-NAS Pose model...")
    pose_model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
    pose_checkpoint_path = os.path.join(weights_dir, "yolo_nas_pose_n_coco_pose.pth")
    print(f"💾 Saving pose model to: {pose_checkpoint_path}")
    
    # Download YOLO-NAS object model  
    print("📦 Downloading YOLO-NAS Object model...")
    object_model = models.get("yolo_nas_s", pretrained_weights="coco")
    object_checkpoint_path = os.path.join(weights_dir, "yolo_nas_s_coco.pth")
    print(f"💾 Saving object model to: {object_checkpoint_path}")
    
    print("✅ Models downloaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)