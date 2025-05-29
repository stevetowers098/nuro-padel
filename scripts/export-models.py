#!/usr/bin/env python3
"""
Model Export Script for NuroPadel AI Services
Exports PyTorch models to ONNX and TensorRT for optimization
Run this on the VM after downloading the base models
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path

# Add service paths
sys.path.append('/opt/padel-docker/services/yolo-nas')
sys.path.append('/opt/padel-docker/services/yolo-combined')
sys.path.append('/opt/padel-docker/services/mmpose')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import optimization utilities
try:
    from services.yolo_nas.utils.model_optimizer import export_pytorch_to_onnx, export_onnx_to_tensorrt
except ImportError:
    logger.error("Could not import model optimizer utilities")
    sys.exit(1)

def export_yolo_nas_models():
    """Export YOLO-NAS models to ONNX and TensorRT"""
    logger.info("🚀 Exporting YOLO-NAS models...")
    
    try:
        from super_gradients.training import models
        weights_dir = Path("/opt/padel-docker/weights")
        optimized_dir = weights_dir / "optimized"
        optimized_dir.mkdir(exist_ok=True)
        
        # Export YOLO-NAS Pose Model
        logger.info("📦 Exporting YOLO-NAS Pose model...")
        pose_checkpoint = weights_dir / "super-gradients" / "yolo_nas_pose_n_coco_pose.pth"
        
        if pose_checkpoint.exists():
            # Load model
            pose_model = models.get("yolo_nas_pose_n", 
                                  checkpoint_path=str(pose_checkpoint), 
                                  num_classes=17)
            pose_model.eval()
            
            if torch.cuda.is_available():
                pose_model.cuda()
            
            # Export to ONNX
            onnx_path = optimized_dir / "yolo_nas_pose_n.onnx"
            input_shape = (1, 3, 640, 640)  # Standard YOLO input
            
            if export_pytorch_to_onnx(pose_model, input_shape, str(onnx_path)):
                logger.info(f"✅ YOLO-NAS Pose exported to ONNX: {onnx_path}")
                
                # Export to TensorRT
                engine_path = optimized_dir / "yolo_nas_pose_n.engine"
                if export_onnx_to_tensorrt(str(onnx_path), str(engine_path)):
                    logger.info(f"✅ YOLO-NAS Pose exported to TensorRT: {engine_path}")
        
        # Export YOLO-NAS Object Model
        logger.info("📦 Exporting YOLO-NAS Object model...")
        object_checkpoint = weights_dir / "super-gradients" / "yolo_nas_s_coco.pth"
        
        if object_checkpoint.exists():
            # Load model
            object_model = models.get("yolo_nas_s", 
                                    checkpoint_path=str(object_checkpoint), 
                                    num_classes=80)
            object_model.eval()
            
            if torch.cuda.is_available():
                object_model.cuda()
            
            # Export to ONNX
            onnx_path = optimized_dir / "yolo_nas_s.onnx"
            input_shape = (1, 3, 640, 640)
            
            if export_pytorch_to_onnx(object_model, input_shape, str(onnx_path)):
                logger.info(f"✅ YOLO-NAS Object exported to ONNX: {onnx_path}")
                
                # Export to TensorRT
                engine_path = optimized_dir / "yolo_nas_s.engine"
                if export_onnx_to_tensorrt(str(onnx_path), str(engine_path)):
                    logger.info(f"✅ YOLO-NAS Object exported to TensorRT: {engine_path}")
    
    except Exception as e:
        logger.error(f"❌ Failed to export YOLO-NAS models: {e}")

def export_yolo_combined_models():
    """Export YOLO Combined models to ONNX and TensorRT"""
    logger.info("🚀 Exporting YOLO Combined models...")
    
    try:
        from ultralytics import YOLO
        weights_dir = Path("/opt/padel-docker/weights")
        optimized_dir = weights_dir / "optimized"
        optimized_dir.mkdir(exist_ok=True)
        
        models_to_export = [
            ("yolo11n-pose.pt", "yolo11n_pose"),
            ("yolo11n.pt", "yolo11n_object"),
            ("yolov8n.pt", "yolov8n_object"),
            ("yolov8n-pose.pt", "yolov8n_pose")
        ]
        
        for model_file, model_name in models_to_export:
            model_path = weights_dir / model_file
            
            if model_path.exists():
                logger.info(f"📦 Exporting {model_name}...")
                
                # Load YOLO model
                model = YOLO(str(model_path))
                
                # Export to ONNX using Ultralytics built-in export
                try:
                    onnx_path = optimized_dir / f"{model_name}.onnx"
                    model.export(format="onnx",
                               dynamic=True,
                               opset=12,
                               simplify=True,
                               optimize=True)
                    
                    # Move exported file to our optimized directory
                    exported_onnx = model_path.with_suffix('.onnx')
                    if exported_onnx.exists():
                        exported_onnx.rename(onnx_path)
                        logger.info(f"✅ {model_name} exported to ONNX: {onnx_path}")
                        
                        # Export to TensorRT
                        engine_path = optimized_dir / f"{model_name}.engine"
                        if export_onnx_to_tensorrt(str(onnx_path), str(engine_path)):
                            logger.info(f"✅ {model_name} exported to TensorRT: {engine_path}")
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to export {model_name}: {e}")
    
    except Exception as e:
        logger.error(f"❌ Failed to export YOLO Combined models: {e}")

def export_tracknet_models():
    """Export TrackNet models to ONNX and TensorRT"""
    logger.info("🚀 Exporting TrackNet models...")
    
    try:
        # Add TrackNet path to sys.path
        import sys
        sys.path.append('/opt/padel-docker/services/yolo-combined')
        
        from tracknet.model import TrackNet, load_tracknet_model
        weights_dir = Path("/opt/padel-docker/weights")
        optimized_dir = weights_dir / "optimized"
        optimized_dir.mkdir(exist_ok=True)
        
        # TrackNet model file
        tracknet_path = weights_dir / "tracknet_v2.pth"
        
        if tracknet_path.exists():
            logger.info("📦 Exporting TrackNet v2...")
            
            # Load TrackNet model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            tracknet_model = load_tracknet_model(str(tracknet_path), device)
            
            if tracknet_model:
                tracknet_model.eval()
                
                # Export to ONNX
                onnx_path = optimized_dir / "tracknet_v2.onnx"
                input_shape = (1, 9, 360, 640)  # Batch, 3 frames * 3 channels, height, width
                
                # Create dynamic axes for TrackNet (batch size can vary)
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
                
                if export_pytorch_to_onnx(tracknet_model, input_shape, str(onnx_path), dynamic_axes):
                    logger.info(f"✅ TrackNet exported to ONNX: {onnx_path}")
                    
                    # Export to TensorRT
                    engine_path = optimized_dir / "tracknet_v2.engine"
                    if export_onnx_to_tensorrt(str(onnx_path), str(engine_path)):
                        logger.info(f"✅ TrackNet exported to TensorRT: {engine_path}")
                else:
                    logger.warning("⚠️ Failed to export TrackNet to ONNX")
            else:
                logger.warning("⚠️ Failed to load TrackNet model")
        else:
            logger.info(f"ℹ️ TrackNet model not found at {tracknet_path} - skipping export")
            logger.info("ℹ️ TrackNet will use random weights for ball tracking")
    
    except Exception as e:
        logger.error(f"❌ Failed to export TrackNet models: {e}")

def export_mmpose_models():
    """Export MMPose models to ONNX"""
    logger.info("🚀 Exporting MMPose models...")
    
    try:
        # MMPose export is more complex and requires specific scripts
        # For now, we'll create a placeholder that logs the process
        logger.info("📦 MMPose model export requires specific configuration...")
        logger.info("ℹ️ MMPose models can be exported using mmdeploy tools:")
        logger.info("   pip install mmdeploy")
        logger.info("   Use mmdeploy.apis.pytorch2onnx for RTMPose models")
        logger.info("   Refer to MMPose documentation for specific export procedures")
        
    except Exception as e:
        logger.error(f"❌ Failed to export MMPose models: {e}")

def main():
    """Main export function"""
    logger.info("🚀 Starting NuroPadel model optimization pipeline...")
    
    # Check if running on CUDA-capable system
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        logger.warning("⚠️ CUDA not available - exports will be CPU-only")
    
    # Create optimized directory
    optimized_dir = Path("/opt/padel-docker/weights/optimized")
    optimized_dir.mkdir(parents=True, exist_ok=True)
    
    # Export models for each service
    export_yolo_nas_models()
    export_yolo_combined_models()
    export_tracknet_models()
    export_mmpose_models()
    
    # Summary
    logger.info("📊 Export Summary:")
    if optimized_dir.exists():
        onnx_files = list(optimized_dir.glob("*.onnx"))
        engine_files = list(optimized_dir.glob("*.engine"))
        
        logger.info(f"   ONNX models: {len(onnx_files)}")
        for f in onnx_files:
            logger.info(f"     - {f.name}")
        
        logger.info(f"   TensorRT engines: {len(engine_files)}")
        for f in engine_files:
            logger.info(f"     - {f.name}")
    
    logger.info("✅ Model optimization pipeline complete!")
    logger.info("🔄 Restart your Docker services to use optimized models")

if __name__ == "__main__":
    main()