#!/usr/bin/env python3
"""
YOLO Compatibility Testing Script
Tests YOLOv8 and YOLO11 compatibility with different ultralytics versions
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, timeout=60):
    """Run command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def test_ultralytics_installation():
    """Test current ultralytics installation"""
    logger.info("=== Testing Current Ultralytics Installation ===")
    
    try:
        import ultralytics
        logger.info(f"✓ Ultralytics version: {ultralytics.__version__}")
        
        # Test YOLO import
        from ultralytics import YOLO
        logger.info("✓ YOLO class import successful")
        
        # Test basic YOLO instance creation
        model = YOLO()  # Creates empty model
        logger.info("✓ YOLO instance creation successful")
        
        return True, ultralytics.__version__
    except Exception as e:
        logger.error(f"✗ Ultralytics installation issue: {e}")
        return False, None

def test_model_loading(model_path, model_name):
    """Test loading a specific model"""
    logger.info(f"Testing {model_name} model loading...")
    
    if not os.path.exists(model_path):
        logger.warning(f"✗ Model file not found: {model_path}")
        return False, f"Model file not found: {model_path}"
    
    try:
        from ultralytics import YOLO
        
        # Test model loading
        model = YOLO(model_path)
        logger.info(f"✓ {model_name} loaded successfully")
        
        # Test model info
        if hasattr(model, 'info'):
            try:
                model.info()
                logger.info(f"✓ {model_name} info() method works")
            except Exception as e:
                logger.warning(f"⚠ {model_name} info() failed: {e}")
        
        # Test model names/classes
        if hasattr(model, 'names'):
            logger.info(f"✓ {model_name} has {len(model.names)} classes")
        
        return True, "Model loaded successfully"
        
    except Exception as e:
        logger.error(f"✗ {model_name} loading failed: {e}")
        return False, str(e)

def test_inference_compatibility(model_path, model_name):
    """Test inference API compatibility"""
    logger.info(f"Testing {model_name} inference compatibility...")
    
    if not os.path.exists(model_path):
        logger.warning(f"✗ Model file not found: {model_path}")
        return False, "Model file not found"
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Load model
        model = YOLO(model_path)
        
        # Create dummy image data (RGB)
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test inference
        results = model(dummy_image, verbose=False)
        logger.info(f"✓ {model_name} inference successful")
        
        # Test result structure
        result = results[0]
        
        # Check common attributes
        attributes_to_check = ['boxes', 'keypoints', 'names', 'path']
        for attr in attributes_to_check:
            if hasattr(result, attr):
                logger.info(f"✓ {model_name} result has '{attr}' attribute")
            else:
                logger.info(f"○ {model_name} result missing '{attr}' attribute (may be normal)")
        
        # Test pose-specific attributes for pose models
        if 'pose' in model_name.lower():
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                logger.info(f"✓ {model_name} keypoints available")
                if hasattr(result.keypoints, 'data') and result.keypoints.data is not None:
                    kp_shape = result.keypoints.data.shape
                    logger.info(f"✓ {model_name} keypoints shape: {kp_shape}")
                else:
                    logger.warning(f"⚠ {model_name} keypoints.data is None")
            else:
                logger.warning(f"⚠ {model_name} missing keypoints for pose model")
        
        # Test object detection attributes
        if hasattr(result, 'boxes') and result.boxes is not None:
            logger.info(f"✓ {model_name} boxes available")
            if hasattr(result.boxes, 'data') and result.boxes.data is not None:
                box_shape = result.boxes.data.shape
                logger.info(f"✓ {model_name} boxes shape: {box_shape}")
        
        return True, "Inference successful"
        
    except Exception as e:
        logger.error(f"✗ {model_name} inference failed: {e}")
        return False, str(e)

def test_version_specific_features():
    """Test version-specific features and API changes"""
    logger.info("=== Testing Version-Specific Features ===")
    
    try:
        from ultralytics import YOLO
        
        # Test if YOLO11 models are supported
        try:
            # This will fail gracefully if YOLO11 not supported
            model = YOLO('yolo11n.pt')  # This might download if not exist
            logger.info("✓ YOLO11 model class supported")
            return True
        except Exception as e:
            logger.warning(f"⚠ YOLO11 support unclear: {e}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Version feature test failed: {e}")
        return False

def main():
    """Main compatibility testing function"""
    logger.info("🔍 YOLO Compatibility Testing Started")
    logger.info("=" * 50)
    
    # Test 1: Current installation
    install_ok, version = test_ultralytics_installation()
    if not install_ok:
        logger.error("Cannot proceed - ultralytics installation issues")
        return False
    
    # Test 2: Version-specific features
    test_version_specific_features()
    
    # Test 3: Model paths
    weights_dir = Path("./weights/ultralytics")
    models_to_test = [
        ("yolo11n.pt", "YOLO11 Object Detection"),
        ("yolo11n-pose.pt", "YOLO11 Pose Detection"),
        ("yolov8n.pt", "YOLOv8 Object Detection"),
        ("yolov8n-pose.pt", "YOLOv8 Pose Detection"),
    ]
    
    logger.info("=== Model Loading Tests ===")
    load_results = {}
    
    for model_file, model_name in models_to_test:
        model_path = weights_dir / model_file
        success, message = test_model_loading(str(model_path), model_name)
        load_results[model_name] = success
        
        if success:
            # Test inference if loading succeeded
            inf_success, inf_message = test_inference_compatibility(str(model_path), model_name)
            logger.info(f"Inference test for {model_name}: {'✓' if inf_success else '✗'}")
    
    # Summary
    logger.info("=" * 50)
    logger.info("🔍 COMPATIBILITY TEST SUMMARY")
    logger.info(f"Ultralytics version: {version}")
    
    total_models = len(models_to_test)
    successful_loads = sum(load_results.values())
    
    logger.info(f"Model loading: {successful_loads}/{total_models} successful")
    
    for model_name, success in load_results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {model_name}")
    
    # Recommendations
    logger.info("=" * 50)
    logger.info("📋 RECOMMENDATIONS")
    
    if successful_loads == total_models:
        logger.info("✅ All models compatible - safe to upgrade ultralytics")
    elif successful_loads >= total_models // 2:
        logger.info("⚠️  Partial compatibility - review failed models before upgrade")
    else:
        logger.info("❌ Major compatibility issues - investigate before upgrade")
    
    # Check for specific YOLOv8 compatibility
    yolov8_models = [name for name in load_results.keys() if "YOLOv8" in name]
    yolov8_success = all(load_results[name] for name in yolov8_models)
    
    if yolov8_success and yolov8_models:
        logger.info("✅ YOLOv8 backward compatibility confirmed")
    elif yolov8_models:
        logger.info("❌ YOLOv8 backward compatibility issues detected")
    
    return successful_loads == total_models

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)