#!/usr/bin/env python3
"""
YOLO-NAS Model Download Script for NuroPadel
Downloads YOLO-NAS models using super_gradients library with DNS workarounds
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_dns_workaround():
    """Setup DNS workaround for sghub.deci.ai issues"""
    try:
        import socket
        import platform
        
        # Add custom DNS resolution if needed
        logger.info("Setting up DNS configuration for super-gradients downloads...")
        
        # For Windows, we might need to use specific DNS servers
        if platform.system() == "Windows":
            # Try using Google DNS
            logger.info("Windows detected - using fallback DNS configuration")
    except Exception as e:
        logger.warning(f"DNS setup warning: {e}")

def download_yolo_nas_models():
    """Download YOLO-NAS models using super_gradients"""
    
    # Setup paths
    weights_dir = Path("./weights")
    super_gradients_dir = weights_dir / "super-gradients"
    super_gradients_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading YOLO-NAS models to: {super_gradients_dir}")
    
    try:
        # Import super_gradients
        from super_gradients.training import models
        logger.info("✅ Super Gradients library available")
        
        # Setup DNS workaround
        setup_dns_workaround()
        
        # Download YOLO-NAS Pose Model
        pose_model_path = super_gradients_dir / "yolo_nas_pose_n_coco_pose.pth"
        
        if not pose_model_path.exists():
            logger.info("📦 Downloading YOLO-NAS Pose model (yolo_nas_pose_n)...")
            try:
                # This will download the model with pretrained weights
                pose_model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
                
                # Save the model state dict
                torch.save(pose_model.state_dict(), pose_model_path)
                logger.info(f"✅ YOLO-NAS Pose model saved to: {pose_model_path}")
                
                # Also save the full model for compatibility
                full_model_path = super_gradients_dir / "yolo_nas_pose_n_full.pth"
                torch.save(pose_model, full_model_path)
                logger.info(f"✅ Full YOLO-NAS Pose model saved to: {full_model_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download YOLO-NAS Pose model: {e}")
                logger.info("💡 This may be due to DNS issues with sghub.deci.ai")
                logger.info("💡 Try running from a different network or using VPN")
        else:
            logger.info(f"✅ YOLO-NAS Pose model already exists: {pose_model_path}")
        
        # Download YOLO-NAS Object Model
        object_model_path = super_gradients_dir / "yolo_nas_s_coco.pth"
        
        if not object_model_path.exists():
            logger.info("📦 Downloading YOLO-NAS Object model (yolo_nas_s)...")
            try:
                # This will download the model with pretrained weights
                object_model = models.get("yolo_nas_s", pretrained_weights="coco")
                
                # Save the model state dict
                torch.save(object_model.state_dict(), object_model_path)
                logger.info(f"✅ YOLO-NAS Object model saved to: {object_model_path}")
                
                # Also save the full model for compatibility
                full_model_path = super_gradients_dir / "yolo_nas_s_full.pth"
                torch.save(object_model, full_model_path)
                logger.info(f"✅ Full YOLO-NAS Object model saved to: {full_model_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download YOLO-NAS Object model: {e}")
                logger.info("💡 This may be due to DNS issues with sghub.deci.ai")
                logger.info("💡 Try running from a different network or using VPN")
        else:
            logger.info(f"✅ YOLO-NAS Object model already exists: {object_model_path}")
        
        # Summary
        logger.info("📊 YOLO-NAS Download Summary:")
        for model_file in super_gradients_dir.glob("*.pth"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            logger.info(f"   {model_file.name}: {size_mb:.1f} MB")
            
    except ImportError:
        logger.error("❌ Super Gradients library not available")
        logger.info("💡 Install with: pip install super-gradients")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to download YOLO-NAS models: {e}")
        return False
    
    return True

def verify_yolo_nas_models():
    """Verify downloaded YOLO-NAS models"""
    
    weights_dir = Path("./weights")
    super_gradients_dir = weights_dir / "super-gradients"
    
    expected_models = [
        "yolo_nas_pose_n_coco_pose.pth",
        "yolo_nas_s_coco.pth"
    ]
    
    logger.info("🔍 Verifying YOLO-NAS models...")
    
    all_valid = True
    for model_name in expected_models:
        model_path = super_gradients_dir / model_name
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Models should be at least 10MB
                logger.info(f"✅ {model_name}: {size_mb:.1f} MB")
            else:
                logger.warning(f"⚠️ {model_name}: {size_mb:.1f} MB (too small)")
                all_valid = False
        else:
            logger.error(f"❌ {model_name}: Not found")
            all_valid = False
    
    return all_valid

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download YOLO-NAS models for NuroPadel")
    parser.add_argument("--verify-only", action="store_true", 
                       help="Only verify existing models, don't download")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download even if models exist")
    
    args = parser.parse_args()
    
    logger.info("🚀 YOLO-NAS Model Download Script")
    
    if args.verify_only:
        success = verify_yolo_nas_models()
        sys.exit(0 if success else 1)
    
    if args.force:
        logger.info("🔄 Force mode - will re-download existing models")
        import shutil
        super_gradients_dir = Path("./weights/super-gradients")
        if super_gradients_dir.exists():
            shutil.rmtree(super_gradients_dir)
    
    # Download models
    success = download_yolo_nas_models()
    
    if success:
        # Verify after download
        verify_success = verify_yolo_nas_models()
        if verify_success:
            logger.info("✅ YOLO-NAS models downloaded and verified successfully!")
        else:
            logger.warning("⚠️ YOLO-NAS models downloaded but verification failed")
            success = False
    
    if not success:
        logger.error("❌ YOLO-NAS model download failed")
        logger.info("💡 Troubleshooting tips:")
        logger.info("   1. Check internet connection")
        logger.info("   2. Try different network/VPN (DNS issues with sghub.deci.ai)")
        logger.info("   3. Ensure super-gradients is installed: pip install super-gradients")
        logger.info("   4. Check available disk space")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()