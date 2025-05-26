#!/bin/bash
# Intelligent environment setup script for NuroPadel
# Groups AI models by compatible dependencies rather than by model type

set -e  # Exit on any error

echo "🚀 Starting intelligent environment setup..."

# Create directory structure
mkdir -p /opt/padel/envs/{modern-torch,legacy-torch,specialized}
mkdir -p /opt/padel/models

create_modern_env() {
    echo "Setting up modern PyTorch environment..."
    cd /opt/padel/envs/modern-torch
    python3 -m venv venv --clear
    source venv/bin/activate
    
    # Install with version pinning for stability
    pip install --upgrade pip
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    pip install ultralytics==8.0.196 opencv-python supervision fastapi uvicorn
    
    # Download YOLO models to their default cache location
    python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
    
    # Verify installation
    python -c "import torch, ultralytics; print(f'✅ Modern env: torch {torch.__version__}, ultralytics {ultralytics.__version__}')"
}

create_legacy_env() {
    echo "Setting up legacy PyTorch environment..."
    cd /opt/padel/envs/legacy-torch
    python3 -m venv venv --clear
    source venv/bin/activate
    
    pip install --upgrade pip
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
    pip install mmpose fastapi uvicorn opencv-python
    
    # Add MMPose model setup here if needed
    # For example: python -c "import mmpose; mmpose.download_model('hrnet_w48')"
    
    python -c "import torch, mmcv; print(f'✅ Legacy env: torch {torch.__version__}, mmcv {mmcv.__version__}')"
}

create_specialized_env() {
    echo "Setting up YOLO-NAS specialized environment..."
    cd /opt/padel/envs/specialized
    python3 -m venv yolo-nas-venv --clear
    source yolo-nas-venv/bin/activate
    
    pip install --upgrade pip
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    pip install super-gradients fastapi uvicorn
    
    # Add YOLO-NAS model setup here if needed
    # For example: python -c "from super_gradients.training import models; models.get('yolo_nas_l', pretrained_weights='coco')"
    
    python -c "import torch, super_gradients; print(f'✅ Specialized env: torch {torch.__version__}')"
}

create_model_tracking() {
    echo "Creating model version tracking file..."
    cat > /opt/padel/models/model_versions.txt << EOL
yolo11n-pose.pt:v1.0
yolo11n.pt:v1.0
yolov8n.pt:v1.0
mmpose:v1.0
yolo-nas:v1.0
EOL
}

# Run all setups
create_modern_env
create_legacy_env  
create_specialized_env
create_model_tracking

# Set permissions
chown -R Towers:Towers /opt/padel/envs
chown -R Towers:Towers /opt/padel/models

echo "🎉 All environments configured successfully!"
echo "Model versions:"
cat /opt/padel/models/model_versions.txt