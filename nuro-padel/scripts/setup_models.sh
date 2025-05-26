#!/bin/bash
# setup_models.sh
# One-time setup script for downloading and configuring models
# This should be run during VM initialization, not during code deployment

set -e  # Exit on any error

echo "Starting model setup..."

# Create necessary directories
mkdir -p /opt/padel/models
cd /opt/padel

# Activate YOLO virtual environment for YOLO models
source yolo/venv/bin/activate

echo "Downloading and caching YOLO models..."

# Download YOLO models to their default cache location
# This ensures they're available to the services
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create model version tracking file
echo "Creating model version tracking file..."
mkdir -p /opt/padel/models
cat > /opt/padel/models/model_versions.txt << EOL
yolo11n-pose.pt:v1.0
yolo11n.pt:v1.0
yolov8n.pt:v1.0
EOL

echo "Setting up MMPose models..."
# Activate MMPose virtual environment
deactivate  # Deactivate YOLO venv first
source /opt/padel/mmpose/venv/bin/activate
# Add MMPose model setup here if needed
# For example:
# python -c "import mmpose; print('MMPose ready')"

echo "Setting up YOLO-NAS models..."
# Activate YOLO-NAS virtual environment
deactivate  # Deactivate MMPose venv first
source /opt/padel/yolo-nas/venv/bin/activate
# Add YOLO-NAS model setup here if needed
# For example:
# python -c "import super_gradients; print('YOLO-NAS ready')"

echo "Setting permissions..."
chown -R Towers:Towers /opt/padel/models

echo "Model setup completed successfully!"
echo "Models are now cached and ready for use by the services."
echo "Model versions:"
cat /opt/padel/models/model_versions.txt