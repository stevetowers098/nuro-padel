#!/bin/bash
# setup_models.sh
# One-time setup script for downloading and configuring models
# This should be run during VM initialization, not during code deployment

set -e  # Exit on any error

echo "Starting model setup..."

# Create necessary directories
mkdir -p /opt/padel/models
cd /opt/padel

# Activate virtual environment
source shared/venv/bin/activate

echo "Downloading and caching YOLO models..."

# Download YOLO models to their default cache location
# This ensures they're available to the services
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" 
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create a model version tracking file
echo "Creating model version tracking file..."
cat > /opt/padel/models/model_versions.txt << EOL
yolo11n-pose.pt:v1.0
yolo11n.pt:v1.0
yolov8n.pt:v1.0
EOL

echo "Setting up MMPose models..."
# Add MMPose model setup here if needed

echo "Setting up YOLO-NAS models..."
# Add YOLO-NAS model setup here if needed

echo "Setting permissions..."
chown -R padel:padel /opt/padel/models

echo "Model setup completed successfully!"
echo "Models are now cached and ready for use by the services."
echo "Model versions:"
cat /opt/padel/models/model_versions.txt