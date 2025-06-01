#!/bin/bash

SERVICE_NAME="mmpose"
SERVICE_PORT="8003"
IMAGE_NAME="ghcr.io/stevetowers098/nuro-padel/${SERVICE_NAME}"

echo "🚀 Deploying ${SERVICE_NAME} service independently..."

# Build and push image
echo "📦 Building ${SERVICE_NAME} image..."
docker build -t ${IMAGE_NAME}:latest .

echo "📤 Pushing ${SERVICE_NAME} image..."
docker push ${IMAGE_NAME}:latest

# Deploy on VM via SSH  
echo "🌐 Deploying to remote server..."
ssh padel-ai << EOF
    cd /opt/padel-docker
    
    # Pull latest image
    echo "📥 Pulling latest ${SERVICE_NAME} image..."
    docker pull ${IMAGE_NAME}:latest
    
    # Stop only this service (others keep running)
    echo "🛑 Stopping existing ${SERVICE_NAME} container..."
    docker stop nuro-padel-${SERVICE_NAME} 2>/dev/null || true
    docker rm nuro-padel-${SERVICE_NAME} 2>/dev/null || true
    
    # Start new container
    echo "▶️ Starting new ${SERVICE_NAME} container..."
    docker run -d \
        --name nuro-padel-${SERVICE_NAME} \
        --restart unless-stopped \
        -p ${SERVICE_PORT}:8000 \
        --gpus all \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e PYTHONUNBUFFERED=1 \
        -e YOLO_OFFLINE=1 \
        -e ULTRALYTICS_OFFLINE=1 \
        -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        -v /opt/padel-docker/weights:/app/weights:ro \
        -v /opt/padel-docker/configs:/app/configs:ro \
        ${IMAGE_NAME}:latest
    
    # Health check
    echo "⏳ Waiting for ${SERVICE_NAME} to be healthy..."
    for i in {1..30}; do
        if curl -sf http://localhost:${SERVICE_PORT}/healthz > /dev/null; then
            echo "✅ ${SERVICE_NAME} deployed successfully!"
            echo "🔗 Service available at http://35.189.53.46:${SERVICE_PORT}"
            exit 0
        fi
        sleep 2
    done
    
    echo "❌ ${SERVICE_NAME} deployment failed!"
    docker logs nuro-padel-${SERVICE_NAME} --tail 20
    exit 1
EOF

if [ $? -eq 0 ]; then
    echo "🎉 ${SERVICE_NAME} deployment completed successfully!"
else
    echo "💥 ${SERVICE_NAME} deployment failed!"
    exit 1
fi