#!/bin/bash
set -euo pipefail
trap 'echo "❌ Error on line $LINENO during YOLO11 deployment" >&2' ERR

# Dry-run mode
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 Dry run enabled: commands will be printed not executed"
fi

# Dry-run mode
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "🔍 Dry run enabled: commands will be printed not executed"
fi

SERVICE_NAME="yolo11"
SERVICE_PORT="8007"
IMAGE_NAME="ghcr.io/stevetowers098/nuro-padel/${SERVICE_NAME}"

echo "🚀 Deploying ${SERVICE_NAME} service independently..."
echo "🔍 Starting deployment script in $(pwd)"

# Build and push image
echo "📦 Building ${SERVICE_NAME} image..."
cd services/yolo11
docker build -t ${IMAGE_NAME}:latest -f Dockerfile .
cd ../..

:start_line:19
-------
echo "📤 Pushing ${SERVICE_NAME} image..."
PUSH_CMD="docker push ${IMAGE_NAME}:latest"
if $DRY_RUN; then
    echo "DRY: $PUSH_CMD"
else
    eval "$PUSH_CMD"
fi
# Authenticate to GitHub Container Registry before push
echo "🔑 Logging into GHCR..."
echo "${GHCR_TOKEN}" | docker login ghcr.io -u stevetowers098 --password-stdin || { echo "❌ GHCR login failed" >&2; exit 1; }
docker push ${IMAGE_NAME}:latest

# Deploy on VM via SSH
echo "🔑 SSH command to run: ssh padel-ai <remote deploy>"
:start_line:26
-------
echo "🌐 Deploying to remote server..."
SSH_CMD="ssh padel-ai << 'EOF'
    cd /opt/padel-docker
    echo \"📥 Pulling latest ${SERVICE_NAME} image...\"
    docker pull ${IMAGE_NAME}:latest
    echo \"🛑 Stopping existing ${SERVICE_NAME} container...\"
    docker stop nuro-padel-${SERVICE_NAME} 2>/dev/null || true
    docker rm nuro-padel-${SERVICE_NAME} 2>/dev/null || true
    echo \"▶️ Starting new ${SERVICE_NAME} container...\"
    docker run -d --name nuro-padel-${SERVICE_NAME} --restart unless-stopped -p ${SERVICE_PORT}:8007 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 -e YOLO_OFFLINE=1 -e ULTRALYTICS_OFFLINE=1 -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python -v /opt/padel-docker/weights:/app/weights:ro ${IMAGE_NAME}:latest
    echo \"⏳ Waiting for ${SERVICE_NAME} to be healthy...\"
    for i in {1..30}; do
        echo \"⏳ Health check attempt \$i...\"
        if curl -sf http://localhost:${SERVICE_PORT}/healthz > /dev/null; then
            echo \"✅ ${SERVICE_NAME} deployed successfully!\"
            exit 0
        fi
        sleep 2
    done
    echo \"❌ ${SERVICE_NAME} deployment failed!\"
    docker logs nuro-padel-${SERVICE_NAME} --tail 20
    exit 1
EOF"
if $DRY_RUN; then
    echo "DRY: $SSH_CMD"
else
    eval "$SSH_CMD"
fi
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
        -p ${SERVICE_PORT}:8007 \
        --gpus all \
        -e CUDA_VISIBLE_DEVICES=0 \
        -e PYTHONUNBUFFERED=1 \
        -e YOLO_OFFLINE=1 \
        -e ULTRALYTICS_OFFLINE=1 \
        -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        -v /opt/padel-docker/weights:/app/weights:ro \
        ${IMAGE_NAME}:latest
    
    # Health check
    echo "⏳ Waiting for ${SERVICE_NAME} to be healthy..."
    for i in {1..30}; do
         echo "⏳ Health check attempt $i..."
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