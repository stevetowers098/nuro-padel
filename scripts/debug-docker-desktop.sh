#!/bin/bash
# Docker Desktop Debug & Test Script for NuroPadel Services
# Tests deployment status and provides Docker Desktop integration

echo "🐳 NURO-PADEL DOCKER DESKTOP DEBUG & TEST"
echo "=========================================="
echo "Timestamp: $(date)"
echo ""

# Change to deployment directory
cd "$(dirname "$0")/../deployment" || exit 1

echo "📍 Current Directory: $(pwd)"
echo "🔍 Testing from: $(basename "$(pwd)")"
echo ""

echo "🚨 PHASE 1: PRE-DEPLOYMENT VALIDATION"
echo "====================================="

# Check if Docker is running
echo "🐳 Docker Status:"
if ! docker info > /dev/null 2>&1; then
    echo "❌ CRITICAL: Docker is not running!"
    echo "   👉 Start Docker Desktop and try again"
    exit 1
else
    echo "✅ Docker is running"
fi

# Check for weights directory
echo ""
echo "📦 Weights Directory Check:"
if [ ! -d "../weights" ]; then
    echo "❌ CRITICAL: weights/ directory not found!"
    echo "   👉 Run download-models.sh first"
    echo "   👉 Expected path: $(pwd)/../weights"
    exit 1
else
    echo "✅ Weights directory exists"
    echo "   📊 Weight files: $(find ../weights -name "*.pt" -o -name "*.pth" -o -name "*.onnx" | wc -l) found"
fi

# Check GPU availability
echo ""
echo "⚡ GPU Availability:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers detected"
    echo "   📊 GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB total"
    echo "   📊 GPU Usage: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MB used"
else
    echo "❌ WARNING: nvidia-smi not found - GPU acceleration may not work"
fi

echo ""
echo "🚨 PHASE 2: DEPLOYMENT TESTING"
echo "=============================="

# Start deployment with better logging
echo "🚀 Starting Docker Compose deployment..."
echo "   👉 You can monitor this in Docker Desktop under 'Containers'"

# Check if already running
if docker-compose ps | grep -q "Up"; then
    echo "⚠️  Services already running - stopping first..."
    docker-compose down
    sleep 3
fi

# Start with explicit logging
docker-compose up -d

echo ""
echo "⏱️  Waiting for services to initialize (60 seconds)..."
echo "   👉 Monitor startup progress in Docker Desktop logs"

# Progress indicator
for i in {1..12}; do
    sleep 5
    echo -n "."
done
echo " Done!"

echo ""
echo "🚨 PHASE 3: SERVICE HEALTH VALIDATION"
echo "===================================="

# Service status check
echo "📊 Container Status:"
docker-compose ps

echo ""
echo "🔍 Individual Service Tests:"

# Test each service
services=("yolo-combined:8001" "mmpose:8003" "yolo-nas:8004" "rf-detr:8005" "vitpose:8006")
healthy_services=0
total_services=${#services[@]}

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    echo ""
    echo "🧪 Testing $name (port $port):"
    
    # Check if container is running
    if docker-compose ps | grep -q "$name.*Up"; then
        echo "   ✅ Container: Running"
        
        # Test health endpoint
        if curl -s -f "http://localhost:$port/healthz" > /dev/null 2>&1; then
            echo "   ✅ Health Check: PASSED"
            ((healthy_services++))
        else
            echo "   ❌ Health Check: FAILED"
            echo "   🔍 Last 10 log lines:"
            docker-compose logs --tail=10 "$name" | sed 's/^/      /'
        fi
    else
        echo "   ❌ Container: NOT RUNNING"
        echo "   🔍 Last 20 log lines:"
        docker-compose logs --tail=20 "$name" | sed 's/^/      /'
    fi
done

echo ""
echo "🚨 PHASE 4: CRITICAL ISSUE VALIDATION"
echo "===================================="

# Run existing validation if YOLO-NAS is up
if docker-compose ps | grep -q "yolo-nas.*Up"; then
    echo "🧪 Running YOLO-NAS Stability Test:"
    
    # Test super-gradients import
    docker-compose exec -T yolo-nas python -c "
import sys
try:
    from super_gradients.training import models
    print('✅ super-gradients import successful')
    
    # Quick model test
    model = models.get('yolo_nas_pose_n', pretrained_weights=None)
    print('✅ YOLO-NAS model creation successful')
    
except Exception as e:
    print(f'❌ YOLO-NAS CRITICAL FAILURE: {e}')
    sys.exit(1)
" 2>&1
    
    yolo_nas_status=$?
else
    echo "❌ YOLO-NAS container not running - skipping detailed test"
    yolo_nas_status=1
fi

echo ""
echo "⚡ GPU Resource Analysis:"
if command -v nvidia-smi &> /dev/null; then
    echo "   📊 Current GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf "      Used: %s MB / %s MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    
    echo "   📊 GPU Processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
        sed 's/^/      /' || echo "      No GPU processes found"
else
    echo "   ❌ Cannot analyze GPU usage - nvidia-smi not available"
fi

echo ""
echo "🚨 PHASE 5: DOCKER DESKTOP INTEGRATION"
echo "======================================"

echo "🐳 Docker Desktop URLs:"
echo "   👉 Containers: docker-desktop://dashboard/containers"
echo "   👉 Images: docker-desktop://dashboard/images"
echo "   👉 Volumes: docker-desktop://dashboard/volumes"

echo ""
echo "📋 Quick Docker Desktop Commands:"
echo "   • View all containers: docker-compose ps"
echo "   • View logs: docker-compose logs [service-name]"
echo "   • Restart service: docker-compose restart [service-name]"
echo "   • Stop all: docker-compose down"
echo "   • Rebuild: docker-compose up --build"

echo ""
echo "📊 FINAL SUMMARY"
echo "================"
echo "✅ Healthy Services: $healthy_services/$total_services"

if [ $healthy_services -eq $total_services ]; then
    echo "🎉 SUCCESS: All services are running and healthy!"
    echo ""
    echo "🌐 Service URLs:"
    echo "   • YOLO Combined: http://localhost:8001/healthz"
    echo "   • MMPose: http://localhost:8003/healthz"
    echo "   • YOLO-NAS: http://localhost:8004/healthz"
    echo "   • RF-DETR: http://localhost:8005/healthz"
    echo "   • ViTPose: http://localhost:8006/healthz"
    echo "   • Nginx Proxy: http://localhost:8080"
    
elif [ $healthy_services -gt 0 ]; then
    echo "⚠️  PARTIAL SUCCESS: $healthy_services/$total_services services healthy"
    echo ""
    echo "🔧 DEBUGGING STEPS:"
    echo "   1. Check failing service logs in Docker Desktop"
    echo "   2. Verify GPU memory isn't exhausted"
    echo "   3. Try restarting individual services"
    
else
    echo "❌ DEPLOYMENT FAILED: No services are healthy"
    echo ""
    echo "🔧 CRITICAL DEBUGGING STEPS:"
    echo "   1. Check Docker Desktop for container errors"
    echo "   2. Verify weights directory: $(pwd)/../weights"
    echo "   3. Check GPU availability and memory"
    echo "   4. Review service logs: docker-compose logs"
fi

if [ $yolo_nas_status -ne 0 ]; then
    echo ""
    echo "⚠️  YOLO-NAS ISSUE DETECTED:"
    echo "   • This service has known stability issues"
    echo "   • Consider disabling it if others work fine"
    echo "   • Command: docker-compose stop yolo-nas"
fi

echo ""
echo "🔍 For detailed debugging, check Docker Desktop logs for each container"
echo "📝 Next: Use Docker Desktop GUI to inspect individual container health"