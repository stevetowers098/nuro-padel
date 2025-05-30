#!/bin/bash
# Weight Mount Fix Validation Script
# Tests the 2 most critical fixes applied

echo "🔍 VALIDATING WEIGHT MOUNT FIXES"
echo "================================="

echo ""
echo "📋 DIAGNOSIS VALIDATION:"
echo "1. ❌ Previous Issue: All services mounted entire '../weights' directory"
echo "2. ❌ Previous Issue: YOLO-NAS user '1000:1000' couldn't access /home/appuser"
echo ""
echo "🔧 APPLIED FIXES:"
echo "1. ✅ Service-specific weight mounts:"
echo "   - yolo-combined: ../weights/ultralytics:/app/weights"
echo "   - mmpose: ../weights/mmpose:/app/weights" 
echo "   - yolo-nas: ../weights/ultralytics:/app/weights/ultralytics"
echo "   - rf-detr: ../weights/rf-detr:/app/weights"
echo "   - vitpose: ../weights/vitpose:/app/weights"
echo "2. ✅ YOLO-NAS user changed from '1000:1000' to 'root'"
echo ""

echo "🧪 TESTING WEIGHT DIRECTORY STRUCTURE..."
echo "----------------------------------------"

# Check if weight subdirectories exist
echo "Required weight directories:"
for dir in "ultralytics" "mmpose" "rf-detr" "vitpose"; do
    if [ -d "../weights/$dir" ]; then
        echo "✅ ../weights/$dir exists"
        echo "   Contents: $(ls -la ../weights/$dir 2>/dev/null | wc -l) files"
    else
        echo "❌ ../weights/$dir MISSING - Service will fail to start"
    fi
done

echo ""
echo "🐳 TESTING DOCKER COMPOSE CHANGES..."
echo "------------------------------------"

# Validate docker-compose syntax
if docker-compose -f deployment/docker-compose.yml config >/dev/null 2>&1; then
    echo "✅ Docker-compose syntax valid"
else
    echo "❌ Docker-compose syntax ERROR:"
    docker-compose -f deployment/docker-compose.yml config
    exit 1
fi

# Check specific mount configurations
echo ""
echo "📂 Mount Configuration Validation:"
docker-compose -f deployment/docker-compose.yml config | grep -A 10 "volumes:" | grep -E "(ultralytics|mmpose|rf-detr|vitpose)" && echo "✅ Service-specific mounts configured" || echo "❌ Mount validation failed"

echo ""
echo "👤 YOLO-NAS User Configuration:"
docker-compose -f deployment/docker-compose.yml config | grep -A 5 "yolo-nas:" | grep 'user: "root"' && echo "✅ YOLO-NAS user set to root" || echo "❌ YOLO-NAS user not configured as root"

echo ""
echo "🎯 PREDICTION: After restart..."
echo "✅ yolo-combined will find ultralytics models"
echo "✅ mmpose will find mmpose-specific models" 
echo "✅ yolo-nas will run as root (no /home/appuser crash)"
echo "✅ rf-detr will find rf-detr models"
echo "✅ vitpose will find vitpose models"

echo ""
echo "⚠️  REQUIRED NEXT STEPS:"
echo "1. Ensure weight directories exist with proper model files"
echo "2. Run: docker-compose down && docker-compose up -d"
echo "3. Monitor service health: docker-compose ps"
echo "4. Test endpoints: curl http://localhost:8001/healthz (and 8003,8004,8005,8006)"