#!/bin/bash
# Deployment Validation Script
# Validates the 2 most critical potential issues

echo "🔍 DEPLOYMENT VALIDATION - Testing Critical Issues"
echo "================================================="

echo ""
echo "🚨 ISSUE #1: YOLO-NAS Service Stability Check"
echo "--------------------------------------------"
echo "Testing super-gradients import and model loading..."

# Test 1: Check if super-gradients can import and load models
docker-compose exec yolo-nas python -c "
import sys
try:
    from super_gradients.training import models
    print('✅ super-gradients import successful')
    
    # Test model loading
    model = models.get('yolo_nas_pose_n', pretrained_weights=None)
    print('✅ YOLO-NAS model creation successful')
    
    # Test if the model can actually run inference
    import torch
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model(dummy_input)
    print('✅ YOLO-NAS inference test successful')
    
except Exception as e:
    print(f'❌ YOLO-NAS CRITICAL FAILURE: {e}')
    sys.exit(1)
" 2>&1

YOLO_NAS_STATUS=$?

echo ""
echo "⚡ ISSUE #2: GPU Resource Contention Check"
echo "----------------------------------------"
echo "Testing GPU allocation across all services..."

# Test 2: Check GPU memory usage and allocation
echo "GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "Docker GPU Allocation:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(yolo|mmpose)"

echo ""
echo "Service GPU Health Checks:"
curl -s http://localhost:8001/healthz | jq -r '.gpu_status // "unknown"' 2>/dev/null || echo "❌ YOLO-Combined unreachable"
curl -s http://localhost:8003/healthz | jq -r '.gpu_status // "unknown"' 2>/dev/null || echo "❌ MMPose unreachable"  
curl -s http://localhost:8004/healthz | jq -r '.gpu_status // "unknown"' 2>/dev/null || echo "❌ YOLO-NAS unreachable"

echo ""
echo "📊 VALIDATION SUMMARY"
echo "===================="
if [ $YOLO_NAS_STATUS -eq 0 ]; then
    echo "✅ YOLO-NAS Service: STABLE"
else
    echo "❌ YOLO-NAS Service: CRITICAL ISSUE CONFIRMED"
fi

echo "⚡ GPU Resource Status: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits) GPU(s) available"
echo ""
echo "🔧 NEXT STEPS:"
echo "- If YOLO-NAS fails: Consider replacing with alternative pose detection"
echo "- If GPU contention detected: Implement GPU sharing or sequential deployment"