# 🚀 Nuro-Padel Upgrade Guide

## Overview

The Nuro-Padel AI system now includes sophisticated upgrade capabilities that allow you to:
- Switch model versions without code changes
- Toggle features dynamically via configuration
- Monitor upgrade readiness across all services
- Override settings with environment variables

## 📋 Quick Reference

### Health Check Endpoints
```bash
# Check all services with enhanced information
curl http://localhost:8001/healthz | jq  # YOLO Combined
curl http://localhost:8003/healthz | jq  # MMPose  
curl http://localhost:8004/healthz | jq  # YOLO-NAS

# Check specific service status
curl http://localhost:8001/healthz | jq '.status, .models, .features'
```

### Demo Script
```bash
chmod +x scripts/demo-upgrade-features.sh
./scripts/demo-upgrade-features.sh
```

## 🔧 Model Configuration

### Configuration Files
Each service has a `config/model_config.json` file:
- `services/yolo-combined/config/model_config.json`
- `services/mmpose/config/model_config.json` 
- `services/yolo-nas/config/model_config.json`

### Example Configuration Structure
```json
{
  "service": "yolo_combined",
  "version": "2.0.0",
  "models": {
    "yolo11_pose": {
      "file": "yolo11n-pose.pt",
      "version": "v1.0.0",
      "enabled": true,
      "fallback": "yolo8n-pose.pt"
    }
  },
  "features": {
    "enhanced_ball_tracking": {
      "enabled": true,
      "description": "Advanced Kalman filtering for ball trajectory"
    },
    "tracknet_v4": {
      "enabled": false,
      "description": "Next generation TrackNet model (ready for testing)"
    }
  },
  "performance": {
    "batch_size": 8,
    "confidence_threshold": 0.3,
    "max_concurrent_requests": 5
  }
}
```

## 🎛️ Feature Flag Management

### Toggle Features via Configuration
```bash
# Enable a feature
jq '.features.tracknet_v4.enabled = true' services/yolo-combined/config/model_config.json > temp.json && mv temp.json services/yolo-combined/config/model_config.json

# Disable a feature
jq '.features.enhanced_ball_tracking.enabled = false' services/yolo-combined/config/model_config.json > temp.json && mv temp.json services/yolo-combined/config/model_config.json

# Check if feature is enabled
curl http://localhost:8001/healthz | jq '.features.tracknet_v4.enabled'
```

### Environment Variable Overrides
```bash
# Override feature flags
export FEATURE_ENHANCED_BALL_TRACKING_ENABLED=false
export FEATURE_TRACKNET_V4_ENABLED=true

# Override model settings
export YOLO11_POSE_ENABLED=true
export CONFIDENCE_THRESHOLD=0.5

# Restart service to apply environment overrides
docker-compose restart yolo-combined
```

## 📊 Enhanced Health Checks

### Health Check Response Format
```json
{
  "status": "healthy",
  "service": {
    "service": "yolo_combined",
    "version": "2.0.0",
    "config_loaded": true
  },
  "models": {
    "yolo11_pose": {
      "loaded": true,
      "enabled": true,
      "version": "v1.0.0",
      "file": "yolo11n-pose.pt",
      "fallback": "yolo8n-pose.pt"
    },
    "yolo11_object": {
      "loaded": true,
      "enabled": true,
      "version": "v1.0.0",
      "file": "yolo11n.pt",
      "fallback": "yolov8n.pt"
    }
  },
  "features": {
    "enhanced_ball_tracking": {
      "enabled": true,
      "description": "Advanced Kalman filtering for ball trajectory"
    },
    "tracknet_v4": {
      "enabled": false,
      "description": "Next generation TrackNet model (ready for testing)"
    }
  },
  "performance": {
    "batch_size": 8,
    "confidence_threshold": 0.3,
    "max_concurrent_requests": 5
  },
  "deployment": {
    "ready_for_upgrade": true,
    "config_hot_reload": true,
    "environment_overrides": false,
    "last_config_check": "2025-05-30T08:45:00Z"
  }
}
```

## 🔄 Upgrade Workflows

### 1. Model Version Upgrade
```bash
# Step 1: Update model configuration
jq '.models.yolo11_pose.file = "yolo11s-pose.pt" | .models.yolo11_pose.version = "v1.1.0"' services/yolo-combined/config/model_config.json > temp.json && mv temp.json services/yolo-combined/config/model_config.json

# Step 2: Check health endpoint for new model info
curl http://localhost:8001/healthz | jq '.models.yolo11_pose'

# Step 3: Test new model (service auto-reloads config)
curl -X POST http://localhost:8001/yolo11/pose -d '{"video_url": "https://example.com/test.mp4"}'
```

### 2. Feature Rollout
```bash
# Step 1: Enable feature gradually
jq '.features.tracknet_v4.enabled = true' services/yolo-combined/config/model_config.json > temp.json && mv temp.json services/yolo-combined/config/model_config.json

# Step 2: Monitor feature status
curl http://localhost:8001/healthz | jq '.features.tracknet_v4'

# Step 3: Rollback if needed
jq '.features.tracknet_v4.enabled = false' services/yolo-combined/config/model_config.json > temp.json && mv temp.json services/yolo-combined/config/model_config.json
```

### 3. Emergency Rollback
```bash
# Using environment variables for immediate rollback
export YOLO11_POSE_ENABLED=false
export YOLOV8_POSE_ENABLED=true
docker-compose restart yolo-combined

# Verify rollback
curl http://localhost:8001/healthz | jq '.deployment.environment_overrides'
```

## 🛠️ Advanced Configuration

### Service-Specific Features

#### YOLO Combined Service
- `enhanced_ball_tracking` - Advanced Kalman filtering
- `tracknet_v4` - Next generation TrackNet model
- `batch_processing` - Process multiple frames in batches
- `half_precision` - Use FP16 for faster inference

#### MMPose Service  
- `biomechanical_analysis` - Calculate joint angles and movement metrics
- `posture_scoring` - Automated posture quality assessment
- `3d_pose_estimation` - Experimental 3D pose reconstruction
- `real_time_feedback` - Live coaching feedback during analysis

#### YOLO-NAS Service
- `high_accuracy_mode` - Use YOLO-NAS for maximum accuracy
- `onnx_optimization` - Use ONNX runtime for faster inference  
- `tensorrt_optimization` - Use TensorRT for maximum speed
- `model_ensembling` - Combine multiple models for better accuracy

## 🚨 Troubleshooting

### Common Issues

#### Config Not Loading
```bash
# Check if config file exists and is valid JSON
cat services/yolo-combined/config/model_config.json | jq .

# Check service logs for config errors
docker-compose logs yolo-combined | grep -i config
```

#### Feature Not Taking Effect
```bash
# Check if config hot-reloaded
curl http://localhost:8001/healthz | jq '.deployment.last_config_check'

# Force config reload by restarting service
docker-compose restart yolo-combined
```

#### Environment Override Not Working
```bash
# Check if environment variable is set correctly
docker-compose exec yolo-combined env | grep FEATURE_

# Verify override is detected in health check
curl http://localhost:8001/healthz | jq '.deployment.environment_overrides'
```

## 📈 Monitoring & Validation

### Upgrade Readiness Check
```bash
# Check all services are ready for upgrade
curl -s http://localhost:8001/healthz | jq '.deployment.ready_for_upgrade'
curl -s http://localhost:8003/healthz | jq '.deployment.ready_for_upgrade'  
curl -s http://localhost:8004/healthz | jq '.deployment.ready_for_upgrade'

# Get upgrade summary
./scripts/demo-upgrade-features.sh
```

### Performance Monitoring
```bash
# Check current performance settings
curl http://localhost:8001/healthz | jq '.performance'
curl http://localhost:8003/healthz | jq '.performance'
curl http://localhost:8004/healthz | jq '.performance'
```

## 🎯 Next Steps

The current implementation provides the foundation for more advanced upgrade strategies:

1. **Blue/Green Deployments** - Deploy new versions alongside old ones
2. **Canary Releases** - Gradually shift traffic to new versions
3. **A/B Testing** - Compare model performance across versions
4. **Automated Rollbacks** - Detect issues and automatically revert

These MVP features make all future upgrade enhancements much easier to implement while providing immediate value for managing model versions and feature rollouts.