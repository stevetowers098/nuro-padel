# NuroPadel Resilient Deployment Guide

## Problem Solved

Previously, if **any** service failed to build, **none** of the services would deploy. This meant that even if YOLO-Combined was working perfectly, a failure in MMPose or YOLO-NAS would prevent any testing.

## Solution: Independent Service Deployment

The resilient deployment system allows working services to deploy and be tested independently, while failed services are gracefully handled.

## Quick Start

### Deploy All Services (Recommended)
```bash
./deploy-resilient.sh all
```
This will:
- Deploy all services that build successfully
- Continue even if some services fail
- Show you which services are working
- Provide access to working services immediately

### Deploy Individual Services
```bash
# Deploy only YOLO Combined (known working)
./deploy-resilient.sh yolo-combined

# Deploy only MMPose 
./deploy-resilient.sh mmpose

# Deploy only YOLO-NAS
./deploy-resilient.sh yolo-nas
```

### Check Service Status
```bash
./deploy-resilient.sh status
```

### Stop All Services
```bash
./deploy-resilient.sh stop
```

## How It Works

### 1. Docker Compose Profiles
Each service has its own profile in [`docker-compose.resilient.yml`](docker-compose.resilient.yml):
- `yolo-combined` profile
- `mmpose` profile  
- `yolo-nas` profile
- `nginx` profile
- `all` profile (includes everything)

### 2. Graceful Failure Handling
The resilient nginx configuration ([`nginx.resilient.conf`](nginx.resilient.conf)) handles missing services gracefully:
- Returns helpful error messages when services are unavailable
- Suggests alternative services
- Maintains API structure even with partial deployments

### 3. Health Check Integration
Each service endpoint provides clear status:
- `/healthz` - Global health (shows which services are available)
- `/yolo-combined/healthz` - YOLO Combined status
- `/mmpose/healthz` - MMPose status
- `/yolo-nas/healthz` - YOLO-NAS status

## Example Scenarios

### Scenario 1: Only YOLO-Combined Works
```bash
$ ./deploy-resilient.sh all

[INFO] Starting resilient deployment of all services...
[SUCCESS] Nginx deployed successfully
[SUCCESS] yolo-combined is running and healthy!
[ERROR] Failed to deploy mmpose
[ERROR] Failed to deploy yolo-nas
[SUCCESS] 1 out of 3 AI services deployed successfully
[SUCCESS] At least one service is available for testing!

Checking service status...
  ✓ yolo-combined (port 8001) - HEALTHY
  ✗ mmpose (port 8003) - DOWN  
  ✗ yolo-nas (port 8004) - DOWN
  ✓ nginx (port 8080) - HEALTHY
```

**Result**: You can immediately test YOLO-Combined at `http://localhost:8080/yolo11/` or `http://localhost:8080/yolov8/`

### Scenario 2: All Services Work
```bash
$ ./deploy-resilient.sh all

[SUCCESS] 3 out of 3 AI services deployed successfully

Checking service status...
  ✓ yolo-combined (port 8001) - HEALTHY
  ✓ mmpose (port 8003) - HEALTHY
  ✓ yolo-nas (port 8004) - HEALTHY
  ✓ nginx (port 8080) - HEALTHY
```

**Result**: Full API available with all endpoints

## API Endpoints (Resilient Mode)

Even with partial deployments, the API structure remains consistent:

### Service Discovery
```bash
curl http://localhost:8080/
```
Returns available services and their endpoints.

### Health Checks
```bash
# Global health
curl http://localhost:8080/healthz

# Individual services  
curl http://localhost:8080/yolo-combined/healthz
curl http://localhost:8080/mmpose/healthz
curl http://localhost:8080/yolo-nas/healthz
```

### Service Endpoints (when available)
```bash
# YOLO Combined
curl -X POST http://localhost:8080/yolo11/pose -F "file=@video.mp4"
curl -X POST http://localhost:8080/yolov8/object -F "file=@video.mp4"

# MMPose  
curl -X POST http://localhost:8080/mmpose/pose -F "file=@video.mp4"

# YOLO-NAS
curl -X POST http://localhost:8080/yolo-nas/pose -F "file=@video.mp4"
```

## Error Handling

When a service is unavailable, you get helpful error messages:

```json
{
  "error": "MMPose service unavailable",
  "service": "mmpose", 
  "alternatives": ["yolo-combined", "yolo-nas"],
  "timestamp": "2025-05-28T21:41:00Z"
}
```

## Migration from Original Setup

### Old Way (All-or-Nothing)
```bash
docker-compose up -d --build  # Fails if any service fails
```

### New Way (Resilient)
```bash
./deploy-resilient.sh all     # Deploys what works, continues testing
```

## Files Created

1. **[`docker-compose.resilient.yml`](docker-compose.resilient.yml)** - Profile-based service definitions
2. **[`nginx.resilient.conf`](nginx.resilient.conf)** - Graceful failure handling
3. **[`deploy-resilient.sh`](deploy-resilient.sh)** - Deployment automation script
4. **[`RESILIENT_DEPLOYMENT.md`](RESILIENT_DEPLOYMENT.md)** - This documentation

## Benefits

✅ **Faster Testing**: Deploy and test working services immediately  
✅ **Independent Development**: Services can be fixed and deployed separately  
✅ **Better Debugging**: Clear visibility into which services are working  
✅ **Graceful Degradation**: API remains functional with partial deployments  
✅ **Production Ready**: Handles service failures in production gracefully  

This allows you to push working models through to the VM for testing even when other services are still being debugged!