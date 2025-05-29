# Docker Container Fixes Applied - Resilient Deployment

## ✅ Issues Fixed

### 1. Permission Errors Fixed
**Problem**: Containers failing with `/home/appuser` permission errors
**Solution**: Added environment variables to `docker-compose.resilient.yml`:
```yaml
environment:
  # Fix permission/directory issues
  - HOME=/tmp
  - MPLCONFIGDIR=/tmp/matplotlib  
  - PYTHONPATH=/app
```

### 2. Configuration Updated
**File**: `docker-compose.resilient.yml`
**Services Updated**: All three services (yolo-combined, mmpose, yolo-nas)

## 📋 Backup Files Created
- `mmpose-service/Dockerfile.backup` - Original Dockerfile backed up

## 🚀 Deployment Commands for VM

### Step 1: Stop Current Containers
```bash
sudo docker-compose -f docker-compose.resilient.yml down
```

### Step 2: Clean Up (Optional - only if needed)
```bash
sudo docker container prune -f
```

### Step 3: Deploy Individual Services (Resilient Approach)
```bash
# Start YOLO Combined first (this was working)
sudo docker-compose -f docker-compose.resilient.yml --profile yolo-combined up -d

# Test YOLO Combined
curl http://localhost:8001/healthz

# Start MMPose with fixes
sudo docker-compose -f docker-compose.resilient.yml --profile mmpose up -d

# Test MMPose
curl http://localhost:8003/healthz

# Start YOLO-NAS with fixes  
sudo docker-compose -f docker-compose.resilient.yml --profile yolo-nas up -d

# Test YOLO-NAS
curl http://localhost:8004/healthz

# Finally start Nginx
sudo docker-compose -f docker-compose.resilient.yml --profile nginx up -d
```

### Step 4: Check All Services
```bash
sudo docker ps
sudo docker-compose -f docker-compose.resilient.yml logs
```

## 🔍 Monitoring Commands
```bash
# Check container status
sudo docker ps -a

# Check logs for specific service
sudo docker-compose -f docker-compose.resilient.yml logs mmpose
sudo docker-compose -f docker-compose.resilient.yml logs yolo-nas

# Check health status
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose  
curl http://localhost:8004/healthz  # YOLO-NAS
curl http://localhost:8080/         # Nginx
```

## 🎯 Expected Results
- All services should start without `/home/appuser` permission errors
- MMPose should resolve MMCV version conflicts using environment fixes
- Nginx should be able to reach all backend services on ports 8001, 8003, 8004

## 🛡️ Rollback Commands (If Needed)
```bash
# Stop everything
sudo docker-compose -f docker-compose.resilient.yml down

# Restore original Dockerfile if needed
cp mmpose-service/Dockerfile.backup mmpose-service/Dockerfile

# Use git to restore docker-compose.resilient.yml if needed
git checkout docker-compose.resilient.yml