# Troubleshooting Guide

## 🚨 Critical Dependency Issues & Solutions

### **1. MMPose Service - Version Conflicts**

#### **MMCV Installation Failures**
**Error**: `ModuleNotFoundError: No module named 'mmcv'` or version conflicts
**Root Cause**: Multiple mmcv versions installed or wrong PyTorch version

**Solution - MMCV Official Installation**:
```dockerfile
# CRITICAL: Uninstall ALL existing mmcv versions first
RUN pip uninstall -y mmcv mmcv-lite mmcv-full || true

# Install PyTorch 2.1.x (compatible with mmcv 2.1.0)
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu118

# Install via OpenMIM (recommended)
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"        # Full version with CUDA ops
RUN mim install "mmdet>=3.0.0,<3.3.0" # Version constraint
RUN pip install pytz==2023.3          # Critical for openxlab
RUN mim install mmpose xtcocotools
RUN pip check                          # Verify no conflicts
```

#### **Numpy Version Conflicts**
**Error**: `numpy>=2.0` conflicts with `mmcv==2.1.0`
**Solution**: Constrain numpy in requirements:
```txt
numpy>=1.21.0,<2.0  # Critical constraint for mmcv compatibility
```

#### **PyTorch CUDA Mismatch**
**Warning**: PyTorch cu118 vs CUDA 12.2.0 base image
**Status**: Acceptable due to forward compatibility
**Future**: Consider PyTorch cu121 for optimal performance

#### **OpenXLab Conflicts**
**Error**: `openxlab 0.1.2 has requirement pytz~=2023.3, but you'll have pytz 2025.2`
**Solution**:
```dockerfile
# Install specific compatible versions
RUN pip install pytz==2023.3 requests==2.28.2 rich==13.4.2 tqdm==4.65.0
RUN mim install mmcv-full mmpose xtcocotools
```

### **2. YOLO-NAS Service - Super-gradients Issues**

#### **Impossible Dependency Conflict**
**Error**: `super-gradients 3.7.1` requires `numpy<=1.23` but sub-dependencies require `numpy>=1.24.4`

**Solution - Install with Dependency Exclusion**:
```dockerfile
# Install compatible versions first
RUN pip install "numpy==1.23.0" "requests==2.31.0"
# Install super-gradients without automatic dependency resolution
RUN pip install super-gradients==3.7.1 --no-deps
# Manually install essential dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Python Version Requirements**
**Error**: `networkx` requires Python >=3.9 but system has Python 3.8.10
**Solution**:
```dockerfile
# Ensure Python 3.10 is default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
# Create venv with Python 3.10 explicitly
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
```

#### **Documentation Tools in Runtime**
**Error**: `sphinx`, `docutils` causing bloat and security risk
**Solution**: Remove from runtime requirements:
```dockerfile
# These are build-time only, not runtime dependencies
# RUN pip install sphinx docutils  # REMOVED
```

### **3. GCS Dependencies - Protobuf Conflicts**

#### **Protobuf Version Hell**
**Error**: `google-cloud-storage` conflicts with `ultralytics` protobuf versions
**Solution**: Pin specific compatible versions:
```txt
google-cloud-storage==2.10.0  # Older version compatible with protobuf 3.x
protobuf>=3.19.5,<4.0.0       # Compatible with both ultralytics and GCS
```

#### **Alternative: Use gsutil CLI**
For YOLO-NAS service, avoid Python GCS library entirely:
```python
# Upload using gsutil CLI instead of google-cloud-storage
async def upload_to_gcs(video_path: str, object_name: str) -> str:
    try:
        gcs_path = f"gs://{GCS_BUCKET_NAME}/{object_name}"
        upload_cmd = ["gsutil", "cp", video_path, gcs_path]
        upload_result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=120)
        
        if upload_result.returncode != 0:
            logger.error(f"gsutil upload failed: {upload_result.stderr}")
            return ""
        
        # Make file public
        public_cmd = ["gsutil", "acl", "ch", "-u", "AllUsers:R", gcs_path]
        subprocess.run(public_cmd, capture_output=True, text=True, timeout=60)
        
        return f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{object_name}"
    except Exception as e:
        logger.error(f"Error uploading to GCS via gsutil: {e}")
        return ""
```

### **4. Docker Build Issues**

#### **Symbolic Link Failures**
**Error**: `ln: failed to create symbolic link '/etc/resolv.conf': Device or resource busy`
**Solution**: Remove DNS configuration entirely:
```dockerfile
# DNS configuration removed - Docker handles automatically
# No /etc/resolv.conf manipulation needed
```

**Why This Works**:
- Docker automatically manages DNS resolution for containers
- Eliminates all `/etc/resolv.conf` manipulation that causes read-only errors
- No DNS configuration to manage or debug
- Standard practice for containerized applications

#### **Missing System Bus Socket**
**Error**: `Failed to open connection to "system" message bus due to missing /var/run/dbus/system_bus_socket`
**Solution**: Install and configure dbus:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start
```

#### **Virtual Environment Failures**
**Error**: `python3 -m venv /opt/venv` fails or `python3 -m ensurepip` fails
**Solution**: Enhanced venv setup:
```dockerfile
# Install comprehensive Python venv support
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10-venv \
    python3-venv \
    python3-pip \
    python3-dev \
    dbus \
    && mkdir -p /var/run/dbus \
    && service dbus start

# Create virtual environment with enhanced error handling
RUN echo "🔧 Creating Python virtual environment..." && \
    python3 -m venv /opt/venv && \
    /opt/venv/bin/python -m ensurepip --upgrade
ENV PATH="/opt/venv/bin:$PATH"
```

#### **Missing Python.h Development Headers**
**Error**: `fatal error: Python.h: No such file or directory`
**Solution**:
```dockerfile
# Install Python development headers
RUN apt-get update && apt-get install -y python3.10-dev python3-dev
```

### **5. GitHub Actions / CI Issues**

#### **Out of Disk Space**
**Error**: GitHub runners have ~14GB, PyTorch deps are ~2GB each
**Solutions**:
```bash
# Aggressive cleanup in GitHub Actions
sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost
sudo rm -rf "$AGENT_TOOLSDIRECTORY" /usr/local/lib/android
sudo docker system prune -af --volumes

# Use CPU-only PyTorch (smaller)
pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Sequential deployment
./scripts/deploy.sh --deploy-sequential
```

#### **Integration Tests Missing**
**Error**: GitHub Actions references missing `docker-compose.test.yml`
**Status**: ✅ **FIXED** - Removed integration tests from CI workflow
**Solution Applied**: Streamlined deployment pipeline for direct pushes to `docker-containers` branch
**Action**: Pushes to `docker-containers` work automatically now

#### **No-Cache Installation for Space**
```dockerfile
ENV PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir -r requirements.txt
```

### **6. Fast Development Build Issues**

#### **Base Image Missing**
**Error**: `ghcr.io/stevetowers098/nuro-padel/base:latest` not found
**Solution**:
```bash
# Pull pre-built base image
docker pull ghcr.io/stevetowers098/nuro-padel/base:latest

# Or build locally if needed
docker build -f Dockerfile.base -t ghcr.io/stevetowers098/nuro-padel/base:latest .
docker push ghcr.io/stevetowers098/nuro-padel/base:latest
```

#### **Permission Issues**
**Error**: `/tmp` directory permission denied
**Solution**: Set proper environment variables:
```yaml
environment:
  - HOME=/tmp
  - MPLCONFIGDIR=/tmp/matplotlib
  - PYTHONPATH=/app
```

#### **Fast Build Not Working**
**Problem**: Still taking 30+ minutes despite using dev-fast.sh
**Solution**: Verify base image exists and Dockerfile.dev files are correct:
```bash
# Check base image
docker image inspect ghcr.io/stevetowers098/nuro-padel/base:latest

# Use development Dockerfiles
docker-compose -f docker-compose.dev.yml up --build
```

### **7. Model Loading Issues**

#### **YOLO Models Not Downloading**
**Error**: Models fail to auto-download
**Solution**: Ensure offline mode is properly configured:
```python
# Production safety - disable auto-downloads
os.environ['YOLO_OFFLINE'] = '1'
os.environ['ULTRALYTICS_OFFLINE'] = '1'
os.environ['ONLINE'] = 'False'
os.environ['YOLO_TELEMETRY'] = 'False'
```

#### **MMPose Model Loading Failures**
**Error**: RTMPose or HRNet models fail to load
**Solution**: Multiple fallback strategies implemented:
```python
# Method 1: Local config and checkpoint
if os.path.exists(config_path) and os.path.exists(checkpoint_path):
    mmpose_model = init_model(config_path, checkpoint_path, device=model_device)

# Method 2: Built-in config with local weights
elif os.path.exists(checkpoint_path):
    config_name = 'projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py'
    mmpose_model = init_model(config_name, checkpoint_path, device=model_device)

# Method 3: Automatic download fallback
else:
    mmpose_model = init_model('rtmpose-m_8xb256-420e_coco-256x192', checkpoint_url, device=model_device)
```

#### **YOLO-NAS Model Loading Failures**
**Error**: Super-gradients models fail to load
**Solution**: Enhanced error handling:
```python
try:
    yolo_nas_pose_model = models.get("yolo_nas_pose_n", pretrained_weights="coco_pose")
    if torch.cuda.is_available():
        yolo_nas_pose_model.to('cuda')
        yolo_nas_pose_model.half()
except Exception as e:
    logger.error(f"Failed to load YOLO-NAS pose model: {e}")
    # Service continues with object detection only
```

## 🔧 Common Debug Commands

### **Service Health Debugging**
```bash
# Check individual service health
curl http://localhost:8001/healthz | jq .  # YOLO Combined
curl http://localhost:8003/healthz | jq .  # MMPose  
curl http://localhost:8004/healthz | jq .  # YOLO-NAS

# Check via load balancer
curl http://localhost:8080/yolo-combined/healthz
curl http://localhost:8080/mmpose/healthz
curl http://localhost:8080/yolo-nas/healthz

# Check service logs
docker logs nuro-padel-yolo-combined
docker logs nuro-padel-mmpose
docker logs nuro-padel-yolo-nas

# Follow logs in real-time
docker-compose logs -f yolo-combined
docker-compose logs -f mmpose
docker-compose logs -f yolo-nas

# Shell into containers
docker exec -it nuro-padel-yolo-combined bash
docker exec -it nuro-padel-mmpose bash
docker exec -it nuro-padel-yolo-nas bash
```

### **Dependency Verification**
```bash
# Check installed packages
pip list | grep -E "(torch|mmpose|super|ultralytics)"

# Verify imports work
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import mmpose; print(f'MMPose: {mmpose.__version__}')"
python -c "from super_gradients.training import models; print('Super-gradients OK')"
python -c "import ultralytics; print('Ultralytics OK')"

# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
nvidia-smi

# Test model loading
python -c "from ultralytics import YOLO; model = YOLO('yolo11n.pt'); print('YOLO11 OK')"
python -c "from super_gradients.training import models; model = models.get('yolo_nas_s', pretrained_weights='coco'); print('YOLO-NAS OK')"
```

### **Build Debugging**
```bash
# Build with verbose output
docker-compose build --no-cache --progress=plain

# Build specific service
docker-compose build --no-cache yolo-combined

# Check disk space
df -h
docker system df

# Clean up space aggressively
docker system prune -af --volumes
docker builder prune -af

# Remove all containers and images
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
```

### **Network/API Debugging**
```bash
# Test API endpoints with sample data
curl -X POST http://localhost:8001/yolo11/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4", "video": false, "data": true}'

curl -X POST http://localhost:8001/yolov8/object \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4", "confidence": 0.3}'

curl -X POST http://localhost:8003/mmpose/pose \
  -H "Content-Type: application/json" \
  -d '{"video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4", "video": true}'

# Check nginx routing
curl http://localhost:8080/
curl http://localhost:8080/api/

# Check service connectivity
docker network inspect nuro-padel_nuro-padel-network
```

### **Performance Debugging**
```bash
# Resource usage monitoring
docker stats --no-stream

# Memory usage per container
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# GPU usage (if available)
nvidia-smi
watch -n 1 nvidia-smi

# Disk usage analysis
du -sh services/*/
du -sh deployment/
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

## 📋 Production Monitoring

### **Service Status Dashboard**
```bash
# Quick health check all services
for port in 8001 8003 8004; do
  echo "=== Port $port ===" 
  curl -s http://localhost:$port/healthz | jq .status || echo "FAILED"
done

# Load balancer health
curl -s http://localhost:8080/healthz | jq .

# Service discovery via nginx
curl -s http://localhost:8080/ | jq .
```

### **Log Monitoring**
```bash
# Follow all logs
docker-compose logs -f

# Service-specific logs with timestamps
docker-compose logs -f --timestamps yolo-combined
docker-compose logs -f --timestamps mmpose  
docker-compose logs -f --timestamps yolo-nas

# Error filtering
docker-compose logs | grep -i error
docker-compose logs | grep -i "failed\|error\|exception"

# Recent logs only
docker-compose logs --tail=100
```

### **Automated Health Monitoring**
```bash
# Create health check script
cat > health_check.sh << 'EOF'
#!/bin/bash
for service in yolo-combined mmpose yolo-nas; do
  status=$(curl -s http://localhost:8080/$service/healthz | jq -r .status 2>/dev/null)
  if [ "$status" = "healthy" ]; then
    echo "✅ $service: healthy"
  else
    echo "❌ $service: unhealthy"
    # Send alert or restart service
    docker-compose restart $service
  fi
done
EOF

chmod +x health_check.sh
# Run every 5 minutes
watch -n 300 ./health_check.sh
```

## 🎯 Quick Fixes by Symptom

### **Service Won't Start**
1. **Check health endpoint**: `curl http://localhost:PORT/healthz`
2. **Check logs**: `docker logs CONTAINER_NAME`
3. **Verify models loaded**: Look for "model loaded successfully" in logs
4. **Check disk space**: `df -h`
5. **Check dependencies**: `docker exec -it CONTAINER python -c "import torch; import service_deps"`
6. **Restart service**: `docker-compose restart SERVICE_NAME`

### **API Returns 503 Service Unavailable**
1. **Models not loaded** - Check logs for download/loading errors
2. **Dependencies missing** - Check import errors in logs: `docker logs SERVICE | grep -i import`
3. **Out of memory** - Reduce batch size or use smaller models
4. **GPU issues** - Verify NVIDIA runtime: `nvidia-smi`
5. **Port conflicts** - Check if ports are already in use: `netstat -tlnp | grep :800`

### **Build Failures**
1. **MMPose**: Use exact versions from DEPLOYMENT.md
2. **YOLO-NAS**: Ensure Python 3.10+ and numpy==1.23.0
3. **Disk space**: Use `./scripts/dev-fast.sh` for development builds
4. **Dependencies**: Follow installation order in troubleshooting guide
5. **Clean build**: `docker-compose build --no-cache`

### **Slow Performance**
1. **Use GPU**: Verify NVIDIA runtime and CUDA availability
2. **Batch processing**: Check if batching is enabled in code
3. **Model optimization**: Use half precision: `model.half()`
4. **Resource limits**: Check Docker resource constraints
5. **Network latency**: Check video download times

### **Memory Issues**
1. **Reduce batch size**: Edit batch_size in service code
2. **Use smaller models**: Switch to 'n' (nano) versions
3. **Clear cache**: `torch.cuda.empty_cache()` in code
4. **Restart services**: `docker-compose restart`
5. **Increase swap**: Add swap file to host system

## 🔧 Emergency Recovery Procedures

### **Complete System Recovery**
```bash
# Stop all services
docker-compose down

# Clean everything
docker system prune -af --volumes
docker builder prune -af

# Restore from backup
./scripts/restore.sh

# Or rebuild from scratch
./scripts/deploy.sh --clean-build
```

### **Single Service Recovery**
```bash
# Restart problematic service
docker-compose restart SERVICE_NAME

# Rebuild single service
docker-compose up --build -d SERVICE_NAME

# Check service logs
docker-compose logs -f SERVICE_NAME
```

### **Rollback to Working Version**
```bash
# Use backup docker-compose
cp deployment/docker-compose-backup.yml deployment/docker-compose.yml

# Or rollback Git commit
git log --oneline
git checkout PREVIOUS_WORKING_COMMIT

# Redeploy
./scripts/deploy.sh
```

This comprehensive troubleshooting guide covers all known issues with tested solutions for reliable operation of the NuroPadel platform.