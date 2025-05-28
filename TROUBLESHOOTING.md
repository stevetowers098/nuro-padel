# Troubleshooting Guide

This guide addresses common deployment and runtime issues for the Nuro Padel AI services.

## 🔧 Fixed Issues in This Repository

### 1. Protobuf Dependency Conflict ✅

**Issue**: `grpcio-status 1.70.0` requires `protobuf>=5.26.1,<6.0dev`, but older protobuf versions cause incompatibility.

**Solution Applied**: Added explicit protobuf version constraints to all service requirements:
- [`mmpose-service/requirements.txt`](mmpose-service/requirements.txt) - Added `protobuf>=5.26.1,<6.0.0`
- [`yolo-nas-service/requirements.txt`](yolo-nas-service/requirements.txt) - Added `protobuf>=5.26.1,<6.0.0`
- [`yolo-combined-service/requirements.txt`](yolo-combined-service/requirements.txt) - Added `protobuf>=5.26.1,<6.0.0`

### 2. Python Encoding Issues ✅

**Issue**: Potential encoding errors due to missing UTF-8 headers in Python files.

**Solution Applied**: Added UTF-8 encoding headers to all main.py files:
- [`mmpose-service/main.py`](mmpose-service/main.py:1) - Added `# -*- coding: utf-8 -*-`
- [`yolo-nas-service/main.py`](yolo-nas-service/main.py:1) - Added `# -*- coding: utf-8 -*-`  
- [`yolo-combined-service/main.py`](yolo-combined-service/main.py:1) - Added `# -*- coding: utf-8 -*-`

## 🚨 Manual Fixes for Existing Deployments

If you have existing deployments experiencing these issues, apply these manual fixes:

### Fix 1: Protobuf Dependency Conflict

```bash
# Update protobuf to compatible version
pip install --upgrade protobuf>=5.26.1

# Or force reinstall with compatible versions
pip install grpcio-status==1.70.0 protobuf>=5.26.1 --force-reinstall
```

### Fix 2: Encoding Error in main.py

```bash
# Check what's on line 18 (if error references line 18)
sed -n '15,20p' /opt/padel/app/app/main.py

# Fix encoding by adding header to main.py
sed -i '1i# -*- coding: utf-8 -*-' /opt/padel/app/app/main.py

# Or manually edit to remove non-ASCII characters
nano /opt/padel/app/app/main.py
```

### Fix 3: Service Path Issue (VM/Systemd Deployments)

**Note**: This repository uses Docker containers. The following applies to VM-based systemd deployments:

```bash
# Check if virtual environment exists
ls -la /opt/padel/shared/venv/bin/python

# If missing, create it or update service config
cd /opt/padel/app/
python3 -m venv /opt/padel/shared/venv
source /opt/padel/shared/venv/bin/activate
pip install -r requirements.txt

# Or update service to use system python
sudo sed -i 's|/opt/padel/shared/venv/bin/python|/usr/bin/python3|g' /etc/systemd/system/padel-api.service
sudo systemctl daemon-reload
sudo systemctl restart padel-api
```

## 🐳 Docker-Specific Solutions

### Container Startup Issues

```bash
# Check container logs
docker-compose logs [service-name]

# Rebuild containers with latest fixes
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Force pull latest images
docker-compose pull
docker-compose up -d
```

### Dependency Issues in Containers

```bash
# Rebuild specific service
docker-compose build --no-cache mmpose-service
docker-compose up -d mmpose-service

# Shell into container for debugging
docker-compose exec mmpose-service bash
pip list | grep protobuf
```

## 🔍 Diagnostic Commands

### Check Service Health

```bash
# Docker deployment
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose
curl http://localhost:8004/healthz  # YOLO-NAS

# VM deployment
curl http://localhost:8000/healthz  # Main API
systemctl status padel-api
```

### Check Dependencies

```bash
# In Docker container
docker-compose exec mmpose-service pip list | grep -E "(protobuf|grpcio)"

# In VM environment
source /opt/padel/shared/venv/bin/activate
pip list | grep -E "(protobuf|grpcio)"
```

### Check Python Encoding

```bash
# Check file encoding
file /path/to/main.py

# Check for non-ASCII characters
grep -P "[^\x00-\x7F]" /path/to/main.py
```

## 📋 Environment Verification

### Docker Environment

```bash
# Verify all services are running
docker-compose ps

# Check resource usage
docker stats

# Verify GPU access (if using CUDA)
docker-compose exec mmpose-service nvidia-smi
```

### VM Environment

```bash
# Check virtual environments
ls -la /opt/padel/*/venv/bin/python

# Check systemd services
systemctl list-units --type=service | grep padel

# Check logs
journalctl -u padel-api -f
```

## 🆘 Common Error Patterns

### Protobuf Errors
```
ERROR: grpcio-status 1.70.0 has requirement protobuf<6.0dev,>=5.26.1, but you'll have protobuf 4.25.7 which is incompatible.
```
**Solution**: Apply Fix 1 above

### Encoding Errors
```
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position X
```
**Solution**: Apply Fix 2 above

### Service Path Errors
```
FileNotFoundError: [Errno 2] No such file or directory: '/opt/padel/shared/venv/bin/python'
```
**Solution**: Apply Fix 3 above

## 📞 Support

If issues persist after applying these fixes:

1. Check the container/service logs for specific error messages
2. Verify all model weights are present and accessible
3. Ensure adequate system resources (RAM, GPU memory)
4. Confirm network connectivity for GCS uploads

For Docker deployments, the fixes in this repository should resolve the major compatibility issues automatically.