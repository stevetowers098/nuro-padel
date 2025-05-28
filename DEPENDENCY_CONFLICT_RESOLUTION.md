# Dependency Conflict Resolution - Protobuf Issue

## 🚨 The Problem

Super-gradients (YOLO-NAS) and Google Cloud Storage have conflicting protobuf requirements:

- `super-gradients==3.7.1` requires `protobuf<4.0.0`
- `google-cloud-storage==2.18.0` requires `protobuf>=5.26.1,<6.0.0` (via grpcio-status)

## 🎯 Solution Strategy

### Option 1: Separate Environments (Recommended)
Install YOLO-NAS services in separate virtual environments without GCS functionality:

```bash
# YOLO-NAS environment (no GCS upload)
cd /opt/padel/yolo-nas
python3 -m venv venv
source venv/bin/activate
pip install -r /opt/padel/app/requirements/yolo-nas.txt
# Let super-gradients install its preferred protobuf version

# Main API environment (with GCS)
cd /opt/padel/shared
source venv/bin/activate
pip install protobuf>=5.26.1,<6.0.0
pip install -r /opt/padel/app/requirements/optimized_main.txt
```

### Option 2: Disable GCS for YOLO-NAS Services
Modify YOLO-NAS services to skip GCS uploads:

```bash
# Install without GCS dependency
pip install -r /opt/padel/app/requirements/yolo-nas.txt
# Skip GCS uploads in YOLO-NAS service code
```

### Option 3: Downgrade GCS (Not Recommended)
Use older google-cloud-storage that's compatible with older protobuf:

```bash
pip install google-cloud-storage==2.10.0  # Older version
pip install protobuf>=3.20.0,<4.0.0
```

## 🔧 Immediate Fix

### Step 1: Install YOLO dependencies first
```bash
cd /opt/padel/yolo
source venv/bin/activate

# Install YOLO-NAS without protobuf constraints
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install super-gradients==3.7.1
pip install ultralytics==8.3.0
pip install fastapi==0.115.12 uvicorn[standard]==0.32.0
pip install opencv-python-headless==4.8.1.78
pip install numpy>=1.24.0,<1.27.0
pip install httpx==0.27.2

# Check what protobuf version super-gradients installed
pip show protobuf
```

### Step 2: Install Main API dependencies
```bash
cd /opt/padel/shared
source venv/bin/activate

# Install main API with newer protobuf
pip install protobuf>=5.26.1,<6.0.0
pip install -r /opt/padel/app/requirements/optimized_main.txt
```

### Step 3: Update Service Configuration
Ensure each service uses its correct virtual environment:

```bash
# Update YOLO-NAS service
sudo sed -i 's|/opt/padel/shared/venv/bin/python|/opt/padel/yolo-nas/venv/bin/python|g' /etc/systemd/system/yolo-nas.service

# Update YOLO11/YOLOv8 services  
sudo sed -i 's|/opt/padel/shared/venv/bin/python|/opt/padel/yolo/venv/bin/python|g' /etc/systemd/system/yolo11.service
sudo sed -i 's|/opt/padel/shared/venv/bin/python|/opt/padel/yolo/venv/bin/python|g' /etc/systemd/system/yolov8.service

# Main API keeps shared environment
# /opt/padel/shared/venv/bin/python for padel-api.service

sudo systemctl daemon-reload
```

## 🧪 Test the Fix

```bash
# Test YOLO-NAS environment
cd /opt/padel/yolo-nas
source venv/bin/activate
python -c "import super_gradients; print('YOLO-NAS OK')"
pip show protobuf

# Test Main API environment  
cd /opt/padel/shared
source venv/bin/activate
python -c "from google.cloud import storage; print('GCS OK')"
pip show protobuf

# Should show different protobuf versions in each environment
```

## 📋 Service Environment Mapping

| Service | Virtual Environment | Protobuf Version | GCS Support |
|---------|-------------------|------------------|-------------|
| padel-api | `/opt/padel/shared/venv` | 5.26.1+ | ✅ Yes |
| mmpose | `/opt/padel/mmpose/venv` | 5.26.1+ | ✅ Yes |
| yolo11 | `/opt/padel/yolo/venv` | Auto | ⚠️ Optional |
| yolov8 | `/opt/padel/yolo/venv` | Auto | ⚠️ Optional |
| yolo-nas | `/opt/padel/yolo-nas/venv` | <4.0.0 | ❌ No |

## 🆘 Rollback if Needed

```bash
# Remove protobuf constraints from all files
sed -i '/protobuf/d' /opt/padel/app/requirements/*.txt

# Let each service install its preferred protobuf version
# Disable GCS uploads in YOLO-NAS services
```

This approach isolates the conflicting dependencies while maintaining functionality.