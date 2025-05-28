# 🚨 CRITICAL DEPENDENCY AUDIT - Bible vs Current State

## ❌ MAJOR INCOMPATIBILITIES FOUND

Comparing current configuration against [`DOCKER_DEPENDENCY_FIXES.md`](DOCKER_DEPENDENCY_FIXES.md) (the bible):

### 🔥 **MMPose Service - COMPLETELY BROKEN**

**Current State:**
```dockerfile
# mmpose-service/Dockerfile (WRONG)
RUN pip install --no-cache-dir --upgrade pip setuptools==60.2.0 wheel && \
    pip install --no-cache-dir pytz==2023.3 requests==2.28.2 "rich>=13.7.1" tqdm==4.65.0 && \
    pip install --no-cache-dir "numpy>=1.21.0,<2.0" xtcocotools mmpose
```

**Bible Requirements:**
```dockerfile
# What it SHOULD be according to bible
RUN pip install --no-cache-dir --upgrade pip==23.1.2 setuptools==60.2.0 wheel && \
    pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 && \
    pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install mmdet && \
    mim install "mmpose>=1.1.0"
```

**❌ Missing Components:**
- PyTorch 1.13.1 (CRITICAL - MMPose won't work with PyTorch 2.x)
- pip==23.1.2 (required version)
- openmim installation
- MIM-based mmcv/mmdet/mmpose installation
- Proper version constraints

### ⚠️ **YOLO Combined Service - Missing Setup**

**Current State:**
```dockerfile
# yolo-combined-service/Dockerfile (MISSING SETUPTOOLS PIN)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
```

**Bible Requirements:**
```dockerfile
# What it SHOULD be
RUN pip install --no-cache-dir --upgrade pip setuptools==65.7.0 wheel
```

**❌ Missing:**
- setuptools==65.7.0 pin (prevents InvalidVersion errors)

### ✅ **YOLO-NAS Service - CORRECT**

**Current State:** ✅ Matches bible exactly
- setuptools==65.7.0 ✅
- super-gradients installed first ✅
- sphinx==4.0.2 constraint ✅
- Minimal requirements.txt ✅

## 🔥 **ROOT CAUSE ANALYSIS**

### Why MMPose is Broken:
1. **PyTorch Version Mismatch**: MMPose requires PyTorch 1.13.1, not 2.x
2. **Missing MIM**: MMPose ecosystem requires OpenMIM for proper installation
3. **Wrong Installation Order**: Should install PyTorch first, then MIM packages
4. **Direct mmpose install**: Should use `mim install mmpose`, not `pip install mmpose`

### Why These Errors Keep Happening:
1. **Current Dockerfile ignores the bible** - doesn't follow documented working configuration
2. **Missing core dependencies** - PyTorch not explicitly installed for MMPose
3. **Wrong installation method** - pip instead of MIM for MMPose ecosystem

## 📋 **COMPATIBILITY MATRIX FROM BIBLE**

| Service | PyTorch Version | setuptools | Installation Method | Status |
|---------|----------------|------------|-------------------|---------|
| **YOLO Combined** | 2.3.1 (current) | ❌ Missing 65.7.0 | pip | ⚠️ Needs Fix |
| **MMPose** | ❌ Missing 1.13.1 | ✅ 60.2.0 | ❌ Should use MIM | 🔥 Broken |
| **YOLO-NAS** | Managed by super-gradients | ✅ 65.7.0 | pip | ✅ Correct |

## 🚨 **CRITICAL FIXES NEEDED**

### 1. MMPose Service - Complete Rebuild Required
```dockerfile
# CORRECT implementation from bible
RUN pip install --no-cache-dir --upgrade pip==23.1.2 setuptools==60.2.0 wheel && \
    pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 && \
    pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv>=2.0.1" && \
    mim install mmdet && \
    mim install "mmpose>=1.1.0" && \
    pip install --no-cache-dir -r requirements.txt
```

### 2. YOLO Combined Service - Add setuptools Pin
```dockerfile
# Add missing setuptools pin
RUN pip install --no-cache-dir --upgrade pip setuptools==65.7.0 wheel
```

### 3. MMPose requirements.txt - Update Versions
```txt
# From bible - these are the TESTED working versions
pytz==2023.3
requests==2.28.2  
rich==13.4.2
tqdm==4.65.0
pycocotools==2.0.7
numpy>=1.21.0,<1.25.0
google-cloud-storage==2.18.0
```

## ⚡ **IMMEDIATE ACTION REQUIRED**

The current MMPose configuration **WILL FAIL** because:
1. No PyTorch 1.13.1 (MMPose incompatible with PyTorch 2.x)
2. No MIM installation (required for MMPose ecosystem)
3. Wrong dependency installation method

**These fixes must be applied to match the bible exactly, or all previous debugging work is lost.**