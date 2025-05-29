# ⚡ Fast Development Workflow - 1-2 Minute Builds

**Problem Solved**: 30-minute Docker builds for every code/requirements change → **1-2 minute builds**

## 🎯 **How It Works**

### **Traditional Approach** 🐌
```bash
# Every change = Full rebuild from scratch
Change code → 30-min build (install PyTorch, MMCV, MMPose, etc.) → Deploy
```

### **Fast Development Approach** ⚡
```bash
# One-time setup: Build base image with heavy dependencies (30 mins)
Heavy deps → Base image (ghcr.io/stevetowers098/nuro-padel/base:latest)

# Daily development: Fast builds using base image (1-2 mins)
Change code → Use base image + add your code (1-2 mins) → Deploy
```

## 🚀 **Quick Start**

### **Step 1: One-Time Base Image Setup** (30 minutes, but only once!)

**Option A: Build Automatically via GitHub Actions**
```bash
# Push the base Dockerfile to trigger automatic build
git add Dockerfile.base .github/workflows/build-base.yml
git commit -m "Add fast development base image"
git push origin docker-containers
# Wait 30 minutes for GitHub Actions to build base image
```

**Option B: Build Locally** 
```bash
# Build base image locally (if you prefer)
docker build -f Dockerfile.base -t ghcr.io/stevetowers098/nuro-padel/base:latest .
docker push ghcr.io/stevetowers098/nuro-padel/base:latest
```

### **Step 2: Fast Development** (1-2 minutes per change!)

```bash
# Make your code changes
vim mmpose-service/main.py  # Edit your code
vim mmpose-service/requirements.txt  # Add new dependencies

# Fast rebuild and deploy (1-2 minutes)
./dev-fast.sh

# Your services are now running with changes!
```

## 📁 **New Files Created**

### **Base Image & Dependencies**
- [`Dockerfile.base`](Dockerfile.base) - Contains all heavy AI dependencies (PyTorch, MMCV, MMPose, etc.)
- [`.github/workflows/build-base.yml`](.github/workflows/build-base.yml) - Builds base image automatically

### **Fast Development Dockerfiles**
- [`mmpose-service/Dockerfile.dev`](mmpose-service/Dockerfile.dev) - Fast MMPose builds
- [`yolo-nas-service/Dockerfile.dev`](yolo-nas-service/Dockerfile.dev) - Fast YOLO-NAS builds  
- [`yolo-combined-service/Dockerfile.dev`](yolo-combined-service/Dockerfile.dev) - Fast YOLO-Combined builds

### **Development Deployment**
- [`docker-compose.dev.yml`](docker-compose.dev.yml) - Uses fast development Dockerfiles
- [`dev-fast.sh`](dev-fast.sh) - One-command fast deployment script

## ⏱️ **Performance Comparison**

| Change Type | Before | After | Improvement |
|-------------|--------|-------|-------------|
| Code changes | 30 mins | 1-2 mins | **15-30x faster** |
| Requirements changes | 30 mins | 1-2 mins | **15-30x faster** |
| First-time setup | 30 mins | 30 mins + 1-2 mins | Same (one-time cost) |

## 🔄 **Development Workflow**

### **Daily Development Cycle**
```bash
# 1. Make changes
git pull origin docker-containers
edit mmpose-service/main.py              # Change your code
edit mmpose-service/requirements.txt     # Add dependencies

# 2. Fast deploy (1-2 minutes)
./dev-fast.sh

# 3. Test your changes
curl http://localhost:8003/healthz       # Test MMPose
curl http://localhost:8080/healthz       # Test full system

# 4. Iterate quickly
edit mmpose-service/main.py              # More changes
./dev-fast.sh                            # Another 1-2 minute build
```

### **Production Deployment** (when ready)
```bash
# Use the normal production deployment for final release
git commit -m "Ready for production"
git push origin docker-containers
# GitHub Actions builds and deploys to VM (30 mins, but optimized for production)
```

## 🛡️ **What's Included in Base Image**

The base image contains all the **slow-to-install** dependencies:

### **Heavy AI Dependencies** (25+ minutes normally)
- PyTorch 2.1.2 + CUDA support
- MMCV 2.1.0 (with CUDA compilation)
- MMEngine 
- MMDetection
- MMPose
- super-gradients 3.7.1 (YOLO-NAS)
- OpenCV with optimizations

### **Common Dependencies** (3-5 minutes normally)
- FastAPI, Pydantic, Uvicorn
- NumPy, Pillow 
- Google Cloud Storage
- Requests, HTTPx
- And all other common packages

## 🎯 **Use Cases**

### **Perfect For Development/Testing Phase** ⚡
- Rapid code iteration
- Testing new features
- Requirements experimentation
- Bug fixes
- Performance tuning

### **When to Use Production Build** 🏭
- Final release deployment
- Version tagging
- Long-term stable deployment
- Production optimizations

## 🔧 **Advanced Usage**

### **Live Code Mounting** (Instant changes)
```yaml
# In docker-compose.dev.yml - uncomment these lines for instant code changes
volumes:
  - ./mmpose-service:/app:ro  # Live code mounting
```

### **Selective Service Building**
```bash
# Build only specific service
docker-compose -f docker-compose.dev.yml build mmpose

# Start only specific service  
docker-compose -f docker-compose.dev.yml up mmpose -d
```

### **Base Image Updates**
```bash
# When you need to update heavy dependencies
edit Dockerfile.base                    # Update PyTorch version, etc.
git push origin docker-containers       # Triggers new base image build
# Wait 30 minutes for new base image, then all dev builds use new base
```

## 🚨 **Troubleshooting**

### **Base Image Not Found**
```bash
# Pull the base image manually
docker pull ghcr.io/stevetowers098/nuro-padel/base:latest

# Or build it locally
docker build -f Dockerfile.base -t ghcr.io/stevetowers098/nuro-padel/base:latest .
```

### **Still Slow Builds**
```bash
# Check if using dev Dockerfiles
docker-compose -f docker-compose.dev.yml build --no-cache

# Verify base image exists
docker image inspect ghcr.io/stevetowers098/nuro-padel/base:latest
```

### **Development vs Production Confusion**
```bash
# Development (fast): 
./dev-fast.sh
docker-compose -f docker-compose.dev.yml up -d

# Production (optimized):
docker-compose -f docker-compose.yml up -d
```

---

## ✅ **Summary**

✅ **One-time setup**: 30-minute base image build (contains all heavy AI dependencies)  
✅ **Daily development**: 1-2 minute builds for code/requirements changes  
✅ **Rapid iteration**: Perfect for testing phase with lots of changes  
✅ **Production ready**: Keep existing production deployment for releases  

**Result**: 15-30x faster development cycle during testing phase! 🚀