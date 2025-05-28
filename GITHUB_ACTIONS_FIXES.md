# GitHub Actions Deployment Fixes

## 🚨 **Issues Identified & Fixed**

### **1. Missing Docker Compose Test File** ✅ FIXED
**Problem**: GitHub Actions referenced missing `docker-compose.test.yml`
**Solution**: Created [`docker-compose.test.yml`](docker-compose.test.yml) with:
- Pre-built image references from GitHub Container Registry
- Correct health check endpoints
- Permission fixes applied (HOME=/tmp, MPLCONFIGDIR=/tmp/matplotlib)
- Test-specific network configuration

### **2. Legacy Docker Compose v1** ✅ FIXED
**Problem**: Workflow using deprecated `docker-compose` commands
**Solution**: Updated [`.github/workflows/docker-deploy.yml`](.github/workflows/docker-deploy.yml):
- Installed `docker-compose-plugin` instead of legacy version
- Changed `docker-compose` commands to `docker compose`
- Updated all workflow steps to use Docker Compose v2

### **3. Incorrect Health Check Endpoints** ✅ FIXED
**Problem**: Health checks calling `http://localhost/healthz` (port 80) instead of correct ports
**Solution**: Fixed endpoints in workflow:
- Individual services: `http://localhost:8001/healthz`, `http://localhost:8003/healthz`, `http://localhost:8004/healthz`
- Nginx proxy: `http://localhost:8080/healthz` (correct port mapping)

### **4. VM Deployment Script Issues** ✅ FIXED
**Problem**: VM deployment using wrong compose file and scripts
**Solution**: Updated VM deployment section:
- Use `docker-compose.resilient.yml` instead of default
- Call `./deploy-resilient.sh` instead of `./deploy.sh`
- Fixed health check endpoint for VM verification

## 📁 **Files Modified**

### **New Files Created**
- [`docker-compose.test.yml`](docker-compose.test.yml) - CI/CD test configuration
- [`GITHUB_ACTIONS_FIXES.md`](GITHUB_ACTIONS_FIXES.md) - This documentation

### **Files Updated**
- [`.github/workflows/docker-deploy.yml`](.github/workflows/docker-deploy.yml) - Fixed Docker Compose v2, endpoints, deployment
- [`docker-compose.resilient.yml`](docker-compose.resilient.yml) - Permission fixes applied earlier

## 🔧 **Technical Changes**

### **Docker Compose Test Configuration**
```yaml
# docker-compose.test.yml
services:
  yolo-combined:
    image: ${REGISTRY}/yolo-combined:latest
    ports: ["8001:8001"]
    environment:
      - HOME=/tmp
      - MPLCONFIGDIR=/tmp/matplotlib
      - PYTHONPATH=/app
```

### **GitHub Actions Workflow Updates**
```yaml
# Updated Docker Compose setup
- name: Setup Docker Compose
  run: |
    sudo apt-get remove docker-compose || true
    sudo apt-get install -y docker-compose-plugin
    docker compose version

# Updated service startup
- name: Start services
  run: |
    export REGISTRY=${{ env.REGISTRY }}/${{ github.repository_owner }}/${{ env.IMAGE_PREFIX }}
    docker compose -f docker-compose.test.yml up -d

# Fixed health check endpoints
- name: Test service endpoints
  run: |
    curl -f http://localhost:8001/healthz || exit 1
    curl -f http://localhost:8003/healthz || exit 1  
    curl -f http://localhost:8004/healthz || exit 1
    curl -f http://localhost:8080/healthz || exit 1
```

### **VM Deployment Updates**
```bash
# Use resilient deployment
docker compose -f docker-compose.resilient.yml pull || echo "Registry pull failed, will build locally"
./deploy-resilient.sh --deploy
curl -f http://localhost:8080/healthz || exit 1
```

## 🎯 **Expected Results**

### **GitHub Actions Pipeline**
- ✅ Docker images build successfully without manifest errors
- ✅ Integration tests pass with correct health check endpoints
- ✅ Docker Compose v2 commands work properly
- ✅ VM deployment uses correct files and scripts

### **Local Development**
- ✅ Use `docker-compose.resilient.yml` for resilient deployment
- ✅ Permission issues resolved with environment variables
- ✅ Health checks accessible on correct ports

### **VM Deployment** 
- ✅ Resilient deployment script handles individual service failures
- ✅ Registry images pulled when available
- ✅ Health verification on correct nginx port (8080)

## 🚀 **Deployment Commands**

### **GitHub Actions (Automatic)**
Push to `docker-containers` branch triggers full pipeline:
1. Build all three services
2. Run integration tests with `docker-compose.test.yml`
3. Deploy to VM using resilient configuration

### **Manual VM Deployment**
```bash
# Use the resilient deployment
sudo docker compose -f docker-compose.resilient.yml --profile yolo-combined up -d
sudo docker compose -f docker-compose.resilient.yml --profile mmpose up -d
sudo docker compose -f docker-compose.resilient.yml --profile yolo-nas up -d
sudo docker compose -f docker-compose.resilient.yml --profile nginx up -d

# Verify health
curl http://localhost:8001/healthz  # YOLO Combined
curl http://localhost:8003/healthz  # MMPose  
curl http://localhost:8004/healthz  # YOLO-NAS
curl http://localhost:8080/healthz  # Nginx
```

## 🛡️ **Rollback Plans**

### **If GitHub Actions Fails**
```bash
# Revert workflow changes
git checkout HEAD~1 -- .github/workflows/docker-deploy.yml

# Remove test file if needed
rm docker-compose.test.yml
```

### **If VM Deployment Fails**
```bash
# Use backup deployment
./deploy.sh --deploy

# Or manual rollback
sudo docker compose -f docker-compose.resilient.yml down
git checkout HEAD~1 -- docker-compose.resilient.yml
```

## 📊 **Monitoring & Verification**

### **GitHub Actions Status**
- Check Actions tab for build success
- Monitor Container Registry for pushed images
- Verify integration test results

### **VM Health Checks**
```bash
# Service status
sudo docker ps -a
sudo docker compose -f docker-compose.resilient.yml logs

# Health endpoints
curl http://localhost:8001/healthz  # Should return 200
curl http://localhost:8003/healthz  # Should return 200  
curl http://localhost:8004/healthz  # Should return 200
curl http://localhost:8080/healthz  # Should return 200
```

---

**Status**: ✅ All GitHub Actions deployment issues have been identified and fixed. Ready for testing with next push to `docker-containers` branch.