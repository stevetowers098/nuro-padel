# GitHub Resilient Deployment Setup Guide

## Overview

Your NuroPadel project now has **GitHub Actions integration** for the resilient deployment system! This allows you to deploy services independently in GitHub runners and optionally to your VM, with graceful failure handling.

## 🚀 **How to Use deploy-resilient in GitHub**

### **1. Automatic Deployments**

The workflow triggers automatically on:
- **Push to `main` or `develop`** - Deploys all services
- **Pull Requests to `main`** - Tests deployment without VM deployment

### **2. Manual Deployments**

Use GitHub's web interface for manual deployments:

1. **Go to your repository** → **Actions** tab
2. **Select "Resilient NuroPadel Deployment"** workflow  
3. **Click "Run workflow"**
4. **Choose options:**

#### **Service Options**:
- `all` - Deploy all services (yolo-combined, mmpose, yolo-nas, nginx)
- `yolo-combined` - Deploy only YOLO Combined service
- `mmpose` - Deploy only MMPose service  
- `yolo-nas` - Deploy only YOLO-NAS service
- `nginx` - Deploy only nginx load balancer

#### **Additional Options**:
- `deploy_to_vm` - Also deploy to your configured VM
- `cleanup_after` - Clean up Docker images after deployment

### **3. Command Line Deployments**

Use GitHub CLI for command-line control:

```bash
# Deploy all services
gh workflow run resilient-deploy.yml -f service=all

# Deploy only working YOLO Combined service
gh workflow run resilient-deploy.yml -f service=yolo-combined

# Deploy to VM as well
gh workflow run resilient-deploy.yml -f service=all -f deploy_to_vm=true

# Deploy specific service without cleanup
gh workflow run resilient-deploy.yml -f service=mmpose -f cleanup_after=false
```

## 🔧 **Configuration Requirements**

### **Repository Secrets** (for VM deployment)

If you want to deploy to your VM, add these secrets in **Settings** → **Secrets and variables** → **Actions**:

```
VM_USER=your-vm-username
VM_HOST=your-vm-ip-or-hostname  
VM_PATH=/path/to/deployment/directory
```

### **Permissions**

The workflow automatically gets:
- ✅ **Container registry access** - For pushing Docker images
- ✅ **Repository access** - For code checkout
- ✅ **Package write permissions** - For GitHub Container Registry

## 📊 **Workflow Features**

### **✅ Resilient Deployment**
- **Independent service deployment** - Working services deploy even if others fail
- **Graceful failure handling** - Continues deployment with available services
- **Health check validation** - Verifies services are running correctly

### **✅ Resource Optimization**  
- **Disk space management** - Cleans up GitHub runner disk space
- **Smart image building** - Only builds changed services
- **Registry integration** - Uses GitHub Container Registry for image storage

### **✅ Multi-Environment Support**
- **GitHub Runner testing** - Tests deployment in GitHub environment
- **VM deployment** - Optionally deploys to your production VM
- **Pull request testing** - Safe testing without production deployment

## 📈 **Usage Examples**

### **Example 1: Quick Test of Working Service**
```bash
# Deploy only the known-working YOLO Combined service
gh workflow run resilient-deploy.yml -f service=yolo-combined
```

### **Example 2: Full Production Deployment**
```bash
# Deploy all services and push to VM
gh workflow run resilient-deploy.yml -f service=all -f deploy_to_vm=true
```

### **Example 3: Debug Specific Service**
```bash
# Deploy only MMPose to test recent fixes
gh workflow run resilient-deploy.yml -f service=mmpose
```

## 🔍 **Monitoring Deployments**

### **View Deployment Status**
1. Go to **Actions** tab in your repository
2. Click on the running/completed workflow
3. View logs for each step:
   - **Build and Push Docker Images** - See which services built successfully
   - **Run Resilient Deployment** - See resilient deployment output
   - **Test Deployed Services** - See health check results
   - **Deployment Summary** - See final status

### **Service Health Checks**
The workflow automatically tests:
- ✅ **Nginx** at `http://localhost:8080/health`
- ✅ **YOLO Combined** at `http://localhost:8001/healthz`  
- ✅ **MMPose** at `http://localhost:8003/healthz`
- ✅ **YOLO-NAS** at `http://localhost:8004/healthz`

### **Expected Output**
```
🚀 Starting resilient deployment for: all
[INFO] Starting resilient deployment of all services...
[SUCCESS] Nginx deployed successfully
[SUCCESS] yolo-combined is running and healthy!
[ERROR] Failed to deploy mmpose
[SUCCESS] yolo-nas is running and healthy!
[SUCCESS] 2 out of 3 AI services deployed successfully
[SUCCESS] At least one service is available for testing!
```

## 🛠 **Best Practices**

1. **Start with individual services** to test before deploying all
2. **Use VM deployment sparingly** to avoid overloading your VM
3. **Monitor workflow logs** to identify and fix failing services
4. **Create backups** of working services before making changes
5. **Test in GitHub runners first** before deploying to VM

## 🔄 **Integration with Existing Workflows**

Your new resilient workflow **complements** your existing workflows:

- **`deploy.yml`** - Traditional all-or-nothing deployment
- **`sequential-deploy.yml`** - Sequential service deployment  
- **`resilient-deploy.yml`** - **NEW!** Independent resilient deployment
- **`docker-deploy.yml`** - Branch-specific Docker deployment

**Recommendation**: Use `resilient-deploy.yml` as your primary deployment method!

## ✅ **Success! You can now:**

- ✅ **Deploy working services independently** in GitHub Actions
- ✅ **Handle service failures gracefully** without blocking deployment
- ✅ **Test services in GitHub runners** before VM deployment
- ✅ **Use manual or automatic deployment** triggers
- ✅ **Monitor deployment status** with detailed logging
- ✅ **Deploy to your VM** with the same resilient approach