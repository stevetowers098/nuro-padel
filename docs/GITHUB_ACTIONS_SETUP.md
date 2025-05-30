# 🚀 GitHub Actions CI/CD Setup Guide

This guide explains how to set up automated deployment to your VM using GitHub Actions with comprehensive logging and testing.

## 📋 Prerequisites

### 1. GitHub Repository Secrets

You need to configure these secrets in your GitHub repository:

**Go to: Repository → Settings → Secrets and variables → Actions**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `VM_IP` | `35.189.53.46` | Your VM's IP address |
| `VM_SSH_KEY` | `[Your private SSH key]` | Private key matching the public key in `/home/Towers/.ssh/authorized_keys` |

### 2. VM Setup

Ensure your VM has the following configured:

```bash
# 1. SSH access for user 'Towers'
sudo mkdir -p /home/Towers/.ssh
sudo chown Towers:Towers /home/Towers/.ssh
sudo chmod 700 /home/Towers/.ssh

# 2. Add your public key to authorized_keys
echo "your-public-ssh-key-here" | sudo tee /home/Towers/.ssh/authorized_keys
sudo chown Towers:Towers /home/Towers/.ssh/authorized_keys
sudo chmod 600 /home/Towers/.ssh/authorized_keys

# 3. Create project directory
sudo mkdir -p /opt/padel-docker
sudo chown Towers:Towers /opt/padel-docker

# 4. Install Docker and Docker Compose
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker Towers

# 5. Install NVIDIA Docker runtime (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

## 🔄 Deployment Workflow

The GitHub Actions workflow consists of 5 comprehensive stages:

### 1. 🔨 Build & Push Docker Images
- Builds 3 services: `yolo-combined`, `mmpose`, `yolo-nas`
- Pushes to GitHub Container Registry (GHCR)
- Uses BuildKit for optimized caching
- Tags with both `latest` and commit SHA

### 2. 🚀 Deploy to Production VM
- SSH into your VM using the configured secrets
- Navigates to `/opt/padel-docker`
- Runs `./scripts/deploy.sh --vm`
- Provides detailed logging of each step

### 3. 🏥 Health Check & Validation
- Waits 90 seconds for services to stabilize
- Tests each service health endpoint:
  - **YOLO Combined**: `http://localhost:8001/healthz`
  - **MMPose**: `http://localhost:8003/healthz`
  - **YOLO-NAS**: `http://localhost:8004/healthz`
  - **Load Balancer**: `http://localhost:8080`
- Shows Docker container status and logs

### 4. 🧪 Integration Tests
- Tests service endpoints with sample requests
- Checks system resource usage (CPU, memory, disk)
- Monitors Docker container resource consumption
- Validates load balancer functionality

### 5. 📈 Deployment Summary
- Generates comprehensive summary report
- Shows status of all jobs
- Provides service URLs for easy access
- Displays build information and timestamps

## 📊 Monitoring & Logging

### Real-time Monitoring
You can monitor the deployment in real-time by:

1. **GitHub Actions Tab**: See live logs of each step
2. **VM SSH Access**: Connect directly to monitor services
3. **Service URLs**: Check health endpoints after deployment

### Log Locations

| Component | Log Location | Command |
|-----------|--------------|---------|
| GitHub Actions | Actions tab in GitHub | - |
| Docker Compose | VM deployment directory | `docker-compose logs` |
| Individual Services | Docker containers | `docker logs <container-name>` |
| System Logs | VM system logs | `journalctl -u docker` |

### Health Check Endpoints

After successful deployment, these endpoints will be available:

```bash
# Service health checks
curl http://35.189.53.46:8001/healthz  # YOLO Combined
curl http://35.189.53.46:8003/healthz  # MMPose  
curl http://35.189.53.46:8004/healthz  # YOLO-NAS
curl http://35.189.53.46:8080/         # Load Balancer

# Detailed service info
curl http://35.189.53.46:8001/         # YOLO Combined API
curl http://35.189.53.46:8003/         # MMPose API
curl http://35.189.53.46:8004/         # YOLO-NAS API
```

## 🔧 Troubleshooting

### Common Issues

#### 1. SSH Connection Fails
```bash
# Check SSH key format (should be OpenSSH format)
ssh-keygen -f ~/.ssh/id_rsa -e -m OpenSSH

# Test SSH connection manually
ssh -i ~/.ssh/id_rsa Towers@35.189.53.46
```

#### 2. Docker Build Fails
- Check available disk space on GitHub Actions runner
- Verify Dockerfile syntax in affected service
- Check for dependency conflicts in requirements.txt

#### 3. Health Checks Fail
```bash
# SSH into VM and check services manually
ssh Towers@35.189.53.46
cd /opt/padel-docker/deployment
docker-compose ps
docker-compose logs [service-name]
```

#### 4. GPU Issues
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check GPU availability in containers
docker-compose exec yolo-combined nvidia-smi
```

### Manual Deployment

If the automated deployment fails, you can deploy manually:

```bash
# SSH into VM
ssh Towers@35.189.53.46

# Navigate to project directory
cd /opt/padel-docker

# Run deployment script
./scripts/deploy.sh --vm

# Check status
cd deployment
docker-compose ps
```

## 🎯 Triggering Deployments

### Automatic Triggers
- **Push to main branch**: Automatically triggers full CI/CD pipeline
- **Pull Request merge**: When PR is merged to main

### Manual Triggers
1. Go to **Actions** tab in your GitHub repository
2. Select **🚀 CI & Deploy to VM** workflow
3. Click **Run workflow** button
4. Select branch and click **Run workflow**

## 📈 Performance Monitoring

The workflow provides detailed performance metrics:

### Build Metrics
- Docker build times for each service
- Cache hit rates for optimized builds
- Image sizes and registry push times

### Deployment Metrics
- SSH connection time
- Service startup time (90-second stabilization period)
- Health check response times

### System Metrics
- CPU usage across all containers
- Memory consumption per service
- GPU utilization (if available)
- Disk space usage

## 🔐 Security Best Practices

1. **SSH Keys**: Use strong SSH keys and rotate regularly
2. **Secrets Management**: Never commit secrets to repository
3. **VM Access**: Limit SSH access to deployment user only
4. **Network Security**: Consider firewall rules for service ports
5. **Container Security**: Keep base images updated

## 🆘 Support

If you encounter issues:

1. **Check GitHub Actions logs** for detailed error messages
2. **SSH into VM** to investigate service status
3. **Review health check endpoints** for service-specific issues
4. **Check system resources** (CPU, memory, disk, GPU)
5. **Validate SSH connectivity** and permissions

The workflow provides comprehensive logging at every step, making it easy to identify and resolve issues quickly.