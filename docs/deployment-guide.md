# NuroPadel Deployment Guide

## Services & Ports

| Service | Port | Command |
|---------|------|---------|
| yolo8 | 8001 | `./services/yolo8/deploy.sh` |
| yolo11 | 8002 | `./services/yolo11/deploy.sh` |
| mmpose | 8003 | `./services/mmpose/deploy.sh` |
| yolo-nas | 8004 | `./services/yolo-nas/deploy.sh` |
| rf-detr | 8005 | `./services/rf-detr/deploy.sh` |
| vitpose | 8006 | `./services/vitpose/deploy.sh` |

## Deployment Options

### Individual Service (Recommended)

```bash
./services/yolo8/deploy.sh    # Deploy only YOLO8
./services/mmpose/deploy.sh   # Deploy only MMPose
```

### All Services

```bash
./scripts/deploy-all.sh       # Deploy everything
```

## Why Individual Deployment?

✅ **Zero Downtime**: Other services keep running  
✅ **Faster**: Deploy only what changed  
✅ **Safer**: Failures are isolated  
✅ **Flexible**: Roll back individual services  

## Quick Commands

```bash
# Health check all services
for port in 8001 8002 8003 8004 8005 8006; do
  echo "Port $port: $(curl -s http://35.189.53.46:$port/healthz || echo 'DOWN')"
done

# Check service logs
ssh padel-ai "docker logs nuro-padel-yolo8 --tail 20"

# Restart failed service
./services/yolo8/deploy.sh
