#!/bin/bash
# health-check.sh
# Script to monitor the health of all NuroPadel services
# and provide detailed diagnostics for any issues

set -e  # Exit on any error

echo "🔍 NuroPadel Health Check"
echo "=========================="
echo "Checking all services at $(date)"
echo

# Define services and their expected ports
declare -A services=(
  ["padel-api"]="8000"
  ["yolo11-service"]="8001"
  ["yolov8-service"]="8002"
  ["mmpose-service"]="8003"
  ["yolo-nas-service"]="8004"
)

# Define environment paths for each service
declare -A env_paths=(
  ["padel-api"]="/opt/padel/envs/modern-torch/venv"
  ["yolo11-service"]="/opt/padel/envs/modern-torch/venv"
  ["yolov8-service"]="/opt/padel/envs/modern-torch/venv"
  ["mmpose-service"]="/opt/padel/envs/legacy-torch/venv"
  ["yolo-nas-service"]="/opt/padel/envs/specialized/yolo-nas-venv"
)

# Check systemd service status
echo "Checking systemd service status..."
echo "--------------------------------"
for service in "${!services[@]}"; do
  status=$(systemctl is-active $service 2>/dev/null || echo "inactive")
  if [ "$status" = "active" ]; then
    echo "✅ $service: ACTIVE"
  else
    echo "❌ $service: $status"
    echo "   Last 5 log lines:"
    sudo journalctl -u $service -n 5 --no-pager | sed 's/^/   /'
  fi
done
echo

# Check if ports are listening
echo "Checking network ports..."
echo "------------------------"
for service in "${!services[@]}"; do
  port="${services[$service]}"
  if netstat -tuln | grep -q ":$port "; then
    echo "✅ Port $port (for $service): LISTENING"
  else
    echo "❌ Port $port (for $service): NOT LISTENING"
  fi
done
echo

# Check virtual environments
echo "Checking virtual environments..."
echo "-------------------------------"
for service in "${!services[@]}"; do
  env_path="${env_paths[$service]}"
  if [ -d "$env_path" ]; then
    if [ -f "$env_path/bin/python" ]; then
      echo "✅ Environment for $service: EXISTS"
      # Check Python version and key packages
      echo "   Python: $($env_path/bin/python --version 2>&1)"
      if [ "$service" = "padel-api" ] || [ "$service" = "yolo11-service" ] || [ "$service" = "yolov8-service" ]; then
        echo "   PyTorch: $($env_path/bin/python -c 'import torch; print(f"v{torch.__version__}")' 2>/dev/null || echo "Not installed")"
        echo "   Ultralytics: $($env_path/bin/python -c 'import ultralytics; print(f"v{ultralytics.__version__}")' 2>/dev/null || echo "Not installed")"
      elif [ "$service" = "mmpose-service" ]; then
        echo "   PyTorch: $($env_path/bin/python -c 'import torch; print(f"v{torch.__version__}")' 2>/dev/null || echo "Not installed")"
        echo "   MMCV: $($env_path/bin/python -c 'import mmcv; print(f"v{mmcv.__version__}")' 2>/dev/null || echo "Not installed")"
      elif [ "$service" = "yolo-nas-service" ]; then
        echo "   PyTorch: $($env_path/bin/python -c 'import torch; print(f"v{torch.__version__}")' 2>/dev/null || echo "Not installed")"
        echo "   Super-Gradients: $($env_path/bin/python -c 'import super_gradients; print("Installed")' 2>/dev/null || echo "Not installed")"
      fi
    else
      echo "❌ Environment for $service: INCOMPLETE (no Python interpreter)"
    fi
  else
    echo "❌ Environment for $service: NOT FOUND at $env_path"
  fi
done
echo

# Check API endpoints
echo "Checking API endpoints..."
echo "------------------------"
# Only check if curl is available
if command -v curl &> /dev/null; then
  # Check main API health endpoint
  health_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/healthz 2>/dev/null || echo "Failed")
  if [ "$health_response" = "200" ]; then
    echo "✅ Main API health endpoint: RESPONDING (200 OK)"
  else
    echo "❌ Main API health endpoint: NOT RESPONDING (got $health_response)"
  fi
else
  echo "⚠️ curl not available, skipping API endpoint checks"
fi
echo

# Summary
echo "Health Check Summary"
echo "-------------------"
active_count=$(systemctl is-active padel-api yolo11-service yolov8-service mmpose-service yolo-nas-service | grep -c "active")
echo "$active_count of 5 services are active"

if [ $active_count -eq 5 ]; then
  echo "✅ All services are running correctly!"
else
  echo "❌ Some services are not running correctly. See details above."
  echo "   Possible solutions:"
  echo "   1. Check logs with: sudo journalctl -u [service-name] -n 50"
  echo "   2. Restart services with: sudo systemctl restart [service-name]"
  echo "   3. Verify environment setup with: bash /opt/padel/scripts/setup-envs.sh"
fi