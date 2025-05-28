#!/bin/bash
# Code-Only Deployment - Instant changes without rebuilds
# For Python code changes that don't require new dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_code() { echo -e "${CYAN}[CODE]${NC} $1"; }

echo "⚡ Code-Only Deployment - Instant Changes"
echo "========================================"

print_status "🔍 Checking if services are running..."

# Check if base services exist
if ! docker-compose -f docker-compose.dev.yml ps | grep -q "Up"; then
    print_warning "⚠️  No services running. Starting base deployment first..."
    ./dev-smart.sh
    echo
fi

start_time=$(date +%s)

print_code "📝 Applying code-only changes (no rebuild required)..."

# Apply the code overlay configuration
docker-compose -f docker-compose.dev.yml -f docker-compose.code.yml up -d

# No need to rebuild - just restart services to pick up code changes
print_code "🔄 Restarting services to pick up code changes..."

# Restart services one by one to minimize downtime
services=("yolo-combined" "mmpose" "yolo-nas")

for service in "${services[@]}"; do
    print_code "♻️  Restarting $service..."
    docker-compose -f docker-compose.dev.yml -f docker-compose.code.yml restart "$service"
    sleep 2
done

end_time=$(date +%s)
duration=$((end_time - start_time))

# Quick health check
print_status "🩺 Quick health check..."
sleep 5

services_config=(
    "yolo-combined:8001"
    "mmpose:8003"
    "yolo-nas:8004"
)

all_healthy=true
for service_config in "${services_config[@]}"; do
    IFS=':' read -r service port <<< "$service_config"
    url="http://localhost:${port}/healthz"
    
    if curl -f -s "$url" >/dev/null 2>&1; then
        print_success "✅ $service is healthy"
    else
        print_warning "⚠️  $service may need a moment to restart"
        all_healthy=false
    fi
done

echo
if [ "$all_healthy" = true ]; then
    print_success "🎉 Code deployment complete!"
else
    print_warning "⏳ Some services are still restarting (this is normal)"
fi

print_code "⚡ Code deployment time: ${duration} seconds"

echo
print_code "💡 Code-Only Mode Features:"
echo "  • ✨ Instant Python code changes (no rebuild)"
echo "  • 🔄 Automatic file sync with running containers"
echo "  • ⚡ Restart only - no rebuild required"
echo "  • 🎯 Perfect for algorithm tweaks and bug fixes"

echo
print_status "🚀 Services ready:"
echo "  • YOLO Combined: http://localhost:8001"
echo "  • MMPose:        http://localhost:8003"
echo "  • YOLO-NAS:      http://localhost:8004"
echo "  • Nginx (API):   http://localhost:8080"

echo
print_code "⚠️  Note: For dependency changes, use './dev-smart.sh' or './dev-service.sh <service>'"