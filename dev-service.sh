#!/bin/bash
# Single Service Rebuild Script - Rebuild just one service in under 2 minutes
# Usage: ./dev-service.sh <service-name>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_service() { echo -e "${PURPLE}[SERVICE]${NC} $1"; }

# Usage function
usage() {
    echo "🎯 Single Service Rebuild Tool"
    echo "=============================="
    echo "Usage: $0 <service-name>"
    echo ""
    echo "Available services:"
    echo "  • yolo-combined  - YOLO11 + YOLOv8 service (port 8001)"
    echo "  • mmpose         - Biomechanics analysis (port 8003)" 
    echo "  • yolo-nas       - High-accuracy detection (port 8004)"
    echo "  • nginx          - Load balancer (port 8080)"
    echo ""
    echo "Examples:"
    echo "  $0 yolo-combined    # Rebuild only YOLO Combined"
    echo "  $0 mmpose          # Rebuild only MMPose"
    echo "  $0 all             # Rebuild all services"
    exit 1
}

# Validate input
if [[ $# -eq 0 ]]; then
    usage
fi

SERVICE="$1"

# Service validation and configuration
declare -A SERVICE_CONFIG=(
    ["yolo-combined"]="yolo-combined-service:8001:/healthz"
    ["mmpose"]="mmpose-service:8003:/healthz" 
    ["yolo-nas"]="yolo-nas-service:8004:/healthz"
    ["nginx"]=".:8080:/healthz"
)

if [[ "$SERVICE" == "all" ]]; then
    print_status "🔄 Rebuilding ALL services..."
    exec ./dev-fast.sh
elif [[ -z "${SERVICE_CONFIG[$SERVICE]}" ]]; then
    print_error "❌ Unknown service: $SERVICE"
    usage
fi

# Parse service configuration
IFS=':' read -r service_dir port health_path <<< "${SERVICE_CONFIG[$SERVICE]}"

echo "🎯 Single Service Rebuild: $SERVICE"
echo "=================================="

print_status "📁 Service directory: $service_dir"
print_status "🌐 Port: $port"

# Create checksum directory
mkdir -p .dev-checksums

start_time=$(date +%s)

# Stop the specific service
print_status "🛑 Stopping $SERVICE..."
docker-compose -f docker-compose.dev.yml stop "$SERVICE" 2>/dev/null || true

# Build the specific service
print_service "🔨 Building $SERVICE..."
docker-compose -f docker-compose.dev.yml build "$SERVICE"

# Start the specific service
print_service "🚀 Starting $SERVICE..."
docker-compose -f docker-compose.dev.yml up -d "$SERVICE"

# Update checksum for change tracking
if [[ "$service_dir" != "." ]]; then
    calc_checksum() {
        local dir="$1"
        find "$dir" -type f \( -name "*.py" -o -name "*.txt" -o -name "Dockerfile*" \) -exec md5sum {} \; 2>/dev/null | sort | md5sum | cut -d' ' -f1
    }
    
    current_checksum=$(calc_checksum "$service_dir")
    echo "$current_checksum" > ".dev-checksums/${SERVICE}.md5"
    print_status "💾 Updated change tracking for $SERVICE"
fi

# Wait for service to be ready
print_status "⏳ Waiting for $SERVICE to be healthy..."
sleep 10

# Health check
health_url="http://localhost:${port}${health_path}"
max_attempts=6
attempt=1

while [[ $attempt -le $max_attempts ]]; do
    if curl -f -s "$health_url" >/dev/null 2>&1; then
        print_success "✅ $SERVICE is healthy!"
        break
    else
        if [[ $attempt -eq $max_attempts ]]; then
            print_error "❌ $SERVICE failed to become healthy after $max_attempts attempts"
            print_error "🔍 Check logs: docker-compose -f docker-compose.dev.yml logs $SERVICE"
            exit 1
        else
            print_status "⏳ Attempt $attempt/$max_attempts - waiting 5s..."
            sleep 5
            ((attempt++))
        fi
    fi
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo
print_success "🎉 $SERVICE rebuilt successfully!"
print_service "⚡ Rebuild time: ${duration} seconds"
print_service "🌐 Service URL: http://localhost:$port"

if [[ "$SERVICE" != "nginx" ]]; then
    print_service "🩺 Health check: $health_url"
fi

echo
print_status "💡 Next steps:"
echo "  • Test your changes: curl http://localhost:$port$health_path"
echo "  • View logs: docker-compose -f docker-compose.dev.yml logs $SERVICE"
echo "  • Rebuild again: $0 $SERVICE"
echo "  • Smart rebuild all: ./dev-smart.sh"