#!/bin/bash
# Smart Development Script - Only rebuilds changed services
# Detects file changes and rebuilds only affected services

set -e

echo "🧠 Smart Development Deployment for NuroPadel"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_smart() { echo -e "${PURPLE}[SMART]${NC} $1"; }

# Create checksum directory if it doesn't exist
mkdir -p .dev-checksums

# Function to calculate directory checksum
calc_checksum() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        find "$dir" -type f \( -name "*.py" -o -name "*.txt" -o -name "Dockerfile*" \) -exec md5sum {} \; 2>/dev/null | sort | md5sum | cut -d' ' -f1
    else
        echo "missing"
    fi
}

# Service directories
declare -A SERVICES=(
    ["yolo-combined"]="yolo-combined-service"
    ["mmpose"]="mmpose-service" 
    ["yolo-nas"]="yolo-nas-service"
)

# Check what needs rebuilding
declare -A NEEDS_REBUILD=()
changed_services=()

print_status "🔍 Analyzing file changes..."

for service in "${!SERVICES[@]}"; do
    dir="${SERVICES[$service]}"
    current_checksum=$(calc_checksum "$dir")
    checksum_file=".dev-checksums/${service}.md5"
    
    if [[ -f "$checksum_file" ]]; then
        stored_checksum=$(cat "$checksum_file")
        if [[ "$current_checksum" != "$stored_checksum" ]]; then
            NEEDS_REBUILD["$service"]=true
            changed_services+=("$service")
            print_smart "📝 $service has changes - will rebuild"
        else
            print_smart "✅ $service unchanged - keeping running"
        fi
    else
        NEEDS_REBUILD["$service"]=true
        changed_services+=("$service")
        print_smart "🆕 $service is new - will build"
    fi
done

# If no changes detected, offer manual override
if [[ ${#changed_services[@]} -eq 0 ]]; then
    print_success "🎉 No changes detected! All services are up to date."
    echo
    read -p "Force rebuild anyway? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        for service in "${!SERVICES[@]}"; do
            NEEDS_REBUILD["$service"]=true
            changed_services+=("$service")
        done
        print_warning "🔄 Force rebuild enabled for all services"
    else
        print_success "✨ Development environment is ready!"
        exit 0
    fi
fi

echo
print_status "📦 Services to rebuild: ${changed_services[*]}"
print_status "⚡ Services to keep running: $(printf '%s ' $(for s in "${!SERVICES[@]}"; do [[ -z "${NEEDS_REBUILD[$s]}" ]] && echo "$s"; done))"

# Start services that aren't running
print_status "🚀 Ensuring base services are running..."
docker-compose -f docker-compose.dev.yml up -d nginx 2>/dev/null || true

# Rebuild only changed services
total_start_time=$(date +%s)

for service in "${changed_services[@]}"; do
    print_status "🔨 Rebuilding $service..."
    service_start_time=$(date +%s)
    
    # Stop only this service
    docker-compose -f docker-compose.dev.yml stop "$service" 2>/dev/null || true
    
    # Build only this service
    docker-compose -f docker-compose.dev.yml build "$service"
    
    # Start this service
    docker-compose -f docker-compose.dev.yml up -d "$service"
    
    service_end_time=$(date +%s)
    service_duration=$((service_end_time - service_start_time))
    
    print_success "✅ $service rebuilt in ${service_duration}s"
    
    # Update checksum
    dir="${SERVICES[$service]}"
    current_checksum=$(calc_checksum "$dir")
    echo "$current_checksum" > ".dev-checksums/${service}.md5"
done

total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))

# Wait for rebuilt services to be ready
if [[ ${#changed_services[@]} -gt 0 ]]; then
    print_status "⏳ Waiting for rebuilt services to be healthy..."
    sleep 15
    
    # Health check only rebuilt services
    for service in "${changed_services[@]}"; do
        case "$service" in
            "yolo-combined") url="http://localhost:8001/healthz" ;;
            "mmpose") url="http://localhost:8003/healthz" ;;
            "yolo-nas") url="http://localhost:8004/healthz" ;;
        esac
        
        if curl -f -s "$url" >/dev/null 2>&1; then
            print_success "✅ $service is healthy"
        else
            print_error "❌ $service is not responding"
        fi
    done
fi

echo
print_success "🎉 Smart deployment complete!"
echo
echo "📊 Development Services:"
echo "  • YOLO Combined: http://localhost:8001"
echo "  • MMPose:        http://localhost:8003"  
echo "  • YOLO-NAS:      http://localhost:8004"
echo "  • Nginx (API):   http://localhost:8080"
echo
print_smart "⚡ Total rebuild time: ${total_duration}s (only changed services)"
print_smart "🔄 Run './dev-smart.sh' again to rebuild only new changes"
print_smart "🎯 Use './dev-service.sh <service>' to rebuild specific service"
print_smart "👀 Use './dev-watch.sh' for automatic rebuilds on file changes"