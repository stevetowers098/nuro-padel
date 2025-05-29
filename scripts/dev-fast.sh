#!/bin/bash
# Fast Development Script - 1-2 minute builds for code/requirements changes
# Run this during testing phase for rapid iteration

set -e

echo "⚡ Fast Development Deployment for NuroPadel"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if base image exists
print_status "Checking for base image..."
if docker image inspect ghcr.io/stevetowers098/nuro-padel/base:latest >/dev/null 2>&1; then
    print_success "Base image found! Fast builds will take 1-2 minutes."
else
    print_warning "Base image not found. This will trigger a one-time 30-minute build."
    print_warning "After this, all future builds will be fast (1-2 minutes)."
    echo
    read -p "Continue with base image build? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Aborted. Run 'docker pull ghcr.io/stevetowers098/nuro-padel/base:latest' to get pre-built base image."
        exit 1
    fi
fi

echo
print_status "Starting fast development deployment..."

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose -f deployment/docker-compose.dev.yml down 2>/dev/null || true

# Build development images (should be fast!)
print_status "Building development images (fast build using base image)..."
start_time=$(date +%s)

docker-compose -f deployment/docker-compose.dev.yml build

end_time=$(date +%s)
build_duration=$((end_time - start_time))

print_success "Build completed in ${build_duration} seconds!"

# Start services
print_status "Starting development services..."
docker-compose -f deployment/docker-compose.dev.yml up -d

# Wait for services to be ready
print_status "Waiting for services to be healthy..."
sleep 30

# Health checks
print_status "Checking service health..."
services=(
    "http://localhost:8001/healthz|YOLO Combined"
    "http://localhost:8003/healthz|MMPose" 
    "http://localhost:8004/healthz|YOLO-NAS"
    "http://localhost:8080/healthz|Nginx"
)

all_healthy=true
for service in "${services[@]}"; do
    IFS='|' read -r url name <<< "$service"
    if curl -f -s "$url" >/dev/null 2>&1; then
        print_success "$name service is healthy ✅"
    else
        print_error "$name service is not responding ❌"
        all_healthy=false
    fi
done

echo
if [ "$all_healthy" = true ]; then
    print_success "🎉 Fast development deployment successful!"
    echo
    echo "📊 Development Services:"
    echo "  • YOLO Combined: http://localhost:8001"
    echo "  • MMPose:        http://localhost:8003"  
    echo "  • YOLO-NAS:      http://localhost:8004"
    echo "  • Nginx (API):   http://localhost:8080"
    echo
    echo "⚡ Next code/requirements changes will build in 1-2 minutes!"
    echo "🔄 To rebuild after changes: ./dev-fast.sh"
    echo "🛑 To stop: docker-compose -f deployment/docker-compose.dev.yml down"
else
    print_error "Some services are unhealthy. Check logs:"
    echo "  docker-compose -f deployment/docker-compose.dev.yml logs"
fi

echo
print_status "Build time: ${build_duration} seconds (vs 30 minutes with full rebuild)"