#!/bin/bash

# NuroPadel Docker Deployment Script
# Builds and deploys 3 AI services with zero-downtime strategy

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="nuro-padel"
SERVICES=("yolo-combined" "mmpose" "yolo-nas")
REGISTRY="nuro-padel"
VM_HOST="towers@35.189.53.46"
VM_PATH="/opt/padel-docker"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        warning "NVIDIA Docker runtime not detected - GPU acceleration may not work"
    fi
    
    # Check weights directory
    if [ ! -d "./weights" ]; then
        warning "Weights directory not found - creating empty directory"
        mkdir -p ./weights
        echo "Place your model weights in ./weights/ directory" > ./weights/README.md
    fi
    
    success "Prerequisites check completed"
}

# Build individual service
build_service() {
    local service=$1
    log "Building ${service} service..."
    
    cd "${service}-service"
    
    # Build with BuildKit for faster builds
    DOCKER_BUILDKIT=1 docker build \
        --tag "${REGISTRY}/${service}:latest" \
        --tag "${REGISTRY}/${service}:$(date +%Y%m%d-%H%M%S)" \
        --progress=plain \
        .
    
    cd ..
    success "${service} service built successfully"
}

# Build all services
build_all() {
    log "Building all Docker services..."
    
    for service in "${SERVICES[@]}"; do
        build_service "$service"
    done
    
    success "All services built successfully"
}

# Test service health
test_service() {
    local service=$1
    local port=$2
    
    log "Testing ${service} service on port ${port}..."
    
    # Start service
    docker run -d \
        --name "test-${service}" \
        --gpus all \
        -p "${port}:${port}" \
        "${REGISTRY}/${service}:latest"
    
    # Wait for service to start
    sleep 30
    
    # Test health endpoint
    local health_url="http://localhost:${port}/healthz"
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f "$health_url" &> /dev/null; then
            success "${service} health check passed"
            docker stop "test-${service}" && docker rm "test-${service}"
            return 0
        fi
        
        log "Health check attempt ${attempt}/${max_attempts} failed, retrying..."
        sleep 10
        ((attempt++))
    done
    
    error "${service} health check failed after ${max_attempts} attempts"
    docker logs "test-${service}"
    docker stop "test-${service}" && docker rm "test-${service}"
    return 1
}

# Test all services
test_all() {
    log "Testing all services..."
    
    test_service "yolo-combined" 8001
    test_service "mmpose" 8003
    test_service "yolo-nas" 8004
    
    success "All service tests passed"
}

# Deploy with zero downtime
deploy_production() {
    log "Deploying to production with zero downtime..."
    
    # Pull latest images if using registry
    # docker-compose pull
    
    # Deploy with rolling updates
    docker-compose up -d --remove-orphans
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 60
    
    # Check all services are healthy
    local services_healthy=true
    for service in "${SERVICES[@]}"; do
        if ! docker-compose ps "$service" | grep -q "healthy"; then
            error "${service} is not healthy"
            services_healthy=false
        fi
    done
    
    if [ "$services_healthy" = true ]; then
        success "Production deployment completed successfully"
        
        # Show running services
        docker-compose ps
        
        # Show service URLs
        log "Service URLs:"
        echo "  - YOLO Combined: http://localhost:8001/healthz"
        echo "  - MMPose:        http://localhost:8003/healthz"
        echo "  - YOLO-NAS:      http://localhost:8004/healthz"
        echo "  - Load Balancer: http://localhost/healthz"
        
    else
        error "Some services are unhealthy - check logs with: docker-compose logs"
        return 1
    fi
}

# Deploy to VM
deploy_vm() {
    log "Deploying to VM: ${VM_HOST}"
    
    # Create VM directory if it doesn't exist
    ssh "$VM_HOST" "mkdir -p $VM_PATH"
    
    # Sync project files to VM
    rsync -avz --delete \
        --exclude='*.git*' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='node_modules' \
        ./ "$VM_HOST:$VM_PATH/"
    
    # Run deployment on VM
    ssh "$VM_HOST" "cd $VM_PATH && bash deploy.sh --production"
    
    success "VM deployment completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up test containers..."
    docker ps -a --filter "name=test-" --format "table {{.Names}}" | tail -n +2 | xargs -r docker rm -f
    docker images --filter "dangling=true" -q | xargs -r docker rmi
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --build              Build all Docker services"
    echo "  --test               Test all services locally"
    echo "  --deploy             Deploy to production (local)"
    echo "  --vm                 Deploy to VM"
    echo "  --all                Build, test, and deploy"
    echo "  --cleanup            Clean up test containers and images"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --build"
    echo "  $0 --test"
    echo "  $0 --all"
    echo "  $0 --vm"
}

# Main execution
main() {
    case "${1:-}" in
        --build)
            check_prerequisites
            build_all
            ;;
        --test)
            check_prerequisites
            test_all
            ;;
        --deploy|--production)
            check_prerequisites
            deploy_production
            ;;
        --vm)
            deploy_vm
            ;;
        --all)
            check_prerequisites
            build_all
            test_all
            deploy_production
            ;;
        --cleanup)
            cleanup
            ;;
        --help|"")
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
}

# Trap for cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"