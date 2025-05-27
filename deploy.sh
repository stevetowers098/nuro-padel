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
    
    # Enable BuildKit for better caching
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    success "Prerequisites check completed"
}

# Check if service needs rebuilding
needs_rebuild() {
    local service=$1
    local service_dir="${service}-service"
    
    # Check if image exists
    if ! docker images "${REGISTRY}/${service}:latest" --format "table {{.Repository}}" | grep -q "${REGISTRY}/${service}"; then
        log "${service}: No existing image found - build required"
        return 0
    fi
    
    # Check for file changes since last build
    local dockerfile_path="${service_dir}/Dockerfile"
    local requirements_path="${service_dir}/requirements.txt"
    local main_path="${service_dir}/main.py"
    local utils_path="${service_dir}/utils/"
    
    # Get image creation time
    local image_created=$(docker inspect "${REGISTRY}/${service}:latest" --format '{{.Created}}' 2>/dev/null)
    if [ -z "$image_created" ]; then
        log "${service}: Could not get image creation time - build required"
        return 0
    fi
    
    local image_timestamp=$(date -d "$image_created" +%s 2>/dev/null || echo "0")
    
    # Check if any source files are newer than the image
    for file_path in "$dockerfile_path" "$requirements_path" "$main_path"; do
        if [ -f "$file_path" ]; then
            local file_timestamp=$(stat -c %Y "$file_path" 2>/dev/null || echo "999999999")
            if [ "$file_timestamp" -gt "$image_timestamp" ]; then
                log "${service}: ${file_path} modified since last build - rebuild required"
                return 0
            fi
        fi
    done
    
    # Check utils directory
    if [ -d "$utils_path" ]; then
        local newest_util=$(find "$utils_path" -type f -name "*.py" -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
        if [ -n "$newest_util" ] && [ "${newest_util%.*}" -gt "$image_timestamp" ]; then
            log "${service}: Utils directory modified since last build - rebuild required"
            return 0
        fi
    fi
    
    log "${service}: No changes detected - using cached image"
    return 1
}

# Build individual service
build_service() {
    local service=$1
    
    # Check if rebuild is needed
    if ! needs_rebuild "$service"; then
        success "${service} service: Using cached image (no changes detected)"
        return 0
    fi
    
    log "Building ${service} service..."
    
    cd "${service}-service"
    
    # Build with BuildKit and advanced caching
    DOCKER_BUILDKIT=1 docker build \
        --tag "${REGISTRY}/${service}:latest" \
        --tag "${REGISTRY}/${service}:$(date +%Y%m%d-%H%M%S)" \
        --cache-from "${REGISTRY}/${service}:latest" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        .
    
    cd ..
    success "${service} service built successfully"
}

# Build all services with smart caching
build_all_optimized() {
    log "Building Docker services with smart change detection..."
    
    local build_count=0
    local cache_count=0
    
    for service in "${SERVICES[@]}"; do
        if needs_rebuild "$service"; then
            build_service "$service"
            ((build_count++))
        else
            success "${service} service: Using cached image"
            ((cache_count++))
        fi
    done
    
    log "Build summary: ${build_count} rebuilt, ${cache_count} cached"
    success "All services processed successfully"
}

# Build all services (legacy method)
build_all() {
    log "Building all Docker services (force rebuild)..."
    
    for service in "${SERVICES[@]}"; do
        build_service "$service"
    done
    
    success "All services built successfully"
}

# Deploy with smart updates
deploy_smart() {
    log "Deploying with smart change detection..."
    
    # Check if docker-compose.yml changed
    local compose_changed=false
    if [ -f "docker-compose.yml" ]; then
        local compose_running=$(docker-compose ps -q 2>/dev/null | wc -l)
        if [ "$compose_running" -eq 0 ]; then
            log "No running containers detected - full deployment required"
            compose_changed=true
        else
            # Check if compose file was modified
            local running_services=$(docker-compose ps --services)
            for service in $running_services; do
                local container_created=$(docker inspect "nuro-padel-${service}" --format '{{.Created}}' 2>/dev/null)
                if [ -n "$container_created" ]; then
                    local container_timestamp=$(date -d "$container_created" +%s 2>/dev/null || echo "0")
                    local compose_timestamp=$(stat -c %Y "docker-compose.yml" 2>/dev/null || echo "999999999")
                    if [ "$compose_timestamp" -gt "$container_timestamp" ]; then
                        log "docker-compose.yml modified - deployment update required"
                        compose_changed=true
                        break
                    fi
                fi
            done
        fi
    else
        compose_changed=true
    fi
    
    if [ "$compose_changed" = true ]; then
        deploy_production
    else
        # Rolling update for changed services only
        log "Performing rolling update for changed services..."
        docker-compose up -d --remove-orphans
        success "Smart deployment completed"
    fi
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
    
    # Clean previous deployment safely (only project files)
    ssh "$VM_HOST" "cd $VM_PATH && find . -maxdepth 1 -name '*.md' -o -name '*.yml' -o -name '*.sh' -o -name '*.conf' -o -name '*-service' | xargs rm -rf 2>/dev/null || true"
    
    # Sync all project files to VM (simplified and comprehensive)
    rsync -avz \
        --exclude='*.git*' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='node_modules' \
        --exclude='.pytest_cache' \
        --exclude='.vscode' \
        --exclude='*.log' \
        ./ "$VM_HOST:$VM_PATH/"
    
    # Make deploy script executable
    ssh "$VM_HOST" "chmod +x $VM_PATH/deploy.sh"
    
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
    echo "Smart Options (with change detection):"
    echo "  --build              Build services (skip unchanged)"
    echo "  --deploy             Deploy with smart updates"
    echo "  --all                Smart build, test, and deploy"
    echo ""
    echo "Force Options (rebuild everything):"
    echo "  --build-force        Force rebuild all services"
    echo "  --deploy-force       Force full redeployment"
    echo "  --all-force          Force rebuild and redeploy"
    echo ""
    echo "Other Options:"
    echo "  --test               Test all services locally"
    echo "  --vm                 Deploy to VM"
    echo "  --cleanup            Clean up test containers and images"
    echo "  --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --build            # Smart build (detects changes)"
    echo "  $0 --build-force      # Force rebuild all"
    echo "  $0 --all              # Smart full deployment"
    echo "  $0 --all-force        # Force full deployment"
    echo "  $0 --vm               # Deploy to VM"
}

# Main execution
main() {
    case "${1:-}" in
        --build)
            check_prerequisites
            build_all_optimized
            ;;
        --build-force)
            check_prerequisites
            build_all
            ;;
        --test)
            check_prerequisites
            test_all
            ;;
        --deploy|--production)
            check_prerequisites
            deploy_smart
            ;;
        --deploy-force)
            check_prerequisites
            deploy_production
            ;;
        --vm)
            deploy_vm
            ;;
        --all)
            check_prerequisites
            build_all_optimized
            test_all
            deploy_smart
            ;;
        --all-force)
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