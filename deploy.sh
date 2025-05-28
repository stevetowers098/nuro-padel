ne#!/bin/bash

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
    
    # Check Docker Compose (v2 preferred, v1 fallback)
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        DOCKER_COMPOSE="docker compose"
        log "Using Docker Compose v2: $(docker compose version --short)"
    elif command -v docker-compose >/dev/null 2>&1; then
        DOCKER_COMPOSE="docker-compose"
        warning "Using legacy Docker Compose v1: $(docker-compose version --short)"
        warning "Consider upgrading to Docker Compose v2 for better performance"
    else
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
    
    # Enable BuildKit and advanced Docker optimizations for maximum speed
    export DOCKER_BUILDKIT=1
    export BUILDKIT_PROGRESS=plain
    export DOCKER_CLI_EXPERIMENTAL=enabled
    
    # Docker Compose v2 specific optimizations
    if [[ "$DOCKER_COMPOSE" == "docker compose" ]]; then
        export COMPOSE_DOCKER_CLI_BUILD=1
        export COMPOSE_BUILDKIT=1
        log "Docker Compose v2 optimizations enabled"
    else
        export COMPOSE_DOCKER_CLI_BUILD=1
        log "Docker Compose v1 compatibility mode"
    fi
    
    # Configure BuildKit for optimal caching
    docker buildx create --name nuro-builder --use --driver docker-container --driver-opt network=host 2>/dev/null || docker buildx use nuro-builder 2>/dev/null || true
    
    # Create cache directories
    mkdir -p /tmp/.buildx-cache-yolo-combined /tmp/.buildx-cache-mmpose /tmp/.buildx-cache-yolo-nas
    
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
    
    # Cross-platform date parsing (Linux vs macOS/BSD)
    local image_timestamp
    if date -d "$image_created" +%s 2>/dev/null; then
        image_timestamp=$(date -d "$image_created" +%s 2>/dev/null)
    elif date -j -f "%Y-%m-%dT%H:%M:%S" "$image_created" +%s 2>/dev/null; then
        image_timestamp=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$image_created" +%s 2>/dev/null)
    else
        image_timestamp=0
    fi
    
    # Check if any source files are newer than the image
    for file_path in "$dockerfile_path" "$requirements_path" "$main_path"; do
        if [ -f "$file_path" ]; then
            # Cross-platform file timestamp (Linux vs macOS/BSD)
            local file_timestamp
            if stat -c %Y "$file_path" 2>/dev/null; then
                file_timestamp=$(stat -c %Y "$file_path" 2>/dev/null)
            elif stat -f %m "$file_path" 2>/dev/null; then
                file_timestamp=$(stat -f %m "$file_path" 2>/dev/null)
            else
                file_timestamp=999999999
            fi
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
    
    # Try BuildKit first, fallback to regular docker build
    if command -v docker-buildx >/dev/null 2>&1 || docker buildx version >/dev/null 2>&1; then
        log "Using BuildKit for optimized caching..."
        docker buildx build \
            --builder nuro-builder \
            --tag "${REGISTRY}/${service}:latest" \
            --tag "${REGISTRY}/${service}:$(date +%Y%m%d-%H%M%S)" \
            --cache-from type=local,src=/tmp/.buildx-cache-${service} \
            --cache-to type=local,dest=/tmp/.buildx-cache-${service},mode=max \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --progress=plain \
            --load \
            . || {
                warning "BuildKit failed, falling back to regular docker build"
                DOCKER_BUILDKIT=1 docker build \
                    --tag "${REGISTRY}/${service}:latest" \
                    --tag "${REGISTRY}/${service}:$(date +%Y%m%d-%H%M%S)" \
                    --cache-from "${REGISTRY}/${service}:latest" \
                    --build-arg BUILDKIT_INLINE_CACHE=1 \
                    --progress=plain \
                    .
            }
    else
        log "BuildKit not available, using regular docker build..."
        DOCKER_BUILDKIT=1 docker build \
            --tag "${REGISTRY}/${service}:latest" \
            --tag "${REGISTRY}/${service}:$(date +%Y%m%d-%H%M%S)" \
            --cache-from "${REGISTRY}/${service}:latest" \
            --build-arg BUILDKIT_INLINE_CACHE=1 \
            --progress=plain \
            .
    fi
    
    cd ..
    success "${service} service built successfully"
}

# Build all services with smart caching and parallel processing
build_all_optimized() {
    log "Building Docker services with smart change detection and parallel processing..."
    
    local build_count=0
    local cache_count=0
    local services_to_build=()
    
    # Determine which services need rebuilding
    for service in "${SERVICES[@]}"; do
        if needs_rebuild "$service"; then
            services_to_build+=("$service")
            ((build_count++))
        else
            success "${service} service: Using cached image"
            ((cache_count++))
        fi
    done
    
    # Build services sequentially to avoid directory conflicts
    # Parallel building can cause issues with cd commands and shared state
    if [ ${#services_to_build[@]} -gt 0 ]; then
        log "Building ${#services_to_build[@]} service(s)..."
        for service in "${services_to_build[@]}"; do
            build_service "$service"
        done
    fi
    
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
        local compose_running=$($DOCKER_COMPOSE ps -q 2>/dev/null | wc -l)
        if [ "$compose_running" -eq 0 ]; then
            log "No running containers detected - full deployment required"
            compose_changed=true
        else
            # Check if compose file was modified
            local running_services=$($DOCKER_COMPOSE ps --services)
            for service in $running_services; do
                local container_created=$(docker inspect "nuro-padel-${service}" --format '{{.Created}}' 2>/dev/null)
                if [ -n "$container_created" ]; then
                    # Cross-platform date and stat commands
                    local container_timestamp
                    if date -d "$container_created" +%s 2>/dev/null; then
                        container_timestamp=$(date -d "$container_created" +%s 2>/dev/null)
                    elif date -j -f "%Y-%m-%dT%H:%M:%S" "$container_created" +%s 2>/dev/null; then
                        container_timestamp=$(date -j -f "%Y-%m-%dT%H:%M:%S" "$container_created" +%s 2>/dev/null)
                    else
                        container_timestamp=0
                    fi
                    
                    local compose_timestamp
                    if stat -c %Y "docker-compose.yml" 2>/dev/null; then
                        compose_timestamp=$(stat -c %Y "docker-compose.yml" 2>/dev/null)
                    elif stat -f %m "docker-compose.yml" 2>/dev/null; then
                        compose_timestamp=$(stat -f %m "docker-compose.yml" 2>/dev/null)
                    else
                        compose_timestamp=999999999
                    fi
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
        $DOCKER_COMPOSE up -d --remove-orphans
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
    # $DOCKER_COMPOSE pull
    
    # Deploy with rolling updates
    $DOCKER_COMPOSE up -d --remove-orphans
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy..."
    sleep 60
    
    # Check all services are healthy
    local services_healthy=true
    for service in "${SERVICES[@]}"; do
        if ! $DOCKER_COMPOSE ps "$service" | grep -q "healthy"; then
            error "${service} is not healthy"
            services_healthy=false
        fi
    done
    
    if [ "$services_healthy" = true ]; then
        success "Production deployment completed successfully"
        
        # Show running services
        $DOCKER_COMPOSE ps
        
        # Show service URLs
        log "Service URLs:"
        echo "  - YOLO Combined: http://localhost:8001/healthz"
        echo "  - MMPose:        http://localhost:8003/healthz"
        echo "  - YOLO-NAS:      http://localhost:8004/healthz"
        echo "  - Load Balancer: http://localhost:8080/healthz"
        
    else
        error "Some services are unhealthy - check logs with: $DOCKER_COMPOSE logs"
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