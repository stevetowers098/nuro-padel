#!/bin/bash

# NuroPadel Resilient Deployment Script
# Deploy services independently - working services proceed even if others fail

set -e

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

# Function to check if a service is running
check_service() {
    local service=$1
    local port=$2
    
    if curl -f -s http://localhost:$port/healthz > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to deploy a single service
deploy_service() {
    local service=$1
    local port=$2
    
    print_status "Deploying $service service..."
    
    # Try to build and start the service
    if docker-compose -f docker-compose.resilient.yml --profile $service up -d --build 2>/dev/null; then
        
        # Wait for service to be healthy (max 3 minutes)
        print_status "Waiting for $service to be healthy (port $port)..."
        for i in {1..36}; do
            if check_service $service $port; then
                print_success "$service is running and healthy!"
                return 0
            fi
            sleep 5
            echo -n "."
        done
        
        print_warning "$service deployed but not responding on health check"
        return 1
    else
        print_error "Failed to deploy $service"
        return 1
    fi
}

# Function to show service status
show_status() {
    print_status "Checking service status..."
    echo
    
    # Check each service
    services=("yolo-combined:8001" "mmpose:8003" "yolo-nas:8004" "nginx:8080")
    
    for service_port in "${services[@]}"; do
        IFS=':' read -r service port <<< "$service_port"
        
        if [ "$service" == "nginx" ]; then
            # Special check for nginx
            if curl -f -s http://localhost:$port/health > /dev/null 2>&1; then
                echo -e "  ${GREEN}✓${NC} $service (port $port) - ${GREEN}HEALTHY${NC}"
            else
                echo -e "  ${RED}✗${NC} $service (port $port) - ${RED}DOWN${NC}"
            fi
        else
            if check_service $service $port; then
                echo -e "  ${GREEN}✓${NC} $service (port $port) - ${GREEN}HEALTHY${NC}"
            else
                echo -e "  ${RED}✗${NC} $service (port $port) - ${RED}DOWN${NC}"
            fi
        fi
    done
    
    echo
    print_status "Access the API at: http://localhost:8080"
    print_status "Check individual services:"
    echo "  - YOLO Combined: http://localhost:8080/yolo-combined/healthz"
    echo "  - MMPose: http://localhost:8080/mmpose/healthz" 
    echo "  - YOLO-NAS: http://localhost:8080/yolo-nas/healthz"
}

# Function to deploy all services with resilience
deploy_all_resilient() {
    print_status "Starting resilient deployment of all services..."
    echo
    
    success_count=0
    total_services=3
    
    # Deploy nginx first (no dependencies)
    print_status "Deploying nginx load balancer..."
    if docker-compose -f docker-compose.resilient.yml --profile nginx up -d --build 2>/dev/null; then
        print_success "Nginx deployed successfully"
    else
        print_error "Failed to deploy nginx - this may affect routing"
    fi
    
    echo
    
    # Deploy each AI service independently
    if deploy_service "yolo-combined" "8001"; then
        ((success_count++))
    fi
    
    echo
    
    if deploy_service "mmpose" "8003"; then
        ((success_count++))
    fi
    
    echo
    
    if deploy_service "yolo-nas" "8004"; then
        ((success_count++))
    fi
    
    echo
    print_status "Deployment Summary:"
    print_success "$success_count out of $total_services AI services deployed successfully"
    
    if [ $success_count -gt 0 ]; then
        print_success "At least one service is available for testing!"
        echo
        show_status
    else
        print_error "No services deployed successfully"
        return 1
    fi
}

# Function to stop all services
stop_all() {
    print_status "Stopping all services..."
    docker-compose -f docker-compose.resilient.yml --profile all down
    print_success "All services stopped"
}

# Function to show usage
show_usage() {
    echo "NuroPadel Resilient Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  all             Deploy all services (resilient mode)"
    echo "  yolo-combined   Deploy only YOLO Combined service"
    echo "  mmpose          Deploy only MMPose service"  
    echo "  yolo-nas        Deploy only YOLO-NAS service"
    echo "  nginx           Deploy only nginx load balancer"
    echo "  status          Check status of all services"
    echo "  stop            Stop all services"
    echo "  help            Show this help message"
    echo
    echo "Examples:"
    echo "  $0 all                    # Deploy all services, continue even if some fail"
    echo "  $0 yolo-combined          # Deploy only the working YOLO Combined service"
    echo "  $0 status                 # Check which services are running"
}

# Main script logic
case "${1:-all}" in
    "all")
        deploy_all_resilient
        ;;
    "yolo-combined")
        deploy_service "yolo-combined" "8001"
        deploy_service "nginx" "8080"  # Also deploy nginx for routing
        show_status
        ;;
    "mmpose")
        deploy_service "mmpose" "8003"
        deploy_service "nginx" "8080"  # Also deploy nginx for routing
        show_status
        ;;
    "yolo-nas")
        deploy_service "yolo-nas" "8004"
        deploy_service "nginx" "8080"  # Also deploy nginx for routing
        show_status
        ;;
    "nginx")
        print_status "Deploying nginx load balancer..."
        docker-compose -f docker-compose.resilient.yml --profile nginx up -d --build
        show_status
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_all
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_usage
        exit 1
        ;;
esac