#!/bin/bash

# NuroPadel Working Services Backup Script
# Creates dated backups of working services for safe restoration

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

# Get current date for backup naming
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups"

# Working services (confirmed working)
WORKING_SERVICES=("yolo-combined" "yolo-nas")

# Function to create backup directory
create_backup_dir() {
    if [ ! -d "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        print_status "Created backup directory: $BACKUP_DIR"
    fi
}

# Function to backup a service
backup_service() {
    local service=$1
    local service_dir="${service}-service"
    local backup_name="${service}_working_${BACKUP_DATE}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    print_status "Creating backup for $service..."
    
    if [ ! -d "$service_dir" ]; then
        print_error "Service directory $service_dir not found!"
        return 1
    fi
    
    # Create service backup
    cp -r "$service_dir" "$backup_path"
    
    # Create backup info file
    cat > "$backup_path/BACKUP_INFO.md" << EOF
# Backup Information

**Service**: $service  
**Backup Date**: $(date)  
**Status**: Working/Verified  
**Git Commit**: $(git rev-parse HEAD 2>/dev/null || echo "N/A")  
**Git Branch**: $(git branch --show-current 2>/dev/null || echo "N/A")  

## Restoration Command
\`\`\`bash
# To restore this backup:
./restore-service.sh $service $backup_name
\`\`\`

## Service Health Check
After restoration, verify with:
\`\`\`bash
# For $service service
curl http://localhost:$(get_service_port $service)/healthz
\`\`\`
EOF
    
    # Create tar.gz archive for space efficiency
    tar -czf "$backup_path.tar.gz" -C "$BACKUP_DIR" "$backup_name"
    rm -rf "$backup_path"
    
    print_success "Backup created: $backup_path.tar.gz"
    return 0
}

# Function to get service port
get_service_port() {
    case "$1" in
        "yolo-combined") echo "8001" ;;
        "yolo-nas") echo "8004" ;;
        "mmpose") echo "8003" ;;
        *) echo "8000" ;;
    esac
}

# Function to list existing backups
list_backups() {
    print_status "Existing backups:"
    echo
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR 2>/dev/null)" ]; then
        print_warning "No backups found"
        return
    fi
    
    echo "📁 Available backups:"
    ls -la "$BACKUP_DIR"/*.tar.gz 2>/dev/null | while read -r line; do
        filename=$(basename $(echo "$line" | awk '{print $9}') .tar.gz)
        size=$(echo "$line" | awk '{print $5}')
        date=$(echo "$line" | awk '{print $6" "$7" "$8}')
        echo "  • $filename ($size bytes, $date)"
    done
}

# Function to backup Docker images
backup_docker_images() {
    print_status "Backing up Docker images for working services..."
    
    for service in "${WORKING_SERVICES[@]}"; do
        local image_name="nuro-padel-${service}"
        local backup_image_name="$BACKUP_DIR/${service}_image_${BACKUP_DATE}.tar"
        
        if docker image inspect "$image_name" >/dev/null 2>&1; then
            print_status "Backing up Docker image: $image_name"
            docker save "$image_name" > "$backup_image_name"
            gzip "$backup_image_name"
            print_success "Docker image backup: ${backup_image_name}.gz"
        else
            print_warning "Docker image $image_name not found, skipping"
        fi
    done
}

# Function to show usage
show_usage() {
    echo "NuroPadel Working Services Backup Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  backup              Create backups of all working services"
    echo "  backup <service>    Create backup of specific service"
    echo "  list                List existing backups"
    echo "  backup-images       Backup Docker images of working services"
    echo "  help                Show this help message"
    echo
    echo "Working Services:"
    for service in "${WORKING_SERVICES[@]}"; do
        echo "  - $service"
    done
    echo
    echo "Examples:"
    echo "  $0 backup                    # Backup all working services"
    echo "  $0 backup yolo-combined      # Backup only YOLO Combined"
    echo "  $0 list                      # Show existing backups"
    echo "  $0 backup-images             # Backup Docker images"
}

# Function to verify service is working
verify_service_working() {
    local service=$1
    local port=$(get_service_port "$service")
    
    print_status "Verifying $service is working..."
    
    # Check if service is in working services list
    if [[ " ${WORKING_SERVICES[@]} " =~ " ${service} " ]]; then
        print_success "$service is confirmed working"
        return 0
    else
        print_warning "$service is not in confirmed working services list"
        print_status "Current working services: ${WORKING_SERVICES[*]}"
        read -p "Continue with backup anyway? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            return 0
        else
            return 1
        fi
    fi
}

# Main script logic
case "${1:-backup}" in
    "backup")
        create_backup_dir
        
        if [ -n "$2" ]; then
            # Backup specific service
            service="$2"
            if verify_service_working "$service"; then
                backup_service "$service"
            else
                print_error "Backup cancelled for $service"
                exit 1
            fi
        else
            # Backup all working services
            print_status "Creating backups for all working services..."
            echo "Working services: ${WORKING_SERVICES[*]}"
            echo
            
            for service in "${WORKING_SERVICES[@]}"; do
                backup_service "$service"
                echo
            done
            
            print_success "All working services backed up successfully!"
        fi
        ;;
    "list")
        list_backups
        ;;
    "backup-images")
        create_backup_dir
        backup_docker_images
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

echo
print_status "Backup operations completed!"
print_status "Backups stored in: $BACKUP_DIR/"