#!/bin/bash

# NuroPadel Service Restoration Script
# Restores backed up services safely

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

BACKUP_DIR="backups"

# Function to list available backups
list_backups() {
    print_status "Available service backups:"
    echo
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR/*.tar.gz 2>/dev/null)" ]; then
        print_warning "No backups found in $BACKUP_DIR/"
        return 1
    fi
    
    local count=0
    for backup in "$BACKUP_DIR"/*.tar.gz; do
        if [ -f "$backup" ]; then
            count=$((count + 1))
            filename=$(basename "$backup" .tar.gz)
            size=$(du -h "$backup" | cut -f1)
            date=$(stat -c %y "$backup" 2>/dev/null || stat -f %Sm "$backup" 2>/dev/null || echo "Unknown")
            echo "  $count. $filename ($size, $date)"
        fi
    done
    
    if [ $count -eq 0 ]; then
        print_warning "No valid backup files found"
        return 1
    fi
    
    return 0
}

# Function to extract service and date from backup name
parse_backup_name() {
    local backup_name="$1"
    # Format: servicename_working_YYYYMMDD_HHMMSS
    local service=$(echo "$backup_name" | cut -d'_' -f1)
    local date_part=$(echo "$backup_name" | cut -d'_' -f3-4)
    echo "$service:$date_part"
}

# Function to restore a service
restore_service() {
    local service="$1"
    local backup_name="$2"
    local backup_file="$BACKUP_DIR/${backup_name}.tar.gz"
    local service_dir="${service}-service"
    local restore_temp="restore_temp_$$"
    
    # Validate inputs
    if [ -z "$service" ] || [ -z "$backup_name" ]; then
        print_error "Service and backup name are required"
        return 1
    fi
    
    if [ ! -f "$backup_file" ]; then
        print_error "Backup file not found: $backup_file"
        return 1
    fi
    
    print_status "Restoring $service from backup: $backup_name"
    
    # Create backup of current service (safety measure)
    if [ -d "$service_dir" ]; then
        local safety_backup="${service_dir}_before_restore_$(date +%Y%m%d_%H%M%S)"
        print_status "Creating safety backup of current service: $safety_backup"
        cp -r "$service_dir" "$BACKUP_DIR/$safety_backup"
        print_success "Safety backup created: $BACKUP_DIR/$safety_backup"
    fi
    
    # Extract backup to temporary location
    print_status "Extracting backup..."
    mkdir -p "$restore_temp"
    tar -xzf "$backup_file" -C "$restore_temp"
    
    # Find the extracted directory
    local extracted_dir=$(find "$restore_temp" -maxdepth 1 -type d -name "*working*" | head -1)
    if [ -z "$extracted_dir" ]; then
        print_error "Could not find extracted service directory"
        rm -rf "$restore_temp"
        return 1
    fi
    
    # Remove current service directory if it exists
    if [ -d "$service_dir" ]; then
        rm -rf "$service_dir"
    fi
    
    # Move extracted service to correct location
    mv "$extracted_dir" "$service_dir"
    rm -rf "$restore_temp"
    
    print_success "Service $service restored successfully!"
    
    # Show backup info if available
    if [ -f "$service_dir/BACKUP_INFO.md" ]; then
        echo
        print_status "Backup Information:"
        cat "$service_dir/BACKUP_INFO.md"
    fi
    
    return 0
}

# Function to show usage
show_usage() {
    echo "NuroPadel Service Restoration Script"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  list                    List available backups"
    echo "  restore <service> <backup_name>   Restore specific service"
    echo "  interactive             Interactive restoration mode"
    echo "  help                    Show this help message"
    echo
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 restore yolo-combined yolo-combined_working_20250528_221045"
    echo "  $0 interactive"
}

# Function for interactive restoration
interactive_restore() {
    print_status "Interactive Service Restoration"
    echo
    
    # List available backups
    if ! list_backups; then
        print_error "No backups available for restoration"
        return 1
    fi
    
    echo
    read -p "Enter the number of the backup to restore (or 'q' to quit): " choice
    
    if [ "$choice" = "q" ] || [ "$choice" = "Q" ]; then
        print_status "Restoration cancelled"
        return 0
    fi
    
    # Validate choice is a number
    if ! [[ "$choice" =~ ^[0-9]+$ ]]; then
        print_error "Invalid choice: $choice"
        return 1
    fi
    
    # Get the selected backup
    local count=0
    local selected_backup=""
    for backup in "$BACKUP_DIR"/*.tar.gz; do
        if [ -f "$backup" ]; then
            count=$((count + 1))
            if [ $count -eq $choice ]; then
                selected_backup=$(basename "$backup" .tar.gz)
                break
            fi
        fi
    done
    
    if [ -z "$selected_backup" ]; then
        print_error "Invalid backup selection: $choice"
        return 1
    fi
    
    # Parse service from backup name
    local service_info=$(parse_backup_name "$selected_backup")
    local service=$(echo "$service_info" | cut -d':' -f1)
    local backup_date=$(echo "$service_info" | cut -d':' -f2)
    
    print_status "Selected backup: $selected_backup"
    print_status "Service: $service"
    print_status "Backup date: $backup_date"
    echo
    
    # Confirm restoration
    read -p "Are you sure you want to restore $service from this backup? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        restore_service "$service" "$selected_backup"
    else
        print_status "Restoration cancelled"
    fi
}

# Main script logic
case "${1:-interactive}" in
    "list")
        list_backups
        ;;
    "restore")
        if [ -z "$2" ] || [ -z "$3" ]; then
            print_error "Usage: $0 restore <service> <backup_name>"
            echo
            show_usage
            exit 1
        fi
        restore_service "$2" "$3"
        ;;
    "interactive")
        interactive_restore
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