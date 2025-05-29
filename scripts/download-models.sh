#!/bin/bash

# NuroPadel Model Download Script
# Downloads all required AI models for services

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WEIGHTS_DIR="./weights"
TEMP_DIR="/tmp/nuro-padel-models"

# Model URLs and filenames
declare -A YOLO_MODELS=(
    ["yolo11n-pose.pt"]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n-pose.pt"
    ["yolov8m.pt"]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    ["yolov8n-pose.pt"]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt"
)

declare -A MMPOSE_MODELS=(
    ["rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"]="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
)

# Logging functions
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

# Create directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p "$WEIGHTS_DIR"
    mkdir -p "$TEMP_DIR"
    
    success "Directories created"
}

# Check if model exists and is valid
check_model_exists() {
    local model_path="$1"
    local expected_min_size="$2"  # in MB
    
    if [ -f "$model_path" ]; then
        local size_mb=$(stat -c%s "$model_path" | awk '{print int($1/1024/1024)}')
        if [ "$size_mb" -ge "$expected_min_size" ]; then
            return 0  # Model exists and is valid
        else
            warning "Model $model_path exists but is too small (${size_mb}MB < ${expected_min_size}MB)"
            return 1  # Model exists but invalid
        fi
    fi
    return 1  # Model doesn't exist
}

# Download single model with retry logic
download_model() {
    local url="$1"
    local filename="$2"
    local expected_size="$3"  # in MB
    local max_retries=3
    
    local temp_file="$TEMP_DIR/$filename"
    local final_path="$WEIGHTS_DIR/$filename"
    
    # Check if model already exists and is valid
    if check_model_exists "$final_path" "$expected_size"; then
        success "$filename already exists and is valid"
        return 0
    fi
    
    log "Downloading $filename..."
    
    for attempt in $(seq 1 $max_retries); do
        log "Attempt $attempt/$max_retries for $filename"
        
        # Download with curl (with resume capability)
        if curl -L --progress-bar --retry 3 --retry-delay 2 \
           -o "$temp_file" "$url"; then
            
            # Verify download
            if check_model_exists "$temp_file" "$expected_size"; then
                mv "$temp_file" "$final_path"
                success "$filename downloaded successfully"
                return 0
            else
                error "Downloaded file $filename is corrupted or incomplete"
                rm -f "$temp_file"
            fi
        else
            error "Download failed for $filename (attempt $attempt)"
        fi
        
        if [ $attempt -lt $max_retries ]; then
            log "Retrying in 5 seconds..."
            sleep 5
        fi
    done
    
    error "Failed to download $filename after $max_retries attempts"
    return 1
}

# Download YOLO models
download_yolo_models() {
    log "Downloading YOLO models..."
    
    local failed_downloads=0
    
    for model in "${!YOLO_MODELS[@]}"; do
        if ! download_model "${YOLO_MODELS[$model]}" "$model" 5; then
            ((failed_downloads++))
        fi
    done
    
    if [ $failed_downloads -eq 0 ]; then
        success "All YOLO models downloaded successfully"
    else
        error "$failed_downloads YOLO model(s) failed to download"
        return 1
    fi
}

# Download MMPose models
download_mmpose_models() {
    log "Downloading MMPose models..."
    
    local failed_downloads=0
    
    for model in "${!MMPOSE_MODELS[@]}"; do
        if ! download_model "${MMPOSE_MODELS[$model]}" "$model" 100; then
            ((failed_downloads++))
        fi
    done
    
    if [ $failed_downloads -eq 0 ]; then
        success "All MMPose models downloaded successfully"
    else
        error "$failed_downloads MMPose model(s) failed to download"
        return 1
    fi
}

# Verify all models
verify_models() {
    log "Verifying all models..."
    
    local total_models=0
    local valid_models=0
    
    # Check YOLO models
    for model in "${!YOLO_MODELS[@]}"; do
        ((total_models++))
        if check_model_exists "$WEIGHTS_DIR/$model" 5; then
            ((valid_models++))
            success "✓ $model"
        else
            error "✗ $model (missing or invalid)"
        fi
    done
    
    # Check MMPose models
    for model in "${!MMPOSE_MODELS[@]}"; do
        ((total_models++))
        if check_model_exists "$WEIGHTS_DIR/$model" 100; then
            ((valid_models++))
            success "✓ $model"
        else
            error "✗ $model (missing or invalid)"
        fi
    done
    
    log "Model verification: $valid_models/$total_models valid"
    
    if [ $valid_models -eq $total_models ]; then
        success "All models verified successfully!"
        return 0
    else
        error "Some models are missing or invalid"
        return 1
    fi
}

# Cleanup temporary files
cleanup() {
    log "Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
    success "Cleanup completed"
}

# Show disk usage
show_disk_usage() {
    log "Model directory disk usage:"
    du -sh "$WEIGHTS_DIR"/* 2>/dev/null || echo "No models found"
    log "Total weights directory size:"
    du -sh "$WEIGHTS_DIR"
}

# Main execution
main() {
    case "${1:-all}" in
        yolo)
            setup_directories
            download_yolo_models
            verify_models
            ;;
        mmpose)
            setup_directories
            download_mmpose_models
            verify_models
            ;;
        all)
            setup_directories
            download_yolo_models
            download_mmpose_models
            verify_models
            show_disk_usage
            ;;
        verify)
            verify_models
            show_disk_usage
            ;;
        clean)
            log "Removing all downloaded models..."
            rm -rf "$WEIGHTS_DIR"/*
            success "All models removed"
            ;;
        --help|help)
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  all      Download all models (default)"
            echo "  yolo     Download YOLO models only"
            echo "  mmpose   Download MMPose models only"
            echo "  verify   Verify existing models"
            echo "  clean    Remove all models"
            echo "  help     Show this help"
            ;;
        *)
            error "Unknown command: $1"
            echo "Use '$0 help' for available commands"
            exit 1
            ;;
    esac
}

# Trap for cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"