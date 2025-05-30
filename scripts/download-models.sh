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

# Model URLs and filenames - Updated with working YOLO11 URLs
declare -A YOLO_MODELS=(
    # YOLO 11 models - Now working with v8.3.0 release
    ["yolo11n.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    ["yolo11n-pose.pt"]="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
    # Keep YOLO v8 as fallback
    ["yolov8n.pt"]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    ["yolov8n-pose.pt"]="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt"
)

declare -A MMPOSE_MODELS=(
    ["rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"]="https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
)

# ViTPose++ models for new service - Optimized for efficiency
declare -A VITPOSE_MODELS=(
    ["vitpose_base_coco_256x192.pth"]="https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
)

# RF-DETR models (downloaded via Python during runtime)
declare -A RF_DETR_MODELS=(
    ["rf_detr_base.pth"]="python_runtime"
)

# TrackNet models - V2 available, V4 pending release
declare -A TRACKNET_MODELS=(
    ["tracknet_v2.pth"]="https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view"
)

# YOLO-NAS models (manually placed, verification only)
declare -A YOLO_NAS_MODELS=(
    ["yolo_nas_pose_n_coco_pose.pth"]="super-gradients"
    ["yolo_nas_s_coco.pth"]="super-gradients"
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

# Diagnostic function to validate assumptions
show_diagnostic_info() {
    log "=== DIAGNOSTIC INFORMATION ==="
    
    log "Configured YOLO models (will go to ultralytics/):"
    for model in "${!YOLO_MODELS[@]}"; do
        echo "  - $model"
    done
    
    log "Configured MMPose models (will go to mmpose/):"
    for model in "${!MMPOSE_MODELS[@]}"; do
        echo "  - $model"
    done
    
    log "Configured ViTPose++ models (will go to vitpose/):"
    for model in "${!VITPOSE_MODELS[@]}"; do
        echo "  - $model"
    done
    
    log "Configured RF-DETR models (downloaded at runtime):"
    for model in "${!RF_DETR_MODELS[@]}"; do
        echo "  - $model (source: ${RF_DETR_MODELS[$model]})"
    done
    
    log "TrackNet models configured: ${#TRACKNET_MODELS[@]}"
    if [ ${#TRACKNET_MODELS[@]} -gt 0 ]; then
        for model in "${!TRACKNET_MODELS[@]}"; do
            echo "  - $model"
        done
    fi
    
    log "YOLO-NAS models (manual verification only):"
    for model in "${!YOLO_NAS_MODELS[@]}"; do
        echo "  - $model (expected in ${YOLO_NAS_MODELS[$model]}/)"
    done
    
    log "Directory structure that will be created:"
    echo "  ./weights/ultralytics/     - YOLO models"
    echo "  ./weights/mmpose/          - MMPose models"
    echo "  ./weights/vitpose/         - ViTPose++ models"
    echo "  ./weights/rf-detr/         - RF-DETR models"
    echo "  ./weights/tracknet/        - TrackNet models"
    echo "  ./weights/super-gradients/ - YOLO-NAS models"
    
    log "=== END DIAGNOSTIC ==="
}

# Create directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p "$WEIGHTS_DIR"
    mkdir -p "$WEIGHTS_DIR/ultralytics"
    mkdir -p "$WEIGHTS_DIR/mmpose"
    mkdir -p "$WEIGHTS_DIR/vitpose"
    mkdir -p "$WEIGHTS_DIR/rf-detr"
    mkdir -p "$WEIGHTS_DIR/tracknet"
    mkdir -p "$WEIGHTS_DIR/super-gradients"
    mkdir -p "$TEMP_DIR"
    
    success "Directories created with organized subdirectories"
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

# Download Google Drive file
download_google_drive() {
    local url="$1"
    local filename="$2"
    local model_type_path="$3"
    local expected_size="$4"
    
    local final_base_dir="$WEIGHTS_DIR/$model_type_path"
    mkdir -p "$final_base_dir"
    local final_path="$final_base_dir/$filename"
    
    # Check if model already exists and is valid
    if check_model_exists "$final_path" "$expected_size"; then
        success "$filename already exists and is valid"
        return 0
    fi
    
    # Extract file ID from Google Drive URL
    local file_id=""
    if [[ "$url" =~ drive\.google\.com/file/d/([a-zA-Z0-9_-]+) ]]; then
        file_id="${BASH_REMATCH[1]}"
    else
        error "Invalid Google Drive URL format: $url"
        return 1
    fi
    
    log "Downloading $filename from Google Drive (ID: $file_id)..."
    
    # Use gdown if available, otherwise provide manual instructions
    if command -v gdown >/dev/null 2>&1; then
        if gdown "https://drive.google.com/uc?id=$file_id" -O "$final_path"; then
            if check_model_exists "$final_path" "$expected_size"; then
                success "$filename downloaded successfully"
                return 0
            else
                error "Downloaded file $filename is corrupted or incomplete"
                rm -f "$final_path"
                return 1
            fi
        else
            error "gdown failed to download $filename"
            return 1
        fi
    else
        warning "gdown not available. Please manually download TrackNet V2:"
        warning "1. Visit: $url"
        warning "2. Download the file"
        warning "3. Save as: $final_path"
        warning "4. Install gdown for automatic downloads: pip install gdown"
        return 1
    fi
}

# Download single model with retry logic
download_model() {
    local url="$1"
    local filename="$2"
    local model_type_path="$3"  # New argument for subdir, e.g., "ultralytics"
    local expected_size="$4"  # Shifted expected_size argument
    local max_retries=3
    
    # Handle Google Drive URLs differently
    if [[ "$url" =~ drive\.google\.com ]]; then
        return $(download_google_drive "$url" "$filename" "$model_type_path" "$expected_size")
    fi
    
    local final_base_dir="$WEIGHTS_DIR/$model_type_path"
    mkdir -p "$final_base_dir"
    local temp_file="$TEMP_DIR/$filename"
    local final_path="$final_base_dir/$filename"
    
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
    log "Downloading YOLO models to ultralytics subdirectory..."
    
    local failed_downloads=0
    
    for model in "${!YOLO_MODELS[@]}"; do
        # Determine expected size based on model type
        local expected_size=5  # Default for nano models (5.9MB actual)
        case "$model" in
            "yolo11n-pose.pt") expected_size=5 ;;  # Actually downloads as 5MB
            *"yolo11n"*) expected_size=5 ;;  # YOLO11n is 5.9MB, accept 5MB+
            *"yolov8n"*) expected_size=6 ;;
            *) expected_size=10 ;;  # For larger models if any
        esac
        
        if ! download_model "${YOLO_MODELS[$model]}" "$model" "ultralytics" "$expected_size"; then
            ((failed_downloads++))
        fi
    done
    
    if [ $failed_downloads -eq 0 ]; then
        success "All YOLO models downloaded successfully to ultralytics/"
    else
        error "$failed_downloads YOLO model(s) failed to download"
        return 1
    fi
}

# Download MMPose models
download_mmpose_models() {
    log "Downloading MMPose models to mmpose subdirectory..."
    
    local failed_downloads=0
    
    for model in "${!MMPOSE_MODELS[@]}"; do
        # Updated expected size from 100MB to ~50MB as recommended
        if ! download_model "${MMPOSE_MODELS[$model]}" "$model" "mmpose" 50; then
            ((failed_downloads++))
        fi
    done
    
    if [ $failed_downloads -eq 0 ]; then
        success "All MMPose models downloaded successfully to mmpose/"
    else
        error "$failed_downloads MMPose model(s) failed to download"
        return 1
    fi
}

# Download ViTPose++ models - Optimized for efficiency
download_vitpose_models() {
    log "Downloading efficient ViTPose++ models to vitpose subdirectory..."
    
    local failed_downloads=0
    
    for model in "${!VITPOSE_MODELS[@]}"; do
        # ViTPose Base models are ~200MB - most efficient option
        if ! download_model "${VITPOSE_MODELS[$model]}" "$model" "vitpose" 180; then
            ((failed_downloads++))
        fi
    done
    
    if [ $failed_downloads -eq 0 ]; then
        success "All efficient ViTPose++ models downloaded successfully to vitpose/"
    else
        error "$failed_downloads ViTPose++ model(s) failed to download"
        return 1
    fi
}

# Download RF-DETR models (placeholder - actual download happens at runtime)
download_rf_detr_models() {
    log "Setting up RF-DETR model directory..."
    
    mkdir -p "$WEIGHTS_DIR/rf-detr"
    
    # Create a placeholder file to indicate the directory is ready
    echo "RF-DETR models will be downloaded automatically at runtime via Python" > "$WEIGHTS_DIR/rf-detr/README.txt"
    echo "The RF-DETR service uses rfdetr==0.1.0 which downloads models on first use" >> "$WEIGHTS_DIR/rf-detr/README.txt"
    
    success "RF-DETR model directory prepared (models download at runtime)"
}

# Download TrackNet models (if URLs available)
download_tracknet_models() {
    log "Downloading TrackNet models to tracknet subdirectory..."
    
    if [ ${#TRACKNET_MODELS[@]} -eq 0 ]; then
        warning "No TrackNet model URLs configured - skipping TrackNet downloads"
        warning "Please manually place tracknet_v2.pth in ./weights/tracknet/ if available"
        return 0
    fi
    
    local failed_downloads=0
    
    for model in "${!TRACKNET_MODELS[@]}"; do
        if ! download_model "${TRACKNET_MODELS[$model]}" "$model" "tracknet" 20; then
            ((failed_downloads++))
        fi
    done
    
    if [ $failed_downloads -eq 0 ]; then
        success "All TrackNet models downloaded successfully to tracknet/"
    else
        error "$failed_downloads TrackNet model(s) failed to download"
        return 1
    fi
}

# Download YOLO-NAS models using Python script
download_yolo_nas_models() {
    log "Downloading YOLO-NAS models using Python downloader..."
    
    # Check if Python script exists
    if [ ! -f "scripts/download-yolo-nas.py" ]; then
        error "YOLO-NAS download script not found: scripts/download-yolo-nas.py"
        return 1
    fi
    
    # Try to run the Python script
    if python3 scripts/download-yolo-nas.py; then
        success "YOLO-NAS models downloaded successfully via Python script"
        return 0
    else
        warning "YOLO-NAS Python download failed - this is often due to DNS issues with sghub.deci.ai"
        warning "YOLO-NAS services will fall back to online download during startup"
        return 1
    fi
}

# Verify YOLO-NAS models
verify_yolo_nas_models() {
    log "Verifying YOLO-NAS models..."
    
    # Use Python script for verification
    if [ -f "scripts/download-yolo-nas.py" ]; then
        if python3 scripts/download-yolo-nas.py --verify-only; then
            success "YOLO-NAS models verified successfully"
            return 0
        else
            warning "YOLO-NAS model verification failed"
            return 1
        fi
    else
        # Fallback to bash verification
        local total_models=0
        local valid_models=0
        
        for model in "${!YOLO_NAS_MODELS[@]}"; do
            ((total_models++))
            local subdir="${YOLO_NAS_MODELS[$model]}"
            local model_path="$WEIGHTS_DIR/$subdir/$model"
            
            if check_model_exists "$model_path" 10; then
                ((valid_models++))
                success "✓ $model (in $subdir/)"
            else
                warning "✗ $model (missing from $subdir/)"
            fi
        done
        
        log "YOLO-NAS verification: $valid_models/$total_models found"
        
        if [ $valid_models -gt 0 ]; then
            success "Some YOLO-NAS models are available"
            return 0
        else
            warning "No YOLO-NAS models found"
            return 1
        fi
    fi
}

# Verify ViTPose++ models
verify_vitpose_models() {
    log "Verifying ViTPose++ models..."
    
    local total_models=0
    local valid_models=0
    
    for model in "${!VITPOSE_MODELS[@]}"; do
        ((total_models++))
        if check_model_exists "$WEIGHTS_DIR/vitpose/$model" 150; then
            ((valid_models++))
            success "✓ vitpose/$model"
        else
            error "✗ vitpose/$model (missing or invalid)"
        fi
    done
    
    log "ViTPose++ verification: $valid_models/$total_models found"
    
    if [ $valid_models -eq $total_models ]; then
        success "All ViTPose++ models verified successfully"
        return 0
    else
        warning "Some ViTPose++ models are missing"
        return 1
    fi
}

# Verify RF-DETR models (check directory setup)
verify_rf_detr_models() {
    log "Verifying RF-DETR model setup..."
    
    if [ -d "$WEIGHTS_DIR/rf-detr" ] && [ -f "$WEIGHTS_DIR/rf-detr/README.txt" ]; then
        success "✓ RF-DETR model directory is ready"
        return 0
    else
        warning "✗ RF-DETR model directory not set up"
        return 1
    fi
}

# Verify all models
verify_models() {
    log "Verifying all models in organized subdirectories..."
    
    local total_models=0
    local valid_models=0
    
    # Check YOLO models in ultralytics subdirectory
    for model in "${!YOLO_MODELS[@]}"; do
        ((total_models++))
        local expected_size=6
        case "$model" in
            "yolo11n-pose.pt") expected_size=5 ;;  # Actually downloads as 5MB
            *"yolo11n"*) expected_size=6 ;;
            *"yolov8n"*) expected_size=6 ;;
            *) expected_size=10 ;;
        esac
        
        if check_model_exists "$WEIGHTS_DIR/ultralytics/$model" "$expected_size"; then
            ((valid_models++))
            success "✓ ultralytics/$model"
        else
            error "✗ ultralytics/$model (missing or invalid)"
        fi
    done
    
    # Check MMPose models in mmpose subdirectory
    for model in "${!MMPOSE_MODELS[@]}"; do
        ((total_models++))
        if check_model_exists "$WEIGHTS_DIR/mmpose/$model" 50; then
            ((valid_models++))
            success "✓ mmpose/$model"
        else
            error "✗ mmpose/$model (missing or invalid)"
        fi
    done
    
    # Check ViTPose++ models in vitpose subdirectory
    for model in "${!VITPOSE_MODELS[@]}"; do
        ((total_models++))
        if check_model_exists "$WEIGHTS_DIR/vitpose/$model" 150; then
            ((valid_models++))
            success "✓ vitpose/$model"
        else
            error "✗ vitpose/$model (missing or invalid)"
        fi
    done
    
    # Check TrackNet models in tracknet subdirectory
    for model in "${!TRACKNET_MODELS[@]}"; do
        ((total_models++))
        if check_model_exists "$WEIGHTS_DIR/tracknet/$model" 20; then
            ((valid_models++))
            success "✓ tracknet/$model"
        else
            error "✗ tracknet/$model (missing or invalid)"
        fi
    done
    
    log "Downloaded model verification: $valid_models/$total_models valid"
    
    # Also verify other models
    verify_vitpose_models
    verify_rf_detr_models
    verify_yolo_nas_models
    
    if [ $valid_models -eq $total_models ]; then
        success "All downloadable models verified successfully!"
        return 0
    else
        error "Some downloadable models are missing or invalid"
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
    log "Model directory disk usage by subdirectory:"
    for subdir in ultralytics mmpose vitpose rf-detr tracknet super-gradients; do
        if [ -d "$WEIGHTS_DIR/$subdir" ]; then
            echo "  $subdir/: $(du -sh "$WEIGHTS_DIR/$subdir" 2>/dev/null | cut -f1)"
        fi
    done
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
        vitpose)
            setup_directories
            download_vitpose_models
            verify_models
            ;;
        rf-detr)
            setup_directories
            download_rf_detr_models
            verify_models
            ;;
        tracknet)
            setup_directories
            download_tracknet_models
            verify_models
            ;;
        all)
            setup_directories
            download_yolo_models
            download_mmpose_models
            download_vitpose_models
            download_rf_detr_models
            download_tracknet_models
            verify_models
            show_disk_usage
            ;;
        verify)
            verify_models
            show_disk_usage
            ;;
        yolo-nas)
            setup_directories
            download_yolo_nas_models
            verify_yolo_nas_models
            ;;
        yolo-nas-verify)
            verify_yolo_nas_models
            ;;
        diagnose|diagnostic)
            show_diagnostic_info
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
            echo "  all        Download all available models (default)"
            echo "  yolo       Download YOLO models only (to ultralytics/)"
            echo "  mmpose     Download MMPose models only (to mmpose/)"
            echo "  vitpose    Download ViTPose++ models only (to vitpose/)"
            echo "  rf-detr    Set up RF-DETR model directory (models download at runtime)"
            echo "  tracknet   Download TrackNet models only (to tracknet/)"
            echo "  yolo-nas         Download YOLO-NAS models using Python script (to super-gradients/)"
            echo "  yolo-nas-verify  Verify existing YOLO-NAS models only"
            echo "  verify     Verify all existing models"
            echo "  diagnose   Show diagnostic information about configured models"
            echo "  clean      Remove all models from all subdirectories"
            echo "  help       Show this help"
            echo ""
            echo "Models are organized into subdirectories:"
            echo "  ./weights/ultralytics/     - YOLO models"
            echo "  ./weights/mmpose/          - MMPose models"
            echo "  ./weights/vitpose/         - ViTPose++ models"
            echo "  ./weights/rf-detr/         - RF-DETR models (downloaded at runtime)"
            echo "  ./weights/tracknet/        - TrackNet models"
            echo "  ./weights/super-gradients/ - YOLO-NAS models (super-gradients format)"
            echo ""
            echo "YOLO-NAS Special Notes:"
            echo "  - Uses Python script (scripts/download-yolo-nas.py) for downloads"
            echo "  - May fail due to DNS issues with sghub.deci.ai"
            echo "  - Services will fall back to online download if models missing"
            echo "  - Try different network/VPN if downloads fail"
            echo ""
            echo "TrackNet Notes:"
            echo "  - Currently: TrackNet V2 (Google Drive download)"
            echo "  - Future: TrackNet V4 upgrade path ready (plug-and-play)"
            echo "  - Enhanced YOLO ball tracking may outperform basic TrackNet V2"
            echo "  - Install gdown for automatic downloads: pip install gdown"
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