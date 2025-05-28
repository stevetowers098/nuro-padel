#!/bin/bash
# File Watcher Script - Auto-rebuild services when files change
# Monitors service directories and rebuilds only what changed

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_watch() { echo -e "${CYAN}[WATCH]${NC} $1"; }

echo "👁️  File Watcher for NuroPadel Development"
echo "========================================="

# Check if inotify is available (Linux) or use polling fallback
if command -v inotifywait >/dev/null 2>&1; then
    WATCH_METHOD="inotify"
    print_success "✅ Using inotify for efficient file watching"
elif command -v fswatch >/dev/null 2>&1; then
    WATCH_METHOD="fswatch"
    print_success "✅ Using fswatch for file watching"
else
    WATCH_METHOD="polling"
    print_warning "⚠️  Using polling method (install inotify-tools for better performance)"
fi

# Service directories to watch
declare -A WATCH_DIRS=(
    ["yolo-combined"]="yolo-combined-service"
    ["mmpose"]="mmpose-service"
    ["yolo-nas"]="yolo-nas-service"
)

# Create checksum directory
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

# Initialize checksums
for service in "${!WATCH_DIRS[@]}"; do
    dir="${WATCH_DIRS[$service]}"
    checksum=$(calc_checksum "$dir")
    echo "$checksum" > ".dev-checksums/${service}.md5"
done

print_status "🚀 Starting initial deployment..."
./dev-smart.sh

echo
print_watch "👁️  Watching for file changes..."
print_watch "📁 Monitoring: ${!WATCH_DIRS[*]}"
print_watch "🛑 Press Ctrl+C to stop watching"

# Function to rebuild service
rebuild_service() {
    local service="$1"
    local timestamp=$(date '+%H:%M:%S')
    
    print_watch "🔄 [$timestamp] Rebuilding $service due to file changes..."
    
    if ./dev-service.sh "$service"; then
        print_success "✅ [$timestamp] $service rebuilt successfully!"
    else
        print_error "❌ [$timestamp] Failed to rebuild $service"
    fi
    
    echo
    print_watch "👁️  Continuing to watch for changes..."
}

# Function to check for changes (polling method)
check_changes() {
    for service in "${!WATCH_DIRS[@]}"; do
        dir="${WATCH_DIRS[$service]}"
        current_checksum=$(calc_checksum "$dir")
        stored_checksum_file=".dev-checksums/${service}.md5"
        
        if [[ -f "$stored_checksum_file" ]]; then
            stored_checksum=$(cat "$stored_checksum_file")
            if [[ "$current_checksum" != "$stored_checksum" ]]; then
                echo "$current_checksum" > "$stored_checksum_file"
                rebuild_service "$service"
            fi
        fi
    done
}

# Cleanup function
cleanup() {
    echo
    print_watch "🛑 Stopping file watcher..."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

case "$WATCH_METHOD" in
    "inotify")
        # Use inotify for efficient watching (Linux)
        while true; do
            for service in "${!WATCH_DIRS[@]}"; do
                dir="${WATCH_DIRS[$service]}"
                (
                    inotifywait -r -e modify,create,delete,move "$dir" \
                        --include '\.(py|txt|yml|yaml|json|toml|cfg|conf|sh|md)$|Dockerfile.*' \
                        --quiet --timeout 1 >/dev/null 2>&1 && {
                        current_checksum=$(calc_checksum "$dir")
                        stored_checksum=$(cat ".dev-checksums/${service}.md5" 2>/dev/null || echo "")
                        if [[ "$current_checksum" != "$stored_checksum" ]]; then
                            echo "$current_checksum" > ".dev-checksums/${service}.md5"
                            rebuild_service "$service"
                        fi
                    }
                ) &
            done
            wait
        done
        ;;
    "fswatch")
        # Use fswatch for macOS
        {
            for service in "${!WATCH_DIRS[@]}"; do
                dir="${WATCH_DIRS[$service]}"
                echo "$dir"
            done
        } | xargs fswatch -o --event Created --event Updated --event Removed | while read f; do
            # Small delay to avoid multiple rapid rebuilds
            sleep 2
            check_changes
        done
        ;;
    "polling")
        # Fallback polling method
        print_warning "📊 Polling every 3 seconds for changes..."
        while true; do
            check_changes
            sleep 3
        done
        ;;
esac