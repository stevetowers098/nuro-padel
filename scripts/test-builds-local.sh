#!/bin/bash

# Quick Local Testing Script for Network Connectivity Fixes
# Tests Docker builds locally before pushing to GitHub (saves 30+ minutes)

set -e

echo "🧪 NURO-PADEL: Quick Local Build Testing"
echo "========================================"
echo "Testing network connectivity fixes locally before GitHub push..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log file
LOG_FILE="test-build-$(date +%Y%m%d-%H%M%S).log"
echo "📝 Logging to: $LOG_FILE"

# Function to log with timestamp
log() {
    echo "[$(date +'%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to test Docker buildx
test_buildx() {
    log "🔧 Testing Docker Buildx setup..."
    
    if ! docker buildx version &>/dev/null; then
        echo -e "${RED}❌ Docker Buildx not available${NC}"
        log "ERROR: Docker Buildx not found"
        return 1
    fi
    
    # Create/use buildx builder
    docker buildx create --name test-builder --use &>/dev/null || docker buildx use test-builder &>/dev/null || true
    
    echo -e "${GREEN}✅ Docker Buildx ready${NC}"
    log "SUCCESS: Docker Buildx setup complete"
    return 0
}

# Function to test network fixes in Dockerfile
test_network_fixes() {
    local service=$1
    log "🌐 Testing network fixes for $service..."
    
    # Create a test Dockerfile that only tests the network portion
    cat > "test-${service}.Dockerfile" << EOF
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Test the network fix
RUN sed -i 's|http://archive.ubuntu.com|http://azure.archive.ubuntu.com|g' /etc/apt/sources.list && \\
    apt-get update && \\
    apt-get install -y --no-install-recommends software-properties-common && \\
    echo "✅ Network fix working for $service"
EOF

    # Try to build just this test layer
    if docker buildx build -f "test-${service}.Dockerfile" --progress=plain . 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✅ Network fix working for $service${NC}"
        log "SUCCESS: Network connectivity test passed for $service"
        rm "test-${service}.Dockerfile"
        return 0
    else
        echo -e "${RED}❌ Network fix failed for $service${NC}"
        log "ERROR: Network connectivity test failed for $service"
        rm "test-${service}.Dockerfile"
        return 1
    fi
}

# Function to test specific service build (first few layers only)
test_service_build() {
    local service=$1
    log "🏗️ Testing $service build (network layers only)..."
    
    # Create a minimal test version that stops after network setup
    local dockerfile="services/$service/Dockerfile"
    
    if [ ! -f "$dockerfile" ]; then
        echo -e "${RED}❌ Dockerfile not found: $dockerfile${NC}"
        return 1
    fi
    
    # Extract just the network-related layers for testing
    head -20 "$dockerfile" > "test-build-${service}.Dockerfile"
    echo 'RUN echo "✅ Network setup successful for '"$service"'"' >> "test-build-${service}.Dockerfile"
    
    if docker buildx build -f "test-build-${service}.Dockerfile" --progress=plain "services/$service" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✅ $service network setup working${NC}"
        log "SUCCESS: $service network layers built successfully"
        rm "test-build-${service}.Dockerfile"
        return 0
    else
        echo -e "${RED}❌ $service network setup failed${NC}"
        log "ERROR: $service network layers failed"
        rm "test-build-${service}.Dockerfile"
        return 1
    fi
}

# Function to validate mirror speed
test_mirror_speed() {
    log "⚡ Testing mirror speed comparison..."
    
    echo "Testing archive.ubuntu.com speed:"
    timeout 10 curl -s -w "%{time_total}s\n" -o /dev/null http://archive.ubuntu.com/ubuntu/ls-lR.gz 2>/dev/null || echo "TIMEOUT/FAILED"
    
    echo "Testing azure.archive.ubuntu.com speed:"
    timeout 10 curl -s -w "%{time_total}s\n" -o /dev/null http://azure.archive.ubuntu.com/ubuntu/ls-lR.gz 2>/dev/null || echo "TIMEOUT/FAILED"
    
    log "Mirror speed test completed"
}

# Main execution
main() {
    log "Starting local build testing..."
    
    # Test 1: Docker Buildx
    if ! test_buildx; then
        echo -e "${RED}❌ Docker Buildx test failed - cannot continue${NC}"
        exit 1
    fi
    
    # Test 2: Mirror speed
    test_mirror_speed
    
    # Test 3: Network fixes for each service
    services=("yolo-combined" "mmpose" "yolo-nas")
    failed_services=()
    
    for service in "${services[@]}"; do
        echo ""
        if test_service_build "$service"; then
            echo -e "${GREEN}✅ $service: PASSED${NC}"
        else
            echo -e "${RED}❌ $service: FAILED${NC}"
            failed_services+=("$service")
        fi
    done
    
    # Summary
    echo ""
    echo "🎯 TESTING SUMMARY"
    echo "=================="
    log "Testing completed"
    
    if [ ${#failed_services[@]} -eq 0 ]; then
        echo -e "${GREEN}✅ ALL TESTS PASSED${NC}"
        echo "🚀 Ready to push to GitHub!"
        log "SUCCESS: All tests passed - ready for GitHub push"
    else
        echo -e "${RED}❌ FAILED SERVICES: ${failed_services[*]}${NC}"
        echo "🔧 Fix issues before pushing to GitHub"
        log "ERROR: Some services failed - review before GitHub push"
        exit 1
    fi
    
    echo ""
    echo "📝 Full log saved to: $LOG_FILE"
    echo "⏱️  Local testing completed in ~2-3 minutes vs 30+ minutes on GitHub"
}

# Run if executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi