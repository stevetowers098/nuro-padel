#!/bin/bash

# SSH Connectivity Test Script for Padel-AI VM
# Helps diagnose and fix SSH connection issues

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Configuration
VM_INSTANCE="padel-ai"
VM_ZONE="australia-southeast1-a"
SSH_USERNAMES=("Towers" "towers" "ubuntu" "user")

echo "🔍 SSH Connectivity Test for Padel-AI VM"
echo "=========================================="

# Test 1: Get VM status and IP
log "1. Checking VM status and external IP..."
if ! command -v gcloud &> /dev/null; then
    error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

VM_STATUS=$(gcloud compute instances describe $VM_INSTANCE --zone=$VM_ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
if [ "$VM_STATUS" = "NOT_FOUND" ]; then
    error "VM instance '$VM_INSTANCE' not found in zone '$VM_ZONE'"
    exit 1
fi

VM_EXTERNAL_IP=$(gcloud compute instances describe $VM_INSTANCE --zone=$VM_ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "NO_IP")

log "VM Status: $VM_STATUS"
log "VM External IP: $VM_EXTERNAL_IP"

if [ "$VM_STATUS" != "RUNNING" ]; then
    warning "VM is not running. Starting VM..."
    gcloud compute instances start $VM_INSTANCE --zone=$VM_ZONE
    log "Waiting for VM to start..."
    sleep 30
    VM_STATUS=$(gcloud compute instances describe $VM_INSTANCE --zone=$VM_ZONE --format="value(status)")
    log "VM Status after start: $VM_STATUS"
fi

# Test 2: Basic network connectivity
log "2. Testing basic network connectivity..."
if ping -c 3 $VM_EXTERNAL_IP >/dev/null 2>&1; then
    success "VM is reachable via ping"
else
    error "VM is not reachable via ping - network issue"
    exit 1
fi

# Test 3: SSH port accessibility
log "3. Testing SSH port accessibility..."
if timeout 5 bash -c "echo >/dev/tcp/$VM_EXTERNAL_IP/22" 2>/dev/null; then
    success "SSH port 22 is accessible"
else
    error "SSH port 22 is not accessible - firewall issue"
    
    log "Checking VM firewall tags..."
    VM_TAGS=$(gcloud compute instances describe $VM_INSTANCE --zone=$VM_ZONE --format="value(tags.items[])" 2>/dev/null || echo "no-tags")
    log "VM Tags: $VM_TAGS"
    
    log "Checking firewall rules for SSH..."
    gcloud compute firewall-rules list --filter="direction=INGRESS AND allowed.ports:22" --format="table(name,direction,priority,sourceRanges.list():label=SRC_RANGES,targetTags.list():label=TARGET_TAGS)"
    
    error "Fix firewall rules to allow SSH access"
    exit 1
fi

# Test 4: SSH key and username testing
log "4. Testing SSH authentication..."

# Check if SSH key is provided via environment variable or file
SSH_KEY_FILE=""
if [ -n "${VM_SSH_KEY:-}" ]; then
    log "Using SSH key from VM_SSH_KEY environment variable"
    echo "$VM_SSH_KEY" > /tmp/test_ssh_key
    chmod 600 /tmp/test_ssh_key
    SSH_KEY_FILE="/tmp/test_ssh_key"
elif [ -f "$HOME/.ssh/id_rsa" ]; then
    log "Using default SSH key: $HOME/.ssh/id_rsa"
    SSH_KEY_FILE="$HOME/.ssh/id_rsa"
elif [ -f "$HOME/.ssh/google_compute_engine" ]; then
    log "Using Google Compute Engine SSH key"
    SSH_KEY_FILE="$HOME/.ssh/google_compute_engine"
else
    error "No SSH key found. Set VM_SSH_KEY environment variable or ensure SSH key exists"
    exit 1
fi

# Test different usernames
WORKING_USERNAME=""
for username in "${SSH_USERNAMES[@]}"; do
    log "Testing SSH with username: $username"
    if timeout 10 ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY_FILE" "$username@$VM_EXTERNAL_IP" "echo 'SSH test successful'" 2>/dev/null; then
        success "SSH authentication successful with username: $username"
        WORKING_USERNAME="$username"
        break
    else
        warning "SSH failed with username: $username"
    fi
done

if [ -z "$WORKING_USERNAME" ]; then
    error "SSH authentication failed with all usernames: ${SSH_USERNAMES[*]}"
    
    log "Trying verbose SSH for debugging..."
    ssh -vvv -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY_FILE" "Towers@$VM_EXTERNAL_IP" "echo 'test'" 2>&1 | tail -20
    
    # Clean up
    [ -f "/tmp/test_ssh_key" ] && rm -f /tmp/test_ssh_key
    exit 1
fi

# Test 5: Verify VM setup
log "5. Verifying VM setup..."
ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i "$SSH_KEY_FILE" "$WORKING_USERNAME@$VM_EXTERNAL_IP" << 'EOF'
echo "🔍 VM System Information:"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Docker: $(docker --version 2>/dev/null || echo 'Not installed')"
echo "Docker Compose: $(docker-compose --version 2>/dev/null || echo 'Not installed')"
echo "Disk Space: $(df -h / | tail -1 | awk '{print $4}') available"
echo "Memory: $(free -h | grep Mem | awk '{print $7}') available"

echo ""
echo "🔍 Checking deployment directory:"
if [ -d "/opt/padel-docker" ]; then
    echo "✅ /opt/padel-docker exists"
    ls -la /opt/padel-docker/ 2>/dev/null || echo "Directory is empty"
else
    echo "❌ /opt/padel-docker does not exist"
fi

echo ""
echo "🔍 Checking running containers:"
docker ps 2>/dev/null || echo "Cannot access Docker (permission issue or not installed)"

echo ""
echo "🔍 Checking SSH service:"
systemctl is-active ssh 2>/dev/null || echo "SSH service status unknown"
EOF

# Test 6: Generate GitHub Actions secrets update
log "6. Generating GitHub Actions secrets recommendations..."
echo ""
echo "📋 GITHUB ACTIONS SECRETS RECOMMENDATIONS:"
echo "=========================================="
echo ""
echo "Based on the test results, update your GitHub repository secrets:"
echo ""
echo "1. VM_HOST should be set to: $VM_EXTERNAL_IP"
echo "2. Working SSH username is: $WORKING_USERNAME"
echo ""
echo "GitHub Actions workflow should use:"
echo "  - host: \${{ secrets.VM_HOST }}  # Set to $VM_EXTERNAL_IP"
echo "  - username: $WORKING_USERNAME"
echo "  - key: \${{ secrets.VM_SSH_KEY }}  # Ensure this matches your SSH private key"
echo ""

# Clean up
[ -f "/tmp/test_ssh_key" ] && rm -f /tmp/test_ssh_key

success "SSH connectivity test completed successfully!"
echo ""
echo "🎉 SUMMARY:"
echo "  ✅ VM is running and accessible"
echo "  ✅ SSH port 22 is open"
echo "  ✅ SSH authentication works with username: $WORKING_USERNAME"
echo "  ✅ VM IP address: $VM_EXTERNAL_IP"
echo ""
echo "💡 Next steps:"
echo "  1. Update GitHub secrets with the values above"
echo "  2. Re-run the deployment pipeline"
echo "  3. Monitor deployment logs for any remaining issues"