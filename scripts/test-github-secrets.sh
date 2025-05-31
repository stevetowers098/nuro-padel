#!/bin/bash

# GitHub Secrets Validation Script
# This script helps diagnose GitHub Actions deployment issues

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

echo "🔍 GitHub Actions Deployment Diagnostics"
echo "========================================"
echo ""

# Check if we're testing locally with environment variables
if [ -n "${VM_IP:-}" ] && [ -n "${VM_SSH_KEY:-}" ]; then
    log "Testing with local environment variables (simulating GitHub Actions)"
    
    # Clean and validate IP
    CLEAN_VM_IP=$(echo "$VM_IP" | tr -d '[:space:]')
    echo "🌐 VM_IP: $CLEAN_VM_IP"
    
    # Validate IP format
    if [[ ! "$CLEAN_VM_IP" =~ ^[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}$ ]]; then
        error "VM_IP format is invalid: '$CLEAN_VM_IP'"
        error "Expected format: XXX.XXX.XXX.XXX (e.g., 35.189.53.46)"
        exit 1
    else
        success "VM_IP format is valid"
    fi
    
    # Check SSH key format
    echo "🔑 SSH Key validation:"
    if echo "$VM_SSH_KEY" | head -1 | grep -q "BEGIN.*PRIVATE KEY"; then
        success "SSH key appears to be in correct format"
        echo "   First line: $(echo "$VM_SSH_KEY" | head -1)"
        echo "   Lines: $(echo "$VM_SSH_KEY" | wc -l)"
    else
        error "SSH key doesn't appear to be in correct format"
        echo "   First line: $(echo "$VM_SSH_KEY" | head -1)"
        warning "SSH private key should start with '-----BEGIN OPENSSH PRIVATE KEY-----' or similar"
    fi
    
    # Test SSH connection
    echo ""
    log "Testing SSH connection to VM..."
    
    # Setup temporary SSH
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    mkdir -p "$TEMP_DIR/.ssh"
    chmod 700 "$TEMP_DIR/.ssh"
    echo "$VM_SSH_KEY" > "$TEMP_DIR/.ssh/id_rsa"
    chmod 600 "$TEMP_DIR/.ssh/id_rsa"
    
    # Test ssh-keyscan
    if ssh-keyscan -H "$CLEAN_VM_IP" >> "$TEMP_DIR/.ssh/known_hosts" 2>/dev/null; then
        success "ssh-keyscan successful for $CLEAN_VM_IP"
    else
        warning "ssh-keyscan failed - will use fallback method in deployment"
    fi
    
    # Test SSH connection
    if ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no -i "$TEMP_DIR/.ssh/id_rsa" "Towers@$CLEAN_VM_IP" "echo 'SSH connection test successful'" 2>/dev/null; then
        success "SSH connection test successful!"
    else
        error "SSH connection test failed"
        log "This could indicate:"
        echo "  - Incorrect VM IP address"
        echo "  - SSH key doesn't match the public key on VM"
        echo "  - VM is not accessible"
        echo "  - User 'Towers' doesn't exist or has wrong permissions"
    fi
    
else
    log "Local environment test (no VM_IP/VM_SSH_KEY provided)"
    echo ""
    echo "To test GitHub secrets locally, run:"
    echo "  export VM_IP='your.vm.ip.address'"
    echo "  export VM_SSH_KEY='-----BEGIN OPENSSH PRIVATE KEY-----"
    echo "  ...your private key content..."
    echo "  -----END OPENSSH PRIVATE KEY-----'"
    echo "  ./scripts/test-github-secrets.sh"
fi

echo ""
echo "📋 GitHub Repository Secret Configuration Checklist:"
echo "======================================================"
echo ""
echo "1. Go to your GitHub repository"
echo "2. Navigate to: Settings > Secrets and variables > Actions"
echo "3. Ensure these secrets are configured:"
echo ""
echo "   VM_IP:"
echo "   - Name: VM_IP"
echo "   - Value: 35.189.53.46 (or your VM's IP address)"
echo "   - ✅ Should be exactly the IP, no spaces or extra characters"
echo ""
echo "   VM_SSH_KEY:"
echo "   - Name: VM_SSH_KEY"
echo "   - Value: Your private SSH key content (starts with -----BEGIN)"
echo "   - ✅ Should be the complete private key including header/footer"
echo "   - ✅ Should match the public key in /home/Towers/.ssh/authorized_keys on VM"
echo ""
echo "4. Test SSH access manually:"
echo "   ssh -i ~/.ssh/your_private_key Towers@35.189.53.46"
echo ""
echo "5. If SSH works manually but GitHub Actions fails:"
echo "   - Check the private key format (OpenSSH vs RSA)"
echo "   - Ensure no trailing spaces in secret values"
echo "   - Verify secret names are exactly: VM_IP and VM_SSH_KEY"
echo ""

echo "🔧 Troubleshooting Steps:"
echo "========================"
echo ""
echo "If deployment still fails:"
echo ""
echo "1. Check GitHub Actions logs for detailed error messages"
echo "2. Look for 'ssh-keyscan' errors or SSH connection failures"
echo "3. Verify VM is accessible: ping 35.189.53.46"
echo "4. Test SSH manually from your local machine"
echo "5. Check VM user permissions: ls -la /home/Towers/.ssh/"
echo "6. Ensure authorized_keys has correct permissions: chmod 600 /home/Towers/.ssh/authorized_keys"
echo ""

echo "✅ Next Steps:"
echo "=============="
echo ""
echo "1. Configure the GitHub secrets as shown above"
echo "2. Push changes to trigger GitHub Actions"
echo "3. Monitor the deployment logs for the improved error messages"
echo "4. The workflow now includes:"
echo "   - Secret validation"
echo "   - IP format checking"
echo "   - Robust SSH setup with fallbacks"
echo "   - Clear error messages"
echo ""

success "Diagnostic script completed!"