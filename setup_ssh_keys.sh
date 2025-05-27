#!/bin/bash
set -e

echo "🔑 NuroPadel SSH Key Setup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VM_HOST="35.189.53.46"
VM_USER="Towers"
KEY_NAME="nuro_padel_deploy"

echo -e "${BLUE}This script will:${NC}"
echo "1. Generate a new SSH key pair for GitHub Actions"
echo "2. Copy the public key to your VM"
echo "3. Test the connection"
echo "4. Show you what to add to GitHub Secrets"
echo ""

# Check if VM is accessible
echo -e "${YELLOW}Step 1: Testing VM accessibility...${NC}"
if ping -c 1 -W 5 $VM_HOST > /dev/null 2>&1; then
    echo -e "${GREEN}✅ VM is reachable at $VM_HOST${NC}"
else
    echo -e "${RED}❌ VM is not reachable at $VM_HOST${NC}"
    echo "Please check your network connection and VM status"
    exit 1
fi

# Generate new SSH key pair
echo -e "${YELLOW}Step 2: Generating new SSH key pair...${NC}"
if [ -f ~/.ssh/$KEY_NAME ]; then
    echo -e "${RED}Key ~/.ssh/$KEY_NAME already exists!${NC}"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting..."
        exit 1
    fi
    rm -f ~/.ssh/$KEY_NAME ~/.ssh/$KEY_NAME.pub
fi

ssh-keygen -t ed25519 -C "github-actions@nuro-padel" -f ~/.ssh/$KEY_NAME -N ""
echo -e "${GREEN}✅ SSH key pair generated${NC}"

# Test if we can connect to VM with existing credentials
echo -e "${YELLOW}Step 3: Testing current VM access...${NC}"
echo "Trying to connect to VM to install the new public key..."

# Try multiple connection methods
CONNECTED=false

# Method 1: Try with existing SSH keys
echo "Attempting connection with existing SSH keys..."
if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "echo 'Connected with existing keys'" 2>/dev/null; then
    CONNECTED=true
    echo -e "${GREEN}✅ Connected with existing SSH keys${NC}"
fi

# Method 2: Try with gcloud (if available)
if [ "$CONNECTED" = false ] && command -v gcloud >/dev/null 2>&1; then
    echo "Attempting connection with gcloud..."
    if gcloud compute ssh padel-ai --zone=australia-southeast1-a --project=surf-coach --command="echo 'Connected with gcloud'" 2>/dev/null; then
        CONNECTED=true
        echo -e "${GREEN}✅ Connected with gcloud${NC}"
        # Use gcloud for subsequent operations
        SSH_CMD="gcloud compute ssh padel-ai --zone=australia-southeast1-a --project=surf-coach --command"
    fi
fi

if [ "$CONNECTED" = false ]; then
    echo -e "${RED}❌ Cannot connect to VM with any method${NC}"
    echo ""
    echo -e "${YELLOW}Manual steps required:${NC}"
    echo "1. Connect to your VM using the Google Cloud Console"
    echo "2. Run these commands on the VM:"
    echo ""
    echo -e "${BLUE}# On the VM as user $VM_USER:${NC}"
    echo "mkdir -p ~/.ssh"
    echo "chmod 700 ~/.ssh"
    echo "echo '$(cat ~/.ssh/$KEY_NAME.pub)' >> ~/.ssh/authorized_keys"
    echo "chmod 600 ~/.ssh/authorized_keys"
    echo "sudo systemctl restart ssh"
    echo ""
    echo -e "${YELLOW}Then add this to GitHub Secrets as VM_SSH_KEY:${NC}"
    echo "$(cat ~/.ssh/$KEY_NAME)"
    exit 1
fi

# Install public key on VM
echo -e "${YELLOW}Step 4: Installing public key on VM...${NC}"
if [ -n "$SSH_CMD" ]; then
    # Using gcloud
    $SSH_CMD "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    $SSH_CMD "echo '$(cat ~/.ssh/$KEY_NAME.pub)' >> ~/.ssh/authorized_keys"
    $SSH_CMD "chmod 600 ~/.ssh/authorized_keys"
    $SSH_CMD "sudo systemctl restart ssh"
else
    # Using direct SSH
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "mkdir -p ~/.ssh && chmod 700 ~/.ssh"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "echo '$(cat ~/.ssh/$KEY_NAME.pub)' >> ~/.ssh/authorized_keys"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "chmod 600 ~/.ssh/authorized_keys"
    ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "sudo systemctl restart ssh"
fi

echo -e "${GREEN}✅ Public key installed on VM${NC}"

# Wait a moment for SSH to restart
echo "Waiting for SSH service to restart..."
sleep 5

# Test new key
echo -e "${YELLOW}Step 5: Testing new SSH key...${NC}"
if ssh -i ~/.ssh/$KEY_NAME -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VM_USER@$VM_HOST "echo 'New key works!'" 2>/dev/null; then
    echo -e "${GREEN}✅ New SSH key works perfectly!${NC}"
else
    echo -e "${RED}❌ New SSH key test failed${NC}"
    echo "Please check the VM configuration manually"
    exit 1
fi

# Show GitHub Secrets configuration
echo ""
echo -e "${GREEN}🎉 SSH Key Setup Complete!${NC}"
echo ""
echo -e "${YELLOW}GitHub Secrets Configuration:${NC}"
echo "================================"
echo ""
echo -e "${BLUE}Add these secrets to your GitHub repository:${NC}"
echo ""
echo -e "${YELLOW}VM_HOST:${NC}"
echo "$VM_HOST"
echo ""
echo -e "${YELLOW}VM_USER:${NC}" 
echo "$VM_USER"
echo ""
echo -e "${YELLOW}VM_SSH_KEY:${NC}"
echo "$(cat ~/.ssh/$KEY_NAME)"
echo ""
echo -e "${BLUE}Steps to add to GitHub:${NC}"
echo "1. Go to your GitHub repository"
echo "2. Click Settings → Secrets and variables → Actions"
echo "3. Add/Update the three secrets above"
echo ""
echo -e "${GREEN}Your deployment should now work!${NC}"

# Clean up - offer to remove local key
echo ""
read -p "Do you want to remove the local private key file (recommended for security)? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    rm ~/.ssh/$KEY_NAME
    echo -e "${GREEN}✅ Local private key removed${NC}"
    echo "The key content is already copied above for GitHub Secrets"
fi

echo ""
echo -e "${GREEN}Setup complete! You can now run your GitHub Actions deployment.${NC}"