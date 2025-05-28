# 🔐 GitHub Actions SSH Setup Guide

## Issue: Permission denied (publickey)

The GitHub Actions deployment is failing because SSH authentication to your VM isn't configured.

## ✅ **Solution: Configure GitHub Secrets**

### **Step 1: Generate SSH Key Pair (if needed)**

On your local machine or VM:
```bash
# Generate a new SSH key pair for GitHub Actions
ssh-keygen -t ed25519 -C "github-actions@nuro-padel" -f ~/.ssh/github-actions-key

# This creates:
# ~/.ssh/github-actions-key (private key)
# ~/.ssh/github-actions-key.pub (public key)
```

### **Step 2: Add Public Key to VM**

SSH to your VM and add the public key:
```bash
ssh towers@35.189.53.46

# Add the public key to authorized_keys
mkdir -p ~/.ssh
echo "YOUR_PUBLIC_KEY_CONTENT" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```

### **Step 3: Configure GitHub Repository Secrets**

Go to your GitHub repository: `https://github.com/stevetowers098/nuro-padel`

1. **Settings** → **Secrets and variables** → **Actions**
2. **Click "New repository secret"**
3. **Add these secrets:**

#### **Required Secrets:**

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `VM_SSH_KEY` | `[PRIVATE KEY CONTENT]` | Private key from ~/.ssh/github-actions-key |
| `VM_HOST` | `35.189.53.46` | Your VM IP address |
| `VM_USER` | `towers` | Your VM username |
| `VM_PATH` | `/opt/padel-docker` | Deployment directory |

#### **VM_SSH_KEY Content:**
```
-----BEGIN OPENSSH PRIVATE KEY-----
[Your private key content from ~/.ssh/github-actions-key]
-----END OPENSSH PRIVATE KEY-----
```

### **Step 4: Test SSH Connection**

Test that the key works:
```bash
ssh -i ~/.ssh/github-actions-key towers@35.189.53.46
```

## 🚀 **Alternative: Manual Deployment**

If you prefer to skip GitHub Actions automation, you can deploy manually:

### **Local Deployment to VM:**
```bash
# Test SSH connection first
ssh towers@35.189.53.46

# If SSH works, deploy manually
./deploy.sh --vm
```

### **Direct VM Commands:**
```bash
# SSH to VM
ssh towers@35.189.53.46

# Create directory and clone
mkdir -p /opt/padel-docker
cd /opt/padel-docker

# Clone the repository directly
git clone https://github.com/stevetowers098/nuro-padel.git .
git checkout docker-containers

# Run deployment
chmod +x deploy.sh
./deploy.sh --all
```

## 🔧 **Troubleshooting**

### **If SSH still fails:**

1. **Check SSH key permissions:**
   ```bash
   chmod 600 ~/.ssh/github-actions-key
   chmod 644 ~/.ssh/github-actions-key.pub
   ```

2. **Test with verbose SSH:**
   ```bash
   ssh -v -i ~/.ssh/github-actions-key towers@35.189.53.46
   ```

3. **Check VM firewall:**
   ```bash
   # On VM, ensure SSH port 22 is open
   sudo ufw status
   sudo ufw allow ssh
   ```

### **Alternative SSH Key Setup:**

If you already have SSH access to the VM, you can use your existing key:

1. **Copy your existing private key content**
2. **Add it as `VM_SSH_KEY` secret in GitHub**
3. **Ensure the corresponding public key is in VM's `~/.ssh/authorized_keys`**

## ✅ **Verification**

Once secrets are configured:

1. **Push any change to trigger GitHub Actions:**
   ```bash
   git add .
   git commit -m "Test GitHub Actions deployment"
   git push origin docker-containers
   ```

2. **Check Actions tab** in your GitHub repository
3. **Deployment should succeed** with "🎉 VM deployment successful!"

## 🎯 **Quick Setup Summary**

1. ✅ **Generate SSH key pair**
2. ✅ **Add public key to VM** (`~/.ssh/authorized_keys`)
3. ✅ **Add private key to GitHub secrets** (`VM_SSH_KEY`)
4. ✅ **Configure other secrets** (`VM_HOST`, `VM_USER`, `VM_PATH`)
5. ✅ **Test by pushing to `docker-containers` branch**

Once configured, your GitHub Actions will automatically deploy to `/opt/padel-docker/` on every push!