# SSH Authentication Troubleshooting Guide

## Issues Fixed in deploy.yml

### 1. ✅ Deploy Cleanup Command Fixed
**Problem:** Line 73 had invalid shell syntax with `***` placeholder
```bash
# BEFORE (broken):
find /opt/padel/app -maxdepth 1 -type d ! -name weights ! -name uploads ! -name processed -exec rm -rf *** + 2>/dev/null || true

# AFTER (fixed):
find /opt/padel/app -maxdepth 1 -mindepth 1 -type d ! -name weights ! -name uploads ! -name processed -exec rm -rf {} + 2>/dev/null || true
```

**What this does:**
- Deletes all first-level directories in `/opt/padel/app` 
- Preserves `weights`, `uploads`, and `processed` directories
- Uses `-mindepth 1` to prevent trying to delete the search directory itself
- Uses proper `{}` placeholder instead of `***`

### 2. ✅ Enhanced SSH Debugging
**Added:**
- `debug: true` to all SSH action steps for verbose logging
- New "Test SSH Connectivity" step to validate connection before deployment
- Better error reporting for SSH failures

## SSH Authentication Error Resolution

### Error Message:
```
ssh: handshake failed: ssh: unable to authenticate, attempted methods [none publickey], no supported methods remain
```

### Root Causes & Solutions:

#### 1. GitHub Secrets Configuration
**Check these GitHub repository secrets:**

```bash
VM_HOST=35.189.53.46
VM_USER=Towers  
VM_SSH_KEY=<your-private-key-content>
```

**Verify VM_SSH_KEY format:**
```
-----BEGIN OPENSSH PRIVATE KEY-----
b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAA...
[key content]
...
-----END OPENSSH PRIVATE KEY-----
```

#### 2. VM User Account Verification
**SSH into your VM manually to verify:**
```bash
# Test from your local machine:
ssh -i path/to/your/private_key Towers@35.189.53.46

# Once connected, verify user setup:
whoami  # Should return: Towers
id      # Should show Towers user groups
ls -la ~/.ssh/  # Check authorized_keys exists
```

#### 3. SSH Key Setup on VM
**Ensure the public key is properly installed:**
```bash
# On the VM as user Towers:
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add your public key to authorized_keys:
echo "your-public-key-content" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# Verify ownership:
ls -la ~/.ssh/
# Should show: drwx------ Towers Towers .ssh/
#              -rw------- Towers Towers authorized_keys
```

#### 4. SSH Service Configuration
**Check SSH daemon config on VM:**
```bash
# Verify SSH is running:
sudo systemctl status ssh

# Check SSH config allows key authentication:
sudo grep -E "PubkeyAuthentication|PasswordAuthentication" /etc/ssh/sshd_config
# Should show:
# PubkeyAuthentication yes
# PasswordAuthentication no (or yes)

# Restart SSH if needed:
sudo systemctl restart ssh
```

#### 5. Firewall & Network
**Ensure SSH port is open:**
```bash
# Check if SSH port 22 is listening:
sudo netstat -tlnp | grep :22

# Verify firewall allows SSH:
sudo ufw status
# Should show: 22/tcp ALLOW

# Google Cloud firewall (if applicable):
gcloud compute firewall-rules list --filter="name~ssh"
```

### Manual Testing Steps:

#### Step 1: Test Local SSH Connection
```bash
# From your local machine:
ssh -v -i path/to/your/private_key Towers@35.189.53.46

# The -v flag will show verbose output to help diagnose issues
```

#### Step 2: Test with GitHub Action SSH Key
```bash
# Extract the private key from GitHub secrets and test locally:
# (Never commit this key to git!)
echo "$VM_SSH_KEY_CONTENT" > temp_key
chmod 600 temp_key
ssh -v -i temp_key Towers@35.189.53.46
rm temp_key  # Delete immediately after test
```

#### Step 3: Check VM Logs
```bash
# On the VM, check SSH auth logs:
sudo tail -f /var/log/auth.log
# Then attempt SSH connection from another terminal
# Look for authentication failures and reasons
```

### Common Solutions:

#### Solution 1: Regenerate SSH Key Pair
```bash
# Generate new key pair:
ssh-keygen -t ed25519 -C "github-actions@nuro-padel" -f ~/.ssh/nuro_padel_key

# Copy public key to VM:
ssh-copy-id -i ~/.ssh/nuro_padel_key.pub Towers@35.189.53.46

# Update GitHub secret VM_SSH_KEY with private key content:
cat ~/.ssh/nuro_padel_key
```

#### Solution 2: Fix File Permissions
```bash
# On VM as user Towers:
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
chown -R Towers:Towers ~/.ssh
```

#### Solution 3: SELinux/AppArmor Issues
```bash
# If SELinux is enabled:
sudo restorecon -Rv ~/.ssh

# Check for denied operations:
sudo ausearch -m avc -ts recent
```

### Next Steps:

1. **Verify GitHub Secrets** - Double-check all three secrets are correctly set
2. **Test Manual SSH** - Use the commands above to test connection manually  
3. **Check VM Logs** - Monitor auth.log during connection attempts
4. **Regenerate Keys** - If all else fails, create new SSH key pair
5. **Run Updated Workflow** - The enhanced debugging will provide more details

### GitHub Actions Debug Output:

With the updated workflow, you'll now see detailed SSH connection information in the "Test SSH Connectivity" step, including:
- User verification
- Host information  
- SSH client version
- Permission details
- Connection success confirmation

This will help pinpoint exactly where the authentication is failing.