# Working Services Backup Guide

## 📋 **Overview**

This guide covers how to backup and restore your **working services** (YOLO-Combined and YOLO-NAS) to protect against regressions when making changes.

## ✅ **Confirmed Working Services**

- **YOLO Combined** - Port 8001 ✅ **WORKING**
- **YOLO-NAS** - Port 8004 ✅ **WORKING**  
- **MMPose** - Port 8003 ⚠️ **NEEDS FIXES**

## 🗂️ **Backup Scripts**

### **1. [`backup-working-services.sh`](backup-working-services.sh)**
Creates dated backups of your working services

### **2. [`restore-service.sh`](restore-service.sh)**  
Safely restores services from backups

## 🚀 **Quick Start**

### **Create Backup of All Working Services**
```bash
./backup-working-services.sh backup
```

### **Create Backup of Specific Service**
```bash
# Backup only YOLO Combined
./backup-working-services.sh backup yolo-combined

# Backup only YOLO-NAS
./backup-working-services.sh backup yolo-nas
```

### **List Existing Backups**
```bash
./backup-working-services.sh list
```

### **Restore Service Interactively**
```bash
./restore-service.sh interactive
```

### **Restore Specific Service**
```bash
./restore-service.sh restore yolo-combined yolo-combined_working_20250528_221045
```

## 📁 **Backup Structure**

Backups are stored in the `backups/` directory:

```
backups/
├── yolo-combined_working_20250528_221045.tar.gz
├── yolo-nas_working_20250528_221045.tar.gz
├── yolo-combined_image_20250528_221045.tar.gz    # Docker images
└── yolo-nas_image_20250528_221045.tar.gz         # Docker images
```

## 🔄 **Backup Workflow**

### **Before Making Changes**
```bash
# 1. Create backup of working services
./backup-working-services.sh backup

# 2. Verify backup was created
./backup-working-services.sh list

# 3. Make your changes
# ... edit code ...

# 4. Test changes
./deploy-resilient.sh all

# 5. If something breaks, restore working version
./restore-service.sh interactive
```

## 📊 **Backup Information**

Each backup includes:
- **Complete service source code**
- **Dockerfile and requirements**
- **Backup metadata** with git commit, date, status
- **Restoration instructions**
- **Health check commands**

### **Example Backup Info**
```markdown
# Backup Information

**Service**: yolo-combined  
**Backup Date**: Tue May 28 22:10:45 2024  
**Status**: Working/Verified  
**Git Commit**: a1b2c3d4e5f6...  
**Git Branch**: docker-containers  

## Restoration Command
```bash
./restore-service.sh yolo-combined yolo-combined_working_20250528_221045
```

## Service Health Check
curl http://localhost:8001/healthz
```

## 🛡️ **Safety Features**

### **Automatic Safety Backups**
When restoring, the script automatically creates a safety backup of your current service:
```bash
backups/yolo-combined-service_before_restore_20250528_221500/
```

### **Validation**
- ✅ **Service verification** - Only backs up confirmed working services
- ✅ **File integrity** - Uses tar.gz compression with validation
- ✅ **Metadata tracking** - Records git commit, branch, and date
- ✅ **Interactive confirmation** - Confirms before destructive operations

## 📈 **Use Cases**

### **1. Experimentation Protection**
```bash
# Before experimenting with MMPose fixes
./backup-working-services.sh backup

# Experiment with changes...
# If MMPose breaks other services, restore working versions
./restore-service.sh interactive
```

### **2. Version Control**
```bash
# Create milestone backups
./backup-working-services.sh backup
# Creates: yolo-combined_working_20250528_221045.tar.gz

# Later reference this known-good version
./restore-service.sh restore yolo-combined yolo-combined_working_20250528_221045
```

### **3. CI/CD Integration**
```bash
# In your deployment pipeline
./backup-working-services.sh backup
./deploy-resilient.sh all
# If deployment fails, automatic rollback available
```

## 🔧 **Advanced Usage**

### **Backup Docker Images**
```bash
# Also backup the built Docker images
./backup-working-services.sh backup-images
```

### **List Backups with Details**
```bash
./backup-working-services.sh list
```
Output:
```
📁 Available backups:
  • yolo-combined_working_20250528_221045 (2.5M bytes, May 28 22:10)
  • yolo-nas_working_20250528_221045 (1.8M bytes, May 28 22:10)
```

### **Selective Restoration**
```bash
# Restore only specific service without affecting others
./restore-service.sh restore yolo-nas yolo-nas_working_20250528_221045
```

## ⚡ **Integration with Resilient Deployment**

The backup system works seamlessly with your resilient deployment:

```bash
# 1. Backup working services
./backup-working-services.sh backup

# 2. Deploy with resilient approach
./deploy-resilient.sh all

# 3. If issues arise, restore and re-deploy
./restore-service.sh interactive
./deploy-resilient.sh all
```

## 🎯 **Best Practices**

1. **✅ Backup before major changes** - Always create backups before significant modifications
2. **✅ Use descriptive names** - Backups include timestamps and status
3. **✅ Test restoration** - Periodically test restoration process
4. **✅ Monitor disk space** - Cleanup old backups when needed
5. **✅ Combine with git** - Backups complement, don't replace version control

## 🗑️ **Cleanup**

To cleanup old backups:
```bash
# Remove backups older than 30 days
find backups/ -name "*.tar.gz" -mtime +30 -delete

# Or manually remove specific backups
rm backups/old_backup_name.tar.gz
```

## ✅ **Summary**

You now have a robust backup system that:

- ✅ **Protects your working services** (YOLO-Combined, YOLO-NAS)
- ✅ **Creates dated, compressed backups** with metadata  
- ✅ **Provides safe restoration** with automatic safety backups
- ✅ **Integrates with your resilient deployment** workflow
- ✅ **Enables confident experimentation** with easy rollback

**Next**: Use this system to safely work on fixing MMPose while keeping your working services protected!