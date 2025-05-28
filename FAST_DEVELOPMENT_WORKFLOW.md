# ⚡ Fast Development Workflow

Transform your 30+ minute rebuilds into **under 5-minute deployments** with smart incremental development.

## 🎯 Quick Start Commands

```bash
# Make scripts executable (run once)
chmod +x dev-*.sh

# Smart rebuild (only changed services) - 2-5 minutes
./dev-smart.sh

# Single service rebuild - 1-2 minutes  
./dev-service.sh yolo-combined

# Code-only changes (no rebuild) - 10-30 seconds
./dev-code.sh

# Auto-rebuild on file changes
./dev-watch.sh
```

## 🧠 Smart Development Scripts

### 1. **`./dev-smart.sh`** - Intelligent Change Detection
- 🔍 **Analyzes file changes** using checksums
- 🎯 **Rebuilds only modified services**
- ⚡ **Keeps working services running**
- ⏱️ **2-5 minutes** vs 30+ minutes

```bash
./dev-smart.sh
# Output: 📝 yolo-combined has changes - will rebuild
#         ✅ mmpose unchanged - keeping running
#         ✅ yolo-nas unchanged - keeping running
```

### 2. **`./dev-service.sh <service>`** - Single Service Rebuild
- 🎯 **Rebuild just one service**
- ⚡ **1-2 minute deployments**
- 🚀 **Perfect for focused development**

```bash
./dev-service.sh yolo-combined    # Rebuild YOLO Combined only
./dev-service.sh mmpose          # Rebuild MMPose only
./dev-service.sh yolo-nas        # Rebuild YOLO-NAS only
./dev-service.sh all             # Rebuild all (same as dev-fast.sh)
```

### 3. **`./dev-code.sh`** - Instant Code Changes
- ✨ **Zero rebuild required**
- 🔄 **Live code sync with containers**
- ⚡ **10-30 second deployments**
- 🎯 **Perfect for Python code tweaks**

```bash
./dev-code.sh
# Output: ⚡ Code deployment time: 15 seconds
```

### 4. **`./dev-watch.sh`** - Automatic File Watching
- 👁️ **Monitors all service directories**
- 🔄 **Auto-rebuilds on file changes**
- 🎯 **Rebuilds only what changed**
- 🛑 **Ctrl+C to stop**

```bash
./dev-watch.sh
# Output: 👁️ Watching for file changes...
#         🔄 Rebuilding yolo-combined due to file changes...
```

## 🚀 Development Workflow Examples

### Scenario 1: Algorithm Development (Code Only)
```bash
# 1. Start with smart deployment
./dev-smart.sh

# 2. Switch to code-only mode for rapid iteration
./dev-code.sh

# 3. Edit your Python files - changes sync instantly!
# No rebuild needed for code changes
```

### Scenario 2: Adding New Dependencies
```bash
# 1. Update requirements.txt
vim yolo-combined-service/requirements.txt

# 2. Smart rebuild (detects dependency changes)
./dev-smart.sh
# Only rebuilds yolo-combined, keeps others running
```

### Scenario 3: Multi-Service Development
```bash
# 1. Start file watcher for automatic rebuilds
./dev-watch.sh

# 2. Edit any files - auto-rebuilds affected services
# 3. Ctrl+C when done
```

### Scenario 4: Focused Single Service Work
```bash
# Working on MMPose biomechanics
./dev-service.sh mmpose

# Working on YOLO detection
./dev-service.sh yolo-combined

# Working on high-accuracy detection  
./dev-service.sh yolo-nas
```

## 📊 Performance Comparison

| Operation | Old Method | New Method | Time Saved |
|-----------|------------|------------|------------|
| Full rebuild | 30+ minutes | 30+ minutes | - |
| Code changes | 30+ minutes | **15 seconds** | 99.2% faster |
| Single service | 30+ minutes | **2 minutes** | 93% faster |
| Smart rebuild | 30+ minutes | **5 minutes** | 83% faster |

## 🎯 Choose the Right Tool

### Use `./dev-code.sh` when:
- ✅ Changing Python algorithm logic
- ✅ Fixing bugs in existing code
- ✅ Tweaking model parameters
- ✅ Updating API endpoints

### Use `./dev-service.sh <service>` when:
- ✅ Adding new Python dependencies
- ✅ Changing Dockerfile
- ✅ Updating service configuration
- ✅ Working on one specific service

### Use `./dev-smart.sh` when:
- ✅ Multiple services changed
- ✅ Not sure what changed
- ✅ First deployment of the day
- ✅ Want automatic change detection

### Use `./dev-watch.sh` when:
- ✅ Active development session
- ✅ Frequent code changes
- ✅ Want hands-free rebuilds
- ✅ Testing across multiple services

## 🔧 How It Works

### Change Detection
- **File checksums** track modifications
- **Smart comparison** identifies affected services
- **Incremental rebuilds** save massive time

### Service Isolation
- **Independent containers** can rebuild separately
- **Shared dependencies** through base images
- **Rolling updates** minimize downtime

### Code Mounting
- **Volume mounts** sync code instantly
- **No rebuild required** for Python changes
- **Live reload** capabilities (where supported)

## 🛠️ Troubleshooting

### Scripts not executable?
```bash
chmod +x dev-*.sh
```

### Services not starting?
```bash
# Check logs
docker-compose -f docker-compose.dev.yml logs <service>

# Force rebuild all
./dev-fast.sh
```

### Code changes not visible?
```bash
# Ensure you're using code-only mode
./dev-code.sh

# Or restart specific service
./dev-service.sh <service>
```

### File watcher not working?
```bash
# Install inotify-tools (Linux) for better performance
sudo apt-get install inotify-tools

# Or use polling fallback (automatic)
./dev-watch.sh
```

## 📁 Files Created

- `dev-smart.sh` - Smart change detection and rebuilds
- `dev-service.sh` - Single service rebuild tool
- `dev-code.sh` - Instant code-only deployments
- `dev-watch.sh` - Automatic file watching and rebuilds
- `docker-compose.code.yml` - Code-only overlay configuration
- `.dev-checksums/` - Change tracking directory (auto-created)

## 🎉 Result

**Transform your development experience:**
- ⚡ **10-30 second** code deployments
- 🎯 **1-2 minute** single service rebuilds  
- 🧠 **2-5 minute** smart multi-service rebuilds
- 👁️ **Automatic** file watching and rebuilds

**No more 30-minute rebuilds for simple code changes!**