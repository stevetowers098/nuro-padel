# 🔧 MMPose Package Compatibility Fix

## Issue
MMPose environment has package version conflicts, particularly with NumPy, xtcocotools, and pycocotools that cause import errors.

## ⚡ Quick Manual Fix (2 minutes)

If your deployment fails on MMPose service startup, run these commands on your VM:

```bash
# Stop the service first
sudo systemctl stop mmpose-service

# Fix the environment with proper versions
source /opt/padel/mmpose/venv/bin/activate

# Uninstall problematic packages
pip uninstall -y xtcocotools pycocotools numpy

# Install compatible versions in correct order
pip install numpy==1.21.6
pip install cython
pip install pycocotools
pip install xtcocotools --no-deps --force-reinstall

# Verify installation
python -c "
try:
    import numpy as np
    print(f'NumPy version: {np.__version__}')
    import xtcocotools
    print('xtcocotools imported successfully')
    import mmpose
    print('MMPose imported successfully')
    print('✅ All packages working correctly')
except Exception as e:
    print('❌ Error:', e)
"

deactivate

# Restart the service
sudo systemctl start mmpose-service
sudo systemctl status mmpose-service
```

## 🔄 Automated Fix (In Deployment)

I've updated the deployment workflow to automatically:

1. **Uninstall conflicting packages** before fresh installation
2. **Install NumPy 1.21.6** first (compatible version)
3. **Install cython** before cocotools (required dependency)
4. **Install pycocotools and xtcocotools** in correct order
5. **Verify installation** before continuing

### Updated MMPose Setup Process:
```bash
# Clean environment
pip uninstall openxlab torch torchvision torchaudio triton xtcocotools pycocotools numpy -y

# Install in correct order
pip install numpy==1.21.6
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install cython
pip install pycocotools
pip install xtcocotools --no-deps --force-reinstall
pip install openmim
mim install "mmpose>=1.0.0"
mim install "mmcv>=2.0.1"

# Verification
python -c "import numpy, xtcocotools, mmpose; print('✅ All working')"
```

## 🎯 Why This Happens

**Root Cause:** Package version conflicts between:
- MMPose requiring specific NumPy versions
- xtcocotools and pycocotools having overlapping dependencies
- PyTorch installation potentially upgrading NumPy to incompatible versions

**Solution:** Install packages in specific order with pinned versions to prevent conflicts.

## 🔍 Troubleshooting

### Check Service Status:
```bash
sudo systemctl status mmpose-service
sudo journalctl -u mmpose-service -f
```

### Test MMPose Manually:
```bash
source /opt/padel/mmpose/venv/bin/activate
python -c "import mmpose; print('MMPose version:', mmpose.__version__)"
```

### Check Package Versions:
```bash
source /opt/padel/mmpose/venv/bin/activate
pip list | grep -E "(numpy|xtcocotools|pycocotools|mmpose)"
```

## ✅ Expected Results

After fix:
- NumPy version: 1.21.6
- xtcocotools imports successfully
- MMPose imports successfully
- mmpose-service runs without errors

The deployment workflow now automatically prevents this issue, but the manual fix is available if needed.