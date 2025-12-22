# Complete Installation Guide: PyTorch 2.7.1 + LeRobot on Ubuntu 24.04 with RTX 5090

This guide provides detailed steps for installing PyTorch 2.7.1 with CUDA 12.6 support and LeRobot on Ubuntu 24.04 with an NVIDIA RTX 5090 GPU.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Update System](#step-1-update-system)
- [Step 2: Disable Nouveau Driver](#step-2-disable-nouveau-driver)
- [Step 3: Disable Secure Boot](#step-3-disable-secure-boot)
- [Step 4: Install NVIDIA Open Kernel Driver](#step-4-install-nvidia-open-kernel-driver)
- [Step 5: Install Anaconda](#step-5-install-anaconda)
- [Step 6: Install PyTorch with CUDA via Conda](#step-6-install-pytorch-with-cuda-via-conda)
- [Step 7: Install LeRobot](#step-7-install-lerobot)
- [Step 8: Verify Installation](#step-8-verify-installation)
- [Troubleshooting](#troubleshooting)
- [Important Notes for RTX 5090](#important-notes-for-rtx-5090)

---

## Prerequisites

- Ubuntu 24.04 LTS
- NVIDIA RTX 5090 GPU
- Internet connection
- Sudo/root access

---

## step 0: pre-install
- hp backup, backupvilla
- boot with usb ubuntu
  - choose safe graphics version instead of "install ubuntu"
  - manual install partition instead of choosing automatic install with win bootmanager
    - efi boot partition: change but use existing partition without formatting
      - /boot/efi
    - choose free space: /: 150GB, /home: 1.2T


## Step 1: Update System

```bash
sudo apt update && sudo apt upgrade -y
```

---

## Step 2: Disable Nouveau Driver

The Nouveau open-source driver conflicts with NVIDIA's proprietary drivers and must be disabled.

### Create Blacklist Configuration

```bash
# Create the blacklist configuration file
sudo bash -c "cat > /etc/modprobe.d/blacklist-nouveau.conf << EOF
blacklist nouveau
options nouveau modeset=0
EOF"
```

**Or create it manually:**

```bash
sudo nano /etc/modprobe.d/blacklist-nouveau.conf
```

Add these two lines:
```
blacklist nouveau
options nouveau modeset=0
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

### Update initramfs and Reboot

```bash
# Regenerate the kernel initramfs
sudo update-initramfs -u

# Reboot to apply changes
sudo reboot
```

### Verify Nouveau is Disabled

After reboot:

```bash
lsmod | grep nouveau
```

**Expected output:** Nothing (empty). If you see output, Nouveau is still loaded.

---

## Step 3: Disable Secure Boot

Secure Boot prevents the NVIDIA driver from loading. You must disable it in BIOS.

### Steps:

1. **Reboot your computer**
2. **Enter BIOS/UEFI Setup** during boot:
   - For HP OMEN: press **F10**
   - Other systems: try F2, F12, Del, or Esc
3. **Navigate to Security or Boot menu**
4. **Find "Secure Boot" setting**
5. **Change it to "Disabled"**
6. **Save and Exit** (usually F10)
7. **Boot into Ubuntu**

---

## Step 4: Install NVIDIA Open Kernel Driver

**CRITICAL:** RTX 5090 requires the open kernel modules, NOT the standard proprietary driver.

### Install the Open Driver

```bash
# Add the graphics drivers PPA
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install the OPEN driver (required for RTX 5090)
sudo apt install nvidia-driver-580-open

# Reboot
sudo reboot
```

### Verify Driver Installation

After reboot:

```bash
# Check driver is working
nvidia-smi
```

**Expected output:** You should see your RTX 5090 listed with driver version, CUDA version, temperature, and memory information.

**Additional verification:**

```bash
# Verify correct modules are loaded
lsmod | grep nvidia
# Should show: nvidia, nvidia_uvm, nvidia_modeset, nvidia_drm

# Check driver version
cat /proc/driver/nvidia/version
```

### Troubleshooting Driver Issues

If you see "couldn't communicate with the NVIDIA driver":

```bash
# Check for key rejection error
sudo modprobe nvidia
```

If you see **"Key was rejected by service"**, Secure Boot is still enabled. Go back to Step 3.

If you see **"requires use of the NVIDIA open kernel modules"**:

```bash
# Remove proprietary driver and install open driver
sudo apt remove --purge nvidia-driver-580 nvidia-dkms-580
sudo apt install nvidia-driver-580-open
sudo reboot
```

---

## Step 5: Install Anaconda

```bash
# Download Anaconda (adjust version as needed)
wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh

# Install
bash Anaconda3-2024.06-1-Linux-x86_64.sh

# Follow prompts, then restart terminal or:
source ~/.bashrc
```

---

## Step 6: Install CUDA and PyTorch

You have two options for installing CUDA: manual system-wide installation (Option A) or using conda's cuda-toolkit (Option B - **Recommended**).

---

### Option A: Manual CUDA 12.6 Installation (System-Wide)

This approach installs CUDA system-wide. Use this if you need CUDA for applications outside of conda environments.

#### Download and Install CUDA Toolkit 12.6

```bash
# Download CUDA 12.6 runfile
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run

# Install (deselect driver since you already installed the open driver)
sudo sh cuda_12.6.0_560.28.03_linux.run
```

**During installation:**
- **Deselect** "Driver" (you already installed nvidia-driver-580-open)
- **Select** "CUDA Toolkit 12.6"
- Accept default installation path: `/usr/local/cuda-12.6`

#### Set Environment Variables

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

Apply changes:

```bash
source ~/.bashrc
```

Verify CUDA:

```bash
nvcc --version
```

#### Install cuDNN (Optional but Recommended)

```bash
# Download cuDNN 9.x for CUDA 12.6 from NVIDIA
# (requires NVIDIA developer account)
# https://developer.nvidia.com/cudnn

# Extract and copy files
tar -xvf cudnn-linux-x86_64-9.x.x.x_cuda12-archive.tar.xz
cd cudnn-linux-x86_64-9.x.x.x_cuda12-archive

sudo cp include/cudnn*.h /usr/local/cuda-12.6/include
sudo cp lib/libcudnn* /usr/local/cuda-12.6/lib64
sudo chmod a+r /usr/local/cuda-12.6/include/cudnn*.h /usr/local/cuda-12.6/lib64/libcudnn*
```

#### Create Conda Environment and Install PyTorch

```bash
# Create environment with Python 3.10 (recommended by LeRobot)
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install ffmpeg (required by LeRobot for video processing)
conda install ffmpeg -c conda-forge

# Install PyTorch 2.7.1 with CUDA 12.6 support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

---

### Option B: Using Conda's cuda-toolkit (Environment-Isolated) ⭐ **RECOMMENDED**

This approach keeps CUDA isolated within the conda environment. This is cleaner and easier to manage.

#### Create Conda Environment for LeRobot

```bash
# Create environment with Python 3.10 (recommended by LeRobot)
conda create -y -n lerobot python=3.10
conda activate lerobot
```

#### Install CUDA Toolkit via Conda

This installs CUDA 12.6 in your conda environment (isolated from system-wide installations):

```bash
# Install CUDA toolkit 12.6 from NVIDIA channel
conda install -c nvidia cuda-toolkit=12.6 -y
```

**Why use conda's cuda-toolkit?**
- ✅ Keeps CUDA isolated within the conda environment
- ✅ No need for system-wide CUDA installation
- ✅ Easy to manage multiple CUDA versions in different environments
- ✅ No PATH/LD_LIBRARY_PATH configuration needed
- ✅ Clean uninstallation (just delete the environment)

#### Install FFmpeg

```bash
# Install ffmpeg (required by LeRobot for video processing)
conda install ffmpeg -c conda-forge
```

#### Install PyTorch 2.7.1 with CUDA 12.6

```bash
# Install PyTorch 2.7.1 with CUDA 12.6 support
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Note:** Use `pip` or `pip3` - they're the same in conda environments.

**Important:** You still need the NVIDIA driver installed system-wide (Step 4), but the CUDA toolkit is managed by conda in this option.

---

## Recommendation: Which Option to Choose?

**Use Option B (conda cuda-toolkit)** if:
- ✅ You only need CUDA for Python/PyTorch/LeRobot work
- ✅ You want easy environment management
- ✅ You prefer clean, isolated installations
- ✅ You might need different CUDA versions for different projects

**Use Option A (manual installation)** if:
- You need system-wide CUDA for non-Python applications
- You're doing CUDA/driver development work
- You have other tools that require system-wide CUDA

For LeRobot work, **Option B is strongly recommended**.

---

## Step 7: Install LeRobot

### Clone LeRobot Repository

```bash
# Navigate to home directory
cd ~

# Clone LeRobot
git clone https://github.com/huggingface/lerobot.git
cd lerobot
```

### Install Additional Linux Dependencies (if needed)

If you encounter build errors during installation:

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev

python -c "import torch, torchvision; print(f'torch: {torch.__version__}');print(f'torchvision: {torchvision.__version__}')"
```

### Install LeRobot

```bash
# Install LeRobot in editable mode with all features
# NOTE: don't use all due to mujocu conflicts
  # - libero requires mujoco>=3.0.0
  # - gym-xarm requires mujoco>=2.3.7,<3.0.0
# pip install -e ".[all]"

# Install everything except xarm
uv pip install -e ".[dynamixel,gamepad,hopejr,lekiwi,reachy2,kinematics,intelrealsense,pi0, smolvla,hilserl,async,dev,test,video_benchmark,aloha,pusht,phone,libero]"

uv pip install -e ".[aloha,pusht,pi0,smolvla]"

# OR install with specific features you need:
# pip install -e ".[aloha,pusht]"
# pip install -e ".[feetech]"
# pip install -e ".[intelrealsense,dynamixel]"
```

**Available feature tags:**
- `all` - All available features
- `aloha` - ALOHA robot support
- `pusht` - PushT environment
- `xarm` - XArm robot support
- `feetech` - Feetech motor support
- `dynamixel` - Dynamixel servo support
- `intelrealsense` - Intel RealSense camera support
- `gamepad` - Gamepad control support

### Optional: Enable Weights & Biases for Experiment Tracking

```bash
# Install wandb (if not already included)
pip install wandb

# Login to wandb
wandb login
```

---

## Step 8: Verify Installation

### Test PyTorch + CUDA

Create a test script `test_pytorch.py`:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test tensor operation
    x = torch.rand(5, 3).cuda()
    print(f"\nTest tensor on GPU:\n{x}")
    
    # Simple computation test
    a = torch.randn(1000, 1000).cuda()
    b = torch.randn(1000, 1000).cuda()
    c = torch.matmul(a, b)
    
    print("\n✓ GPU is working correctly!")
    print(f"✓ Successfully performed matrix multiplication on {torch.cuda.get_device_name(0)}")
else:
    print("\n✗ CUDA is not available. Check your installation.")
```

Run it:

```bash
python test_pytorch.py
```

**Expected output:**
```
PyTorch version: 2.7.1+cu126
CUDA available: True
CUDA version: 12.6
cuDNN version: 90100
Number of GPUs: 1
GPU Name: NVIDIA GeForce RTX 5090
GPU Compute Capability: (9, 0)

Test tensor on GPU:
tensor([[...]])

✓ GPU is working correctly!
✓ Successfully performed matrix multiplication on NVIDIA GeForce RTX 5090
```

### Test LeRobot

```bash
# Test LeRobot import
python -c "import lerobot; print('LeRobot imported successfully')"

# Optional: Visualize a dataset to test full setup
python -m lerobot.scripts.visualize_dataset \
  --repo-id lerobot/pusht \
  --episode-index 0
```

### Quick Verification Commands

```bash
# One-liner to check everything
python -c "import torch, lerobot; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print('LeRobot: OK')"
```

---

## Troubleshooting

### If PyTorch was installed before CUDA drivers

Check if CUDA is available:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

If it shows `False`, reinstall PyTorch:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Reinstall with CUDA 12.6 support
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

### If nvidia-smi shows "No devices were found"

This means the driver loaded but can't find the GPU. Check error in dmesg:

```bash
sudo dmesg | grep "requires use of the NVIDIA open kernel modules"
```

If you see this error, you installed the wrong driver. Remove and reinstall:

```bash
sudo apt remove --purge nvidia-driver-580 nvidia-dkms-580
sudo apt install nvidia-driver-580-open
sudo reboot
```

### If nvidia-smi shows "couldn't communicate with the NVIDIA driver"

Check for Secure Boot blocking the driver:

```bash
sudo modprobe nvidia
```

If you see **"Key was rejected by service"**, Secure Boot is still enabled. Disable it in BIOS.

### If you get "CUDA out of memory" errors

```python
# Clear CUDA cache
import torch
torch.cuda.empty_cache()
```

### Check detailed GPU info

```bash
# System level
nvidia-smi -L
nvidia-smi -q

# PyTorch level
python -c "import torch; print(torch.cuda.get_device_properties(0))"
```

---

## Important Notes for RTX 5090

### Key Differences from Older GPUs

| Requirement | RTX 5090 | Older GPUs (30/40 series) |
|-------------|----------|---------------------------|
| Driver Type | **Open kernel modules** (`nvidia-driver-XXX-open`) | Proprietary driver works |
| Secure Boot | **Must be disabled** | Can work with MOK enrollment |
| Minimum Driver | 560+ (580+ recommended) | 525+ typically sufficient |
| CUDA Version | 12.6+ for PyTorch 2.7.1 | 11.8+ for most workloads |

### Why Open Kernel Modules?

The RTX 5090 **requires** NVIDIA's open-source kernel modules. The standard proprietary driver will load but fail with the error:

```
NVRM: installed in this system requires use of the NVIDIA open kernel modules.
```

Always use `nvidia-driver-XXX-open` packages for RTX 5090.

### Why Disable Secure Boot?

Even with the open driver, Secure Boot will reject unsigned kernel modules with:

```
modprobe: ERROR: could not insert 'nvidia': Key was rejected by service
```

You must disable Secure Boot in BIOS for the RTX 5090 to work.

---

## Summary of Versions

**Confirmed working configuration:**
- ✅ **OS:** Ubuntu 24.04 LTS
- ✅ **GPU:** NVIDIA RTX 5090
- ✅ **Driver:** nvidia-driver-580-open
- ✅ **Python:** 3.10
- ✅ **CUDA Toolkit:** 12.6 (via conda `cuda-toolkit` package from nvidia channel)
- ✅ **PyTorch:** 2.7.1+cu126
- ✅ **LeRobot:** Latest from GitHub

---

## Advantages of This Setup

1. **Environment Isolation:** CUDA toolkit is contained within conda environment
2. **Easy Management:** Multiple environments with different CUDA versions possible
3. **Clean Uninstallation:** Just delete the conda environment
4. **No System Conflicts:** No PATH/LD_LIBRARY_PATH configuration needed
5. **LeRobot Compatible:** Meets all LeRobot requirements (Python 3.10+, PyTorch 2.2+)

---

## Quick Reference Commands

### Activate Environment
```bash
conda activate lerobot
```

### Check Status
```bash
# GPU status
nvidia-smi

# PyTorch + CUDA status
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# LeRobot status
python -c "import lerobot; print('LeRobot OK')"
```

### Update PyTorch (future updates)
```bash
conda activate lerobot
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

---

## Additional Resources

- **LeRobot Documentation:** https://huggingface.co/docs/lerobot
- **LeRobot GitHub:** https://github.com/huggingface/lerobot
- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html
- **NVIDIA Driver Downloads:** https://www.nvidia.com/en-us/geforce/drivers/

---

## License

This guide is provided as-is for educational purposes. Always refer to official documentation for the most up-to-date information.

---

**Last Updated:** October 2025  
**Tested On:** Ubuntu 24.04 LTS with NVIDIA RTX 5090