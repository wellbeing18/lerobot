# LeRobot Evaluation Script Setup Guide

This document summarizes the setup process and solutions for running LeRobot evaluation scripts, specifically for ALOHA simulation environments.

## Environment Setup

### 1. Conda Environment: `lerobot`

The project uses a conda environment with Python 3.10:

```bash
conda activate lerobot
```

**Important**: Always use `python` (not `python3`) when running scripts to ensure you're using the lerobot environment's Python interpreter.

### 2. PyTorch with CUDA Installation

The lerobot project requires:
- PyTorch: `>=2.2.1,<2.8.0` (currently using `2.7.1+cu126`)
- Torchvision: `>=0.21.0,<0.23.0` (currently using `0.22.1+cu126`)
- CUDA: 12.6

**Installation command** (if reinstall is needed):
```bash
conda run -n lerobot pip uninstall -y torch torchvision
conda run -n lerobot pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

**Verify installation**:
```bash
conda run -n lerobot python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.7.1+cu126
CUDA available: True
```

### 3. NVIDIA Driver Requirements

For CUDA 12.6 support, you need:
- NVIDIA driver version: 525 or newer (tested with 580.65.06)
- GPU properly loaded with kernel modules

**Verify GPU is detected**:
```bash
nvidia-smi
```

## Model Migration

### Issue: Missing Processor Files

Older pretrained models (like `lerobot/act_aloha_sim_transfer_cube_human`) are missing `policy_preprocessor.json` and `policy_postprocessor.json` files. These models were created before the processor pipeline refactoring.

**Error example**:
```
FileNotFoundError: Could not find 'policy_preprocessor.json' on the HuggingFace Hub at 'lerobot/act_aloha_sim_transfer_cube_human'
```

### Solution: Migrate Old Models

Use the migration script to convert old models to the new format:

```bash
python3 -m lerobot.processor.migrate_policy_normalization \
  --pretrained-path lerobot/act_aloha_sim_transfer_cube_human \
  --output-dir outputs/migrated/act_aloha_transfer_cube
```

This script:
1. Loads the pretrained policy model and configuration
2. Extracts normalization statistics from the model's state dict
3. Creates preprocessor and postprocessor pipelines
4. Removes normalization layers from the model
5. Saves the clean model with processor configs

## MuJoCo Rendering Backend Configuration

### Issue: Rendering Errors

MuJoCo requires a proper OpenGL backend for rendering. Different backends work in different environments:

**Common errors**:
- `gladLoadGL error` - OpenGL initialization failed
- `EGL: Failed to get EGL display` - EGL backend not available
- `GLX: No GLXFBConfigs returned` - GLFW backend issues

### Solution: Use EGL Backend

The EGL backend works best with NVIDIA GPU and drivers:

```bash
export MUJOCO_GL=egl
```

Or set it inline with the command:

```bash
MUJOCO_GL=egl python -m lerobot.scripts.lerobot_eval ...
```

**Alternative backends** (if EGL doesn't work):
- `MUJOCO_GL=osmesa` - Software rendering (slower, doesn't require GPU)
- `MUJOCO_GL=glfw` - For systems with display and proper OpenGL setup

**Permanent setup** (optional):
Add to `~/.bashrc`:
```bash
export MUJOCO_GL=egl
```

## Running Evaluation

### Working Command

```bash
MUJOCO_GL=egl python -m lerobot.scripts.lerobot_eval \
  --policy.path=/home/jrobot/project/lerobot/src/outputs/migrated/act_aloha_transfer_cube \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=10 \
  --eval.batch_size=2 \
  --output_dir=outputs/eval/act_aloha_transfer
```

### Key Points

1. **Use absolute paths** for local models: `/home/jrobot/project/lerobot/src/outputs/migrated/act_aloha_transfer_cube`
   - Relative paths like `outputs/migrated/...` will fail with HuggingFace Hub validation errors

2. **Run from project root**: `/home/jrobot/project/lerobot`

3. **GPU usage**: Add `--policy.device=cuda` to use GPU (defaults to CPU if not specified)

4. **Environment activated**: Ensure `(lerobot)` is shown in your prompt

## Troubleshooting

### CUDA Not Available

**Symptom**: `WARNING: Device 'cuda' is not available. Switching to 'cpu'.`

**Checks**:
1. Verify NVIDIA driver is loaded: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure you're in lerobot environment: `which python` should show `/home/jrobot/anaconda3/envs/lerobot/bin/python`

**Solution**: Reinstall PyTorch with CUDA support (see section 2 above)

### Path Errors

**Symptom**: `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**Solution**: Use absolute paths for local models, not relative paths

### Rendering Errors

**Symptom**: `gladLoadGL error`, `GLX: No GLXFBConfigs returned`, etc.

**Solution**: Try different MuJoCo backends in this order:
1. `MUJOCO_GL=egl` (best for GPU)
2. `MUJOCO_GL=osmesa` (software rendering)
3. Install missing packages: `sudo apt-get install -y libosmesa6-dev freeglut3-dev libglew-dev`

## Dependencies Reference

From `pyproject.toml`:
- Core: `torch>=2.2.1,<2.8.0`, `torchvision>=0.21.0,<0.23.0`
- Simulation: `gym-aloha>=0.1.1`
- MuJoCo: `mujoco==2.3.7`
- Python: `>=3.10`

## Summary

The key issues resolved:
1. ✅ PyTorch CUDA installation via pip with correct index URL
2. ✅ Model migration from old format to new processor pipeline format
3. ✅ MuJoCo rendering backend configuration (EGL)
4. ✅ Proper use of absolute paths for local models
5. ✅ Environment activation and Python interpreter selection

The evaluation script should now run successfully with GPU acceleration and proper rendering.
