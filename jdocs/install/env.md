# LeRobot Environment Setup for RTX 5090

**Problem:** LeRobot's upstream pyproject.toml has `torch>=2.2.1,<2.8.0` constraint, but RTX 5090 (Blackwell, sm_120) requires PyTorch 2.9.x+cu128.

---

## Solution: Relaxed pyproject.toml (RECOMMENDED)

We've modified `pyproject.toml` to relax version constraints:

```toml
# Changed from:
"torch>=2.2.1,<2.8.0"
"torchvision>=0.21.0,<0.23.0"

# To:
"torch>=2.2.1"
"torchvision>=0.21.0"
```

Now you can install any extras without torch being downgraded:

```bash
conda activate lerobot
pip install -e ".[smolvla]"  # Won't downgrade torch!
pip install -e ".[pi]"       # Safe
pip install -e ".[groot]"    # Safe
```

---

## Initial Setup (One-time)

### Step 1: Install correct torch for RTX 5090

```bash
conda activate lerobot
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### Step 2: Install flash-attn

```bash
# From cached wheel (fast)
pip install /home/jrobot/.cache/pip/wheels/f5/05/1e/a6726e9eee2e7ee6151dbfed113e89d220dd3964ba617ab32d/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl

# Or rebuild from source (takes ~60 min)
# TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install flash-attn --no-build-isolation
```

### Step 3: Install lerobot with extras

```bash
pip install -e ".[smolvla]"
```

### Step 4: Verification

Run this to verify environment is correct:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'sm_120 support: {\"sm_120\" in torch.cuda.get_arch_list()}')
try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except Exception as e:
    print(f'Flash Attention: ERROR - {e}')
"
```

Expected output:
```
PyTorch: 2.9.1+cu128
CUDA: 12.8
sm_120 support: True
Flash Attention: 2.8.3
```

---

## Quick Reference

### Working RTX 5090 Configuration

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10 | Required |
| torch | 2.9.1+cu128 | Has sm_120 support |
| torchvision | 0.24.1+cu128 | Matches torch |
| flash-attn | 2.8.3 | Built for sm_120 |
| CUDA (torch) | 12.8 | |

### Install Commands Cheatsheet

```bash
# With relaxed pyproject.toml, just install extras directly:
pip install -e ".[smolvla]"
pip install -e ".[pi]"
pip install -e ".[groot]"

# If torch gets messed up, fix with:
pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install /home/jrobot/.cache/pip/wheels/f5/05/1e/a6726e9eee2e7ee6151dbfed113e89d220dd3964ba617ab32d/flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl
```

---

## Reference

- See: `/home/jrobot/project/Isaac-GR00T/custom/jdocs/issue_solved/RTX_5090_GROOT_ENV_SETUP.md`
- Flash-attn cached wheel: `~/.cache/pip/wheels/f5/05/1e/.../flash_attn-2.8.3-cp310-cp310-linux_x86_64.whl`
